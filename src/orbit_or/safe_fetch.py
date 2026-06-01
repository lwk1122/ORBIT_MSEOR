"""Safe content fetching for whitelisted domains.

Two-layer defense:
  Layer 1 — is_fetch_allowed(): business policy (whitelist + sanitization), no network I/O.
  Layer 2 — safe_fetch_content(): safehttpx pins verified public IPs at the TCP socket
             layer, eliminating parser-differential and DNS-rebinding SSRF attacks.

Both layers use httpx.URL as the single URL parser to avoid parser differentials.
"""

import json
import logging

import httpx
import safehttpx
import trafilatura

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fetch whitelist — MUCH stricter than scoring whitelist.
# Only domains with editorial control where we trust the content.
# NO wildcards (.edu, .gov, etc.) — explicit domains only.
# ---------------------------------------------------------------------------

FETCH_WHITELIST: frozenset[str] = frozenset(
    {
        # Wire services / major press
        "reuters.com",
        "apnews.com",
        "bbc.com",
        # Top journals (peer-reviewed, editorial control)
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "pnas.org",
        "cell.com",
        "link.springer.com",
        "onlinelibrary.wiley.com",
        "dl.acm.org",
        "academic.oup.com",
        "ieee.org",
        # Preprints (user-uploaded but standardized format, low injection risk)
        "arxiv.org",
        # Government — specific agencies only
        "www.bls.gov",
        "www.census.gov",
        "federalreserve.gov",
        "stlouisfed.org",
        "www.dallasfed.org",
        "www.frbsf.org",
        "www.newyorkfed.org",
        "www.chicagofed.org",
        "imf.org",
        "worldbank.org",
        "bis.org",
        "ecb.europa.eu",
        "nber.org",
        # AI research labs (corporate editorial control)
        "research.google",
        "deepmind.google",
        "ai.meta.com",
        "openai.com",
        "anthropic.com",
        "mistral.ai",
        "allenai.org",
        "research.ibm.com",
        "machinelearning.apple.com",
        # ML platforms (structured format, low injection risk)
        "paperswithcode.com",
        "mlcommons.org",
        "lmsys.org",
        # Semiconductor
        "semiconductors.org",
        # Conference proceedings
        "proceedings.neurips.cc",
        "proceedings.mlr.press",
        "aclanthology.org",
        # Standards bodies
        "ietf.org",
        "jedec.org",
        # NVIDIA docs
        "developer.nvidia.com",
        "docs.nvidia.com",
    }
)

MAX_RESPONSE_BYTES = 512 * 1024  # 512 KB
MAX_REDIRECTS = 3
FETCH_TIMEOUT = 15.0  # seconds
USER_AGENT = (
    "Mozilla/5.0 (compatible; ORBITBot/1.0; "
    "+https://github.com/orbit-or/orbit-or)"
)


# ---------------------------------------------------------------------------
# Layer 1: Business policy — pure string checks, no network I/O
# ---------------------------------------------------------------------------


def is_fetch_allowed(url: str) -> bool:
    """Business policy: is this domain on our whitelist?

    No network I/O. SSRF defense is handled by Layer 2 (safehttpx) in
    safe_fetch_content().
    """
    if not url:
        return False

    # Block all ASCII control characters and backslashes (parser differential vectors)
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in url) or "\\" in url:
        return False

    # Block percent-encoded null bytes — libraries may decode and truncate
    if "%00" in url.lower():
        return False

    # Unified parser: httpx.URL (same parser Layer 2 uses, no differential)
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.host or ""
    if not hostname:
        return False

    # Reject non-ASCII hostnames (IDN homoglyph defense-in-depth)
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError:
        return False

    # Block non-standard ports (whitelist trusts editorial content on 80/443 only)
    if parsed.port is not None and parsed.port not in (80, 443):
        return False

    # Block credentials in URL (userinfo@host)
    if parsed.userinfo:
        return False

    # Whitelist check — exact match, then strip www. (removeprefix, NOT lstrip)
    normalized = hostname.removeprefix("www.")
    if normalized not in FETCH_WHITELIST and hostname not in FETCH_WHITELIST:
        return False

    return True


# ---------------------------------------------------------------------------
# Layer 2: safehttpx — socket-layer SSRF defense
# ---------------------------------------------------------------------------


async def safe_fetch_content(
    url: str,
    *,
    max_chars: int = 3000,
) -> dict | None:
    """Fetch URL content safely. Returns dict with title/description/content, or None.

    Each redirect hop: safehttpx resolves DNS → verifies public IP → pins the
    verified IP into an AsyncSecureTransport so TCP connects only to that IP.
    Redirect targets are also checked against the business whitelist (Layer 1).
    """
    if not is_fetch_allowed(url):
        logger.debug("[safe_fetch] URL not on fetch whitelist: %s", url[:80])
        return None

    try:
        current_url = url
        raw_html = None

        for _hop in range(MAX_REDIRECTS + 1):
            parsed_url = httpx.URL(current_url)
            hostname = parsed_url.host
            if not hostname:
                return None

            # safehttpx: DNS resolve → verify public IP → pin in transport
            try:
                verified_ip = await safehttpx.async_validate_url(hostname)
            except (ValueError, OSError) as exc:
                logger.warning(
                    "[safe_fetch] SSRF/DNS blocked: %s (%s)", current_url[:80], exc
                )
                return None

            # Belt-and-suspenders: double-check the IP safehttpx returned
            if not safehttpx.is_public_ip(verified_ip):
                logger.warning(
                    "[safe_fetch] IP validation mismatch: %s → %s",
                    hostname,
                    verified_ip,
                )
                return None

            transport = safehttpx.AsyncSecureTransport(verified_ip)
            async with httpx.AsyncClient(
                transport=transport,
                timeout=FETCH_TIMEOUT,
                headers={"User-Agent": USER_AGENT},
            ) as client:
                async with client.stream(
                    "GET", current_url, follow_redirects=False
                ) as resp:
                    # --- Redirect handling (headers available, body not yet read) ---
                    if resp.status_code in (301, 302, 303, 307, 308):
                        location = resp.headers.get("location", "")
                        if not location:
                            return None
                        try:
                            location = str(parsed_url.join(location))
                            redirect_url = httpx.URL(location)
                        except httpx.InvalidURL:
                            logger.warning(
                                "[safe_fetch] Invalid redirect URL from %s",
                                current_url[:60],
                            )
                            return None
                        # Block HTTPS→HTTP downgrade
                        if (
                            parsed_url.scheme == "https"
                            and redirect_url.scheme == "http"
                        ):
                            logger.warning(
                                "[safe_fetch] HTTPS downgrade blocked: %s → %s",
                                current_url[:60],
                                location[:60],
                            )
                            return None
                        # Layer 1: whitelist gate on redirect target
                        if not is_fetch_allowed(location):
                            logger.warning(
                                "[safe_fetch] Redirect left whitelist: %s → %s",
                                current_url[:60],
                                location[:60],
                            )
                            return None
                        current_url = location
                        continue

                    if resp.status_code != 200:
                        return None

                    # --- Content-Type gate (skip non-HTML) ---
                    ct = (resp.headers.get("content-type") or "").lower()
                    if ct and not any(
                        t in ct
                        for t in ("text/html", "text/plain", "application/xhtml")
                    ):
                        return None

                    # --- Content-Length early reject ---
                    content_length = resp.headers.get("content-length")
                    if content_length:
                        try:
                            if int(content_length) > MAX_RESPONSE_BYTES:
                                return None
                        except ValueError:
                            return None

                    # --- Stream body with hard byte cap ---
                    chunks: list[bytes] = []
                    total = 0
                    over_limit = False
                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                        total += len(chunk)
                        if total > MAX_RESPONSE_BYTES:
                            over_limit = True
                            break
                        chunks.append(chunk)
                    if over_limit:
                        return None
                    raw_html = b"".join(chunks).decode(
                        resp.encoding or "utf-8", errors="replace"
                    )
                    break
        else:
            logger.warning("[safe_fetch] Too many redirects for %s", url[:80])
            return None

    except httpx.TimeoutException:
        logger.warning("[safe_fetch] Timeout fetching %s", url[:80])
        return None
    except (httpx.HTTPError, httpx.InvalidURL, OSError) as exc:
        logger.warning("[safe_fetch] Fetch error: %s: %s", url[:80], exc)
        return None
    except Exception as exc:
        logger.error(
            "[safe_fetch] Unexpected error: %s: %s", url[:80], exc, exc_info=True
        )
        return None

    # Extract content via trafilatura
    try:
        result = trafilatura.extract(
            raw_html,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=True,
        )
        if not result:
            metadata = trafilatura.extract_metadata(raw_html)
            if metadata and metadata.description:
                return {
                    "title": metadata.title or "",
                    "description": metadata.description or "",
                    "content": metadata.description or "",
                    "url": url,
                }
            return None

        parsed = json.loads(result)
        content = parsed.get("text", "") or ""
        title = parsed.get("title", "") or ""
        description = parsed.get("description", "") or ""

        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        return {
            "title": title,
            "description": description,
            "content": content,
            "url": url,
        }

    except Exception as exc:
        logger.warning("[safe_fetch] Extraction error for %s: %s", url[:80], exc)
        return None
