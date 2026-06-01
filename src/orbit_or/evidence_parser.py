"""W→F Evidence Pipeline: extract structured data from web evidence into the Ledger."""

import json
import logging
import re
import sqlite3

from . import api
from . import db
from . import ledger as _ledger
from . import topic_config
from .broker import call_text
from .safe_fetch import is_fetch_allowed, safe_fetch_content

logger = logging.getLogger(__name__)


def _fact_provider(topic_id: int) -> str:
    try:
        return topic_config.get_provider_profile_for(topic_id, "fact_provider")
    except sqlite3.OperationalError:
        raise
    except Exception as exc:
        logger.debug("[evidence] Fact provider lookup failed: %s", exc)
        return "minimax"

# ---------------------------------------------------------------------------
# Domain scoring
# ---------------------------------------------------------------------------

DOMAIN_TIERS: dict[str, float] = {
    # Tier 1: Wire services + financial data (1.0)
    "reuters.com": 1.0,
    "apnews.com": 1.0,
    "bloomberg.com": 1.0,
    # Tier 2: Major financial press (0.9)
    "ft.com": 0.9,
    "wsj.com": 0.9,
    "economist.com": 0.9,
    "nytimes.com": 0.9,
    # Tier 3: Specialized financial (0.8)
    "cnbc.com": 0.8,
    "marketwatch.com": 0.8,
    "investing.com": 0.8,
    # Tier 4: Government/institutional (0.85)
    "imf.org": 0.85,
    "worldbank.org": 0.85,
    "federalreserve.gov": 0.85,
    "pbc.gov.cn": 0.85,
    # Tier 5: General news (0.7)
    "bbc.com": 0.7,
    "cnn.com": 0.7,
    # Tier 6: Mixed (0.5)
    "seekingalpha.com": 0.5,
    # Tier 7: Low (0.2)
    "reddit.com": 0.2,
    "twitter.com": 0.2,
    "x.com": 0.2,
    # Academic peer-reviewed (0.85-0.90)
    "nature.com": 0.90,
    "sciencedirect.com": 0.90,
    "ieee.org": 0.85,
    "dl.acm.org": 0.85,
    "link.springer.com": 0.85,
    "onlinelibrary.wiley.com": 0.85,
    "siam.org": 0.85,
    "epubs.siam.org": 0.85,
    "link.aps.org": 0.85,
    "journals.aps.org": 0.85,
    "pnas.org": 0.90,
    # Top venues (0.80)
    "neurips.cc": 0.80,
    "openreview.net": 0.80,
    "aclanthology.org": 0.80,
    "usenix.org": 0.80,
    "pmc.ncbi.nlm.nih.gov": 0.80,
    "jmlr.org": 0.80,
    # Preprints / academic aggregators (0.65-0.70)
    "arxiv.org": 0.65,
    "biorxiv.org": 0.65,
    "medrxiv.org": 0.65,
    "techrxiv.org": 0.65,
    "researchgate.net": 0.65,
    "www.researchgate.net": 0.65,
    "semanticscholar.org": 0.70,
    "www.semanticscholar.org": 0.70,
    # University repos (0.65)
    "tuprints.ulb.tu-darmstadt.de": 0.65,
    # Tech press top (0.85)
    "spectrum.ieee.org": 0.85,
    "technologyreview.com": 0.85,
    "eetimes.com": 0.85,
    # Tech press mid (0.70-0.75)
    "wired.com": 0.70,
    "techcrunch.com": 0.70,
    "venturebeat.com": 0.70,
    "arstechnica.com": 0.70,
    "theverge.com": 0.70,
    # Big tech research (0.85)
    "research.google": 0.85,
    "deepmind.google": 0.85,
    "ai.meta.com": 0.85,
    "openai.com": 0.85,
    # Big tech docs (0.75-0.80)
    "developer.nvidia.com": 0.80,
    "learn.microsoft.com": 0.75,
    "cloud.google.com": 0.75,
    # Data providers (0.80-0.85)
    "gartner.com": 0.85,
    "idc.com": 0.85,
    "trendforce.com": 0.80,
    "pewresearch.org": 0.80,
    # ML platforms / benchmark aggregators (0.70-0.80)
    "huggingface.co": 0.75,
    "llm-stats.com": 0.70,
    "artificialanalysis.ai": 0.70,
    "lmsys.org": 0.80,
    "paperswithcode.com": 0.80,
    "mlcommons.org": 0.90,
    "opencompass.org.cn": 0.75,
    # Conference proceedings (0.80)
    "proceedings.neurips.cc": 0.80,
    "proceedings.mlr.press": 0.80,
    "iclr.cc": 0.80,
    "thecvf.com": 0.80,
    "aaai.org": 0.80,
    # Academic publishers — additions (0.70-0.95)
    "science.org": 0.95,
    "cell.com": 0.85,
    "academic.oup.com": 0.85,
    "www.mdpi.com": 0.70,
    "europepmc.org": 0.75,
    "medinform.jmir.org": 0.75,
    # AI research labs (0.80-0.90)
    "anthropic.com": 0.90,
    "mistral.ai": 0.85,
    "allenai.org": 0.80,
    "research.ibm.com": 0.85,
    # Big tech research — additions (0.85)
    "machinelearning.apple.com": 0.85,
    # NVIDIA docs (0.80)
    "docs.api.nvidia.com": 0.80,
    "docs.nvidia.com": 0.80,
    "build.nvidia.com": 0.80,
    # Semiconductor industry (0.75-0.85)
    "semiconductors.org": 0.85,
    "techinsights.com": 0.80,
    "digitimes.com": 0.75,
    "nextplatform.com": 0.75,
    # Quantitative finance / central banks (0.85-0.90)
    "stlouisfed.org": 0.90,
    "www.dallasfed.org": 0.85,
    "www.frbsf.org": 0.85,
    "www.newyorkfed.org": 0.85,
    "www.chicagofed.org": 0.85,
    "bis.org": 0.90,
    "ecb.europa.eu": 0.90,
    "nber.org": 0.85,
    "www.weforum.org": 0.75,
    # Standards bodies (0.85)
    "jedec.org": 0.85,
    "pcisig.com": 0.85,
    "ietf.org": 0.85,
    # Preprints — finance/economics (0.65)
    "ssrn.com": 0.65,
    # Reference (0.55-0.60)
    "en.wikipedia.org": 0.60,
    "wikipedia.org": 0.55,
    # Code (0.60)
    "github.com": 0.60,
}
UNKNOWN_DOMAIN_SCORE = 0.4
DOMAIN_SCORE_THRESHOLD = 0.6


def score_domain(domain: str) -> float:
    normalized = (domain or "").lower().removeprefix("www.")
    score = DOMAIN_TIERS.get(normalized)
    if score is not None:
        return score
    # Wildcard TLD matching
    if normalized.endswith(".gov"):
        return 0.80
    if normalized.endswith(".edu"):
        return 0.65
    if normalized.endswith(".ac.uk"):
        return 0.65
    if normalized.endswith(".mil"):
        return 0.75
    return UNKNOWN_DOMAIN_SCORE


_STOP_WORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "about",
        "which",
        "their",
        "there",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "more",
        "than",
        "into",
        "also",
        "some",
        "when",
        "what",
        "they",
        "does",
        "other",
        "each",
        "most",
        "very",
        "over",
        "such",
        "only",
        "these",
    }
)


def check_topic_relevance(title: str, snippet: str, topic_keywords: list[str]) -> float:
    """Return 1.0 if strongly relevant (2+ hits), 0.8 if likely (1 hit), 0.5 if uncertain."""
    text = (title + " " + snippet).lower()
    words = set(re.findall(r"\b\w+\b", text))
    hits = sum(1 for kw in topic_keywords if kw in words)
    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.8
    return 0.5  # uncertain, don't hard-block


# ---------------------------------------------------------------------------
# Evidence extraction prompt
# ---------------------------------------------------------------------------


def build_evidence_extraction_prompt(
    topic_id: int,
    web_row: dict,
    entities: list[tuple[int, str]],
    attributes: list[tuple[int, str]],
) -> str | None:
    """Build an extraction prompt for a web evidence row.

    Returns None if there are no entities AND no attributes to reference.
    """
    if not entities and not attributes:
        return None

    domain = web_row.get("source_domain", "")
    domain_score = score_domain(domain)

    # Entity list
    entity_lines = ["0. [UNSPECIFIED] — choose if unclear"]
    for eid, name in entities:
        entity_lines.append(f"{eid}. {name}")
    entity_lines.append("NEW. [NEW] (name: ___)")
    entity_block = "\n".join(entity_lines)

    # Attribute list with STRICT definitions
    from .server import _ATTR_STRICT_DEFS

    attr_lines = ["0. [UNSPECIFIED]"]
    for aid, name in attributes:
        strict = _ATTR_STRICT_DEFS.get(name, "")
        suffix = f" (STRICT: {strict})" if strict else ""
        attr_lines.append(f"{aid}. {name}{suffix}")
    attr_lines.append('NEW. Use "NEW:Metric Name"')
    attr_block = "\n".join(attr_lines)

    query_text = web_row.get("query_text", "")
    title = web_row.get("title", "") or ""
    snippet = web_row.get("snippet", "") or ""
    url = web_row.get("url", "") or ""
    web_id = web_row.get("id", 0)

    from .server import _LEDGER_JSON_SCHEMA

    return (
        f"Extract STRUCTURED DATA from this web search result.\n"
        f"The search query provides context for what was being researched.\n\n"
        f'Search query: "{query_text}"\n'
        f'Title: "{title}"\n'
        f"URL: {url}\n"
        f'Snippet: "{snippet[:2000]}"\n'
        f"Source: {domain} (score: {domain_score})\n"
        f"Default source citation: W{web_id}\n\n"
        f"Entities (pick # or NEW):\n{entity_block}\n\n"
        f"Attributes (pick # or NEW):\n{attr_block}\n\n" + _LEDGER_JSON_SCHEMA
    )


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------


async def extract_evidence_to_ledger(
    topic_id: int,
    subtopic_id: int | None,
    web_row: dict,
    *,
    current_round: int = 0,
) -> list[dict]:
    """Extract structured data from a single web evidence row into the Ledger.

    Returns list of result dicts for logging.
    """
    web_id = web_row.get("id")
    if web_id is None:
        return []
    domain = web_row.get("source_domain", "")
    domain_sc = score_domain(domain)

    # Filter low-quality domains
    if domain_sc < DOMAIN_SCORE_THRESHOLD:
        api.mark_web_evidence_ledger_processed([web_id])
        logger.debug(
            "[evidence] Skipped W%s from %s (score=%.1f < threshold)",
            web_id,
            domain,
            domain_sc,
        )
        return []

    # Fetch entity/attribute lists
    entities = _ledger.get_entity_numbered_list(topic_id, current_round)
    attributes = _ledger.get_attribute_numbered_list(topic_id)

    prompt = build_evidence_extraction_prompt(topic_id, web_row, entities, attributes)
    if prompt is None:
        api.mark_web_evidence_ledger_processed([web_id])
        return []

    provider = _fact_provider(topic_id)
    try:
        resp = await call_text(
            prompt,
            provider=provider,
            strategy="direct",
            allow_web=False,
            system_instruction="You are a data extraction clerk. Extract structured data points from web search results.",
            fallback_role="fact_proposer",
        )
    except Exception as exc:
        logger.warning("[evidence] LLM call failed for W%s: %s", web_id, exc)
        api.mark_web_evidence_ledger_processed([web_id])
        return []

    if not resp:
        api.mark_web_evidence_ledger_processed([web_id])
        return []

    from .server import parse_clerk_ledger_output  # lazy to avoid circular deps

    parsed = parse_clerk_ledger_output(
        resp,
        entities,
        attributes,
        topic_id,
        subtopic_id,
        current_round,
        "evidence_parser",
    )

    results = []
    for entry in parsed:
        if entry.get("type") == "qualitative":
            logger.info(
                "[evidence] Qualitative from W%s: %s",
                web_id,
                entry.get("text", "")[:80],
            )
            results.append(entry)
            continue
        if entry.get("type") == "structured":
            try:
                lid, status = _ledger.normalize_and_upsert(
                    topic_id=entry["topic_id"],
                    subtopic_id=entry.get("subtopic_id"),
                    entity_id=entry.get("entity_id"),
                    attribute_id=entry.get("attribute_id"),
                    raw_value=entry.get("raw_value", ""),
                    raw_timeframe=entry.get("raw_timeframe"),
                    entry_type="web_evidence",
                    source_ref=f"[W{web_id}]",
                    source_domain=domain,
                    domain_score=domain_sc,
                    created_by="evidence_parser",
                    current_round=current_round,
                    valid_from=entry.get("valid_from"),
                    valid_to=entry.get("valid_to"),
                    min_val=entry.get("min_val"),
                    max_val=entry.get("max_val"),
                    unit=entry.get("unit"),
                    stat_type=entry.get("stat_type"),
                    value_mean=entry.get("value_mean"),
                    value_std=entry.get("value_std"),
                    value_p=entry.get("value_p"),
                    value_n=entry.get("value_n"),
                    value_ci_lower=entry.get("value_ci_lower"),
                    value_ci_upper=entry.get("value_ci_upper"),
                    value_ci_level=entry.get("value_ci_level"),
                    baseline_entity_id=entry.get("baseline_entity_id"),
                    split=entry.get("split"),
                    config_json=entry.get("config_json"),
                )
                results.append({"ledger_id": lid, "status": status, **entry})
            except Exception as exc:
                logger.warning(
                    "[evidence] Ledger upsert failed for W%s: %s", web_id, exc
                )

    api.mark_web_evidence_ledger_processed([web_id])
    logger.info(
        "[evidence] Extracted %d entries from W%s (%s, score=%.1f)",
        len(results),
        web_id,
        domain,
        domain_sc,
    )
    return results


# ---------------------------------------------------------------------------
# Pending promotion
# ---------------------------------------------------------------------------


def try_promote_pending_entries(topic_id: int, current_round: int) -> int:
    """Re-attempt resolution for pending entries. Returns count promoted."""
    pending = db.get_active_ledger_pending(topic_id, current_round)
    promoted = 0

    for entry in pending:
        raw_text = entry.get("raw_text", "")
        source_ref = entry.get("source_ref", "")
        pending_id = entry["id"]

        # Resolve both fields — skip if either is still missing to avoid
        # cascading duplicate pending entries in normalize_and_upsert.
        entity_id = _ledger.resolve_entity(
            raw_text, topic_id, round_number=current_round
        )
        attribute_id = _ledger.resolve_attribute(raw_text, topic_id)
        if entity_id is None or attribute_id is None:
            continue

        try:
            lid, status = _ledger.normalize_and_upsert(
                topic_id=topic_id,
                subtopic_id=entry.get("subtopic_id"),
                entity_id=entity_id,
                attribute_id=attribute_id,
                raw_value=raw_text,
                raw_timeframe=None,
                entry_type="promoted_pending",
                source_ref=source_ref or "",
                created_by="evidence_parser",
                current_round=current_round,
            )
            if status != "pending":
                deleted = db.delete_ledger_pending(pending_id)
                if deleted:
                    promoted += 1
                logger.info(
                    "[evidence] Promoted pending %d → ledger %s (%s)",
                    pending_id,
                    lid,
                    status,
                )
        except Exception as exc:
            logger.warning(
                "[evidence] Pending promotion failed for %d: %s", pending_id, exc
            )

    # Cleanup expired
    expired = db.expire_ledger_pending(current_round)
    if expired:
        logger.info(
            "[evidence] Expired %d pending entries at round %d", expired, current_round
        )

    return promoted


# ---------------------------------------------------------------------------
# Pass 2: Fact extraction from WebEvidence (Wikidata-style structured facts)
# ---------------------------------------------------------------------------

_FACT_EXTRACTION_PROMPT_TEMPLATE = (
    "Extract ATOMIC factual statements from this web search result.\n"
    "Each fact MUST be atomic — one statement per fact. Never combine with 'and', 'but', 'however'.\n\n"
    "=== QUALITY RUBRIC ===\n"
    "Extract ONLY statements that are at least one of:\n"
    "- QUANTITATIVE: contains a specific number, percentage, dollar amount, or measurement\n"
    "- TEMPORAL: anchored to a specific date, quarter, year, or date range\n"
    "- COMPARATIVE: explicit comparison (X > Y, X grew by Z%, X outperforms Y)\n"
    "- CAUSAL: states a cause-effect relationship with specifics\n\n"
    "DO NOT extract:\n"
    "- Metadata about data sources ('BLS produces employment statistics')\n"
    "- Vague descriptions ('AI will have significant impact on the economy')\n"
    "- Methodology descriptions ('is measured in basis points')\n"
    "- Source descriptions ('combines multiple data sources')\n"
    "- Release announcements without data ('report was released in March')\n\n"
    "NEGATIVE EXAMPLES (do NOT extract these):\n"
    '- "The Bureau of Labor Statistics produces monthly employment data" → metadata\n'
    '- "Inflation is measured in percentage changes of CPI" → methodology\n'
    '- "The labor market is described as resilient" → vague, no numbers\n'
    '- "The methodology combines survey and administrative data" → methodology\n'
    '- "The quarterly report was released in January 2025" → release, no data\n\n'
    "POSITIVE EXAMPLES (DO extract these):\n"
    '- "Automation could displace 1.5% of the global workforce annually by 2035" → quantitative + temporal\n'
    '- "AI adoption in financial services reached 23% in Q4 2024" → quantitative + temporal\n'
    '- "Routine cognitive tasks are more automatable than routine manual tasks" → comparative\n\n'
    "ENTITY NAMING RULES:\n"
    "- ALWAYS use the full official name for subject_entity and claimed_by.\n"
    "- NEVER use abbreviations or acronyms alone.\n"
    "- Examples: 'People's Bank of China' NOT 'PBOC', 'Federal Reserve' NOT 'Fed', "
    "'International Monetary Fund' NOT 'IMF', 'Bank of Japan' NOT 'BOJ'.\n"
    "- If the snippet only contains an acronym, expand it to the full name.\n\n"
    "TIME FIELD RULES:\n"
    "- Use the most specific format possible: 2035 | Q4 2026 | H1 2025 | Jan 2024 | 2024-2025\n"
    "- Accept year-only (2035), quarter (Q4 2024), half (H1 2025), month-year (Jan 2024), ranges (2024-2025)\n"
    "- Use NONE only for truly timeless facts (e.g., physical constants, definitions)\n"
    "- If the URL path or title contains a year/date, use it as a time qualifier\n\n"
    'Search query: "{query}"\n'
    'Title: "{title}"\n'
    "URL: {url}\n"
    'Snippet: "{snippet}"\n'
    "Source: {domain}\n"
    "{extra_content_block}\n"
    "For each fact, output a JSON object with this schema:\n"
    '{{"proposition": {{"subject_entity": "...", "predicate": "...", '
    '"object": {{"type": "string|quantity|boolean", "value": "..."}}}}, '
    '"qualifiers": [{{"key": "...", "value": "..."}}], '
    '"attribution": {{"claimed_by": "...", "claim_act": "..."}}, '
    '"source_refs": ["W{web_id}"], '
    '"raw_text": "original sentence"}}\n\n'
    "Reply with strict JSON: "
    '{{"facts": [<list of fact objects>]}}\n'
    'If no atomic facts: {{"facts": []}}'
)

_MAX_FACT_EXTRACTION_RETRIES = 2
_MAX_FACTS_PER_EVIDENCE = 5


async def extract_facts_from_evidence(
    topic_id: int,
    subtopic_id: int | None,
    web_row: dict,
) -> list[int]:
    """Pass 2: Extract atomic structured facts from a WebEvidence row.

    Creates FactCandidates for librarian review. Returns list of candidate IDs.
    """
    from .canonical import (
        validate_structured_fact,
        structured_fact_to_columns,
        snap_subject,
        build_canonical_text,
    )
    from .json_utils import extract_json_object

    web_id = web_row.get("id")
    if web_id is None:
        return []

    domain = web_row.get("source_domain", "")
    domain_sc = score_domain(domain)
    if domain_sc < DOMAIN_SCORE_THRESHOLD:
        return []

    query = web_row.get("query_text", "")
    title = web_row.get("title", "") or ""
    snippet = web_row.get("snippet", "") or ""
    url = web_row.get("url", "") or ""

    if not snippet.strip():
        return []

    # Try safe_fetch for enrichment
    extra_content_block = ""
    if url and is_fetch_allowed(url):
        try:
            fetched = await safe_fetch_content(url, max_chars=2000)
            if fetched and fetched.get("content"):
                extra_content_block = f"\nFull page content:\n{fetched['content']}\n\n"
        except Exception:
            pass

    prompt = _FACT_EXTRACTION_PROMPT_TEMPLATE.format(
        query=query,
        title=title,
        url=url,
        snippet=snippet[:2000],
        domain=domain,
        web_id=web_id,
        extra_content_block=extra_content_block,
    )

    provider = _fact_provider(topic_id)
    resp = None
    for attempt in range(_MAX_FACT_EXTRACTION_RETRIES + 1):
        try:
            resp = await call_text(
                prompt,
                provider=provider,
                strategy="direct",
                allow_web=False,
                system_instruction="You are a data extraction clerk. Extract atomic factual statements from web evidence.",
                fallback_role="fact_proposer",
            )
        except Exception as exc:
            logger.warning(
                "[evidence-facts] LLM call failed for W%s attempt %d: %s",
                web_id,
                attempt,
                exc,
            )
            continue
        if resp:
            break

    if not resp:
        return []

    parsed = extract_json_object(resp)
    if not isinstance(parsed, dict):
        return []

    facts_raw = parsed.get("facts", [])
    if not isinstance(facts_raw, list):
        return []

    created_ids: list[int] = []
    for fact_data in facts_raw[:_MAX_FACTS_PER_EVIDENCE]:
        if not isinstance(fact_data, dict):
            continue

        validated, err = validate_structured_fact(fact_data)
        if validated is None:
            logger.debug("[evidence-facts] Validation failed for W%s: %s", web_id, err)
            continue

        # Subject snapping
        snapped_subject = snap_subject(validated.proposition.subject_entity, topic_id)

        source_refs = validated.source_refs or [f"W{web_id}"]
        cols = structured_fact_to_columns(validated)
        canonical_text = build_canonical_text(
            snapped_subject,
            validated.proposition.predicate,
            cols.get("object_json"),
            cols.get("qualifiers_json"),
            cols.get("attribution_json"),
            source_refs,
        )

        if not canonical_text.strip():
            continue

        # Dedup check
        if api.fact_candidate_exists(topic_id, canonical_text, statuses=("pending",)):
            continue
        if api.fact_exists(topic_id, canonical_text):
            continue

        candidate_id = api.create_fact_candidate_with_stage(
            topic_id,
            subtopic_id,
            None,  # writer_msg_id
            canonical_text,
            fact_stage="web_extracted",
            candidate_type="sourced_claim",
            source_kind="web",
            source_refs_json=json.dumps(source_refs, ensure_ascii=False),
            source_excerpt=validated.raw_text[:500] if validated.raw_text else None,
            subject=snapped_subject,
            predicate=validated.proposition.predicate,
            object_json=cols.get("object_json"),
            qualifiers_json=cols.get("qualifiers_json"),
            attribution_json=cols.get("attribution_json"),
        )
        created_ids.append(candidate_id)
        logger.info(
            "[evidence-facts] Created FactCandidate %s from W%s: %s",
            candidate_id,
            web_id,
            canonical_text[:80],
        )

    return created_ids


# ---------------------------------------------------------------------------
# Unified extraction: single LLM call for both Ledger + Facts
# ---------------------------------------------------------------------------

_UNIFIED_EXTRACTION_PROMPT_TEMPLATE = (
    "Extract ALL structured data from this web search result in a SINGLE pass.\n"
    "Return a JSON object with two sections: ledger_entries and facts.\n\n"
    "=== QUALITY RUBRIC (applies to FACTS section) ===\n"
    "Extract ONLY statements that are at least one of:\n"
    "- QUANTITATIVE: contains a specific number, percentage, dollar amount, or measurement\n"
    "- TEMPORAL: anchored to a specific date, quarter, year, or date range\n"
    "- COMPARATIVE: explicit comparison (X > Y, X grew by Z%, X outperforms Y)\n"
    "- CAUSAL: states a cause-effect relationship with specifics\n\n"
    "DO NOT extract:\n"
    "- Metadata about data sources ('BLS produces employment statistics')\n"
    "- Vague descriptions ('AI will have significant impact on the economy')\n"
    "- Methodology descriptions ('is measured in basis points')\n"
    "- Source descriptions ('combines multiple data sources')\n"
    "- Release announcements without data ('report was released in March')\n\n"
    "NEGATIVE EXAMPLES (do NOT extract these):\n"
    '- "The Bureau of Labor Statistics produces monthly employment data" → metadata\n'
    '- "Inflation is measured in percentage changes of CPI" → methodology\n'
    '- "The labor market is described as resilient" → vague, no numbers\n'
    '- "The methodology combines survey and administrative data" → methodology\n'
    '- "The quarterly report was released in January 2025" → release, no data\n\n'
    "POSITIVE EXAMPLES (DO extract these):\n"
    '- "Automation could displace 1.5% of the global workforce annually by 2035" → quantitative + temporal\n'
    '- "AI adoption in financial services reached 23% in Q4 2024" → quantitative + temporal\n'
    '- "Routine cognitive tasks are more automatable than routine manual tasks" → comparative\n\n'
    "ENTITY NAMING RULES:\n"
    "- ALWAYS use the full official name for subjects and entities.\n"
    "- NEVER use abbreviations or acronyms alone.\n"
    "- Examples: 'People's Bank of China' NOT 'PBOC', 'Federal Reserve' NOT 'Fed', "
    "'International Monetary Fund' NOT 'IMF', 'Bank of Japan' NOT 'BOJ'.\n"
    "- If the snippet only contains an acronym, expand it to the full name.\n\n"
    "TIME FIELD RULES:\n"
    "- Use the most specific format possible: 2035 | Q4 2026 | H1 2025 | Jan 2024 | 2024-2025\n"
    "- Accept year-only (2035), quarter (Q4 2024), half (H1 2025), month-year (Jan 2024), ranges (2024-2025)\n"
    "- Use NONE only for truly timeless facts (e.g., physical constants, definitions)\n"
    "- If the URL path or title contains a year/date, use it as a time qualifier\n\n"
    'Search query: "{query}"\n'
    'Title: "{title}"\n'
    "URL: {url}\n"
    'Snippet: "{snippet}"\n'
    "Source: {domain} (score: {domain_score})\n"
    "{extra_content_block}\n"
    "=== SECTION 1: LEDGER ENTRIES (numerical data points) ===\n"
    "Entities (pick #):\n{entity_block}\n\n"
    "Attributes (pick #):\n{attr_block}\n\n"
    "Units (base only): [USD, EUR, CNY, FLOPS, SECONDS, WATTS, NONE, OTHER]\n\n"
    "STRICT NUMERIC RULES:\n"
    "- MIN and MAX must be scientific notation: X.XXXXeN (e.g., 2.6970e10)\n"
    "- If a single value, MIN = MAX\n"
    "- Never put prose, methodology, or text in MIN/MAX fields\n"
    "- VALUE (MIN/MAX) must be a number, range, or percentage. NEVER a sentence, methodology description, or opinion.\n"
    "- GOOD: '0.9302', '85-95', '1.5e3'\n"
    "- BAD: 'the method achieves good performance', 'significant improvement'\n"
    "- TIME must be MM/DD/YYYY-MM/DD/YYYY or NONE\n\n"
    "Per data point, output one line:\n"
    "ENTITY: [#] | ATTR: [#] | MIN: [X.XXXXeN] | MAX: [X.XXXXeN] "
    "| UNIT: [base] | TIME: [MM/DD/YYYY-MM/DD/YYYY or NONE] | SOURCE: [W{web_id}]\n"
    "If NEW: ENTITY: NEW (Bank of Japan) | ...\n"
    "If qualitative: FACT: [standalone sentence]\n"
    "If no numerical data: leave ledger_entries empty.\n\n"
    "=== SECTION 2: FACTS (atomic factual statements) ===\n"
    "Each fact MUST be atomic — one statement per fact. Never combine with 'and', 'but', 'however'.\n"
    "For each fact, use this schema:\n"
    '{{"proposition": {{"subject_entity": "...", "predicate": "...", '
    '"object": {{"type": "string|quantity|boolean", "value": "..."}}}}, '
    '"qualifiers": [{{"key": "...", "value": "..."}}], '
    '"attribution": {{"claimed_by": "...", "claim_act": "..."}}, '
    '"source_refs": ["W{web_id}"], '
    '"raw_text": "original sentence", '
    '"ledger_indices": [0, 2]}}\n'
    "The ledger_indices field is OPTIONAL: list the indices (0-based) into the ledger_entries array "
    "that back this fact, if any.\n\n"
    "Reply with strict JSON:\n"
    '{{"ledger_entries": ["ENTITY: ... | ATTR: ... | ..."], '
    '"facts": [<list of fact objects>]}}\n'
    'If nothing to extract: {{"ledger_entries": [], "facts": []}}'
)


async def extract_all_from_evidence(
    topic_id: int,
    subtopic_id: int | None,
    web_row: dict,
    *,
    current_round: int = 0,
) -> tuple[list[dict], list[int]]:
    """Unified extraction: single LLM call returns both Ledger entries and FactCandidates.

    Returns (ledger_results, fact_candidate_ids).
    """
    from .canonical import (
        validate_structured_fact,
        structured_fact_to_columns,
        snap_subject,
        build_canonical_text,
    )
    from .json_utils import extract_json_object
    from .server import parse_clerk_ledger_output  # lazy to avoid circular deps

    web_id = web_row.get("id")
    if web_id is None:
        return [], []

    domain = web_row.get("source_domain", "")
    domain_sc = score_domain(domain)

    # Filter low-quality domains
    if domain_sc < DOMAIN_SCORE_THRESHOLD:
        api.mark_web_evidence_ledger_processed([web_id])
        logger.debug(
            "[evidence] Skipped W%s from %s (score=%.1f < threshold)",
            web_id,
            domain,
            domain_sc,
        )
        return [], []

    query = web_row.get("query_text", "")
    title = web_row.get("title", "") or ""
    snippet = web_row.get("snippet", "") or ""
    url = web_row.get("url", "") or ""

    if not snippet.strip():
        api.mark_web_evidence_ledger_processed([web_id])
        return [], []

    # FA-5: Topic relevance check
    topic = api.get_topic(topic_id) if topic_id else None
    if topic:
        topic_words = [
            w.lower()
            for w in (topic.get("summary") or "").split()
            if len(w) > 2 and w.lower() not in _STOP_WORDS
        ]
        relevance = check_topic_relevance(title, snippet, topic_words)
        domain_sc *= relevance

    # Try safe_fetch for enrichment
    extra_content_block = ""
    if url and is_fetch_allowed(url):
        try:
            fetched = await safe_fetch_content(url, max_chars=2000)
            if fetched and fetched.get("content"):
                extra_content_block = f"\nFull page content:\n{fetched['content']}\n\n"
        except Exception:
            pass

    # Fetch entity/attribute lists for Ledger section
    entities = _ledger.get_entity_numbered_list(topic_id, current_round)
    attributes = _ledger.get_attribute_numbered_list(topic_id)

    # Build entity/attribute blocks
    entity_lines = ["0. [UNSPECIFIED] — choose if unclear"]
    for eid, name in entities:
        entity_lines.append(f"{eid}. {name}")
    entity_lines.append("NEW. [NEW] (name: ___)")
    entity_block = "\n".join(entity_lines)

    attr_lines = ["0. [UNSPECIFIED] — choose if unclear"]
    for aid, name in attributes:
        attr_lines.append(f"{aid}. {name}")
    attr_lines.append("NEW. [NEW] (name: ___)")
    attr_block = "\n".join(attr_lines)

    prompt = _UNIFIED_EXTRACTION_PROMPT_TEMPLATE.format(
        query=query,
        title=title,
        url=url,
        snippet=snippet[:2000],
        domain=domain,
        domain_score=domain_sc,
        web_id=web_id,
        extra_content_block=extra_content_block,
        entity_block=entity_block,
        attr_block=attr_block,
    )

    provider = _fact_provider(topic_id)
    resp = None
    for attempt in range(_MAX_FACT_EXTRACTION_RETRIES + 1):
        try:
            resp = await call_text(
                prompt,
                provider=provider,
                strategy="direct",
                allow_web=False,
                system_instruction="You are a data extraction clerk. Extract structured data and atomic facts from web evidence.",
                fallback_role="fact_proposer",
            )
        except Exception as exc:
            logger.warning(
                "[evidence-unified] LLM call failed for W%s attempt %d: %s",
                web_id,
                attempt,
                exc,
            )
            continue
        if resp:
            break

    if not resp:
        api.mark_web_evidence_ledger_processed([web_id])
        return [], []

    # Try to parse as unified JSON
    parsed = extract_json_object(resp)

    ledger_results: list[dict] = []
    fact_ids: list[int] = []

    try:
        if isinstance(parsed, dict):
            # --- Ledger entries ---
            ledger_lines = parsed.get("ledger_entries", [])
            if isinstance(ledger_lines, list) and ledger_lines:
                # Re-join lines and parse via existing pipeline
                raw_text = "\n".join(str(line) for line in ledger_lines)
                ledger_parsed = parse_clerk_ledger_output(
                    raw_text,
                    entities,
                    attributes,
                    topic_id,
                    subtopic_id,
                    current_round,
                    "evidence_parser",
                )
                for entry in ledger_parsed:
                    if entry.get("type") == "structured":
                        try:
                            lid, status = _ledger.normalize_and_upsert(
                                topic_id=entry["topic_id"],
                                subtopic_id=entry.get("subtopic_id"),
                                entity_id=entry.get("entity_id"),
                                attribute_id=entry.get("attribute_id"),
                                raw_value=entry.get("raw_value", ""),
                                raw_timeframe=entry.get("raw_timeframe"),
                                entry_type="web_evidence",
                                source_ref=f"[W{web_id}]",
                                source_domain=domain,
                                domain_score=domain_sc,
                                created_by="evidence_parser",
                                current_round=current_round,
                                valid_from=entry.get("valid_from"),
                                valid_to=entry.get("valid_to"),
                                min_val=entry.get("min_val"),
                                max_val=entry.get("max_val"),
                                unit=entry.get("unit"),
                                stat_type=entry.get("stat_type"),
                                value_mean=entry.get("value_mean"),
                                value_std=entry.get("value_std"),
                                value_p=entry.get("value_p"),
                                value_n=entry.get("value_n"),
                                value_ci_lower=entry.get("value_ci_lower"),
                                value_ci_upper=entry.get("value_ci_upper"),
                                value_ci_level=entry.get("value_ci_level"),
                                baseline_entity_id=entry.get("baseline_entity_id"),
                                split=entry.get("split"),
                                config_json=entry.get("config_json"),
                            )
                            ledger_results.append(
                                {"ledger_id": lid, "status": status, **entry}
                            )
                        except Exception as exc:
                            logger.warning(
                                "[evidence-unified] Ledger upsert failed for W%s: %s",
                                web_id,
                                exc,
                            )

            # --- Facts ---
            facts_raw = parsed.get("facts", [])
            # Track ledger_indices per fact for edge creation
            fact_ledger_indices: list[list[int]] = []
            if isinstance(facts_raw, list):
                for fact_data in facts_raw[:_MAX_FACTS_PER_EVIDENCE]:
                    if not isinstance(fact_data, dict):
                        continue
                    # Capture ledger_indices before validation strips it
                    raw_ledger_indices = fact_data.get("ledger_indices") or []
                    if not isinstance(raw_ledger_indices, list):
                        raw_ledger_indices = []
                    validated, err = validate_structured_fact(fact_data)
                    if validated is None:
                        logger.debug(
                            "[evidence-unified] Fact validation failed for W%s: %s",
                            web_id,
                            err,
                        )
                        continue

                    snapped_subject = snap_subject(
                        validated.proposition.subject_entity, topic_id
                    )
                    source_refs = validated.source_refs or [f"W{web_id}"]
                    cols = structured_fact_to_columns(validated)
                    canonical_text = build_canonical_text(
                        snapped_subject,
                        validated.proposition.predicate,
                        cols.get("object_json"),
                        cols.get("qualifiers_json"),
                        cols.get("attribution_json"),
                        source_refs,
                    )
                    if not canonical_text.strip():
                        continue
                    if api.fact_candidate_exists(
                        topic_id, canonical_text, statuses=("pending",)
                    ):
                        continue
                    if api.fact_exists(topic_id, canonical_text):
                        continue

                    candidate_id = api.create_fact_candidate_with_stage(
                        topic_id,
                        subtopic_id,
                        None,
                        canonical_text,
                        fact_stage="web_extracted",
                        candidate_type="sourced_claim",
                        source_kind="web",
                        source_refs_json=json.dumps(source_refs, ensure_ascii=False),
                        source_excerpt=(
                            validated.raw_text[:500] if validated.raw_text else None
                        ),
                        subject=snapped_subject,
                        predicate=validated.proposition.predicate,
                        object_json=cols.get("object_json"),
                        qualifiers_json=cols.get("qualifiers_json"),
                        attribution_json=cols.get("attribution_json"),
                    )
                    fact_ids.append(candidate_id)
                    fact_ledger_indices.append(raw_ledger_indices)
                    logger.info(
                        "[evidence-unified] Created FactCandidate %s from W%s: %s",
                        candidate_id,
                        web_id,
                        canonical_text[:80],
                    )

            # Phase C: Create KnowledgeEdge links
            # Note: Ledger→FactCandidate and Web→FactCandidate edges are NOT
            # created here because fact_ids are FactCandidate IDs (not Fact IDs).
            # Those edges are created in librarian_processor.py when the candidate
            # is promoted to a Fact.

            # derived_from edges: web_evidence → ledger (Ledger IDs are real)
            for lr in ledger_results:
                if lr.get("ledger_id"):
                    db.insert_knowledge_edge(
                        topic_id,
                        web_id,
                        "web_evidence",
                        lr["ledger_id"],
                        "ledger",
                        "derived_from",
                        created_by="extract_all_from_evidence",
                    )
        else:
            # Fallback: try old pipe-delimited format (ledger only)
            ledger_parsed = parse_clerk_ledger_output(
                resp,
                entities,
                attributes,
                topic_id,
                subtopic_id,
                current_round,
                "evidence_parser",
            )
            for entry in ledger_parsed:
                if entry.get("type") == "structured":
                    try:
                        lid, status = _ledger.normalize_and_upsert(
                            topic_id=entry["topic_id"],
                            subtopic_id=entry.get("subtopic_id"),
                            entity_id=entry.get("entity_id"),
                            attribute_id=entry.get("attribute_id"),
                            raw_value=entry.get("raw_value", ""),
                            raw_timeframe=entry.get("raw_timeframe"),
                            entry_type="web_evidence",
                            source_ref=f"[W{web_id}]",
                            source_domain=domain,
                            domain_score=domain_sc,
                            created_by="evidence_parser",
                            current_round=current_round,
                            valid_from=entry.get("valid_from"),
                            valid_to=entry.get("valid_to"),
                            min_val=entry.get("min_val"),
                            max_val=entry.get("max_val"),
                            unit=entry.get("unit"),
                            stat_type=entry.get("stat_type"),
                            value_mean=entry.get("value_mean"),
                            value_std=entry.get("value_std"),
                            value_p=entry.get("value_p"),
                            value_n=entry.get("value_n"),
                            value_ci_lower=entry.get("value_ci_lower"),
                            value_ci_upper=entry.get("value_ci_upper"),
                            value_ci_level=entry.get("value_ci_level"),
                            baseline_entity_id=entry.get("baseline_entity_id"),
                            split=entry.get("split"),
                            config_json=entry.get("config_json"),
                        )
                        ledger_results.append(
                            {"ledger_id": lid, "status": status, **entry}
                        )
                    except Exception as exc:
                        logger.warning(
                            "[evidence-unified] Fallback ledger upsert failed for W%s: %s",
                            web_id,
                            exc,
                        )
    finally:
        api.mark_web_evidence_ledger_processed([web_id])

    logger.info(
        "[evidence-unified] W%s (%s, score=%.1f): %d ledger + %d facts",
        web_id,
        domain,
        domain_sc,
        len(ledger_results),
        len(fact_ids),
    )
    return ledger_results, fact_ids


# ---------------------------------------------------------------------------
# Ledger backfill: fill missing unit/time on incomplete entries
# ---------------------------------------------------------------------------

_BACKFILL_PROMPT_TEMPLATE = (
    "You have a structured data entry with MISSING fields. "
    "Using the source material below, fill in ONLY the missing fields.\n\n"
    "Current entry:\n"
    "  Entity: {entity_name}\n"
    "  Attribute: {attribute_name}\n"
    "  Value: {value} (min={min_val}, max={max_val})\n"
    "  Unit: {unit}\n"
    "  Time: {time}\n\n"
    "Source URL: {url}\n"
    "Source snippets:\n{snippets}\n\n"
    "{extra_content_block}"
    "Fill in the missing fields. Reply with strict JSON:\n"
    '{{"unit": "...", "time": "..."}}\n'
    "Rules:\n"
    "- unit: one of [USD, EUR, CNY, FLOPS, SECONDS, WATTS, %, NONE, OTHER] "
    "or a specific unit from the snippet\n"
    "- time: use format 2035 | Q4 2026 | H1 2025 | Jan 2024 | 2024-2025 | NONE\n"
    "- Only fill what is actually stated in the source — do NOT guess\n"
    "- Return the EXISTING value if that field is already filled\n"
    '- If truly unknown, use null: {{"unit": null, "time": null}}'
)


async def backfill_ledger_entry(
    topic_id: int, entry: dict, web_rows: list[dict]
) -> dict | None:
    """Try to fill missing unit/time on a Ledger entry. Returns updates or None."""
    from .json_utils import extract_json_object

    entry_id = entry.get("id")
    unit = entry.get("unit")
    valid_from = entry.get("valid_from")

    # Only backfill if something is missing
    if unit and valid_from:
        return None

    source_ref = entry.get("source_ref", "")
    # Find matching web evidence by source_ref [W<id>]
    w_ids = [int(m) for m in re.findall(r"\[?W(\d+)\]?", source_ref)]
    matching_rows = [r for r in web_rows if r.get("id") in w_ids]
    if not matching_rows:
        return None

    # Collect snippets and URL
    url = matching_rows[0].get("url", "") or ""
    snippets = "\n---\n".join(
        (r.get("snippet", "") or "").strip() for r in matching_rows if r.get("snippet")
    )
    if not snippets:
        return None

    # Try safe_fetch for extra context
    extra_content_block = ""
    if url and is_fetch_allowed(url):
        try:
            fetched = await safe_fetch_content(url, max_chars=1500)
            if fetched and fetched.get("content"):
                extra_content_block = f"Full page content:\n{fetched['content']}\n\n"
        except Exception:
            pass

    # Resolve entity/attribute names
    entity_name = "Unknown"
    attr_name = "Unknown"
    eid = entry.get("entity_id")
    aid = entry.get("attribute_id")
    if eid:
        e = db.get_ledger_entity(eid)
        if e:
            entity_name = e.get("canonical_name", "Unknown")
    if aid:
        a = db.get_ledger_attribute(aid)
        if a:
            attr_name = a.get("canonical_name", "Unknown")

    prompt = _BACKFILL_PROMPT_TEMPLATE.format(
        entity_name=entity_name,
        attribute_name=attr_name,
        value=entry.get("value", ""),
        min_val=entry.get("value_numeric_min"),
        max_val=entry.get("value_numeric_max"),
        unit=unit or "MISSING",
        time=valid_from or "MISSING",
        url=url,
        snippets=snippets[:2000],
        extra_content_block=extra_content_block,
    )

    provider = _fact_provider(topic_id)
    try:
        resp = await call_text(
            prompt,
            provider=provider,
            strategy="direct",
            allow_web=False,
            system_instruction="You are a data extraction clerk. Fill in missing fields from source material.",
            fallback_role="fact_proposer",
        )
    except Exception as exc:
        logger.debug("[backfill] LLM call failed for L%s: %s", entry_id, exc)
        return None

    if not resp:
        return None

    parsed = extract_json_object(resp)
    if not isinstance(parsed, dict):
        return None

    updates = {}
    if not unit and parsed.get("unit") and parsed["unit"] != "null":
        updates["unit"] = str(parsed["unit"])
    if not valid_from and parsed.get("time") and parsed["time"] != "null":
        updates["time"] = str(parsed["time"])

    if not updates:
        return None

    # Apply updates to DB
    with db.get_db() as conn:
        if "unit" in updates:
            conn.execute(
                "UPDATE Ledger SET unit = ? WHERE id = ?",
                (updates["unit"], entry_id),
            )
        if "time" in updates:
            from .ledger import parse_time_field

            vf, vt = parse_time_field(updates["time"])
            conn.execute(
                "UPDATE Ledger SET valid_from = ?, valid_to = ?, normalized_timeframe = ? WHERE id = ?",
                (vf, vt, updates["time"], entry_id),
            )

    logger.info(
        "[backfill] Updated L%s: %s",
        entry_id,
        updates,
    )
    return updates


async def backfill_incomplete_ledger_entries(topic_id: int, current_round: int) -> int:
    """Backfill incomplete Ledger entries (missing unit or time).

    Only processes entries created in the current round. Returns count updated.
    """
    # Get incomplete entries
    incomplete = db.get_ledger_entries(topic_id)
    incomplete = [
        e
        for e in incomplete
        if e.get("status") == "accepted"
        and (not e.get("unit") or not e.get("valid_from"))
    ]
    if not incomplete:
        return 0

    # Load all web evidence for this topic (for snippet lookup)
    web_rows = db.get_web_evidence_for_topic(topic_id)

    updated = 0
    for entry in incomplete[:10]:  # cap per batch
        try:
            result = await backfill_ledger_entry(topic_id, entry, web_rows)
            if result:
                updated += 1
        except Exception as exc:
            logger.debug("[backfill] Failed for L%s: %s", entry.get("id"), exc)

    if updated:
        logger.info(
            "[backfill] Backfilled %d/%d incomplete entries for topic %s",
            updated,
            len(incomplete),
            topic_id,
        )
    return updated


# ---------------------------------------------------------------------------
# Code Evidence → Ledger + FactCandidate extraction (CE-1)
# ---------------------------------------------------------------------------


async def extract_from_code_evidence(
    topic_id: int,
    subtopic_id: int | None,
    *,
    code_evidence_id: int,
    hypothesis: str,
    stdout: str,
    current_round: int = 0,
) -> tuple[list[dict], list[int]]:
    """Extract structured Ledger entries and FactCandidates from code evidence output.

    Mirrors extract_all_from_evidence() but uses hypothesis+stdout as the snippet.
    Returns (ledger_results, fact_candidate_ids).
    """
    from .canonical import (
        validate_structured_fact,
        structured_fact_to_columns,
        snap_subject,
        build_canonical_text,
    )
    from .json_utils import extract_json_object
    from .server import parse_clerk_ledger_output

    snippet = (
        f"Hypothesis: {hypothesis[:500]}\nCode output:\n{(stdout or '')[:3000]}".strip()
    )
    if not snippet:
        return [], []

    # Fetch entity/attribute lists for Ledger section
    entities = _ledger.get_entity_numbered_list(topic_id, current_round)
    attributes = _ledger.get_attribute_numbered_list(topic_id)

    entity_lines = ["0. [UNSPECIFIED] — choose if unclear"]
    for eid, name in entities:
        entity_lines.append(f"{eid}. {name}")
    entity_lines.append("NEW. [NEW] (name: ___)")
    entity_block = "\n".join(entity_lines)

    attr_lines = ["0. [UNSPECIFIED] — choose if unclear"]
    for aid, name in attributes:
        attr_lines.append(f"{aid}. {name}")
    attr_lines.append("NEW. [NEW] (name: ___)")
    attr_block = "\n".join(attr_lines)

    prompt = _UNIFIED_EXTRACTION_PROMPT_TEMPLATE.format(
        query=hypothesis,
        title=f"Code experiment E{code_evidence_id}",
        url="",
        snippet=snippet[:2000],
        domain="code_sandbox",
        domain_score=1.0,
        web_id=code_evidence_id,
        extra_content_block="",
        entity_block=entity_block,
        attr_block=attr_block,
    )
    # Fix citation prefix: template produces [W{id}] but code evidence uses [E{id}]
    prompt = prompt.replace(f"[W{code_evidence_id}]", f"[E{code_evidence_id}]").replace(
        f'"W{code_evidence_id}"', f'"E{code_evidence_id}"'
    )

    provider = _fact_provider(topic_id)
    resp = None
    for attempt in range(_MAX_FACT_EXTRACTION_RETRIES + 1):
        try:
            resp = await call_text(
                prompt,
                provider=provider,
                strategy="direct",
                allow_web=False,
                system_instruction="You are a data extraction clerk. Extract structured data and atomic facts from code experiment results.",
                fallback_role="fact_proposer",
            )
        except Exception as exc:
            logger.warning(
                "[evidence-code] LLM call failed for E%s attempt %d: %s",
                code_evidence_id,
                attempt,
                exc,
            )
            continue
        if resp:
            break

    if not resp:
        return [], []

    parsed = extract_json_object(resp)
    ledger_results: list[dict] = []
    fact_ids: list[int] = []

    try:
        if not isinstance(parsed, dict):
            return [], []

        # --- Ledger entries ---
        ledger_lines = parsed.get("ledger_entries", [])
        if isinstance(ledger_lines, list) and ledger_lines:
            raw_text = "\n".join(str(line) for line in ledger_lines)
            ledger_parsed = parse_clerk_ledger_output(
                raw_text,
                entities,
                attributes,
                topic_id,
                subtopic_id,
                current_round,
                "evidence_parser",
            )
            for entry in ledger_parsed:
                if entry.get("type") == "structured":
                    try:
                        lid, status = _ledger.normalize_and_upsert(
                            topic_id=entry["topic_id"],
                            subtopic_id=entry.get("subtopic_id"),
                            entity_id=entry.get("entity_id"),
                            attribute_id=entry.get("attribute_id"),
                            raw_value=entry.get("raw_value", ""),
                            raw_timeframe=entry.get("raw_timeframe"),
                            entry_type="code_evidence",
                            source_ref=f"[E{code_evidence_id}]",
                            source_domain="code_sandbox",
                            domain_score=1.0,
                            created_by="evidence_parser",
                            current_round=current_round,
                            valid_from=entry.get("valid_from"),
                            valid_to=entry.get("valid_to"),
                            min_val=entry.get("min_val"),
                            max_val=entry.get("max_val"),
                            unit=entry.get("unit"),
                            stat_type=entry.get("stat_type"),
                            value_mean=entry.get("value_mean"),
                            value_std=entry.get("value_std"),
                            value_p=entry.get("value_p"),
                            value_n=entry.get("value_n"),
                            value_ci_lower=entry.get("value_ci_lower"),
                            value_ci_upper=entry.get("value_ci_upper"),
                            value_ci_level=entry.get("value_ci_level"),
                            baseline_entity_id=entry.get("baseline_entity_id"),
                            split=entry.get("split"),
                            config_json=entry.get("config_json"),
                        )
                        ledger_results.append(
                            {"ledger_id": lid, "status": status, **entry}
                        )
                        # code_evidence → ledger edge
                        db.insert_knowledge_edge(
                            topic_id,
                            code_evidence_id,
                            "code_evidence",
                            lid,
                            "ledger",
                            "derived_from",
                            created_by="extract_from_code_evidence",
                        )
                    except Exception as exc:
                        logger.warning(
                            "[evidence-code] Ledger upsert failed for E%s: %s",
                            code_evidence_id,
                            exc,
                        )

        # --- Facts ---
        facts_raw = parsed.get("facts", [])
        if isinstance(facts_raw, list):
            for fact_data in facts_raw[:_MAX_FACTS_PER_EVIDENCE]:
                if not isinstance(fact_data, dict):
                    continue
                validated, err = validate_structured_fact(fact_data)
                if validated is None:
                    logger.debug(
                        "[evidence-code] Fact validation failed for E%s: %s",
                        code_evidence_id,
                        err,
                    )
                    continue

                snapped_subject = snap_subject(
                    validated.proposition.subject_entity, topic_id
                )
                source_refs = [f"E{code_evidence_id}"]
                cols = structured_fact_to_columns(validated)
                canonical_text = build_canonical_text(
                    snapped_subject,
                    validated.proposition.predicate,
                    cols.get("object_json"),
                    cols.get("qualifiers_json"),
                    cols.get("attribution_json"),
                    source_refs,
                )
                if not canonical_text.strip():
                    continue
                if api.fact_candidate_exists(
                    topic_id, canonical_text, statuses=("pending",)
                ):
                    continue
                if api.fact_exists(topic_id, canonical_text):
                    continue

                candidate_id = api.create_fact_candidate_with_stage(
                    topic_id,
                    subtopic_id,
                    None,
                    canonical_text,
                    fact_stage="code_verified",
                    candidate_type="code_evidence",
                    source_kind="code",
                    source_refs_json=json.dumps(source_refs, ensure_ascii=False),
                    source_excerpt=(
                        validated.raw_text[:500] if validated.raw_text else None
                    ),
                    subject=snapped_subject,
                    predicate=validated.proposition.predicate,
                    object_json=cols.get("object_json"),
                    qualifiers_json=cols.get("qualifiers_json"),
                    attribution_json=cols.get("attribution_json"),
                )
                fact_ids.append(candidate_id)
                logger.info(
                    "[evidence-code] Created FactCandidate %s from E%s: %s",
                    candidate_id,
                    code_evidence_id,
                    canonical_text[:80],
                )
    except Exception as exc:
        logger.warning(
            "[evidence-code] Extraction failed for E%s: %s",
            code_evidence_id,
            exc,
        )

    logger.info(
        "[evidence-code] E%s: %d ledger + %d facts",
        code_evidence_id,
        len(ledger_results),
        len(fact_ids),
    )
    return ledger_results, fact_ids
