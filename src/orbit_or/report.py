"""5-pass LLM blog generation pipeline.

Pass 1: Dossier extraction from DB data
Pass 2: Blog outline generation
Pass 3: Final blog text
Pass 4: Review (fact-check)
Pass 5: Auto-fix (apply review fixes to blog, if needed)
"""

import json
import logging
import sqlite3
from typing import Any

from . import api
from .broker import DEFAULT_MAX_TOKENS, call_text
from . import topic_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# i18n for report rendering
# ---------------------------------------------------------------------------

_I18N = {
    "zh": {
        "appendix": "附录：关键发现",
        "warnings": "数据准确性警告",
        "review": "审校意见",
        "quality_label": "质量评估",
        "confidence": "可信度",
        "meta_prefix": "ORBIT 报告",
    },
    "en": {
        "appendix": "Appendix: Key Findings",
        "warnings": "Accuracy Warnings",
        "review": "Review",
        "quality_label": "Quality",
        "confidence": "confidence",
        "meta_prefix": "ORBIT Report",
    },
}


def _t(lang: str, key: str) -> str:
    """Get translated string for the given language."""
    return _I18N.get(lang, _I18N["en"]).get(key, _I18N["en"].get(key, key))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def generate_report(topic_id: int) -> dict[str, Any] | None:
    """Generate a full report for a closed topic. Returns report JSON or None on failure."""
    topic = api.get_topic(topic_id)
    if not topic:
        logger.warning("[report] Topic %d not found", topic_id)
        return None

    lang = topic_config.get(topic_id, "blog_language")
    lang_instruction = (
        "Write the blog in Chinese (zh-CN)." if lang == "zh" else "Write in English."
    )
    try:
        writer_provider = topic_config.get_provider_profile_for(
            topic_id, "writer_provider"
        )
    except sqlite3.OperationalError:
        raise
    except Exception as exc:
        logger.debug("[report] Writer provider lookup failed: %s", exc)
        writer_provider = "minimax"

    try:
        dossier = await _pass1_dossier(topic_id, topic, writer_provider)
        if not dossier:
            logger.warning(
                "[report] Pass 1 dossier extraction failed for topic %d", topic_id
            )
            return None

        outline = await _pass2_outline(dossier, lang_instruction, writer_provider)
        if not outline:
            logger.warning(
                "[report] Pass 2 outline generation failed for topic %d", topic_id
            )
            return None

        blog = await _pass3_blog(dossier, outline, lang_instruction, writer_provider)
        if not blog:
            logger.warning(
                "[report] Pass 3 blog generation failed for topic %d", topic_id
            )
            return None

        review = await _pass4_review(blog, dossier, lang_instruction, writer_provider)

        # Pass 5: auto-fix if review says needs_revision
        review_applied = False
        if review and review.get("overall_quality") == "needs_revision":
            fixed_blog = await _pass5_autofix(
                blog, review, lang_instruction, writer_provider
            )
            if fixed_blog:
                blog = fixed_blog
                review_applied = True
                logger.info("[report] Pass 5 auto-fix applied for topic %d", topic_id)

        report = {
            "topic_id": topic_id,
            "topic_summary": topic["summary"],
            "language": lang,
            "dossier": dossier,
            "outline": outline,
            "blog": blog,
            "review": review,
            "review_applied": review_applied,
        }

        api.save_report(topic_id, json.dumps(report, ensure_ascii=False))
        logger.info("[report] Report generated and saved for topic %d", topic_id)
        return report

    except Exception as exc:
        logger.error(
            "[report] Report generation failed for topic %d: %s", topic_id, exc
        )
        return None


# ---------------------------------------------------------------------------
# Pass 1: Dossier
# ---------------------------------------------------------------------------


async def _pass1_dossier(
    topic_id: int, topic: dict, provider: str = "minimax"
) -> dict[str, Any] | None:
    """Collect data from DB and extract structured dossier via LLM."""
    subtopics = api.get_current_subtopics(topic_id)
    facts = api.get_facts(topic_id, limit=200)
    claims = api.get_claims(topic_id, limit=200)
    web_evidence = api.get_web_evidence_for_topic(topic_id)
    code_evidence = api.get_code_evidence_for_topic(topic_id)
    ledger_entries = api.get_ledger_entries_with_names(topic_id)
    _ = api.get_knowledge_edges(topic_id)  # available for future dossier enrichment

    context_parts = [
        f"# Topic: {topic['summary']}",
        f"Detail: {topic['detail']}",
        f"Conclusion: {topic.get('conclusion', 'N/A')}",
        "",
        "## Subtopics and Conclusions:",
    ]
    for st in subtopics:
        conclusion = st.get("conclusion") or "(no conclusion)"
        context_parts.append(f"- {st['summary']}: {conclusion[:500]}")

    context_parts.append("\n## Key Facts:")
    for f in facts[:50]:
        context_parts.append(f"- [F{f['id']}] {f['content'][:200]}")

    context_parts.append("\n## Claims:")
    for c in claims[:30]:
        context_parts.append(f"- [C{c['id']}] {c['content'][:200]}")

    context_parts.append("\n## Web Evidence:")
    for w in web_evidence[:20]:
        context_parts.append(
            f"- [W{w['id']}] {w.get('title', '')} -- {w.get('snippet', '')[:150]}"
        )

    context_parts.append("\n## Code Evidence:")
    for ce in code_evidence[:10]:
        status = "success" if ce.get("success") else "failed"
        context_parts.append(f"- [Code{ce['id']}] {ce['hypothesis'][:150]} ({status})")

    context_parts.append("\n## Ledger Entries:")
    for le in ledger_entries[:30]:
        context_parts.append(
            f"- [L{le['id']}] {le.get('entity_name', '?')}.{le.get('attribute_name', '?')} = {le.get('value', '?')}"
        )

    context = "\n".join(context_parts)

    prompt = f"""You are a research analyst. Extract a structured dossier from this ORBIT workspace data.

{context}

Output strictly JSON with these fields:
{{
  "topic": "topic summary",
  "why_it_matters": "1-2 sentences",
  "subtopics": [{{"name": "...", "key_finding": "...", "confidence": "high/medium/low"}}],
  "dramatic_moments": ["moment1", "moment2"],
  "internal_conflicts": ["conflict1"],
  "unstable_claims": ["claim1"],
  "final_consensus": "the consensus if any",
  "best_quotes": ["quote1"],
  "recommended_blog_angle": "suggested narrative angle",
  "accuracy_warnings": ["warning1"]
}}"""

    result = await call_text(
        prompt,
        provider=provider,
        strategy="direct",
        temperature=0.5,
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=True,
        fallback_role="skynet",
    )
    if not result or not result.strip():
        return None
    try:
        from .json_utils import extract_json_object

        parsed = extract_json_object(result)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pass 2: Outline
# ---------------------------------------------------------------------------


async def _pass2_outline(
    dossier: dict, lang_instruction: str, provider: str = "minimax"
) -> dict[str, Any] | None:
    """Generate blog outline from dossier."""
    prompt = f"""{lang_instruction}

You are a blog editor. Given this research dossier, create a blog outline.

Dossier:
{json.dumps(dossier, ensure_ascii=False, indent=2)[:6000]}

Output strictly JSON:
{{
  "title_candidates": ["title1", "title2", "title3"],
  "chosen_angle": "the narrative angle",
  "opening_hook": "dramatic opening hook",
  "section_outline": [
    {{"title": "section title", "dramatic_function": "what this section does", "factual_core": "key facts to include"}}
  ],
  "closing_thesis": "the closing argument"
}}"""

    result = await call_text(
        prompt,
        provider=provider,
        strategy="direct",
        temperature=0.7,
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=True,
        fallback_role="skynet",
    )
    if not result or not result.strip():
        return None
    try:
        from .json_utils import extract_json_object

        parsed = extract_json_object(result)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pass 3: Blog
# ---------------------------------------------------------------------------


async def _pass3_blog(
    dossier: dict,
    outline: dict,
    lang_instruction: str,
    provider: str = "minimax",
) -> str | None:
    """Generate final blog text."""
    prompt = f"""{lang_instruction}

You are a sharp, dramatic science/tech blogger. Write a blog post based on this outline and dossier.

Style: dramatic, sharp, scene-by-scene progression, no hype. Preserve citation markers like [F1], [C2], [W3] where relevant.

Outline:
{json.dumps(outline, ensure_ascii=False, indent=2)[:4000]}

Dossier:
{json.dumps(dossier, ensure_ascii=False, indent=2)[:6000]}

Write the complete blog post. Use markdown formatting."""

    result = await call_text(
        prompt,
        provider=provider,
        strategy="direct",
        temperature=0.8,
        max_tokens=DEFAULT_MAX_TOKENS,
        fallback_role="skynet",
    )
    return result if result and result.strip() else None


# ---------------------------------------------------------------------------
# Pass 4: Review
# ---------------------------------------------------------------------------


async def _pass4_review(
    blog: str,
    dossier: dict,
    lang_instruction: str,
    provider: str = "minimax",
) -> dict[str, Any] | None:
    """Review pass for overclaims and fabricated precision."""
    prompt = f"""{lang_instruction}

You are a fact-checker. Review this blog post against the dossier for:
1. Overclaims (stated as fact when source is uncertain)
2. Fabricated precision (specific numbers not in the data)
3. Missing important caveats

IMPORTANT: Output all issue descriptions, locations, and fixes in the SAME language as the blog.
The "overall_quality" and "type" fields should remain in English for programmatic use.

Blog:
{blog}

Dossier accuracy_warnings:
{json.dumps(dossier.get('accuracy_warnings', []), ensure_ascii=False)}

Output strictly JSON:
{{
  "issues_found": [{{"type": "overclaim|fabricated_precision|missing_caveat", "location": "...", "fix": "..."}}],
  "overall_quality": "good|acceptable|needs_revision",
  "suggested_fixes": ["fix1"]
}}"""

    result = await call_text(
        prompt,
        provider=provider,
        strategy="direct",
        temperature=0.3,
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=True,
        fallback_role="skynet",
    )
    if not result or not result.strip():
        return None
    try:
        from .json_utils import extract_json_object

        parsed = extract_json_object(result)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Pass 5: Auto-fix
# ---------------------------------------------------------------------------


async def _pass5_autofix(
    blog: str, review: dict, lang_instruction: str, provider: str = "minimax"
) -> str | None:
    """Apply review fixes to the blog text. Returns revised blog or None on failure."""
    issues = review.get("issues_found", [])
    fixes = review.get("suggested_fixes", [])
    if not issues and not fixes:
        return None

    issues_text = json.dumps(issues, ensure_ascii=False, indent=2)
    fixes_text = json.dumps(fixes, ensure_ascii=False, indent=2)

    prompt = f"""{lang_instruction}

You are a blog editor. Revise the following blog post to address ALL review issues below.
Apply each fix precisely. Do NOT add new content or change the narrative structure.
Preserve all citation markers like [F1], [C2], [W3].

Issues to fix:
{issues_text}

Suggested fixes:
{fixes_text}

Original blog:
{blog}

Output the COMPLETE revised blog in markdown. No commentary, just the revised text."""

    result = await call_text(
        prompt,
        provider=provider,
        strategy="direct",
        temperature=0.3,
        max_tokens=DEFAULT_MAX_TOKENS,
        fallback_role="skynet",
    )
    return result if result and result.strip() else None


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _build_citation_map(topic_id: int) -> dict[str, str]:
    """Build a map of citation IDs to content for tooltip display."""
    cmap: dict[str, str] = {}
    try:
        for f in api.get_facts(topic_id, limit=500):
            cmap[f"F{f['id']}"] = (f.get("summary") or f["content"])[:200]
        for c in api.get_claims(topic_id, limit=500):
            cmap[f"C{c['id']}"] = (c.get("summary") or c["content"])[:200]
        for doc in api.list_corpus_documents(topic_id, limit=200):
            for chunk in api.get_corpus_chunks_for_document(doc["id"]):
                title = doc.get("title") or f"Document {doc['id']}"
                cmap[f"D{chunk['id']}"] = f"{title}: {chunk['text'][:180]}"
        for w in api.get_web_evidence_for_topic(topic_id):
            cmap[f"W{w['id']}"] = (w.get("title") or w.get("snippet", ""))[:200]
        for ce in api.get_code_evidence_for_topic(topic_id):
            cmap[f"E{ce['id']}"] = (ce.get("summary") or ce["hypothesis"])[:200]
        for le in api.get_ledger_entries_with_names(topic_id):
            cmap[f"L{le['id']}"] = (
                f"{le.get('entity_name', '?')}.{le.get('attribute_name', '?')}={le.get('value', '?')}"
            )
    except Exception as exc:
        logger.warning("[report] Failed to build citation map: %s", exc)
    return cmap


def render_html_report(report: dict) -> str:
    """Render a self-contained HTML report with proper markdown and citation tooltips."""
    import re
    import mistune

    lang = report.get("language", "en")
    topic_id = report.get("topic_id")
    blog = report.get("blog", "")
    dossier = report.get("dossier", {})
    outline = report.get("outline", {})
    topic_summary = report.get("topic_summary", "Report")
    title = "Report"
    if outline and outline.get("title_candidates"):
        title = outline["title_candidates"][0]

    # Build citation tooltips from DB
    citation_map = _build_citation_map(topic_id) if topic_id else {}

    # Render markdown → HTML using mistune, then sanitize with nh3
    import nh3

    _SAFE_TAGS = {
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "table",
        "tr",
        "td",
        "th",
        "thead",
        "tbody",
        "blockquote",
        "code",
        "pre",
        "em",
        "strong",
        "a",
        "br",
        "hr",
        "span",
        "div",
        "details",
        "summary",
        "img",
    }
    _SAFE_ATTRS = {
        "a": {"href"},
        "img": {"src", "alt"},
        "span": {"class", "data-ref", "title"},
        "div": {"class"},
    }
    blog_html = mistune.html(blog) if blog else ""
    blog_html = nh3.clean(blog_html, tags=_SAFE_TAGS, attributes=_SAFE_ATTRS)

    # Replace [F1], [C2], etc. with tooltip spans containing real content
    def _cite_repl(m):
        cid = f"{m.group(1)}{m.group(2)}"
        tip = _esc(citation_map.get(cid, cid))
        return f'<span class="cite" data-ref="{cid}">[{cid}]<span class="cite-tip">{tip}</span></span>'

    blog_html = re.sub(r"\[(F|C|W|L|M|E)(\d+)\]", _cite_repl, blog_html)

    conf_label = _t(lang, "confidence")
    facts_html = ""
    for st in dossier.get("subtopics", []):
        facts_html += f"<li><strong>{_esc(st.get('name', ''))}</strong>: {_esc(st.get('key_finding', ''))} ({conf_label}: {_esc(st.get('confidence', '?'))})</li>\n"

    warnings_html = ""
    for w in dossier.get("accuracy_warnings", []):
        warnings_html += f"<li>{_esc(w)}</li>\n"

    review = report.get("review", {})
    review_applied = report.get("review_applied", False)
    review_html = ""
    if review:
        quality = review.get("overall_quality", "N/A")
        quality_label = _t(lang, "quality_label")
        applied_note = " ✓" if review_applied else ""
        review_html = (
            f"<p><strong>{quality_label}:</strong> {_esc(quality)}{applied_note}</p>"
        )
        for issue in review.get("issues_found", []):
            review_html += f'<p class="issue"><strong>[{_esc(issue.get("type", ""))}]</strong> {_esc(issue.get("location", ""))}: {_esc(issue.get("fix", ""))}</p>'

    html_lang = "zh" if lang == "zh" else "en"
    meta_prefix = _t(lang, "meta_prefix")
    appendix_title = _t(lang, "appendix")
    warnings_title = _t(lang, "warnings")
    review_title = _t(lang, "review")

    issue_count = len(review.get("issues_found", [])) if review else 0
    review_section = ""
    if review_html:
        review_section = (
            f"<details><summary>{review_title} ({issue_count})</summary>"
            f"{review_html}</details>"
        )

    warnings_section = ""
    if warnings_html:
        warnings_section = f"<h2>{warnings_title}</h2><ul>{warnings_html}</ul>"

    return f"""<!DOCTYPE html>
<html lang="{html_lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ --bg: #0d1117; --fg: #e6edf3; --accent: #58a6ff; --card-bg: #161b22; --border: #30363d; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--fg); line-height: 1.7; padding: 2rem; max-width: 900px; margin: 0 auto; }}
  h1 {{ color: var(--accent); margin: 1.5rem 0 0.5rem; font-size: 1.8rem; }}
  h2 {{ color: var(--accent); margin: 1.2rem 0 0.4rem; font-size: 1.4rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }}
  h3 {{ margin: 1rem 0 0.3rem; font-size: 1.1rem; }}
  p {{ margin: 0.5rem 0; }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }}
  li {{ margin: 0.3rem 0; margin-left: 1.5rem; }}
  ul, ol {{ margin: 0.5rem 0; padding-left: 1.5rem; }}
  table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
  th, td {{ border: 1px solid var(--border); padding: 0.4rem 0.8rem; text-align: left; }}
  th {{ background: var(--card-bg); }}
  blockquote {{ border-left: 3px solid var(--accent); padding-left: 1rem; margin: 0.5rem 0; color: #8b949e; }}
  code {{ background: #1f2937; padding: 1px 4px; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #1f2937; padding: 1rem; border-radius: 6px; overflow-x: auto; margin: 1rem 0; }}
  pre code {{ background: none; padding: 0; }}
  .cite {{ background: #1f2937; padding: 2px 6px; border-radius: 3px; color: var(--accent); cursor: pointer; font-size: 0.85em; user-select: text; position: relative; display: inline-block; }}
  .cite:hover {{ background: #2d3748; }}
  .cite-tip {{ display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; padding: 0.5rem 0.75rem; font-size: 0.8rem; color: var(--fg); white-space: normal; width: max-content; max-width: 400px; z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.4); line-height: 1.4; margin-bottom: 4px; }}
  .cite:hover .cite-tip {{ display: block; }}
  .appendix {{ margin-top: 3rem; padding-top: 1rem; border-top: 2px solid var(--border); }}
  .appendix ul {{ list-style: disc; padding-left: 1.5rem; }}
  .appendix li {{ margin: 0.3rem 0; font-size: 0.9rem; }}
  .issue {{ background: #2d1b1b; padding: 0.5rem; border-radius: 4px; margin: 0.3rem 0; font-size: 0.9rem; }}
  .meta {{ color: #8b949e; font-size: 0.85rem; margin-bottom: 2rem; }}
  details {{ margin-top: 1rem; }}
  summary {{ cursor: pointer; color: var(--accent); font-size: 1.1rem; }}
</style>
</head>
<body>
<div class="meta">{meta_prefix} -- {_esc(topic_summary)}</div>

{blog_html}

<div class="appendix">
<h2>{appendix_title}</h2>
<ul>{facts_html}</ul>

{warnings_section}

{review_section}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown_report(report: dict) -> str:
    """Render clean markdown."""
    lang = report.get("language", "en")
    blog = report.get("blog", "")
    dossier = report.get("dossier", {})
    conf_label = _t(lang, "confidence")
    appendix_title = _t(lang, "appendix")
    warnings_title = _t(lang, "warnings")

    parts = [blog, f"\n\n---\n\n## {appendix_title}\n"]
    for st in dossier.get("subtopics", []):
        parts.append(
            f"- **{st.get('name', '')}**: {st.get('key_finding', '')} ({conf_label}: {st.get('confidence', '?')})"
        )
    warnings = dossier.get("accuracy_warnings", [])
    if warnings:
        parts.append(f"\n## {warnings_title}\n")
        for w in warnings:
            parts.append(f"- {w}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(text) -> str:
    """HTML-escape a value, handling None and non-string types."""
    import html

    return html.escape(str(text) if text is not None else "", quote=True)
