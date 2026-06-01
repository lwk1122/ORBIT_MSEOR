"""Ledger normalization module: four-layer stack + Skynet seeding.

L1: Timeframe normalizer (pure regex)
L2: Entity/Attribute alias resolution
L3: Value/Unit normalizer (scale-invariant)
L4: UNIQUE constraint on normalized keys (handled by db.upsert_ledger_entry)
"""

import datetime
import json
import logging
import math
import re
from typing import Callable

from . import db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# L1: Timeframe normalizer
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

_MONTH_PATTERN = "|".join(_MONTH_MAP)

_TIMEFRAME_PATTERNS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    (
        re.compile(r"Q([1-4])\s*(\d{4})", re.IGNORECASE),
        lambda m: f"{m.group(2)}-Q{m.group(1)}",
    ),
    (
        re.compile(r"(\d{4})\s*Q([1-4])", re.IGNORECASE),
        lambda m: f"{m.group(1)}-Q{m.group(2)}",
    ),
    (
        re.compile(r"H([12])\s*(\d{4})", re.IGNORECASE),
        lambda m: f"{m.group(2)}-H{m.group(1)}",
    ),
    (
        re.compile(r"(\d{4})\s*H([12])", re.IGNORECASE),
        lambda m: f"{m.group(1)}-H{m.group(2)}",
    ),
    (
        re.compile(rf"({_MONTH_PATTERN})\w*\s+(\d{{4}})", re.IGNORECASE),
        lambda m: f"{m.group(2)}-{_MONTH_MAP[m.group(1)[:3].lower()]}",
    ),
    (
        re.compile(r"(?:end\s+of|late)\s+(\d{4})", re.IGNORECASE),
        lambda m: f"{m.group(1)}-Q4",
    ),
    (
        re.compile(r"(?:early|start\s+of)\s+(\d{4})", re.IGNORECASE),
        lambda m: f"{m.group(1)}-Q1",
    ),
    (re.compile(r"mid[- ]?(\d{4})", re.IGNORECASE), lambda m: f"{m.group(1)}-Q2"),
    (re.compile(r"^(\d{4})$"), lambda m: m.group(1)),
]


def normalize_timeframe(raw: str) -> str:
    """Returns normalized timeframe string or '' if no match."""
    text = raw.strip()
    if not text:
        return ""
    for pattern, replacer in _TIMEFRAME_PATTERNS:
        match = pattern.search(text)
        if match:
            return replacer(match)
    return ""


# ---------------------------------------------------------------------------
# L1b: Timeframe → interval conversion (ISO date ranges)
# ---------------------------------------------------------------------------

_MONTH_LAST_DAY = {
    "01": "31",
    "02": "28",
    "03": "31",
    "04": "30",
    "05": "31",
    "06": "30",
    "07": "31",
    "08": "31",
    "09": "30",
    "10": "31",
    "11": "30",
    "12": "31",
}

_QUARTER_RANGES = {
    "1": ("01-01", "03-31"),
    "2": ("04-01", "06-30"),
    "3": ("07-01", "09-30"),
    "4": ("10-01", "12-31"),
}

_HALF_RANGES = {
    "1": ("01-01", "06-30"),
    "2": ("07-01", "12-31"),
}


def _month_range(m: re.Match[str]) -> tuple[str, str]:
    m1 = _MONTH_MAP.get(m.group(1)[:3].lower())
    m2 = _MONTH_MAP.get(m.group(2)[:3].lower())
    year = m.group(3)
    if m1 and m2:
        return (f"{year}-{m1}-01", f"{year}-{m2}-{_MONTH_LAST_DAY[m2]}")
    return (f"{year}-01-01", f"{year}-12-31")


def _year_range(m: re.Match[str]) -> tuple[str, str]:
    y1, y2 = m.group(1), m.group(2)
    if int(y2) > 12:
        return (f"{y1}-01-01", f"{y2}-12-31")
    return (f"{y1}-01-01", f"{y1}-12-31")


_INTERVAL_PATTERNS: list[
    tuple[re.Pattern[str], Callable[[re.Match[str]], tuple[str, str]]]
] = [
    # Quarters
    (
        re.compile(r"Q([1-4])\s*(\d{4})", re.IGNORECASE),
        lambda m: (
            f"{m.group(2)}-{_QUARTER_RANGES[m.group(1)][0]}",
            f"{m.group(2)}-{_QUARTER_RANGES[m.group(1)][1]}",
        ),
    ),
    (
        re.compile(r"(\d{4})[\s-]*Q([1-4])", re.IGNORECASE),
        lambda m: (
            f"{m.group(1)}-{_QUARTER_RANGES[m.group(2)][0]}",
            f"{m.group(1)}-{_QUARTER_RANGES[m.group(2)][1]}",
        ),
    ),
    # Halves
    (
        re.compile(r"H([12])\s*(\d{4})", re.IGNORECASE),
        lambda m: (
            f"{m.group(2)}-{_HALF_RANGES[m.group(1)][0]}",
            f"{m.group(2)}-{_HALF_RANGES[m.group(1)][1]}",
        ),
    ),
    (
        re.compile(r"(\d{4})[\s-]*H([12])", re.IGNORECASE),
        lambda m: (
            f"{m.group(1)}-{_HALF_RANGES[m.group(2)][0]}",
            f"{m.group(1)}-{_HALF_RANGES[m.group(2)][1]}",
        ),
    ),
    # Month ranges: "Jan-Mar 2024"
    (
        re.compile(
            rf"({_MONTH_PATTERN})\w*\s*[-–]\s*({_MONTH_PATTERN})\w*\s+(\d{{4}})",
            re.IGNORECASE,
        ),
        _month_range,
    ),
    # Single month: "March 2025"
    (
        re.compile(rf"({_MONTH_PATTERN})\w*\s+(\d{{4}})", re.IGNORECASE),
        lambda m: (
            f"{m.group(2)}-{_MONTH_MAP[m.group(1)[:3].lower()]}-01",
            f"{m.group(2)}-{_MONTH_MAP[m.group(1)[:3].lower()]}-{_MONTH_LAST_DAY[_MONTH_MAP[m.group(1)[:3].lower()]]}",
        ),
    ),
    # Relative qualifiers
    (
        re.compile(r"(?:end\s+of|late)\s+(\d{4})", re.IGNORECASE),
        lambda m: (f"{m.group(1)}-10-01", f"{m.group(1)}-12-31"),
    ),
    (
        re.compile(r"(?:early|start\s+of)\s+(\d{4})", re.IGNORECASE),
        lambda m: (f"{m.group(1)}-01-01", f"{m.group(1)}-03-31"),
    ),
    (
        re.compile(r"mid[- ]?(\d{4})", re.IGNORECASE),
        lambda m: (f"{m.group(1)}-04-01", f"{m.group(1)}-09-30"),
    ),
    (
        re.compile(r"(?:by|through|until|before)\s+(\d{4})", re.IGNORECASE),
        lambda m: (f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"),
    ),
    (
        re.compile(r"(?:after|post[-\s]?|since)\s*(\d{4})", re.IGNORECASE),
        lambda m: (f"{int(m.group(1))+1}-01-01", f"{int(m.group(1))+5}-12-31"),
    ),
    # Specific date: "2024-07-01"
    (
        re.compile(r"^(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"),
        lambda m: (m.group(0), m.group(0)),
    ),
    # Year-month: "2025-03"
    (
        re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$"),
        lambda m: (
            f"{m.group(1)}-{m.group(2)}-01",
            f"{m.group(1)}-{m.group(2)}-{_MONTH_LAST_DAY[m.group(2)]}",
        ),
    ),
    # Year ranges: "2024-2026"
    (re.compile(r"(\d{4})\s*[-–—]+\s*(\d{4})(?!\d)"), _year_range),
    (
        re.compile(r"(\d{4})\s+to\s+(\d{4})", re.IGNORECASE),
        lambda m: (f"{m.group(1)}-01-01", f"{m.group(2)}-12-31"),
    ),
    # Decades: "2020s"
    (
        re.compile(r"^(\d{4})s$", re.IGNORECASE),
        lambda m: (f"{m.group(1)}-01-01", f"{int(m.group(1))+9}-12-31"),
    ),
    # Fiscal year: "FY2025"
    (
        re.compile(r"FY\s*(\d{4})", re.IGNORECASE),
        lambda m: (f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"),
    ),
    # Bare year (most generic — must be last)
    (
        re.compile(r"^(\d{4})$"),
        lambda m: (f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"),
    ),
]


def timeframe_to_interval(raw: str) -> tuple[str | None, str | None]:
    """Convert human-readable timeframe to (valid_from, valid_to) ISO dates.

    Returns (None, None) for empty/unrecognizable input (timeless entry).
    """
    text = raw.strip()
    if not text:
        return (None, None)
    for pattern, converter in _INTERVAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return converter(match)
    return (None, None)


def intervals_overlap(
    a: tuple[str | None, str | None], b: tuple[str | None, str | None]
) -> bool:
    """Check if two intervals overlap.

    None bounds mean unbounded (timeless).
    Two timeless entries (None, None) are considered overlapping.
    """
    a_from, a_to = a
    b_from, b_to = b
    if a_from is None and a_to is None and b_from is None and b_to is None:
        return True
    if (a_from is None and a_to is None) or (b_from is None and b_to is None):
        return True
    if a_to is not None and b_from is not None and a_to < b_from:
        return False
    if b_to is not None and a_from is not None and b_to < a_from:
        return False
    return True


# ---------------------------------------------------------------------------
# L1c: parse_time_field (MM/DD/YYYY prompt format + fallback)
# ---------------------------------------------------------------------------

_MMDDYYYY_RE = re.compile(r"(\d{2})/(\d{2})/(\d{4})\s*-\s*(\d{2})/(\d{2})/(\d{4})")


def parse_time_field(raw: str | None) -> tuple[str | None, str | None]:
    """Parse TIME field from prompt output.

    Tries MM/DD/YYYY-MM/DD/YYYY format first (new prompt), then falls back
    to timeframe_to_interval() for legacy/non-compliant output.
    Returns (valid_from, valid_to) as ISO dates or (None, None).
    """
    if not raw or not raw.strip():
        return (None, None)
    text = raw.strip()
    if text.upper() == "NONE":
        return (None, None)
    m = _MMDDYYYY_RE.search(text)
    if m:
        try:
            vf_date = datetime.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
            vt_date = datetime.date(int(m.group(6)), int(m.group(4)), int(m.group(5)))
            return (vf_date.isoformat(), vt_date.isoformat())
        except ValueError:
            pass  # invalid date — fall through to timeframe_to_interval
    return timeframe_to_interval(text)


# ---------------------------------------------------------------------------
# Credibility-based upsert logic
# ---------------------------------------------------------------------------

CREDIBILITY_SIMILAR_THRESHOLD = 0.1
VALUE_EPSILON = 1e-6
DOMAIN_SCORE_THRESHOLD = 0.6

_REJECTED_ENTRY_TYPES = {"agent_claim", "synthesized_conclusion"}


# ---------------------------------------------------------------------------
# Timeframe fallback: derive from valid_from/valid_to when normalize_timeframe
# returns empty
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(\d{4})")
_QUARTER_FROM_MONTH = {
    "01": "Q1",
    "02": "Q1",
    "03": "Q1",
    "04": "Q2",
    "05": "Q2",
    "06": "Q2",
    "07": "Q3",
    "08": "Q3",
    "09": "Q3",
    "10": "Q4",
    "11": "Q4",
    "12": "Q4",
}


def _derive_timeframe_from_dates(valid_from: str | None, valid_to: str | None) -> str:
    """Derive a normalized timeframe string from ISO date bounds.

    Falls back to extracting year (and quarter if month available).
    Returns '' if nothing can be derived.
    """
    date_str = valid_from or valid_to
    if not date_str:
        return ""
    # Try to extract YYYY-MM or just YYYY
    parts = date_str.split("-")
    if len(parts) >= 2 and len(parts[0]) == 4:
        year = parts[0]
        month = parts[1] if len(parts[1]) == 2 else None
        if month and month in _QUARTER_FROM_MONTH:
            return f"{year}-{_QUARTER_FROM_MONTH[month]}"
        return year
    m = _YEAR_RE.search(date_str)
    if m:
        return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Ledger validation gate (Phase D: LE-1)
# ---------------------------------------------------------------------------

_PROSE_WORDS_RE = re.compile(
    r"\b(is|was|are|were|should|could|would|the|this|because|therefore|"
    r"methodology|measured|approach|however|although|furthermore)\b",
    re.IGNORECASE,
)

_UNIT_ENUM_SET = frozenset(
    [
        "NONE",
        "USD",
        "EUR",
        "CNY",
        "JPY",
        "%",
        "BPS",
        "USD/CNY",
        "USD/JPY",
        "TRILLION_USD",
        "BILLION_USD",
        "MILLION_USD",
        "MULTIPLIER",
        "FLOPS",
        "SECONDS",
        "WATTS",
        "OTHER",
    ]
)


def _validate_ledger_entry(
    *,
    raw_value: str,
    vmin: float | None,
    vmax: float | None,
    unit: str | None,
    entity_id: int | None = None,
    attribute_id: int | None = None,
) -> str | None:
    """Validate a ledger entry before insert.

    Returns None if valid, or a rejection reason string.
    """
    # Rule 1: Numeric bounds — must be finite and min <= max
    if vmin is not None and not math.isfinite(vmin):
        return "non-finite vmin"
    if vmax is not None and not math.isfinite(vmax):
        return "non-finite vmax"
    if vmin is not None and vmax is not None and vmin > vmax:
        return "vmin > vmax"

    # Rule 2: Unit must be in UNIT_ENUM or None
    if unit is not None and unit not in _UNIT_ENUM_SET:
        return f"invalid unit: {unit}"

    # Rule 3: Prose detection — reject if raw_value has >5 whitespace tokens
    # and no numeric was parsed
    word_count = len(raw_value.split())
    if word_count > 5 and vmin is None and vmax is None:
        return "prose without numeric"

    # Rule 4: Prose patterns — reject if value matches common prose words
    # when word count > 3
    if word_count > 3 and _PROSE_WORDS_RE.search(raw_value):
        return "prose pattern detected"

    # Rule 5: Entity/Attribute not UNSPECIFIED (id == 0)
    if entity_id is not None and entity_id == 0:
        return "unspecified entity"
    if attribute_id is not None and attribute_id == 0:
        return "unspecified attribute"

    return None


def _values_equal(min_a: float, max_a: float, min_b: float, max_b: float) -> bool:
    """Check if two value ranges are effectively identical."""

    def close(a: float, b: float) -> bool:
        if a == 0 and b == 0:
            return True
        return abs(a - b) / max(abs(a), abs(b), 1e-30) < VALUE_EPSILON

    return close(min_a, min_b) and close(max_a, max_b)


def determine_upsert_action(
    new_entry: dict, existing_entries: list[dict]
) -> tuple[str, int | None]:
    """Determine what action to take for a new ledger entry vs existing entries.

    new_entry keys: entity_id, attribute_id, valid_from, valid_to, min_val, max_val, unit, domain_score
    existing_entries: list of dicts with same keys + 'id'

    Returns: (action, matched_entry_id) where action is one of
    INSERT, DUPLICATE, REPLACE, DISCARD, CONFLICT, REJECT.
    matched_entry_id is the id of the existing entry that triggered the action (None for INSERT/REJECT).
    """
    ds = new_entry.get("domain_score")
    if ds is not None and ds < DOMAIN_SCORE_THRESHOLD:
        return "REJECT", None

    new_min = new_entry.get("min_val")
    new_max = new_entry.get("max_val")
    if new_min is None or new_max is None:
        return "INSERT", None
    try:
        new_min, new_max = float(new_min), float(new_max)
    except (ValueError, TypeError):
        return "INSERT", None
    new_unit = new_entry.get("unit") or ""

    # Sort by domain_score descending so highest-credibility entry is compared first
    sorted_existing = sorted(
        existing_entries, key=lambda e: (e.get("domain_score") or 0), reverse=True
    )
    for existing in sorted_existing:
        if existing.get("entity_id") != new_entry.get("entity_id"):
            continue
        if existing.get("attribute_id") != new_entry.get("attribute_id"):
            continue
        ex_unit = existing.get("unit") or ""
        if ex_unit != new_unit:
            continue
        # Check time overlap
        new_interval = (new_entry.get("valid_from"), new_entry.get("valid_to"))
        ex_interval = (existing.get("valid_from"), existing.get("valid_to"))
        if not intervals_overlap(new_interval, ex_interval):
            continue
        # Overlapping time + same entity/attr/unit → compare values
        ex_min = existing.get("min_val")
        ex_max = existing.get("max_val")
        if ex_min is None or ex_max is None:
            continue
        try:
            ex_min, ex_max = float(ex_min), float(ex_max)
        except (ValueError, TypeError):
            continue
        score_diff = (new_entry.get("domain_score") or 0) - (
            existing.get("domain_score") or 0
        )
        matched_id = existing.get("id")

        if _values_equal(new_min, new_max, ex_min, ex_max):
            if score_diff > 0:
                return "REPLACE", matched_id
            return "DUPLICATE", matched_id

        if score_diff > CREDIBILITY_SIMILAR_THRESHOLD:
            return "REPLACE", matched_id
        elif score_diff < -CREDIBILITY_SIMILAR_THRESHOLD:
            return "DISCARD", matched_id
        else:
            return "CONFLICT", matched_id

    return "INSERT", None


# ---------------------------------------------------------------------------
# L2: Entity/Attribute alias resolution
# ---------------------------------------------------------------------------

ALIAS_CONFIRMATION_THRESHOLD = 2


def resolve_entity(
    raw_text: str, topic_id: int, round_number: int | None = None, conn=None
) -> int | None:
    """Lookup entity by alias. Returns entity_id or None.

    Confirmed alias -> return immediately, update last_mentioned_round.
    Unconfirmed -> increment match_count, promote if >= threshold, return entity_id.
    No match -> return None (Phase 2 Clerk will handle).
    All operations run in a single transaction for atomicity.
    """
    alias = raw_text.strip().lower()
    if not alias:
        return None
    return db.resolve_entity_alias_atomic(
        alias, topic_id, ALIAS_CONFIRMATION_THRESHOLD, round_number, conn=conn
    )


def resolve_attribute(raw_text: str, topic_id: int, conn=None) -> int | None:
    """Same pattern for attributes. Single-transaction resolution."""
    alias = raw_text.strip().lower()
    if not alias:
        return None
    return db.resolve_attribute_alias_atomic(
        alias, topic_id, ALIAS_CONFIRMATION_THRESHOLD, conn=conn
    )


def add_entity_with_aliases(
    topic_id: int,
    canonical_name: str,
    entity_type: str | None,
    aliases: list[str],
    confirmed: bool = False,
) -> int:
    """Create entity + aliases in a single transaction. Handles UNIQUE conflicts."""
    return db.create_entity_with_aliases_batch(
        topic_id, canonical_name, entity_type, aliases, confirmed
    )


def add_attribute_with_aliases(
    topic_id: int,
    canonical_name: str,
    value_type: str | None,
    aliases: list[str],
    confirmed: bool = False,
) -> int:
    """Create attribute + aliases in a single transaction. Handles UNIQUE conflicts."""
    return db.create_attribute_with_aliases_batch(
        topic_id, canonical_name, value_type, aliases, confirmed
    )


# ---------------------------------------------------------------------------
# L3: Value/Unit normalizer
# ---------------------------------------------------------------------------

UNIT_ENUM = [
    "USD",
    "EUR",
    "CNY",
    "JPY",
    "%",
    "BPS",
    "USD/CNY",
    "USD/JPY",
    "TRILLION_USD",
    "BILLION_USD",
    "MILLION_USD",
    "MULTIPLIER",
    "OTHER",
]

SCALE_WORDS: dict[str, float] = {
    "trillion": 1e12,
    "trn": 1e12,
    "billion": 1e9,
    "bln": 1e9,
    "bn": 1e9,
    "million": 1e6,
    "mln": 1e6,
    "mn": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}

_UNIT_ALIASES: dict[str, str] = {
    "$": "USD",
    "usd": "USD",
    "dollars": "USD",
    "dollar": "USD",
    "€": "EUR",
    "eur": "EUR",
    "euros": "EUR",
    "euro": "EUR",
    # ¥ is ambiguous between CNY and JPY — omitted; use explicit aliases
    "cny": "CNY",
    "yuan": "CNY",
    "rmb": "CNY",
    "jpy": "JPY",
    "yen": "JPY",
    "%": "%",
    "percent": "%",
    "percentage": "%",
    "pct": "%",
    "bps": "BPS",
    "basis points": "BPS",
    "basis point": "BPS",
    "usd/cny": "USD/CNY",
    "usd/jpy": "USD/JPY",
}
_UNIT_ALIASES_SORTED = sorted(_UNIT_ALIASES.items(), key=lambda x: -len(x[0]))

_NUMBER_RE = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?")
_SCALE_PATTERN = "|".join(
    re.escape(k) for k in sorted(SCALE_WORDS, key=len, reverse=True)
)
# Allow scale words to follow digits without whitespace (e.g., "100k", "1.5bn")
_SCALE_RE = re.compile(rf"(?:\d\s*)?({_SCALE_PATTERN})\b", re.IGNORECASE)


def _parse_number(s: str) -> float | None:
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _detect_scale(text: str) -> float:
    m = _SCALE_RE.search(text)
    if m:
        return SCALE_WORDS[m.group(1).lower()]
    return 1.0


def _detect_unit(text: str) -> str | None:
    lower = text.lower()
    for alias, unit in _UNIT_ALIASES_SORTED:
        if alias in lower:
            return unit
    return None


def normalize_value(raw: str) -> tuple[float | None, float | None, str | None]:
    """Scale-invariant parsing. Returns (min, max, unit).

    '6.8000' -> (6.8, 6.8, None)
    '1.5 Trillion USD' -> (1.5e12, 1.5e12, 'USD')
    '1500 Billion USD' -> (1.5e12, 1.5e12, 'USD')
    '2.5%-3.0%' -> (2.5, 3.0, '%')
    'up to 50 billion' -> (None, 5e10, None)
    """
    text = raw.strip()
    if not text:
        return None, None, None

    unit = _detect_unit(text)
    scale = _detect_scale(text)

    # Check for range patterns: X-Y, X to Y, X–Y
    range_match = re.search(
        r"([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:%\s*)?(?:[-–]|to)\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%)?",
        text,
        re.IGNORECASE,
    )
    if range_match:
        lo = _parse_number(range_match.group(1))
        hi = _parse_number(range_match.group(2))
        if range_match.group(3) == "%":
            unit = "%"
        if lo is not None:
            lo *= scale
        if hi is not None:
            hi *= scale
        return lo, hi, unit

    # Check for one-sided ranges: "up to X", "at least X", "over X"
    up_to_match = re.search(
        r"(?:up\s+to|at\s+most|no\s+more\s+than)\s+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if up_to_match:
        hi = _parse_number(up_to_match.group(1))
        if hi is not None:
            hi *= scale
        return None, hi, unit

    at_least_match = re.search(
        r"(?:at\s+least|over|more\s+than|above)\s+([+-]?\d+(?:,\d{3})*(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if at_least_match:
        lo = _parse_number(at_least_match.group(1))
        if lo is not None:
            lo *= scale
        return lo, None, unit

    # Single number
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        val = _parse_number(numbers[0])
        if val is not None:
            val *= scale
            return val, val, unit

    return None, None, unit


# ---------------------------------------------------------------------------
# Source reference normalization
# ---------------------------------------------------------------------------

_CITATION_RE = re.compile(r"\[?([DMWLFCEA]\d+)\]?")


def normalize_source_ref(raw: str) -> str:
    """Extract and normalize citation markers from a free-text source_ref.

    "dreamer M415 projection"               → "M415"
    "[M426][M427][M429]"                     → "M426 M427 M429"
    '[L9] "the mathematical floor..."'       → "L9"
    "M418 M422 M423"                         → "M418 M422 M423"
    "L9"                                     → "L9"
    ""                                       → ""
    """
    if not raw or not raw.strip():
        return ""
    markers = _CITATION_RE.findall(raw)
    if not markers:
        # No citation markers — keep truncated original as fallback
        return raw.strip()[:100]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in markers:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return " ".join(unique)


# ---------------------------------------------------------------------------
# Decontextualization
# ---------------------------------------------------------------------------


def decontextualize_ledger_entry(
    entity_name: str,
    attr_name: str,
    value: str,
    timeframe: str,
    source_ref: str,
    source_domain: str | None = None,
    stat_type: str | None = None,
    value_std: float | None = None,
    value_p: float | None = None,
    value_n: int | None = None,
    baseline_entity_name: str | None = None,
    split: str | None = None,
    config_json: str | None = None,
) -> str:
    """Template-based decontextualization. No LLM."""
    # Enrich entity name with config details
    display_entity = entity_name
    if config_json:
        try:
            cfg = json.loads(config_json)
            if isinstance(cfg, dict):
                cfg_parts = [f"{k}={v}" for k, v in cfg.items() if k != "variant"]
                if cfg_parts:
                    display_entity = f"{entity_name} ({', '.join(cfg_parts)})"
                elif cfg.get("variant") and cfg["variant"] != "default":
                    display_entity = f"{entity_name} ({cfg['variant']})"
        except (json.JSONDecodeError, TypeError):
            pass

    # Build the main description based on stat_type
    if stat_type == "mean_std" and value_std is not None:
        main = f"{display_entity} {attr_name} = {value} +/- {value_std}"
    elif stat_type == "delta":
        vs = f" vs {baseline_entity_name}" if baseline_entity_name else ""
        main = f"{display_entity}{vs} {attr_name} delta = {value}"
    elif stat_type == "p_value":
        vs = f" vs {baseline_entity_name}" if baseline_entity_name else ""
        main = f"{display_entity}{vs} {attr_name} p = {value}"
    elif stat_type == "ratio":
        main = f"{display_entity} {attr_name} ratio = {value}"
    elif stat_type == "percentage":
        val_str = str(value).rstrip("%") if value else ""
        main = f"{display_entity} {attr_name} = {val_str}%"
    else:
        main = f"{display_entity} {attr_name}"
        if value:
            main += f" = {value}"

    parts = [main]
    # Append statistical context
    stat_extras = []
    if value_p is not None and stat_type != "p_value":
        stat_extras.append(f"p={value_p}")
    if value_n is not None:
        stat_extras.append(f"n={value_n}")
    if split:
        stat_extras.append(f"split={split}")
    if stat_extras:
        parts.append(f"({', '.join(stat_extras)})")
    if timeframe:
        parts.append(f"({timeframe})")
    if source_ref:
        domain = f" {source_domain}" if source_domain else ""
        parts.append(f"source: {source_ref}{domain}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------


def expand_abbreviations(text: str, topic_id: int) -> str:
    """Replace confirmed entity aliases with canonical names before embedding."""
    with db.get_db() as conn:
        rows = conn.execute(
            """SELECT a.alias_text, e.canonical_name
               FROM LedgerEntityAlias a
               JOIN LedgerEntity e ON e.id = a.entity_id
               WHERE e.topic_id = ? AND a.confirmed = 1""",
            (topic_id,),
        ).fetchall()

    alias_map = {
        r["alias_text"]: r["canonical_name"]
        for r in rows
        if r["alias_text"].lower() != r["canonical_name"].lower()
    }

    if not alias_map:
        return text

    # Sort longest-first to avoid partial replacements
    for alias in sorted(alias_map, key=len, reverse=True):
        escaped = re.escape(alias)
        left_b = r"\b" if alias[0].isalnum() or alias[0] == "_" else ""
        right_b = r"\b" if alias[-1].isalnum() or alias[-1] == "_" else ""
        pattern = re.compile(rf"{left_b}{escaped}{right_b}", re.IGNORECASE)
        text = pattern.sub(lambda _, a=alias: alias_map[a], text)

    return text


# ---------------------------------------------------------------------------
# Normalize-and-upsert orchestrator (Phase 2 entry point)
# ---------------------------------------------------------------------------


_PENDING_TTL_ROUNDS = 3


def normalize_and_upsert(
    *,
    topic_id: int,
    subtopic_id: int | None = None,
    raw_entity: str | None = None,
    entity_id: int | None = None,
    raw_attribute: str | None = None,
    attribute_id: int | None = None,
    raw_value: str,
    raw_timeframe: str | None = None,
    entry_type: str,
    source_ref: str,
    source_domain: str | None = None,
    domain_score: float | None = None,
    created_by: str | None = None,
    current_round: int | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    min_val: float | None = None,
    max_val: float | None = None,
    unit: str | None = None,
    # Phase G: rich statistical fields
    stat_type: str | None = None,
    value_mean: float | None = None,
    value_std: float | None = None,
    value_p: float | None = None,
    value_n: int | None = None,
    value_ci_lower: float | None = None,
    value_ci_upper: float | None = None,
    value_ci_level: float | None = None,
    baseline_entity_id: int | None = None,
    split: str | None = None,
    config_json: str | None = None,
) -> tuple[int | None, str]:
    """Full L1-L4 pipeline in a single DB connection. Returns (ledger_id, status).

    status: 'inserted', 'updated', 'skipped', 'pending', 'deduplicated',
            'rejected', 'replaced', 'discarded', 'conflict'.
    """
    # Entry-type gate: reject agent_claim / synthesized_conclusion
    if entry_type in _REJECTED_ENTRY_TYPES:
        logger.debug(
            "[ledger] Rejected entry_type=%s (source_ref=%s)", entry_type, source_ref
        )
        return None, "rejected"

    # Domain score gate
    if domain_score is not None and domain_score < DOMAIN_SCORE_THRESHOLD:
        logger.debug(
            "[ledger] Rejected low domain_score=%.2f (source_ref=%s)",
            domain_score,
            source_ref,
        )
        return None, "rejected"

    # Reject empty values early
    if not raw_value or not raw_value.strip():
        logger.debug("[ledger] Rejected empty raw_value (source_ref=%s)", source_ref)
        return None, "skipped"

    # L1: Timeframe — use pre-parsed valid_from/valid_to if provided, else parse
    if valid_from is None and valid_to is None:
        valid_from, valid_to = parse_time_field(raw_timeframe or "")
    normalized_tf = normalize_timeframe(raw_timeframe or "")
    # LE-4: Derive timeframe from valid_from/valid_to when normalize_timeframe empty
    if not normalized_tf and (valid_from or valid_to):
        normalized_tf = _derive_timeframe_from_dates(valid_from, valid_to)
    ttl = (current_round + _PENDING_TTL_ROUNDS) if current_round is not None else None

    # L3: Value/Unit — use pre-parsed min_val/max_val if provided, else parse
    # Preserve caller-provided unit; only derive from raw_value if not provided
    caller_unit = unit
    if min_val is not None or max_val is not None:
        vmin = min_val if min_val is not None else max_val
        vmax = max_val if max_val is not None else min_val
        if caller_unit is None or caller_unit in ("", "NONE"):
            _, _, unit = normalize_value(raw_value)
        else:
            unit = caller_unit
    else:
        vmin, vmax, parsed_unit = normalize_value(raw_value)
        if caller_unit and caller_unit not in ("", "NONE"):
            unit = caller_unit
        else:
            unit = parsed_unit
        # Normalize one-sided ranges to point values for dedup
        if vmin is not None and vmax is None:
            vmax = vmin
        elif vmax is not None and vmin is None:
            vmin = vmax

    # LE-1: Validation gate (rules 1-4: numeric bounds, unit, prose)
    rejection = _validate_ledger_entry(
        raw_value=raw_value,
        vmin=vmin,
        vmax=vmax,
        unit=unit,
    )
    if rejection:
        logger.debug(
            "[ledger] Validation rejected: %s (source_ref=%s)", rejection, source_ref
        )
        return None, "rejected"

    # Normalize source_ref to citation markers only
    source_ref = normalize_source_ref(source_ref)

    with db.get_db() as conn:
        # L2: Entity resolution
        if entity_id is None and raw_entity:
            entity_id = resolve_entity(raw_entity, topic_id, conn=conn)
        if entity_id is None:
            db.create_ledger_pending(
                topic_id,
                subtopic_id,
                raw_value,
                source_ref,
                None,
                "entity",
                current_round,
                ttl,
                conn=conn,
            )
            return None, "pending"

        # L2: Attribute resolution
        if attribute_id is None and raw_attribute:
            attribute_id = resolve_attribute(raw_attribute, topic_id, conn=conn)
        if attribute_id is None:
            db.create_ledger_pending(
                topic_id,
                subtopic_id,
                raw_value,
                source_ref,
                None,
                "attribute",
                current_round,
                ttl,
                conn=conn,
            )
            return None, "pending"

        # LE-1: Validation gate rule 5 (entity/attribute not UNSPECIFIED)
        rejection5 = _validate_ledger_entry(
            raw_value=raw_value,
            vmin=vmin,
            vmax=vmax,
            unit=unit,
            entity_id=entity_id,
            attribute_id=attribute_id,
        )
        if rejection5:
            logger.debug(
                "[ledger] Validation rejected: %s (source_ref=%s)",
                rejection5,
                source_ref,
            )
            return None, "rejected"

        # Credibility-based upsert: query existing entries for same (entity, attribute)
        existing_rows = conn.execute(
            """SELECT id, entity_id, attribute_id, value_numeric_min AS min_val,
                      value_numeric_max AS max_val, unit, domain_score,
                      valid_from, valid_to, source_ref
               FROM Ledger
               WHERE topic_id = ? AND entity_id = ? AND attribute_id = ?
                     AND status = 'accepted'""",
            (topic_id, entity_id, attribute_id),
        ).fetchall()
        existing_entries = [dict(r) for r in existing_rows]

        new_entry = {
            "entity_id": entity_id,
            "attribute_id": attribute_id,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "min_val": vmin,
            "max_val": vmax,
            "unit": unit,
            "domain_score": domain_score,
        }
        action, matched_id = determine_upsert_action(new_entry, existing_entries)

        if action == "REJECT":
            return None, "rejected"
        if action == "DISCARD":
            return None, "discarded"
        if action == "DUPLICATE":
            if matched_id is not None:
                db.merge_ledger_source_ref(matched_id, source_ref, conn=conn)
                return matched_id, "deduplicated"
            # Should not reach here — determine_upsert_action guarantees a match
            logger.warning("[ledger] DUPLICATE action but no matched_id returned")
            return None, "deduplicated"

        if action == "REPLACE":
            if matched_id is not None:
                # Merge old source_ref before overwriting fields
                db.merge_ledger_source_ref(matched_id, source_ref, conn=conn)
                # Recompute decontextualized for the replacement values
                entity = db.get_ledger_entity(entity_id, conn=conn)
                entity_name = entity["canonical_name"] if entity else "Unknown"
                attr = db.get_ledger_attribute(attribute_id, conn=conn)
                attr_name = attr["canonical_name"] if attr else "Unknown"
                # Resolve baseline entity name for decontextualization
                baseline_name = None
                if baseline_entity_id:
                    bl = db.get_ledger_entity(baseline_entity_id, conn=conn)
                    baseline_name = bl["canonical_name"] if bl else None
                decontext = decontextualize_ledger_entry(
                    entity_name,
                    attr_name,
                    raw_value,
                    normalized_tf,
                    source_ref,
                    source_domain,
                    stat_type=stat_type,
                    value_std=value_std,
                    value_p=value_p,
                    value_n=value_n,
                    baseline_entity_name=baseline_name,
                    split=split,
                    config_json=config_json,
                )
                conn.execute(
                    """UPDATE Ledger SET value = ?, value_numeric_min = ?,
                       value_numeric_max = ?, unit = ?, domain_score = ?,
                       source_domain = ?, valid_from = ?, valid_to = ?,
                       normalized_timeframe = ?, decontextualized = ?,
                       value_mean = ?, value_std = ?, value_ci_lower = ?,
                       value_ci_upper = ?, value_ci_level = ?, value_p = ?,
                       value_n = ?, value_stat_type = ?,
                       baseline_entity_id = ?, split = ?,
                       config_json = ?
                       WHERE id = ?""",
                    (
                        raw_value,
                        vmin,
                        vmax,
                        unit,
                        domain_score,
                        source_domain,
                        valid_from,
                        valid_to,
                        normalized_tf,
                        decontext,
                        value_mean,
                        value_std,
                        value_ci_lower,
                        value_ci_upper,
                        value_ci_level,
                        value_p,
                        value_n,
                        stat_type,
                        baseline_entity_id,
                        split,
                        config_json,
                        matched_id,
                    ),
                )
                return matched_id, "replaced"
            logger.warning(
                "[ledger] REPLACE action but no matched_id returned; falling through to INSERT"
            )

        # Decontextualize
        entity = db.get_ledger_entity(entity_id, conn=conn)
        entity_name = entity["canonical_name"] if entity else "Unknown"
        attr = db.get_ledger_attribute(attribute_id, conn=conn)
        attr_name = attr["canonical_name"] if attr else "Unknown"
        baseline_name = None
        if baseline_entity_id:
            bl = db.get_ledger_entity(baseline_entity_id, conn=conn)
            baseline_name = bl["canonical_name"] if bl else None
        decontext = decontextualize_ledger_entry(
            entity_name,
            attr_name,
            raw_value,
            normalized_tf,
            source_ref,
            source_domain,
            stat_type=stat_type,
            value_std=value_std,
            value_p=value_p,
            value_n=value_n,
            baseline_entity_name=baseline_name,
            split=split,
            config_json=config_json,
        )

        # L4: Upsert (INSERT or CONFLICT)
        lid, was_inserted = db.upsert_ledger_entry(
            topic_id=topic_id,
            subtopic_id=subtopic_id,
            entity_id=entity_id,
            attribute_id=attribute_id,
            value=raw_value,
            value_numeric_min=vmin,
            value_numeric_max=vmax,
            unit=unit,
            normalized_timeframe=normalized_tf,
            entry_type=entry_type,
            source_ref=source_ref,
            source_domain=source_domain,
            domain_score=domain_score,
            decontextualized=decontext,
            created_by=created_by,
            valid_from=valid_from,
            valid_to=valid_to,
            value_mean=value_mean,
            value_std=value_std,
            value_ci_lower=value_ci_lower,
            value_ci_upper=value_ci_upper,
            value_ci_level=value_ci_level,
            value_p=value_p,
            value_n=value_n,
            value_stat_type=stat_type,
            baseline_entity_id=baseline_entity_id,
            split=split,
            config_json=config_json,
            conn=conn,
        )

        # If this was a CONFLICT action, create a conflicts_with edge
        if action == "CONFLICT":
            new_unit_norm = unit or ""
            for ex in existing_entries:
                if ex["id"] == lid:
                    continue  # skip self-edge (ON CONFLICT UPDATE can reuse id)
                ex_unit = ex.get("unit") or ""
                if ex_unit != new_unit_norm:
                    continue  # only conflict with same-unit entries
                ex_interval = (ex.get("valid_from"), ex.get("valid_to"))
                new_interval = (valid_from, valid_to)
                if not intervals_overlap(new_interval, ex_interval):
                    continue
                # Skip entries with identical values — not a real conflict
                ex_min_v = ex.get("min_val")
                ex_max_v = ex.get("max_val")
                if (
                    ex_min_v is not None
                    and ex_max_v is not None
                    and vmin is not None
                    and vmax is not None
                ):
                    try:
                        if _values_equal(vmin, vmax, float(ex_min_v), float(ex_max_v)):
                            continue
                    except (ValueError, TypeError):
                        pass
                db.create_ledger_edge(
                    topic_id,
                    lid,
                    ex["id"],
                    "conflicts_with",
                    created_by="auto",
                    conn=conn,
                )

        if action == "CONFLICT":
            return lid, "conflict"
        return lid, "inserted" if was_inserted else "updated"


# ---------------------------------------------------------------------------
# Skynet seeding
# ---------------------------------------------------------------------------

SEED_PROMPT = """Analyze this topic/subtopic. Identify key entities and measurable attributes.

ENTITY = the subject being measured or observed (company, product, model, country, system).
  Examples: "NVIDIA", "Qwen2.5-3B", "USD/CNY", "Linux kernel"
  NOT metrics, benchmarks, or scores — those are attributes.

ATTRIBUTE = the measurable property of an entity (benchmark score, revenue, market share, latency).
  Examples: "MMLU Score", "Revenue", "Market Share", "HumanEval Score", "Inference Speed"
  NOT the entity itself.

Output strict JSON: {"entities": [{"name": "...", "type": "...", "aliases": [...]}],
                     "attributes": [{"name": "...", "value_type": "...", "aliases": [...]}]}"""


async def seed_ledger_from_topic(
    topic_id: int,
    subtopic_id: int,
    topic_summary: str,
    topic_detail: str,
    subtopic_summary: str,
    subtopic_detail: str,
) -> None:
    """Seeds entities/attributes/aliases for a new subtopic.

    1. Existing entities/attributes are already topic-scoped (shared across subtopics)
    2. Call the configured control model to generate additional ones for this subtopic
    3. Insert with confirmed=True (Skynet-seeded)
    4. Handles UNIQUE conflicts gracefully (INSERT OR IGNORE)
    Logs warning on LLM failure, does not crash.
    """
    from .master_graph import ask_control_model

    context = (
        f"Topic: {topic_summary}\n"
        f"Topic Detail: {topic_detail}\n"
        f"Subtopic: {subtopic_summary}\n"
        f"Subtopic Detail: {subtopic_detail}"
    )

    try:
        data = await ask_control_model(SEED_PROMPT, context, "skynet", topic_id=topic_id)
    except Exception as exc:
        logger.warning("Ledger seed LLM call failed: %s", exc)
        return

    if not isinstance(data, dict):
        logger.warning("Ledger seed returned non-dict response: %s", type(data))
        return
    if data.get("error"):
        logger.warning("Ledger seed returned error: %s", data)
        return
    if "entities" not in data and "attributes" not in data:
        logger.warning(
            "Ledger seed response missing expected keys: %s", list(data.keys())
        )
        return

    entities = data.get("entities", [])
    if isinstance(entities, list):
        for item in entities:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            aliases = [a for a in aliases if isinstance(a, str) and a.strip()]
            add_entity_with_aliases(
                topic_id, name.strip(), item.get("type"), aliases, confirmed=True
            )

    attributes = data.get("attributes", [])
    if isinstance(attributes, list):
        for item in attributes:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            aliases = [a for a in aliases if isinstance(a, str) and a.strip()]
            add_attribute_with_aliases(
                topic_id, name.strip(), item.get("value_type"), aliases, confirmed=True
            )

    logger.info(
        "Ledger seeded for topic=%s subtopic=%s: %d entities, %d attributes",
        topic_id,
        subtopic_id,
        len(entities) if isinstance(entities, list) else 0,
        len(attributes) if isinstance(attributes, list) else 0,
    )


# ---------------------------------------------------------------------------
# Numbered list helpers (for Clerk prompt)
# ---------------------------------------------------------------------------


def get_entity_numbered_list(
    topic_id: int, round_number: int, max_items: int = 15
) -> list[tuple[int, str]]:
    """Return entities ordered by last_mentioned_round DESC, limited to max_items."""
    with db.get_db() as conn:
        rows = conn.execute(
            """SELECT id, canonical_name FROM LedgerEntity
            WHERE topic_id = ?
            ORDER BY CASE WHEN last_mentioned_round IS NULL THEN 0 ELSE last_mentioned_round END DESC, id
            LIMIT ?""",
            (topic_id, max_items),
        ).fetchall()
        return [(r["id"], r["canonical_name"]) for r in rows]


def get_attribute_numbered_list(
    topic_id: int, max_items: int = 15
) -> list[tuple[int, str]]:
    """Return attributes ordered by id, limited to max_items."""
    with db.get_db() as conn:
        rows = conn.execute(
            """SELECT id, canonical_name FROM LedgerAttribute
            WHERE topic_id = ?
            ORDER BY id
            LIMIT ?""",
            (topic_id, max_items),
        ).fetchall()
        return [(r["id"], r["canonical_name"]) for r in rows]


# ---------------------------------------------------------------------------
# Auto-generate conflict edges
# ---------------------------------------------------------------------------


def auto_generate_conflict_edges(topic_id: int) -> int:
    """Create ``conflicts_with`` edges between all contested entry pairs.

    Returns the number of newly created edges.
    """
    contested_groups = db.get_contested_ledger_pairs(topic_id)
    pairs: list[tuple[int, int]] = []
    for group in contested_groups:
        ids = [e["id"] for e in group["entries"]]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
    return db.bulk_create_ledger_edges(
        topic_id, pairs, "conflicts_with", created_by="auto"
    )
