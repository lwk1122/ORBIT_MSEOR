"""Per-topic configuration with typed accessors and defaults."""

import json
import re
import sqlite3

from .db import get_db

DEFAULTS: dict[str, str] = {
    "llm_provider": "minimax",
    "search_api": "minimax",
    "api_consult_provider": "minimax",
    "control_provider": "minimax",
    "vote_provider": "minimax",
    "rag_provider": "minimax",
    "writer_provider": "minimax",
    "fact_provider": "minimax",
    "code_provider": "minimax",
    "web_provider": "minimax",
    "temperature": "0.7",
    "max_rounds": "7",
    "subtopic_candidate_count": "3",
    "decision_pass_ratio": "0.667",
    "sandbox_packages": "[]",  # JSON list of extra pip packages
    "hitl_plan_approval": "0",  # PAUSE① default OFF
    "hitl_subtopic_review": "0",  # PAUSE② default OFF
    "hitl_final_review": "0",  # PAUSE③ default OFF
    "hitl_replan_pause": "0",  # Replan HITL default OFF
    "max_replan_rounds": "1",  # Max replan cycles (0 = no replan, 1 = one replan)
    "blog_language": "zh",  # zh | en
    "domain_profile": "base",  # base | mse
    "mse_workflow_mode": "modeling_fast",  # modeling_fast | modeling_reviewed
    "mse_candidate_count": "3",  # pass@k modeling candidates for MiniMax tournament helpers
}

# Map llm_provider config values to broker profile constants
_PROVIDER_PROFILE_MAP = {
    "minimax": "minimax",
}

_PROVIDER_CONFIG_KEYS = frozenset(
    {
        "llm_provider",
        "api_consult_provider",
        "control_provider",
        "vote_provider",
        "rag_provider",
        "writer_provider",
        "fact_provider",
        "code_provider",
        "web_provider",
    }
)


def _is_missing_topic_config_table(exc: sqlite3.OperationalError) -> bool:
    return "no such table: topicconfig" in str(exc).lower()


def get(topic_id: int, key: str) -> str:
    """Get config value with default fallback."""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT config_value FROM TopicConfig WHERE topic_id = ? AND config_key = ?",
                (topic_id, key),
            ).fetchone()
            if row:
                return row["config_value"]
    except sqlite3.OperationalError as exc:
        if not _is_missing_topic_config_table(exc):
            raise
    return DEFAULTS.get(key, "")


def get_explicit(topic_id: int, key: str) -> str | None:
    """Get a stored config value without applying defaults."""
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT config_value FROM TopicConfig WHERE topic_id = ? AND config_key = ?",
                (topic_id, key),
            ).fetchone()
            return row["config_value"] if row else None
    except sqlite3.OperationalError as exc:
        if not _is_missing_topic_config_table(exc):
            raise
        return None


def get_int(topic_id: int, key: str) -> int:
    try:
        return int(get(topic_id, key))
    except (ValueError, TypeError):
        return int(DEFAULTS.get(key, "0"))


def get_float(topic_id: int, key: str) -> float:
    try:
        return float(get(topic_id, key))
    except (ValueError, TypeError):
        return float(DEFAULTS.get(key, "0.0"))


def get_bool(topic_id: int, key: str) -> bool:
    return get(topic_id, key) == "1"


def get_provider_profile(topic_id: int) -> str:
    """Map config value to broker PROFILE_* constant."""
    return get_provider_profile_for(topic_id, "llm_provider")


def get_provider_profile_for(topic_id: int, key: str, fallback_key: str = "") -> str:
    """Return broker provider profile for a provider config key.

    Provider routing is MiniMax-only in this build. Unknown legacy values are
    normalized to MiniMax.
    """
    if key not in _PROVIDER_CONFIG_KEYS:
        raise ValueError(f"Unknown provider config key: {key}")

    provider = get_explicit(topic_id, key)
    if provider is None and key == "web_provider":
        legacy_search = get_explicit(topic_id, "search_api")
        if legacy_search:
            provider = legacy_search
    if provider is None and fallback_key:
        provider = get_explicit(topic_id, fallback_key)
    if provider is None:
        provider = get(topic_id, key)
    if not provider and fallback_key:
        provider = get(topic_id, fallback_key)
    return _PROVIDER_PROFILE_MAP.get(provider, "minimax")


def get_provider_profile_override(topic_id: int) -> str | None:
    """Return explicit provider override, or None to use role defaults."""
    provider = get_explicit(topic_id, "llm_provider")
    if provider is None:
        return None
    return _PROVIDER_PROFILE_MAP.get(provider, "minimax")


def get_sandbox_packages(topic_id: int) -> list[str]:
    raw = get(topic_id, "sandbox_packages")
    try:
        packages = json.loads(raw)
        if isinstance(packages, list):
            return [str(p).strip() for p in packages if str(p).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def set_config(topic_id: int, key: str, value: str) -> None:
    """Set a single config value."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO TopicConfig (topic_id, config_key, config_value) VALUES (?, ?, ?)",
            (topic_id, key, value),
        )


def get_all(topic_id: int) -> dict[str, str]:
    """Get all config values merged with defaults."""
    result = dict(DEFAULTS)
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT config_key, config_value FROM TopicConfig WHERE topic_id = ?",
                (topic_id,),
            ).fetchall()
            for row in rows:
                result[row["config_key"]] = row["config_value"]
    except sqlite3.OperationalError as exc:
        if not _is_missing_topic_config_table(exc):
            raise
    return result


_VALID_CONFIG_KEYS = frozenset(DEFAULTS.keys())

# Value validators per key type — returns (ok, sanitized_value)
_BOOL_KEYS = {
    "hitl_plan_approval",
    "hitl_subtopic_review",
    "hitl_final_review",
    "hitl_replan_pause",
}
_INT_KEYS = {
    "max_rounds": (1, 20),
    "subtopic_candidate_count": (1, 10),
    "max_replan_rounds": (0, 10),
    "mse_candidate_count": (1, 8),
}
_FLOAT_KEYS = {"temperature": (0.0, 2.0), "decision_pass_ratio": (0.5, 1.0)}
_ENUM_KEYS = {
    "llm_provider": {"minimax"},
    "api_consult_provider": {"minimax"},
    "control_provider": {"minimax"},
    "vote_provider": {"minimax"},
    "rag_provider": {"minimax"},
    "writer_provider": {"minimax"},
    "fact_provider": {"minimax"},
    "code_provider": {"minimax"},
    "web_provider": {"minimax"},
    "search_api": {"minimax"},
    "blog_language": {"zh", "en"},
    "domain_profile": {"base", "mse", "management_science_engineering"},
    "mse_workflow_mode": {"modeling_fast", "modeling_reviewed"},
}


def validate_config_value(key: str, value: str) -> tuple[bool, str]:
    """Validate a config value. Returns (is_valid, error_message)."""
    if key not in _VALID_CONFIG_KEYS:
        return False, f"Unknown config key: {key}"
    if key in _BOOL_KEYS:
        if value not in ("0", "1"):
            return False, f"{key} must be '0' or '1'"
    elif key in _INT_KEYS:
        lo, hi = _INT_KEYS[key]
        try:
            v = int(value)
            if not (lo <= v <= hi):
                return False, f"{key} must be between {lo} and {hi}"
        except (ValueError, TypeError):
            return False, f"{key} must be an integer"
    elif key in _FLOAT_KEYS:
        lo, hi = _FLOAT_KEYS[key]
        try:
            v = float(value)
            if not (lo <= v <= hi):
                return False, f"{key} must be between {lo} and {hi}"
        except (ValueError, TypeError):
            return False, f"{key} must be a number"
    elif key in _ENUM_KEYS:
        if value not in _ENUM_KEYS[key]:
            return False, f"{key} must be one of: {', '.join(sorted(_ENUM_KEYS[key]))}"
    elif key == "sandbox_packages":
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                return False, "sandbox_packages must be a JSON list"
            from .code_sandbox import _SAFE_PKG_RE, _ALLOWED_PIP_PACKAGES

            for pkg in parsed:
                if not isinstance(pkg, str):
                    return (
                        False,
                        f"sandbox_packages: each item must be a string, got {type(pkg).__name__}",
                    )
                base_name = re.split(r"[<>=!~\[]", pkg)[0].strip().lower()
                if base_name not in _ALLOWED_PIP_PACKAGES:
                    return (
                        False,
                        f"sandbox_packages: '{pkg}' not in allowed packages list",
                    )
                if not _SAFE_PKG_RE.fullmatch(pkg):
                    return False, f"sandbox_packages: '{pkg}' has invalid format"
        except (json.JSONDecodeError, TypeError):
            return False, "sandbox_packages must be valid JSON"
    return True, ""


def set_bulk(topic_id: int, config: dict[str, str]) -> tuple[bool, str]:
    """Set multiple config values at once. Validates keys and values.

    Returns (success, error_message).
    """
    # Validate all keys and values before writing anything
    for key, value in config.items():
        value_str = str(value)
        ok, err = validate_config_value(key, value_str)
        if not ok:
            return False, err

    with get_db() as conn:
        for key, value in config.items():
            conn.execute(
                "INSERT OR REPLACE INTO TopicConfig (topic_id, config_key, config_value) VALUES (?, ?, ?)",
                (topic_id, key, str(value)),
            )
    return True, ""
