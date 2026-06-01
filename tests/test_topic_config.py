import os
import sqlite3
import pytest

from orbit_or.db import get_db, get_db_path, init_db
from orbit_or import api
from orbit_or import topic_config


@pytest.fixture(autouse=True)
def setup_teardown():
    os.environ["TESTING"] = "1"
    db_path = get_db_path()
    if os.path.exists(db_path):
        os.remove(db_path)
    init_db()
    yield
    if os.path.exists(db_path):
        os.remove(db_path)


def test_topic_config_table_exists():
    with get_db() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='TopicConfig'"
        ).fetchone()
        assert row is not None


def test_set_and_get_config():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_config(topic_id, "temperature", "0.9")
    assert topic_config.get(topic_id, "temperature") == "0.9"


def test_get_default_fallback():
    topic_id = api.create_topic("Test topic", "Test detail")
    # Should return default value when not explicitly set
    assert topic_config.get(topic_id, "temperature") == "0.7"
    assert topic_config.get(topic_id, "max_rounds") == "7"
    assert topic_config.get(topic_id, "llm_provider") == "minimax"
    assert topic_config.get(topic_id, "api_consult_provider") == "minimax"
    assert topic_config.get(topic_id, "web_provider") == "minimax"
    assert topic_config.get(topic_id, "mse_workflow_mode") == "modeling_fast"
    assert topic_config.get(topic_id, "mse_candidate_count") == "3"


def test_get_int():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_config(topic_id, "max_rounds", "10")
    assert topic_config.get_int(topic_id, "max_rounds") == 10


def test_get_float():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_config(topic_id, "decision_pass_ratio", "0.75")
    assert topic_config.get_float(topic_id, "decision_pass_ratio") == 0.75


def test_get_bool():
    topic_id = api.create_topic("Test topic", "Test detail")
    assert topic_config.get_bool(topic_id, "hitl_plan_approval") is False
    assert topic_config.get_bool(topic_id, "hitl_subtopic_review") is False
    topic_config.set_config(topic_id, "hitl_plan_approval", "0")
    assert topic_config.get_bool(topic_id, "hitl_plan_approval") is False


def test_get_provider_profile():
    topic_id = api.create_topic("Test topic", "Test detail")
    assert topic_config.get_provider_profile(topic_id) == "minimax"
    assert topic_config.get_provider_profile_override(topic_id) is None
    topic_config.set_config(topic_id, "llm_provider", "legacy_provider")
    assert topic_config.get_provider_profile(topic_id) == "minimax"
    assert topic_config.get_provider_profile_override(topic_id) == "minimax"
    topic_config.set_config(topic_id, "llm_provider", "minimax")
    assert topic_config.get_provider_profile(topic_id) == "minimax"
    assert topic_config.get_provider_profile_override(topic_id) == "minimax"


def test_get_provider_profile_for_stage_defaults_and_overrides():
    topic_id = api.create_topic("Test topic", "Test detail")
    assert topic_config.get_provider_profile_for(topic_id, "llm_provider") == "minimax"
    assert (
        topic_config.get_provider_profile_for(topic_id, "api_consult_provider")
        == "minimax"
    )
    assert topic_config.get_provider_profile_for(topic_id, "web_provider") == "minimax"

    topic_config.set_config(topic_id, "web_provider", "minimax")
    assert topic_config.get_provider_profile_for(topic_id, "web_provider") == "minimax"


def test_stage_provider_defaults_do_not_inherit_llm_provider_override():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_config(topic_id, "llm_provider", "minimax")
    assert topic_config.get_provider_profile_for(topic_id, "fact_provider") == "minimax"
    assert topic_config.get_provider_profile_for(topic_id, "code_provider") == "minimax"


def test_provider_profile_for_can_use_explicit_fallback_before_primary_default():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_config(topic_id, "control_provider", "legacy_provider")
    assert topic_config.get_provider_profile_for(topic_id, "web_provider") == "minimax"
    assert (
        topic_config.get_provider_profile_for(
            topic_id, "web_provider", fallback_key="control_provider"
        )
        == "minimax"
    )
    topic_config.set_config(topic_id, "web_provider", "minimax")
    assert (
        topic_config.get_provider_profile_for(
            topic_id, "web_provider", fallback_key="control_provider"
        )
        == "minimax"
    )


def test_web_provider_honors_search_api_when_not_explicit():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_config(topic_id, "search_api", "minimax")
    assert topic_config.get_provider_profile_for(topic_id, "web_provider") == "minimax"
    topic_config.set_config(topic_id, "web_provider", "minimax")
    assert topic_config.get_provider_profile_for(topic_id, "web_provider") == "minimax"


def test_get_sandbox_packages():
    topic_id = api.create_topic("Test topic", "Test detail")
    assert topic_config.get_sandbox_packages(topic_id) == []
    topic_config.set_config(topic_id, "sandbox_packages", '["networkx", "requests"]')
    assert topic_config.get_sandbox_packages(topic_id) == ["networkx", "requests"]


def test_get_all():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_config(topic_id, "temperature", "0.5")
    all_cfg = topic_config.get_all(topic_id)
    assert all_cfg["temperature"] == "0.5"
    # Defaults should be present
    assert "max_rounds" in all_cfg
    assert all_cfg["max_rounds"] == "7"


def test_set_bulk():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_bulk(topic_id, {"temperature": "0.3", "max_rounds": "5"})
    assert topic_config.get(topic_id, "temperature") == "0.3"
    assert topic_config.get_int(topic_id, "max_rounds") == 5


def test_create_topic_with_config():
    topic_id = api.create_topic(
        "Configured topic", "Detail", config={"temperature": "0.5", "max_rounds": "10"}
    )
    assert topic_config.get(topic_id, "temperature") == "0.5"
    assert topic_config.get_int(topic_id, "max_rounds") == 10


def test_api_wrappers():
    topic_id = api.create_topic("Test topic", "Test detail")
    api.set_topic_config(topic_id, "temperature", "0.8")
    assert api.get_topic_config(topic_id, "temperature") == "0.8"
    all_cfg = api.get_all_topic_config(topic_id)
    assert all_cfg["temperature"] == "0.8"


def test_insert_or_replace_on_duplicate():
    topic_id = api.create_topic("Test topic", "Test detail")
    topic_config.set_config(topic_id, "temperature", "0.5")
    topic_config.set_config(topic_id, "temperature", "0.9")
    assert topic_config.get(topic_id, "temperature") == "0.9"


# --- Validation tests ---


def test_validate_rejects_unknown_key():
    ok, err = topic_config.validate_config_value("nonexistent_key", "foo")
    assert ok is False
    assert "Unknown config key" in err


def test_validate_bool_keys():
    ok, _ = topic_config.validate_config_value("hitl_plan_approval", "1")
    assert ok is True
    ok, _ = topic_config.validate_config_value("hitl_plan_approval", "0")
    assert ok is True
    ok, err = topic_config.validate_config_value("hitl_plan_approval", "yes")
    assert ok is False


def test_validate_int_range():
    ok, _ = topic_config.validate_config_value("max_rounds", "7")
    assert ok is True
    ok, _ = topic_config.validate_config_value("mse_candidate_count", "5")
    assert ok is True
    ok, err = topic_config.validate_config_value("max_rounds", "0")
    assert ok is False
    ok, err = topic_config.validate_config_value("mse_candidate_count", "9")
    assert ok is False
    ok, err = topic_config.validate_config_value("max_rounds", "999")
    assert ok is False
    ok, err = topic_config.validate_config_value("max_rounds", "abc")
    assert ok is False


def test_validate_float_range():
    ok, _ = topic_config.validate_config_value("temperature", "0.7")
    assert ok is True
    ok, err = topic_config.validate_config_value("temperature", "-1")
    assert ok is False
    ok, err = topic_config.validate_config_value("temperature", "5.0")
    assert ok is False


def test_validate_enum_keys():
    ok, _ = topic_config.validate_config_value("llm_provider", "minimax")
    assert ok is True
    ok, _ = topic_config.validate_config_value("llm_provider", "minimax")
    assert ok is True
    ok, _ = topic_config.validate_config_value("api_consult_provider", "minimax")
    assert ok is True
    ok, err = topic_config.validate_config_value("web_provider", "legacy_provider")
    assert ok is False
    ok, _ = topic_config.validate_config_value("mse_workflow_mode", "modeling_reviewed")
    assert ok is True
    ok, err = topic_config.validate_config_value("mse_workflow_mode", "legacy")
    assert ok is False
    ok, err = topic_config.validate_config_value("llm_provider", "openai")
    assert ok is False


def test_validate_sandbox_packages():
    ok, _ = topic_config.validate_config_value("sandbox_packages", '["numpy"]')
    assert ok is True
    ok, err = topic_config.validate_config_value("sandbox_packages", "not json")
    assert ok is False


def test_topic_config_missing_table_falls_back_to_defaults(monkeypatch):
    class MissingTableConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("no such table: TopicConfig")

    monkeypatch.setattr(topic_config, "get_db", lambda: MissingTableConn())

    assert topic_config.get(1, "llm_provider") == "minimax"
    assert topic_config.get_explicit(1, "llm_provider") is None
    assert topic_config.get_all(1)["llm_provider"] == "minimax"


def test_topic_config_reraises_non_bootstrap_operational_errors(monkeypatch):
    class LockedConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(topic_config, "get_db", lambda: LockedConn())

    with pytest.raises(sqlite3.OperationalError):
        topic_config.get(1, "llm_provider")
    with pytest.raises(sqlite3.OperationalError):
        topic_config.get_explicit(1, "llm_provider")
    with pytest.raises(sqlite3.OperationalError):
        topic_config.get_all(1)
    ok, err = topic_config.validate_config_value("sandbox_packages", '"just a string"')
    assert ok is False


def test_set_bulk_rejects_invalid():
    topic_id = api.create_topic("Test", "Detail")
    ok, err = topic_config.set_bulk(topic_id, {"unknown_key": "value"})
    assert ok is False
    assert "Unknown" in err
    # Valid keys with bad values
    ok, err = topic_config.set_bulk(topic_id, {"max_rounds": "abc"})
    assert ok is False
    # Atomic: if one key fails, nothing is written
    ok, err = topic_config.set_bulk(
        topic_id, {"temperature": "0.5", "max_rounds": "999"}
    )
    assert ok is False
    # temperature should NOT have been written since max_rounds failed
    assert topic_config.get(topic_id, "temperature") == "0.7"  # default
