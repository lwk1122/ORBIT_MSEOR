import os
import pytest

from orbit_or.db import get_db, get_db_path, init_db
from orbit_or import api
from orbit_or import topic_config
from orbit_or.master_graph import (
    TopicState,
    node_hitl_plan_check,
    node_hitl_subtopic_check,
    node_hitl_final_check,
    node_hitl_replan_check,
    route_after_hitl,
)


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


def test_user_injection_table_exists():
    with get_db() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='UserInjection'"
        ).fetchone()
        assert row is not None


def test_topic_paused_at_stage_column():
    topic_id = api.create_topic("Test", "Detail")
    api.pause_topic(topic_id, "plan_approval")
    topic = api.get_topic(topic_id)
    assert topic["status"] == "Paused"
    assert topic["paused_at_stage"] == "plan_approval"


def test_pause_and_resume():
    topic_id = api.create_topic("Test", "Detail")
    api.pause_topic(topic_id, "plan_approval")
    topic = api.get_topic(topic_id)
    assert topic["status"] == "Paused"
    api.resume_topic(topic_id)
    topic = api.get_topic(topic_id)
    assert topic["status"] == "Running"
    assert topic["paused_at_stage"] is None


def test_inject_knowledge():
    topic_id = api.create_topic("Test", "Detail")
    inj_id = api.inject_knowledge(topic_id, "url", "https://example.com")
    assert inj_id is not None
    pending = api.get_pending_injections(topic_id)
    assert len(pending) == 1
    assert pending[0]["injection_type"] == "url"
    assert pending[0]["content"] == "https://example.com"
    assert pending[0]["status"] == "pending"


def test_mark_injection_processed():
    topic_id = api.create_topic("Test", "Detail")
    inj_id = api.inject_knowledge(topic_id, "text", "Some fact to inject")
    api.mark_injection_processed(inj_id)
    pending = api.get_pending_injections(topic_id)
    assert len(pending) == 0


def test_inject_multiple():
    topic_id = api.create_topic("Test", "Detail")
    api.inject_knowledge(topic_id, "url", "https://example.com")
    api.inject_knowledge(topic_id, "text", "Some knowledge")
    api.inject_knowledge(topic_id, "search_query", "What is AI?")
    pending = api.get_pending_injections(topic_id)
    assert len(pending) == 3


def test_hitl_plan_check_pauses_when_enabled():
    topic_id = api.create_topic("Test", "Detail")
    topic_config.set_config(topic_id, "hitl_plan_approval", "1")
    state: TopicState = {"topic_id": topic_id}
    result = node_hitl_plan_check(state)
    assert result["deferred"] is True
    topic = api.get_topic(topic_id)
    assert topic["status"] == "Paused"
    assert topic["paused_at_stage"] == "plan_approval"


def test_hitl_plan_check_skips_when_disabled():
    topic_id = api.create_topic("Test", "Detail")
    topic_config.set_config(topic_id, "hitl_plan_approval", "0")
    state: TopicState = {"topic_id": topic_id}
    result = node_hitl_plan_check(state)
    assert result["deferred"] is False


def test_hitl_subtopic_check_skips_by_default():
    topic_id = api.create_topic("Test", "Detail")
    state: TopicState = {"topic_id": topic_id}
    result = node_hitl_subtopic_check(state)
    assert result["deferred"] is False


def test_hitl_subtopic_check_pauses_when_enabled():
    topic_id = api.create_topic("Test", "Detail")
    topic_config.set_config(topic_id, "hitl_subtopic_review", "1")
    state: TopicState = {"topic_id": topic_id}
    result = node_hitl_subtopic_check(state)
    assert result["deferred"] is True
    topic = api.get_topic(topic_id)
    assert topic["paused_at_stage"] == "subtopic_review"


def test_hitl_final_check_skips_by_default():
    topic_id = api.create_topic("Test", "Detail")
    state: TopicState = {"topic_id": topic_id}
    result = node_hitl_final_check(state)
    assert result["deferred"] is False


def test_hitl_replan_check_skips_by_default():
    topic_id = api.create_topic("Test", "Detail")
    state: TopicState = {"topic_id": topic_id}
    result = node_hitl_replan_check(state)
    assert result["deferred"] is False


def test_route_after_hitl_deferred():
    state: TopicState = {"topic_id": 1, "deferred": True}
    assert route_after_hitl(state) == "defer_topic"


def test_route_after_hitl_continue():
    state: TopicState = {"topic_id": 1, "deferred": False}
    assert route_after_hitl(state) == "continue"


def test_set_topic_status_accepts_paused():
    topic_id = api.create_topic("Test", "Detail")
    api.set_topic_status(topic_id, "Paused")
    topic = api.get_topic(topic_id)
    assert topic["status"] == "Paused"


def test_get_current_topic_prefers_active():
    # Create a closed topic and a paused topic
    t1 = api.create_topic("Closed topic", "Detail")
    api.set_topic_status(t1, "Closed")
    t2 = api.create_topic("Paused topic", "Detail")
    api.pause_topic(t2, "plan_approval")
    current = api.get_current_topic()
    assert current["id"] == t2
    assert current["status"] == "Paused"
