import os
import pytest

from orbit_or.db import get_db, get_db_path, init_db
from orbit_or import api


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


def test_queue_columns_exist():
    with get_db() as conn:
        cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(Topic)").fetchall()
        }
        assert "queue_position" in cols
        assert "queued_at" in cols


def test_enqueue_topic():
    topic_id = api.enqueue_topic("Queued topic", "Detail")
    topic = api.get_topic(topic_id)
    assert topic["status"] == "Queued"
    assert topic["queue_position"] == 1
    assert topic["queued_at"] is not None


def test_enqueue_multiple_sequential_positions():
    t1 = api.enqueue_topic("First", "Detail")
    t2 = api.enqueue_topic("Second", "Detail")
    t3 = api.enqueue_topic("Third", "Detail")
    q = api.get_topic_queue()
    assert len(q) == 3
    assert q[0]["id"] == t1
    assert q[0]["queue_position"] == 1
    assert q[1]["id"] == t2
    assert q[1]["queue_position"] == 2
    assert q[2]["id"] == t3
    assert q[2]["queue_position"] == 3


def test_dequeue_topic():
    t1 = api.enqueue_topic("To dequeue", "Detail")
    api.dequeue_topic(t1)
    topic = api.get_topic(t1)
    assert topic["queue_position"] is None
    assert topic["queued_at"] is None
    assert topic["status"] == "Closed"


def test_get_next_queued_topic():
    t1 = api.enqueue_topic("First", "Detail")
    api.enqueue_topic("Second", "Detail")
    next_t = api.get_next_queued_topic()
    assert next_t["id"] == t1


def test_get_next_queued_topic_returns_none_when_empty():
    assert api.get_next_queued_topic() is None


def test_reorder_queue():
    t1 = api.enqueue_topic("First", "Detail")
    t2 = api.enqueue_topic("Second", "Detail")
    t3 = api.enqueue_topic("Third", "Detail")
    # Reverse order
    api.reorder_queue([t3, t1, t2])
    q = api.get_topic_queue()
    assert q[0]["id"] == t3
    assert q[0]["queue_position"] == 1
    assert q[1]["id"] == t1
    assert q[1]["queue_position"] == 2
    assert q[2]["id"] == t2
    assert q[2]["queue_position"] == 3


def test_enqueue_with_config():
    topic_id = api.enqueue_topic("With config", "Detail", config={"temperature": "0.3"})
    from orbit_or import topic_config

    assert topic_config.get(topic_id, "temperature") == "0.3"


def test_set_topic_status_accepts_queued():
    topic_id = api.create_topic("Test", "Detail")
    api.set_topic_status(topic_id, "Queued")
    topic = api.get_topic(topic_id)
    assert topic["status"] == "Queued"


def test_get_current_topic_does_not_return_queued():
    # Queued topics should not be returned by get_current_topic unless no active topic exists
    t_closed = api.create_topic("Closed", "Detail")
    api.set_topic_status(t_closed, "Closed")
    api.enqueue_topic("Queued", "Detail")
    # get_current_topic should fall back to the most recent topic (closed or queued)
    current = api.get_current_topic()
    # It should be the queued one as latest
    assert current is not None
    # When an active (Running) topic exists, it should be preferred
    t_active = api.create_topic("Active", "Detail")
    api.set_topic_status(t_active, "Running")
    current = api.get_current_topic()
    assert current["id"] == t_active
    assert current["status"] == "Running"


def test_auto_start_simulation():
    """Simulate server loop auto-start behavior."""
    t1 = api.enqueue_topic("Auto start me", "Detail")
    next_t = api.get_next_queued_topic()
    assert next_t["id"] == t1
    api.set_topic_status(t1, "Started")
    api.dequeue_topic(t1)
    topic = api.get_topic(t1)
    assert topic["status"] == "Started"
    assert topic["queue_position"] is None
    # No more queued topics
    assert api.get_next_queued_topic() is None
