import os
import json

import pytest

from orbit_or import api
from orbit_or.db import get_db_path, init_db
from orbit_or.web import (
    build_dashboard_snapshot,
    create_app,
    handle_ingest_corpus_document,
    render_dashboard_html,
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


def test_dashboard_snapshot_handles_empty_database():
    snapshot = build_dashboard_snapshot()

    assert snapshot["topic"] is None
    assert snapshot["plan"] is None
    assert snapshot["subtopics"] == []
    assert snapshot["messages"] == []
    assert snapshot["facts"] == []
    assert snapshot["fact_candidates"] == []
    assert snapshot["mse_review"] is None
    assert snapshot["status"]["db_path"].endswith("test_orbit.db")


def test_dashboard_snapshot_includes_plan_messages_facts_and_pending_candidates():
    topic_id = api.create_topic("Topic", "Detail")
    plan_id = api.create_plan(
        topic_id,
        '[{"summary":"Subtopic A","detail":"Detail A"},{"summary":"Subtopic B","detail":"Detail B"}]',
        current_index=1,
    )
    assert plan_id is not None

    subtopic_id = api.create_subtopic(topic_id, "Subtopic A", "Detail A")
    api.post_message(topic_id, subtopic_id, "skynet", "Grounding brief", round_number=1, turn_kind="base")
    api.post_message(topic_id, subtopic_id, "dog", "Please narrow the claim", round_number=2, turn_kind="base")
    api.insert_fact(topic_id, "Accepted fact", "Librarian")
    api.create_fact_candidate(topic_id, subtopic_id, None, "Pending fact")

    snapshot = build_dashboard_snapshot()

    assert snapshot["topic"]["id"] == topic_id
    assert snapshot["plan"]["current_index"] == 1
    assert len(snapshot["plan"]["items"]) == 2
    assert snapshot["current_subtopic"]["id"] == subtopic_id
    assert [message["sender"] for message in snapshot["messages"]] == ["skynet", "dog"]
    assert snapshot["status"]["current_round"] == 2
    assert snapshot["status"]["current_phase"] == "evidence"
    assert snapshot["facts"][0]["content"] == "Accepted fact"
    assert snapshot["fact_candidates"][0]["candidate_text"] == "Pending fact"
    assert snapshot["mse_review"]["review_counts"]["documents"] == 0
    assert snapshot["mse_review"]["review_counts"]["problems"] == 0


def test_dashboard_snapshot_can_target_closed_topic_explicitly():
    closed_topic_id = api.create_topic("Closed Topic", "Closed Detail")
    api.set_topic_status(closed_topic_id, "Closed")
    api.save_report(closed_topic_id, '{"topic_id": 1, "title": "done"}')

    active_topic_id = api.create_topic("Active Topic", "Active Detail")
    api.set_topic_status(active_topic_id, "Running")

    snapshot = build_dashboard_snapshot(topic_id=closed_topic_id)

    assert snapshot["topic"]["id"] == closed_topic_id
    assert snapshot["topic"]["status"] == "Closed"
    assert snapshot["topic"]["report_json"] == '{"topic_id": 1, "title": "done"}'


def test_create_app_registers_read_only_routes():
    app = create_app()
    routes = {(route.method, route.resource.canonical) for route in app.router.routes()}

    assert ("GET", "/") in routes
    assert ("GET", "/api/dashboard") in routes
    assert ("GET", "/api/health") in routes
    assert ("GET", "/api/topics") in routes
    assert ("GET", "/api/topic/{id}/mse_review") in routes
    assert ("GET", "/api/topic/{id}/mse_report") in routes
    assert ("GET", "/api/topic/{id}/mse_report/markdown") in routes
    assert ("POST", "/api/topic/{id}/corpus/ingest") in routes
    assert ("POST", "/api/mse/component/{id}/review") in routes
    assert ("POST", "/api/mse/diagnostic/{id}/status") in routes


@pytest.mark.asyncio
async def test_handle_ingest_corpus_document_creates_indexed_chunks():
    topic_id = api.create_topic("MSE Corpus", "Capacity planning.")

    class Request:
        match_info = {"id": str(topic_id)}

        async def json(self):
            return {
                "title": "Capacity memo",
                "doc_type": "markdown",
                "text": "# Capacity\n\nCapacity is 10 pallets per day.",
            }

    response = await handle_ingest_corpus_document(Request())
    payload = json.loads(response.text)

    assert response.status == 200
    assert payload["ok"] is True
    assert payload["chunk_count"] == 1
    docs = api.list_corpus_documents(topic_id)
    assert docs[0]["title"] == "Capacity memo"
    chunks = api.get_corpus_chunks_for_document(payload["document_id"])
    assert "Capacity is 10 pallets" in chunks[0]["text"]


def test_dashboard_html_escapes_dynamic_content_in_client_renderer():
    html = render_dashboard_html()

    assert "function esc(value)" in html
    assert "linkCitations(message.content)" in html
    assert "linkCitations(f.content)" in html
    assert "esc(c.candidate_text)" in html
    assert 'id="tab-mse"' in html
    assert "function reviewComponent" in html
    assert "function resolveDiagnostic" in html
    assert "Provenance JSON" in html
    assert "/mse_report/markdown" in html
    assert 'id="topic-select"' in html
    assert "currentTopicId" in html
    assert "/api/topics" in html
