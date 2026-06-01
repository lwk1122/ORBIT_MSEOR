import os
import json
import pytest

from orbit_or.db import get_db, get_db_path, init_db
from orbit_or import api
from orbit_or.report import render_html_report, render_markdown_report


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


def test_report_json_column_exists():
    with get_db() as conn:
        cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(Topic)").fetchall()
        }
        assert "report_json" in cols


def test_save_and_get_report():
    topic_id = api.create_topic("Test", "Detail")
    report_data = {"topic_id": topic_id, "blog": "Hello world"}
    api.save_report(topic_id, json.dumps(report_data))
    raw = api.get_report(topic_id)
    assert raw is not None
    loaded = json.loads(raw)
    assert loaded["blog"] == "Hello world"


def test_get_report_returns_none_when_empty():
    topic_id = api.create_topic("Test", "Detail")
    assert api.get_report(topic_id) is None


def _sample_report():
    return {
        "topic_summary": "Test Topic",
        "language": "en",
        "dossier": {
            "topic": "Test Topic",
            "why_it_matters": "Testing is important",
            "subtopics": [
                {
                    "name": "Sub A",
                    "key_finding": "Found something interesting",
                    "confidence": "high",
                }
            ],
            "dramatic_moments": ["Moment 1"],
            "internal_conflicts": [],
            "unstable_claims": [],
            "final_consensus": "We agree",
            "best_quotes": [],
            "recommended_blog_angle": "Direct angle",
            "accuracy_warnings": ["Be careful with X"],
        },
        "outline": {
            "title_candidates": ["Great Blog Title", "Another Title"],
            "chosen_angle": "Angle 1",
            "opening_hook": "Once upon a time...",
            "section_outline": [
                {
                    "title": "Section 1",
                    "dramatic_function": "Introduction",
                    "factual_core": "Facts here",
                }
            ],
            "closing_thesis": "Done.",
        },
        "blog": "# My Blog Post\n\nThis is the blog [F1] with citations [C2].\n\n## Section 1\n\nMore content here.",
        "review": {
            "issues_found": [
                {
                    "type": "overclaim",
                    "location": "paragraph 2",
                    "fix": "Add qualifier",
                }
            ],
            "overall_quality": "good",
            "suggested_fixes": ["Add a caveat"],
        },
    }


def test_render_html_report():
    report = _sample_report()
    html = render_html_report(report)
    assert "<!DOCTYPE html>" in html
    assert "Great Blog Title" in html
    assert "My Blog Post" in html
    assert "[F1]" in html
    assert "Accuracy Warnings" in html
    assert "Be careful with X" in html
    assert "Quality" in html


def test_render_html_report_escapes_xss():
    report = _sample_report()
    report["blog"] = "# <script>alert(1)</script>\n\nNormal text"
    html = render_html_report(report)
    assert "<script>alert(1)</script>" not in html
    # nh3 strips dangerous tags entirely (not escaped)
    assert "alert(1)" not in html or "&lt;" in html


def test_render_markdown_report():
    report = _sample_report()
    md = render_markdown_report(report)
    assert "# My Blog Post" in md
    assert "Sub A" in md
    assert "Accuracy Warnings" in md


def test_render_html_report_no_review():
    report = _sample_report()
    report["review"] = None
    html = render_html_report(report)
    assert "<!DOCTYPE html>" in html
    assert "My Blog Post" in html
