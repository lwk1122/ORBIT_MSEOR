"""Render a topic transcript from an ORBIT SQLite database."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def get_data(db_path: Path, topic_id: int):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        topic = _dict(
            conn.execute("SELECT * FROM Topic WHERE id = ?", (topic_id,)).fetchone()
        )
        plan = _dict(
            conn.execute(
                "SELECT * FROM Plan WHERE topic_id = ? ORDER BY id DESC LIMIT 1",
                (topic_id,),
            ).fetchone()
        )
        subtopics = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM Subtopic WHERE topic_id = ? ORDER BY id ASC",
                (topic_id,),
            ).fetchall()
        ]
        messages = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM Message
                WHERE topic_id = ?
                ORDER BY
                    CASE WHEN subtopic_id IS NULL THEN 1 ELSE 0 END,
                    COALESCE(subtopic_id, 0),
                    id ASC
                """,
                (topic_id,),
            ).fetchall()
        ]
        return topic, plan, subtopics, messages
    finally:
        conn.close()


def render_log(
    topic: dict[str, Any] | None,
    plan: dict[str, Any] | None,
    subtopics: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    if topic:
        lines.append(f"=== Topic {topic['id']}: {topic['summary']} ===")
        lines.append(topic.get("detail") or "")
    if plan:
        lines.append("\n=== Latest Plan ===")
        lines.append(plan.get("content") or "")

    subtopic_by_id = {item["id"]: item for item in subtopics}
    topic_level = [msg for msg in messages if msg.get("subtopic_id") is None]
    by_subtopic: dict[int, list[dict[str, Any]]] = {}
    for msg in messages:
        sid = msg.get("subtopic_id")
        if sid is not None:
            by_subtopic.setdefault(int(sid), []).append(msg)

    for subtopic in subtopics:
        lines.append(f"\n=== Subtopic {subtopic['id']}: {subtopic['summary']} ===")
        if subtopic.get("detail"):
            lines.append(subtopic["detail"])
        for msg in by_subtopic.get(subtopic["id"], []):
            lines.append(_render_message(msg))

    orphan_subtopic_ids = set(by_subtopic) - set(subtopic_by_id)
    for sid in sorted(orphan_subtopic_ids):
        lines.append(f"\n=== Subtopic {sid} ===")
        for msg in by_subtopic[sid]:
            lines.append(_render_message(msg))

    if topic_level:
        lines.append("\n=== Topic-level Messages ===")
        for msg in topic_level:
            lines.append(_render_message(msg))
    return "\n".join(line for line in lines if line is not None)


def _render_message(msg: dict[str, Any]) -> str:
    sender = msg.get("sender") or "unknown"
    msg_type = msg.get("msg_type") or "standard"
    content = msg.get("content") or ""
    return f"[{msg['id']}] {sender} ({msg_type}): {content}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("topic_id", type=int)
    args = parser.parse_args()
    print(render_log(*get_data(args.db_path, args.topic_id)))


if __name__ == "__main__":
    main()
