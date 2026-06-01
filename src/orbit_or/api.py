import json

from .db import (
    supersede_facts as db_supersede_facts,
    update_fact_summary_and_embedding as db_update_fact_summary_and_embedding,
    update_claim_summary as db_update_claim_summary,
    insert_code_evidence as db_insert_code_evidence,
    get_code_evidence_for_topic as db_get_code_evidence_for_topic,
    get_code_evidence_for_topic_full as db_get_code_evidence_for_topic_full,
    get_code_evidence_by_id as db_get_code_evidence_by_id,
    increment_code_evidence_review_count as db_increment_review_count,
    reset_code_evidence_review_count as db_reset_review_count,
    insert_api_evidence as db_insert_api_evidence,
    get_api_evidence_for_topic as db_get_api_evidence_for_topic,
    get_api_evidence_by_id as db_get_api_evidence_by_id,
    get_web_evidence_for_topic as db_get_web_evidence,
    get_unprocessed_web_evidence as db_get_unprocessed_web_evidence,
    mark_web_evidence_ledger_processed as db_mark_web_evidence_ledger_processed,
    get_messages_since as db_get_messages_since,
    get_ledger_entries_with_names as db_get_ledger_entries_with_names,
    get_contested_ledger_pairs as db_get_contested_ledger_pairs,
    get_ledger_edges as db_get_ledger_edges,
    get_active_ledger_pending as db_get_active_ledger_pending,
    get_max_round_number as db_get_max_round_number,
    get_web_evidence_count as db_get_web_evidence_count,
    insert_tool_trace as db_insert_tool_trace,
    insert_corpus_document as db_insert_corpus_document,
    get_corpus_document as db_get_corpus_document,
    list_corpus_documents as db_list_corpus_documents,
    create_corpus_ingest_run as db_create_corpus_ingest_run,
    update_corpus_ingest_run as db_update_corpus_ingest_run,
    insert_corpus_chunk as db_insert_corpus_chunk,
    insert_corpus_chunk_with_embedding as db_insert_corpus_chunk_with_embedding,
    get_corpus_chunks_for_document as db_get_corpus_chunks_for_document,
    reindex_corpus_document as db_reindex_corpus_document,
    get_corpus_neighbor_chunks as db_get_corpus_neighbor_chunks,
    search_corpus_chunks as db_search_corpus_chunks,
    search_corpus_chunks_lexical as db_search_corpus_chunks_lexical,
    insert_optimization_problem as db_insert_optimization_problem,
    get_optimization_problem as db_get_optimization_problem,
    list_optimization_problems as db_list_optimization_problems,
    insert_optimization_component as db_insert_optimization_component,
    get_optimization_components as db_get_optimization_components,
    update_optimization_component_review as db_update_optimization_component_review,
    insert_optimization_model_ir as db_insert_optimization_model_ir,
    get_optimization_model_irs as db_get_optimization_model_irs,
    insert_optimization_artifact as db_insert_optimization_artifact,
    get_optimization_artifacts as db_get_optimization_artifacts,
    insert_solver_run as db_insert_solver_run,
    get_solver_runs as db_get_solver_runs,
    insert_model_diagnostic as db_insert_model_diagnostic,
    get_model_diagnostics as db_get_model_diagnostics,
    update_model_diagnostic_status as db_update_model_diagnostic_status,
    upsert_modeling_experience as db_upsert_modeling_experience,
    list_modeling_experiences as db_list_modeling_experiences,
    record_modeling_experience_event as db_record_modeling_experience_event,
    update_fact_structured_columns as db_update_fact_structured_columns,
    update_claim_structured_columns as db_update_claim_structured_columns,
    claim_candidate_exists as db_claim_candidate_exists,
    close_subtopic as db_close_subtopic,
    create_claim_candidate as db_create_claim_candidate,
    create_fact_candidate as db_create_fact_candidate,
    fact_exists as db_fact_exists,
    fact_candidate_exists as db_fact_candidate_exists,
    get_db,
    get_claim_candidates as db_get_claim_candidates,
    get_open_subtopic as db_get_open_subtopic,
    get_db_path,
    get_facts_by_ids as db_get_facts_by_ids,
    get_fact_by_content as db_get_fact_by_content,
    get_claim_by_content as db_get_claim_by_content,
    get_claims_by_support_fact_set as db_get_claims_by_support_fact_set,
    update_claim_superseded as db_update_claim_superseded,
    merge_fact_source_ref as db_merge_fact_source_ref,
    merge_claim_support_facts as db_merge_claim_support_facts,
    insert_claim_and_supersede as db_insert_claim_and_supersede,
    get_fact_candidates as db_get_fact_candidates,
    get_vote_records as db_get_vote_records,
    insert_claim as db_insert_claim,
    insert_vote_record as db_insert_vote_record,
    insert_web_evidence as db_insert_web_evidence,
    clone_web_evidence_to_topic as db_clone_web_evidence_to_topic,
    insert_fact,
    insert_fact_with_embedding,
    insert_message_with_embedding,
    search_claims_lexical,
    search_facts,
    search_facts_lexical,
    search_messages,
    search_messages_lexical,
    search_web_evidence_cross_topic as db_search_web_evidence_cross_topic,
    search_web_evidence_same_topic as db_search_web_evidence_same_topic,
    update_claim_candidate_review as db_update_claim_candidate_review,
    update_fact_candidate_review as db_update_fact_candidate_review,
    update_subtopic_start_msg,
    update_subtopic_locked_scope,
    update_topic_conclusion as db_update_topic_conclusion,
    insert_knowledge_edge as db_insert_knowledge_edge,
    get_knowledge_edges as db_get_knowledge_edges,
    deactivate_knowledge_edge as db_deactivate_knowledge_edge,
    get_active_conflicts as db_get_active_conflicts,
    get_claim_justification_groups as db_get_claim_justification_groups,
    supersede_fact as db_supersede_fact,
    set_topic_config as db_set_topic_config,
    get_topic_config as db_get_topic_config,
    get_all_topic_config as db_get_all_topic_config,
    insert_user_injection as db_insert_user_injection,
    get_pending_injections as db_get_pending_injections,
    mark_injection_processed as db_mark_injection_processed,
    get_next_queued_topic as db_get_next_queued_topic,
    dequeue_topic as db_dequeue_topic,
    get_topic_queue as db_get_topic_queue,
    reorder_queue as db_reorder_queue,
)
from .embedding import aget_embedding
from . import analytics

# Expose db functions that don't need additional api logic
__all__ = [
    "get_current_topic",
    "get_topic",
    "list_topics",
    "create_plan",
    "get_plan",
    "get_current_subtopics",
    "get_latest_subtopic",
    "get_subtopic",
    "get_messages",
    "create_topic",
    "set_topic_status",
    "create_subtopic",
    "post_message",
    "search_facts",
    "insert_fact",
    "insert_fact_with_embedding",
    "insert_message_with_embedding",
    "search_messages",
    "search_facts_lexical",
    "search_messages_lexical",
    "search_facts_hybrid",
    "search_messages_hybrid",
    "search_corpus_chunks_hybrid",
    "reciprocal_rank_fusion",
    "insert_corpus_document",
    "get_corpus_document",
    "list_corpus_documents",
    "create_corpus_ingest_run",
    "update_corpus_ingest_run",
    "insert_corpus_chunk",
    "insert_corpus_chunk_with_embedding",
    "get_corpus_chunks_for_document",
    "reindex_corpus_document",
    "get_corpus_neighbor_chunks",
    "insert_optimization_problem",
    "get_optimization_problem",
    "list_optimization_problems",
    "insert_optimization_component",
    "get_optimization_components",
    "update_optimization_component_review",
    "insert_optimization_model_ir",
    "get_optimization_model_irs",
    "insert_optimization_artifact",
    "get_optimization_artifacts",
    "insert_solver_run",
    "get_solver_runs",
    "insert_model_diagnostic",
    "get_model_diagnostics",
    "update_model_diagnostic_status",
    "upsert_modeling_experience",
    "list_modeling_experiences",
    "record_modeling_experience_event",
    "get_mse_review_snapshot",
    "get_mse_provenance_report",
    "render_mse_provenance_markdown",
    "get_active_plan",
    "advance_plan_cursor",
    "update_subtopic_start_msg",
    "close_subtopic",
    "get_open_subtopic",
    "get_db_path",
    "persist_message",
    "fact_exists",
    "get_fact_by_content",
    "update_subtopic_locked_scope",
    "create_fact_candidate",
    "create_fact_candidate_with_stage",
    "get_pending_fact_candidates",
    "get_fact_candidates",
    "get_facts",
    "fact_candidate_exists",
    "update_fact_candidate_review",
    "create_claim_candidate",
    "get_pending_claim_candidates",
    "get_claim_candidates",
    "claim_candidate_exists",
    "update_claim_candidate_review",
    "insert_claim",
    "get_facts_by_ids",
    "search_claims_hybrid",
    "insert_web_evidence",
    "search_web_evidence_same_topic",
    "search_web_evidence_cross_topic",
    "insert_vote_record",
    "get_vote_records",
    "get_messages_since",
    "insert_api_evidence",
    "get_api_evidence_for_topic",
    "get_api_evidence_by_id",
]


def _dict(row):
    return dict(row) if row else None


def reciprocal_rank_fusion(
    *groups,
    limit: int,
    id_key: str = "id",
    k: int = 60,
    score_key: str = "rrf_score",
):
    """Fuse ranked result lists with reciprocal rank fusion.

    Each input group is assumed to be ordered from best to worst. The returned
    rows are plain dicts with an added ``rrf_score`` field.
    """
    scores: dict[object, float] = {}
    rows_by_id: dict[object, dict] = {}
    first_seen: dict[object, int] = {}
    sequence = 0

    for group in groups:
        for rank, raw_row in enumerate(list(group or []), start=1):
            row = dict(raw_row)
            row_id = row[id_key]
            if row_id not in rows_by_id:
                rows_by_id[row_id] = row
                first_seen[row_id] = sequence
                sequence += 1
            scores[row_id] = scores.get(row_id, 0.0) + (1.0 / (k + rank))

    ranked_ids = sorted(
        scores,
        key=lambda row_id: (-scores[row_id], first_seen[row_id]),
    )
    fused = []
    for row_id in ranked_ids[:limit]:
        fused.append({**rows_by_id[row_id], score_key: scores[row_id]})
    return fused


def _merge_ranked_rows(*groups, limit: int):
    return reciprocal_rank_fusion(*groups, limit=limit)


def get_current_topic():
    with get_db() as conn:
        # Priority: active topics first
        row = conn.execute(
            "SELECT * FROM Topic WHERE status IN ('Running','Started','Paused') ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row:
            return _dict(row)
        # Fallback: latest topic
        row = conn.execute("SELECT * FROM Topic ORDER BY id DESC LIMIT 1").fetchone()
        return _dict(row)


def get_topic(topic_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM Topic WHERE id = ?", (topic_id,)).fetchone()
        return _dict(row)


def list_topics():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM Topic ORDER BY id DESC").fetchall()
        return [_dict(row) for row in rows]


def create_plan(topic_id: int, content: str, current_index: int = 0) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Plan (topic_id, content, current_index) VALUES (?, ?, ?)",
            (topic_id, content, current_index),
        )
        return cursor.lastrowid


def get_plan(topic_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM Plan WHERE topic_id = ? ORDER BY id DESC LIMIT 1",
            (topic_id,),
        ).fetchone()
        return _dict(row)


def get_active_plan(topic_id: int):
    return get_plan(topic_id)


def get_current_subtopics(topic_id):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM Subtopic WHERE topic_id = ? ORDER BY id ASC", (topic_id,)
        ).fetchall()
        return [_dict(r) for r in rows]


def get_latest_subtopic(topic_id):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM Subtopic WHERE topic_id = ? ORDER BY id DESC LIMIT 1",
            (topic_id,),
        ).fetchone()
        return _dict(row)


def get_subtopic(subtopic_id: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM Subtopic WHERE id = ?", (subtopic_id,)
        ).fetchone()
        return _dict(row)


def get_messages(topic_id, subtopic_id=None, limit=50, msg_type=None):
    with get_db() as conn:
        clauses = ["topic_id = ?"]
        params = [topic_id]
        if subtopic_id is not None:
            clauses.append("subtopic_id = ?")
            params.append(subtopic_id)
        if msg_type is not None:
            clauses.append("msg_type = ?")
            params.append(msg_type)
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM Message WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        # Return in chronological order
        return [_dict(r) for r in reversed(rows)]


def get_messages_since(
    topic_id: int, subtopic_id: int, since_id: int, msg_type: str = "standard"
):
    """Return messages newer than *since_id* in chronological order."""
    return db_get_messages_since(topic_id, subtopic_id, since_id, msg_type)


def create_topic(summary, detail, config=None):
    from . import topic_config

    # Validate config before creating topic
    if config and isinstance(config, dict):
        for key, value in config.items():
            ok, err = topic_config.validate_config_value(key, str(value))
            if not ok:
                raise ValueError(err)
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Topic (summary, detail, status) VALUES (?, ?, 'Started')",
            (summary, detail),
        )
        topic_id = cursor.lastrowid
        if config and isinstance(config, dict):
            for key, value in config.items():
                conn.execute(
                    "INSERT OR REPLACE INTO TopicConfig (topic_id, config_key, config_value) VALUES (?, ?, ?)",
                    (topic_id, key, str(value)),
                )
    analytics.capture(
        f"topic_{topic_id}",
        "topic_created",
        {"summary_length": len(summary), "has_config": bool(config)},
    )
    return topic_id


def set_topic_status(topic_id, status):
    if status not in ["Closed", "Started", "Running", "Paused", "Queued"]:
        raise ValueError("Invalid status")
    with get_db() as conn:
        conn.execute("UPDATE Topic SET status = ? WHERE id = ?", (status, topic_id))


def create_subtopic(topic_id, summary, detail, start_msg_id=None):
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO Subtopic (topic_id, summary, detail, start_msg_id, status) VALUES (?, ?, ?, ?, 'Open')",
            (topic_id, summary, detail, start_msg_id),
        )
        return cursor.lastrowid


def post_message(
    topic_id,
    subtopic_id,
    sender,
    content,
    msg_type="standard",
    confidence_score=None,
    round_number=None,
    turn_kind=None,
    summary=None,
):
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO Message (topic_id, subtopic_id, sender, content, msg_type, confidence_score, round_number, turn_kind, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                sender,
                content,
                msg_type,
                confidence_score,
                round_number,
                turn_kind,
                summary,
            ),
        )
        # For FTS, we combine content and summary
        fts_content = content
        if summary:
            fts_content = f"{summary}\n\n{content}"

        conn.execute(
            "INSERT OR REPLACE INTO messages_fts(rowid, content, topic_id, msg_type, sender) VALUES (?, ?, ?, ?, ?)",
            (cursor.lastrowid, fts_content, str(topic_id), msg_type, sender),
        )
        return cursor.lastrowid


async def persist_message(
    topic_id,
    subtopic_id,
    sender,
    content,
    msg_type="standard",
    confidence_score=None,
    round_number=None,
    turn_kind=None,
    summary=None,
):
    if msg_type == "standard":
        embedding = await aget_embedding(content)
        if embedding:
            return insert_message_with_embedding(
                topic_id,
                subtopic_id,
                sender,
                content,
                msg_type,
                embedding,
                confidence_score,
                round_number,
                turn_kind,
                summary=summary,
            )
    return post_message(
        topic_id,
        subtopic_id,
        sender,
        content,
        msg_type,
        confidence_score,
        round_number,
        turn_kind,
        summary=summary,
    )


def advance_plan_cursor(plan_id: int):
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE Plan SET current_index = current_index + 1 WHERE id = ?",
            (plan_id,),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Plan {plan_id} not found")


def close_subtopic(subtopic_id: int, conclusion: str):
    db_close_subtopic(subtopic_id, conclusion)


def get_open_subtopic(topic_id: int):
    return db_get_open_subtopic(topic_id)


def fact_exists(topic_id: int, content: str, source: str | None = None):
    return db_fact_exists(topic_id, content, source)


def get_fact_by_content(topic_id: int, content: str):
    return db_get_fact_by_content(topic_id, content)


def get_claim_by_content(topic_id: int, content: str):
    return db_get_claim_by_content(topic_id, content)


def get_claims_by_support_fact_set(topic_id: int, sorted_fact_ids_json: str):
    return db_get_claims_by_support_fact_set(topic_id, sorted_fact_ids_json)


def update_claim_superseded(claim_id: int, superseded_by_id: int) -> None:
    db_update_claim_superseded(claim_id, superseded_by_id)


def merge_fact_source_ref(fact_id: int, new_refs: list[str]) -> None:
    db_merge_fact_source_ref(fact_id, new_refs)


def merge_claim_support_facts(claim_id: int, new_fact_ids: list[int]) -> None:
    db_merge_claim_support_facts(claim_id, new_fact_ids)


def insert_claim_and_supersede(
    topic_id: int,
    subtopic_id: int | None,
    content: str,
    *,
    supersede_claim_id: int,
    summary: str | None = None,
    support_fact_ids_json: str | None = None,
    rationale_short: str | None = None,
    claim_score: float | None = None,
    status: str = "active",
    candidate_id: int | None = None,
    # G.4: Structured claim fields
    claim_type: str | None = None,
    scope_tags: str | None = None,
    scope_context: str | None = None,
    falsification_criteria: str | None = None,
    inference_logic: str | None = None,
    conclusion: str | None = None,
    evidence_strength: float | None = None,
    scope_breadth: float | None = None,
    submitted_by: str | None = None,
) -> int:
    return db_insert_claim_and_supersede(
        topic_id,
        subtopic_id,
        content,
        supersede_claim_id=supersede_claim_id,
        summary=summary,
        support_fact_ids_json=support_fact_ids_json,
        rationale_short=rationale_short,
        claim_score=claim_score,
        status=status,
        candidate_id=candidate_id,
        claim_type=claim_type,
        scope_tags=scope_tags,
        scope_context=scope_context,
        falsification_criteria=falsification_criteria,
        inference_logic=inference_logic,
        conclusion=conclusion,
        evidence_strength=evidence_strength,
        scope_breadth=scope_breadth,
        submitted_by=submitted_by,
    )


def create_fact_candidate(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: int | None,
    candidate_text: str,
    **kwargs,
) -> int:
    return db_create_fact_candidate(
        topic_id, subtopic_id, writer_msg_id, candidate_text, **kwargs
    )


def create_fact_candidate_with_stage(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: int | None,
    candidate_text: str,
    *,
    summary: str | None = None,
    fact_stage: str,
    candidate_type: str = "sourced_claim",
    source_kind: str | None = None,
    evidence_note: str | None = None,
    source_refs_json: str | None = None,
    source_excerpt: str | None = None,
    verification_status: str | None = None,
    round_number: int | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    object_json: str | None = None,
    qualifiers_json: str | None = None,
    attribution_json: str | None = None,
) -> int:
    return db_create_fact_candidate(
        topic_id,
        subtopic_id,
        writer_msg_id,
        candidate_text,
        summary=summary,
        fact_stage=fact_stage,
        candidate_type=candidate_type,
        source_kind=source_kind,
        evidence_note=evidence_note,
        source_refs_json=source_refs_json,
        source_excerpt=source_excerpt,
        verification_status=verification_status,
        round_number=round_number,
        subject=subject,
        predicate=predicate,
        object_json=object_json,
        qualifiers_json=qualifiers_json,
        attribution_json=attribution_json,
    )


def create_claim_candidate(
    topic_id: int,
    subtopic_id: int,
    clerk_msg_id: int | None,
    candidate_text: str,
    *,
    summary: str | None = None,
    support_fact_ids_json: str | None = None,
    rationale_short: str | None = None,
    claim_type: str | None = None,
    scope_tags: str | None = None,
    scope_context: str | None = None,
    falsification_criteria: str | None = None,
    inference_logic: str | None = None,
    conclusion: str | None = None,
    evidence_strength: float | None = None,
    scope_breadth: float | None = None,
    submitted_by: str | None = None,
) -> int:
    return db_create_claim_candidate(
        topic_id,
        subtopic_id,
        clerk_msg_id,
        candidate_text,
        summary=summary,
        support_fact_ids_json=support_fact_ids_json,
        rationale_short=rationale_short,
        claim_type=claim_type,
        scope_tags=scope_tags,
        scope_context=scope_context,
        falsification_criteria=falsification_criteria,
        inference_logic=inference_logic,
        conclusion=conclusion,
        evidence_strength=evidence_strength,
        scope_breadth=scope_breadth,
        submitted_by=submitted_by,
    )


def get_pending_fact_candidates(topic_id: int, subtopic_id: int):
    return db_get_fact_candidates(topic_id, subtopic_id=subtopic_id, status="pending")


def get_pending_claim_candidates(topic_id: int, subtopic_id: int):
    return db_get_claim_candidates(topic_id, subtopic_id=subtopic_id, status="pending")


def get_fact_candidates(
    topic_id: int,
    subtopic_id: int | None = None,
    status: str | None = None,
    limit: int | None = None,
):
    candidates = db_get_fact_candidates(
        topic_id, subtopic_id=subtopic_id, status=status
    )
    if limit is not None:
        return candidates[:limit]
    return candidates


def get_claim_candidates(
    topic_id: int,
    subtopic_id: int | None = None,
    status: str | None = None,
    limit: int | None = None,
):
    candidates = db_get_claim_candidates(
        topic_id, subtopic_id=subtopic_id, status=status
    )
    if limit is not None:
        return candidates[:limit]
    return candidates


def get_facts(topic_id: int, limit: int | None = None):
    with get_db() as conn:
        params = [topic_id]
        query = "SELECT * FROM Fact WHERE topic_id = ? ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_dict(row) for row in rows]


def count_facts(topic_id: int, subtopic_id: int | None = None) -> int:
    with get_db() as conn:
        if subtopic_id is not None:
            row = conn.execute(
                "SELECT COUNT(*) FROM Fact WHERE topic_id = ? AND subtopic_id = ?",
                (topic_id, subtopic_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM Fact WHERE topic_id = ?",
                (topic_id,),
            ).fetchone()
        return row[0] if row else 0


def fact_candidate_exists(topic_id: int, candidate_text: str, statuses=None):
    return db_fact_candidate_exists(topic_id, candidate_text, statuses=statuses)


def claim_candidate_exists(topic_id: int, candidate_text: str, statuses=None):
    return db_claim_candidate_exists(topic_id, candidate_text, statuses=statuses)


def update_fact_candidate_review(
    candidate_id: int,
    status: str,
    reviewed_text: str | None = None,
    review_note: str | None = None,
    evidence_note: str | None = None,
    confidence_score: float | None = None,
    reviewer: str | None = None,
    accepted_fact_id: int | None = None,
):
    db_update_fact_candidate_review(
        candidate_id,
        status,
        reviewed_text=reviewed_text,
        review_note=review_note,
        evidence_note=evidence_note,
        confidence_score=confidence_score,
        reviewer=reviewer,
        accepted_fact_id=accepted_fact_id,
    )


def update_claim_candidate_review(
    candidate_id: int,
    status: str,
    reviewed_text: str | None = None,
    review_note: str | None = None,
    claim_score: float | None = None,
    accepted_claim_id: int | None = None,
):
    db_update_claim_candidate_review(
        candidate_id,
        status,
        reviewed_text=reviewed_text,
        review_note=review_note,
        claim_score=claim_score,
        accepted_claim_id=accepted_claim_id,
    )


def insert_claim(
    topic_id: int,
    subtopic_id: int | None,
    content: str,
    *,
    summary: str | None = None,
    support_fact_ids_json: str | None = None,
    rationale_short: str | None = None,
    claim_score: float | None = None,
    status: str = "active",
    candidate_id: int | None = None,
    claim_type: str | None = None,
    scope_tags: str | None = None,
    scope_context: str | None = None,
    falsification_criteria: str | None = None,
    inference_logic: str | None = None,
    conclusion: str | None = None,
    evidence_strength: float | None = None,
    scope_breadth: float | None = None,
    submitted_by: str | None = None,
) -> int:
    return db_insert_claim(
        topic_id,
        subtopic_id,
        content,
        summary=summary,
        support_fact_ids_json=support_fact_ids_json,
        rationale_short=rationale_short,
        claim_score=claim_score,
        status=status,
        candidate_id=candidate_id,
        claim_type=claim_type,
        scope_tags=scope_tags,
        scope_context=scope_context,
        falsification_criteria=falsification_criteria,
        inference_logic=inference_logic,
        conclusion=conclusion,
        evidence_strength=evidence_strength,
        scope_breadth=scope_breadth,
        submitted_by=submitted_by,
    )


def insert_web_evidence(
    origin_topic_id: int,
    origin_subtopic_id: int | None,
    query_text: str,
    title: str,
    snippet: str,
    url: str,
    source_domain: str,
    result_rank: int,
    search_provider: str,
    search_role: str,
    summary: str | None = None,
) -> int | None:
    return db_insert_web_evidence(
        origin_topic_id,
        origin_subtopic_id,
        query_text,
        title,
        snippet,
        url,
        source_domain,
        result_rank,
        search_provider,
        search_role,
        summary=summary,
    )


def clone_web_evidence_to_topic(
    source_rows: list[dict],
    target_topic_id: int,
    target_subtopic_id: int | None = None,
) -> dict[int, int]:
    return db_clone_web_evidence_to_topic(
        source_rows, target_topic_id, target_subtopic_id
    )


def insert_vote_record(
    topic_id: int,
    subtopic_id: int | None,
    round_number: int | None,
    vote_kind: str,
    subject: str,
    prompt_text: str,
    voter: str,
    parsed_ok: bool,
    decision: str | None,
    reason: str | None,
    raw_response: str,
    metadata_json: str | None = None,
) -> int:
    return db_insert_vote_record(
        topic_id,
        subtopic_id,
        round_number,
        vote_kind,
        subject,
        prompt_text,
        voter,
        parsed_ok,
        decision,
        reason,
        raw_response,
        metadata_json,
    )


def get_vote_records(
    topic_id: int,
    *,
    subtopic_id: int | None = None,
    vote_kind: str | None = None,
    round_number: int | None = None,
    limit: int | None = None,
):
    return db_get_vote_records(
        topic_id,
        subtopic_id=subtopic_id,
        vote_kind=vote_kind,
        round_number=round_number,
        limit=limit,
    )


def get_facts_by_ids(topic_id: int, fact_ids: list[int]):
    return db_get_facts_by_ids(topic_id, fact_ids)


def search_facts_hybrid(
    topic_id: int, query_text: str, query_embedding, top_k: int = 12
):
    dense = search_facts(topic_id, query_embedding, top_k=top_k)
    lexical = search_facts_lexical(topic_id, query_text, top_k=top_k)
    return _merge_ranked_rows(dense, lexical, limit=top_k)


def search_claims_hybrid(
    topic_id: int, query_text: str, query_embedding=None, top_k: int = 8
):
    _ = query_embedding
    return search_claims_lexical(topic_id, query_text, top_k=top_k)


def search_messages_hybrid(
    topic_id: int,
    query_text: str,
    query_embedding,
    msg_type: str | None = None,
    top_k: int = 8,
    exclude_ids=None,
):
    dense = search_messages(
        topic_id,
        query_embedding,
        msg_type=msg_type,
        top_k=top_k,
        exclude_ids=exclude_ids,
    )
    lexical = search_messages_lexical(
        topic_id, query_text, msg_type=msg_type, top_k=top_k, exclude_ids=exclude_ids
    )
    return _merge_ranked_rows(dense, lexical, limit=top_k)


def search_corpus_chunks_hybrid(
    topic_id: int | None,
    query_text: str,
    query_embedding=None,
    top_k: int = 8,
    include_global: bool = True,
):
    dense = []
    if query_embedding is not None:
        dense = db_search_corpus_chunks(
            topic_id,
            query_embedding,
            top_k=top_k,
            include_global=include_global,
        )
    lexical = db_search_corpus_chunks_lexical(
        topic_id,
        query_text,
        top_k=top_k,
        include_global=include_global,
    )
    return _merge_ranked_rows(dense, lexical, limit=top_k)


def search_web_evidence_same_topic(
    topic_id: int, query_text: str, top_k: int = 8, max_age_days: int = 30
):
    return db_search_web_evidence_same_topic(
        topic_id, query_text, top_k=top_k, max_age_days=max_age_days
    )


def search_web_evidence_cross_topic(
    topic_id: int, query_text: str, top_k: int = 8, max_age_days: int = 30
):
    return db_search_web_evidence_cross_topic(
        topic_id, query_text, top_k=top_k, max_age_days=max_age_days
    )


def get_web_evidence_for_topic(topic_id: int) -> list[dict]:
    return db_get_web_evidence(topic_id)


def insert_code_evidence(
    origin_topic_id: int,
    origin_subtopic_id: int | None,
    hypothesis: str,
    source_code: str,
    stdout: str | None,
    stderr: str | None,
    exit_code: int,
    execution_time_s: float | None,
    iterations: int,
    success: bool,
    requesting_role: str | None = None,
    summary: str | None = None,
    parent_evidence_id: int | None = None,
) -> int:
    return db_insert_code_evidence(
        origin_topic_id,
        origin_subtopic_id,
        hypothesis,
        source_code,
        stdout,
        stderr,
        exit_code,
        execution_time_s,
        iterations,
        success,
        requesting_role=requesting_role,
        summary=summary,
        parent_evidence_id=parent_evidence_id,
    )


def get_code_evidence_for_topic(topic_id: int) -> list[dict]:
    return db_get_code_evidence_for_topic(topic_id)


def get_code_evidence_for_topic_full(topic_id: int) -> list[dict]:
    """Including source_code — used by the dashboard."""
    return db_get_code_evidence_for_topic_full(topic_id)


def get_code_evidence_by_id(evidence_id: int) -> dict | None:
    return db_get_code_evidence_by_id(evidence_id)


def increment_code_evidence_review_count(evidence_id: int) -> None:
    db_increment_review_count(evidence_id)


def reset_code_evidence_review_count(evidence_id: int) -> None:
    db_reset_review_count(evidence_id)


def insert_api_evidence(
    origin_topic_id: int,
    origin_subtopic_id: int | None,
    question: str,
    answer: str,
    provider: str | None = None,
    requested_provider: str | None = None,
    model: str | None = None,
    requesting_role: str | None = None,
    planner_reason: str | None = None,
    fallback_used: bool = False,
) -> int:
    return db_insert_api_evidence(
        origin_topic_id,
        origin_subtopic_id,
        question,
        answer,
        provider=provider,
        requested_provider=requested_provider,
        model=model,
        requesting_role=requesting_role,
        planner_reason=planner_reason,
        fallback_used=fallback_used,
    )


def get_api_evidence_for_topic(topic_id: int, limit: int = 10) -> list[dict]:
    return db_get_api_evidence_for_topic(topic_id, limit=limit)


def get_api_evidence_by_id(evidence_id: int) -> dict | None:
    return db_get_api_evidence_by_id(evidence_id)


def supersede_facts(fact_ids: list[int]) -> None:
    db_supersede_facts(fact_ids)


def get_web_evidence_count(topic_id: int) -> int:
    return db_get_web_evidence_count(topic_id)


def insert_tool_trace(
    topic_id: int,
    tool_type: str,
    query: str | None = None,
    result_count: int | None = None,
    metadata_json: str | None = None,
) -> int:
    return db_insert_tool_trace(topic_id, tool_type, query, result_count, metadata_json)


def insert_corpus_document(**kwargs) -> int:
    return db_insert_corpus_document(**kwargs)


def get_corpus_document(document_id: int) -> dict | None:
    return db_get_corpus_document(document_id)


def list_corpus_documents(
    topic_id: int | None = None,
    *,
    include_global: bool = True,
    access_scope: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    return db_list_corpus_documents(
        topic_id,
        include_global=include_global,
        access_scope=access_scope,
        limit=limit,
    )


def create_corpus_ingest_run(**kwargs) -> int:
    return db_create_corpus_ingest_run(**kwargs)


def update_corpus_ingest_run(run_id: int, **kwargs) -> None:
    db_update_corpus_ingest_run(run_id, **kwargs)


def insert_corpus_chunk(**kwargs) -> int:
    return db_insert_corpus_chunk(**kwargs)


def insert_corpus_chunk_with_embedding(**kwargs) -> int:
    return db_insert_corpus_chunk_with_embedding(**kwargs)


def get_corpus_chunks_for_document(document_id: int) -> list[dict]:
    return db_get_corpus_chunks_for_document(document_id)


def reindex_corpus_document(document_id: int) -> int:
    return db_reindex_corpus_document(document_id)


def get_corpus_neighbor_chunks(chunk_id: int, window: int = 1) -> list[dict]:
    return db_get_corpus_neighbor_chunks(chunk_id, window=window)


def insert_optimization_problem(**kwargs) -> int:
    return db_insert_optimization_problem(**kwargs)


def get_optimization_problem(problem_id: int) -> dict | None:
    return db_get_optimization_problem(problem_id)


def list_optimization_problems(topic_id: int, limit: int | None = None) -> list[dict]:
    return db_list_optimization_problems(topic_id, limit=limit)


def insert_optimization_component(**kwargs) -> int:
    return db_insert_optimization_component(**kwargs)


def get_optimization_components(
    problem_id: int,
    *,
    component_type: str | None = None,
    review_status: str | None = None,
) -> list[dict]:
    return db_get_optimization_components(
        problem_id,
        component_type=component_type,
        review_status=review_status,
    )


def update_optimization_component_review(
    component_id: int,
    *,
    review_status: str,
    validation_notes: str | None = None,
) -> bool:
    updated = db_update_optimization_component_review(
        component_id,
        review_status=review_status,
        validation_notes=validation_notes,
    )
    if updated:
        with get_db() as conn:
            component = conn.execute(
                """
                SELECT topic_id, problem_id
                FROM OptimizationComponent
                WHERE id = ?
                """,
                (component_id,),
            ).fetchone()
        if component:
            from .optimization import propagate_component_status_to_solver_evidence

            propagate_component_status_to_solver_evidence(
                int(component["topic_id"]), problem_id=int(component["problem_id"])
            )
    return updated


def insert_optimization_model_ir(**kwargs) -> int:
    return db_insert_optimization_model_ir(**kwargs)


def get_optimization_model_irs(problem_id: int) -> list[dict]:
    return db_get_optimization_model_irs(problem_id)


def insert_optimization_artifact(**kwargs) -> int:
    return db_insert_optimization_artifact(**kwargs)


def get_optimization_artifacts(problem_id: int) -> list[dict]:
    return db_get_optimization_artifacts(problem_id)


def insert_solver_run(**kwargs) -> int:
    return db_insert_solver_run(**kwargs)


def get_solver_runs(problem_id: int) -> list[dict]:
    return db_get_solver_runs(problem_id)


def insert_model_diagnostic(**kwargs) -> int:
    return db_insert_model_diagnostic(**kwargs)


def get_model_diagnostics(
    problem_id: int, *, status: str | None = None
) -> list[dict]:
    return db_get_model_diagnostics(problem_id, status=status)


def update_model_diagnostic_status(
    diagnostic_id: int,
    *,
    status: str,
    resolution: str | None = None,
) -> bool:
    return db_update_model_diagnostic_status(
        diagnostic_id,
        status=status,
        resolution=resolution,
    )


def upsert_modeling_experience(**kwargs) -> int:
    return db_upsert_modeling_experience(**kwargs)


def list_modeling_experiences(
    *,
    family: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    return db_list_modeling_experiences(
        family=family,
        status=status,
        limit=limit,
    )


def record_modeling_experience_event(
    experience_id: int,
    *,
    outcome: str,
    validation_summary_json: str | None = None,
) -> bool:
    return db_record_modeling_experience_event(
        experience_id,
        outcome=outcome,
        validation_summary_json=validation_summary_json,
    )


def get_mse_review_snapshot(topic_id: int) -> dict:
    documents = []
    for doc in list_corpus_documents(topic_id, limit=500):
        chunks = get_corpus_chunks_for_document(doc["id"])
        documents.append(
            {
                **doc,
                "chunk_count": len(chunks),
                "chunks": chunks[:20],
            }
        )

    problems = []
    for problem in list_optimization_problems(topic_id, limit=500):
        components = get_optimization_components(problem["id"])
        model_irs = get_optimization_model_irs(problem["id"])
        artifacts = get_optimization_artifacts(problem["id"])
        solver_runs = get_solver_runs(problem["id"])
        diagnostics = get_model_diagnostics(problem["id"])
        problems.append(
            {
                **problem,
                "components": components,
                "model_irs": model_irs,
                "artifacts": artifacts,
                "solver_runs": solver_runs,
                "diagnostics": diagnostics,
                "pending_component_count": sum(
                    1 for item in components if item.get("review_status") == "candidate"
                ),
                "open_diagnostic_count": sum(
                    1 for item in diagnostics if item.get("status") == "open"
                ),
            }
        )

    return {
        "topic_id": topic_id,
        "documents": documents,
        "problems": problems,
        "review_counts": {
            "documents": len(documents),
            "problems": len(problems),
            "components": sum(len(problem["components"]) for problem in problems),
            "model_irs": sum(len(problem["model_irs"]) for problem in problems),
            "artifacts": sum(len(problem["artifacts"]) for problem in problems),
            "solver_runs": sum(len(problem["solver_runs"]) for problem in problems),
            "open_diagnostics": sum(
                problem["open_diagnostic_count"] for problem in problems
            ),
            "pending_components": sum(
                problem["pending_component_count"] for problem in problems
            ),
        },
    }


def _json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _extract_solver_run_ids(text: str) -> list[int]:
    import re

    ids: list[int] = []
    for match in re.finditer(r"SolverRun\s+(\d+)", text or "", flags=re.IGNORECASE):
        try:
            ids.append(int(match.group(1)))
        except ValueError:
            continue
    return list(dict.fromkeys(ids))


def get_mse_provenance_report(topic_id: int) -> dict:
    """Build a provider-free provenance report for MSE/OR review workflows."""
    snapshot = get_mse_review_snapshot(topic_id)
    with get_db() as conn:
        topic = conn.execute("SELECT * FROM Topic WHERE id = ?", (topic_id,)).fetchone()
        claim_rows = conn.execute(
            """
            SELECT id, content, status, claim_type, scope_tags, scope_context,
                   falsification_criteria, inference_logic, conclusion,
                   evidence_strength, created_at
            FROM Claim
            WHERE topic_id = ? AND claim_type = 'optimization_result'
            ORDER BY id ASC
            """,
            (topic_id,),
        ).fetchall()
        candidate_rows = conn.execute(
            """
            SELECT id, candidate_text, status, claim_type, scope_tags, scope_context,
                   falsification_criteria, inference_logic, conclusion,
                   evidence_strength, created_at, reviewed_at, review_note,
                   accepted_claim_id
            FROM ClaimCandidate
            WHERE topic_id = ? AND claim_type = 'optimization_result'
            ORDER BY id ASC
            """,
            (topic_id,),
        ).fetchall()
        support_edges = conn.execute(
            """
            SELECT source_id, source_type, target_id, target_type, relation
            FROM KnowledgeEdge
            WHERE topic_id = ?
              AND target_type = 'claim'
              AND relation = 'supports'
              AND is_active = 1
            ORDER BY target_id ASC, source_type ASC, source_id ASC
            """,
            (topic_id,),
        ).fetchall()
        diagnostics = conn.execute(
            """
            SELECT *
            FROM ModelDiagnostic
            WHERE topic_id = ?
            ORDER BY status ASC, severity DESC, id ASC
            """,
            (topic_id,),
        ).fetchall()

    support_by_claim: dict[int, list[dict]] = {}
    for edge in support_edges:
        support_by_claim.setdefault(int(edge["target_id"]), []).append(dict(edge))

    def claim_payload(row, text_key: str) -> dict:
        text = str(row[text_key] or "")
        inference = str(row["inference_logic"] or "")
        conclusion = str(row["conclusion"] or "")
        return {
            "id": row["id"],
            "status": row["status"],
            "text": text,
            "conclusion": conclusion,
            "scope_tags": _json_list(row["scope_tags"]),
            "scope_context": row["scope_context"],
            "falsification_criteria": row["falsification_criteria"],
            "inference_logic": inference,
            "evidence_strength": row["evidence_strength"],
            "solver_run_ids": _extract_solver_run_ids(
                " ".join([text, inference, conclusion])
            ),
            "support_edges": support_by_claim.get(int(row["id"]), []),
        }

    problems = []
    for problem in snapshot["problems"]:
        components = [
            {
                "id": component["id"],
                "component_type": component["component_type"],
                "symbol": component["symbol"],
                "review_status": component["review_status"],
                "source_refs": _json_list(component["source_refs_json"]),
                "natural_text": component["natural_text"],
                "formal_text": component["formal_text"],
            }
            for component in problem["components"]
        ]
        artifacts = [
            {
                "id": artifact["id"],
                "artifact_type": artifact["artifact_type"],
                "model_language": artifact["model_language"],
                "parser_status": artifact["parser_status"],
                "repair_status": artifact["repair_status"],
                "linked_component_ids": _json_list(
                    artifact["linked_component_ids_json"]
                ),
                "has_component_fingerprints": bool(
                    artifact.get("component_fingerprints_json")
                ),
            }
            for artifact in problem["artifacts"]
        ]
        model_irs = [
            {
                "id": model_ir["id"],
                "status": model_ir["status"],
                "linked_component_ids": _json_list(
                    model_ir["linked_component_ids_json"]
                ),
                "has_component_fingerprints": bool(
                    model_ir.get("component_fingerprints_json")
                ),
            }
            for model_ir in problem.get("model_irs", [])
        ]
        solver_runs = [
            {
                "id": run["id"],
                "artifact_id": run["artifact_id"],
                "solver_backend": run["solver_backend"],
                "status": run["status"],
                "objective_value": run["objective_value"],
                "code_evidence_id": run["code_evidence_id"],
            }
            for run in problem["solver_runs"]
        ]
        problems.append(
            {
                "id": problem["id"],
                "title": problem["title"],
                "problem_class": problem["problem_class"],
                "status": problem["status"],
                "components": components,
                "model_irs": model_irs,
                "artifacts": artifacts,
                "solver_runs": solver_runs,
                "open_diagnostics": [
                    dict(diagnostic)
                    for diagnostic in problem["diagnostics"]
                    if diagnostic.get("status") == "open"
                ],
            }
        )

    return {
        "topic": dict(topic) if topic else {"id": topic_id},
        "review_counts": snapshot["review_counts"],
        "documents": [
            {
                "id": doc["id"],
                "title": doc["title"],
                "doc_type": doc["doc_type"],
                "source_path": doc["source_path"],
                "source_url": doc["source_url"],
                "parser_version": doc["parser_version"],
                "index_status": doc["index_status"],
                "chunk_count": doc["chunk_count"],
            }
            for doc in snapshot["documents"]
        ],
        "problems": problems,
        "solver_claims": [claim_payload(row, "content") for row in claim_rows],
        "solver_claim_candidates": [
            {
                **claim_payload(row, "candidate_text"),
                "reviewed_at": row["reviewed_at"],
                "review_note": row["review_note"],
                "accepted_claim_id": row["accepted_claim_id"],
            }
            for row in candidate_rows
        ],
        "diagnostics": [dict(row) for row in diagnostics],
    }


def render_mse_provenance_markdown(report: dict) -> str:
    """Render a compact markdown report from `get_mse_provenance_report`."""
    topic = report.get("topic") or {}
    lines = [
        f"# MSE Provenance Report: {topic.get('summary') or topic.get('id')}",
        "",
        "## Review Counts",
    ]
    for key, value in (report.get("review_counts") or {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Corpus Documents"])
    documents = report.get("documents") or []
    if not documents:
        lines.append("- None")
    for doc in documents:
        source = doc.get("source_url") or doc.get("source_path") or "local"
        lines.append(
            f"- D{doc.get('id')}: {doc.get('title')} "
            f"({doc.get('doc_type')}, chunks={doc.get('chunk_count')}, source={source})"
        )

    lines.extend(["", "## Optimization Problems"])
    problems = report.get("problems") or []
    if not problems:
        lines.append("- None")
    for problem in problems:
        lines.append(f"- Problem {problem.get('id')}: {problem.get('title')}")
        for component in problem.get("components") or []:
            lines.append(
                f"  - C{component.get('id')} {component.get('component_type')} "
                f"{component.get('symbol') or ''} "
                f"[{component.get('review_status')}] refs={component.get('source_refs')}"
            )
        for model_ir in problem.get("model_irs") or []:
            lines.append(
                f"  - IR{model_ir.get('id')} [{model_ir.get('status')}] "
                f"components={model_ir.get('linked_component_ids')}"
            )
        for artifact in problem.get("artifacts") or []:
            lines.append(
                f"  - O{artifact.get('id')} {artifact.get('artifact_type')} "
                f"[{artifact.get('parser_status')}] "
                f"components={artifact.get('linked_component_ids')}"
            )
        for run in problem.get("solver_runs") or []:
            lines.append(
                f"  - SolverRun {run.get('id')} {run.get('solver_backend')} "
                f"[{run.get('status')}] objective={run.get('objective_value')} "
                f"E{run.get('code_evidence_id')}"
            )
        for diagnostic in problem.get("open_diagnostics") or []:
            lines.append(
                f"  - Diagnostic {diagnostic.get('id')} "
                f"{diagnostic.get('diagnostic_type')} "
                f"[{diagnostic.get('severity')}] {diagnostic.get('message')}"
            )

    lines.extend(["", "## Solver Claims"])
    claims = report.get("solver_claims") or []
    if not claims:
        lines.append("- None")
    for claim in claims:
        lines.append(
            f"- Claim {claim.get('id')} [{claim.get('status')}]: "
            f"{claim.get('conclusion') or claim.get('text')}"
        )
        lines.append(f"  - Solver runs: {claim.get('solver_run_ids')}")
        lines.append(f"  - Support edges: {claim.get('support_edges')}")

    lines.extend(["", "## Solver Claim Candidates"])
    candidates = report.get("solver_claim_candidates") or []
    if not candidates:
        lines.append("- None")
    for candidate in candidates:
        lines.append(
            f"- Candidate {candidate.get('id')} [{candidate.get('status')}]: "
            f"{candidate.get('conclusion') or candidate.get('text')}"
        )
    return "\n".join(lines) + "\n"


def update_fact_structured_columns(
    fact_id: int,
    subject: str | None = None,
    predicate: str | None = None,
    object_json: str | None = None,
    qualifiers_json: str | None = None,
    attribution_json: str | None = None,
) -> None:
    db_update_fact_structured_columns(
        fact_id, subject, predicate, object_json, qualifiers_json, attribution_json
    )


def update_claim_structured_columns(
    claim_id: int,
    subject: str | None = None,
    predicate: str | None = None,
    object_json: str | None = None,
    qualifiers_json: str | None = None,
    polarity: str | None = None,
) -> None:
    db_update_claim_structured_columns(
        claim_id, subject, predicate, object_json, qualifiers_json, polarity
    )


def get_claims(
    topic_id: int, limit: int | None = None, include_superseded: bool = False
):
    with get_db() as conn:
        params = [topic_id]
        query = "SELECT * FROM Claim WHERE topic_id = ?"
        if not include_superseded:
            query += " AND superseded_by IS NULL"
        query += " ORDER BY id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def count_claims(topic_id: int) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM Claim WHERE topic_id = ? AND superseded_by IS NULL",
            (topic_id,),
        ).fetchone()
        return row[0] if row else 0


def count_code_evidence(topic_id: int) -> int:
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM CodeEvidence WHERE origin_topic_id = ?",
            (topic_id,),
        ).fetchone()
        return row[0] if row else 0


def update_topic_conclusion(topic_id: int, conclusion: str) -> None:
    db_update_topic_conclusion(topic_id, conclusion)


def update_fact_summary_and_embedding(
    fact_id: int, summary: str, embedding: list[float]
) -> None:
    db_update_fact_summary_and_embedding(fact_id, summary, embedding)


def update_claim_summary(claim_id: int, summary: str) -> None:
    db_update_claim_summary(claim_id, summary)


def get_unprocessed_web_evidence(topic_id: int, limit: int = 20) -> list[dict]:
    return db_get_unprocessed_web_evidence(topic_id, limit)


def mark_web_evidence_ledger_processed(web_ids: list[int]) -> None:
    db_mark_web_evidence_ledger_processed(web_ids)


def get_ledger_entries_with_names(topic_id: int):
    return db_get_ledger_entries_with_names(topic_id)


def get_contested_ledger_pairs(topic_id: int):
    return db_get_contested_ledger_pairs(topic_id)


def get_ledger_edges(topic_id: int):
    return db_get_ledger_edges(topic_id)


def get_active_ledger_pending(topic_id: int, current_round: int):
    return db_get_active_ledger_pending(topic_id, current_round)


def get_max_round_number(topic_id: int, subtopic_id: int):
    return db_get_max_round_number(topic_id, subtopic_id)


# ---------------------------------------------------------------------------
# Phase C: KnowledgeEdge + supersede_fact wrappers
# ---------------------------------------------------------------------------


def insert_knowledge_edge(
    topic_id: int,
    source_id: int,
    source_type: str,
    target_id: int,
    target_type: str,
    relation: str,
    justification_group: str = "default",
    confidence: float | None = None,
    created_by: str | None = None,
) -> int | None:
    return db_insert_knowledge_edge(
        topic_id,
        source_id,
        source_type,
        target_id,
        target_type,
        relation,
        justification_group=justification_group,
        confidence=confidence,
        created_by=created_by,
    )


def get_knowledge_edges(
    topic_id: int,
    source_id: int | None = None,
    source_type: str | None = None,
    target_id: int | None = None,
    target_type: str | None = None,
    relation: str | None = None,
    active_only: bool = True,
) -> list[dict]:
    return db_get_knowledge_edges(
        topic_id,
        source_id=source_id,
        source_type=source_type,
        target_id=target_id,
        target_type=target_type,
        relation=relation,
        active_only=active_only,
    )


def deactivate_knowledge_edge(edge_id: int) -> None:
    db_deactivate_knowledge_edge(edge_id)


def get_active_conflicts(topic_id: int, node_type: str) -> list[dict]:
    return db_get_active_conflicts(topic_id, node_type)


def get_claim_justification_groups(
    topic_id: int, claim_id: int
) -> dict[str, list[dict]]:
    return db_get_claim_justification_groups(topic_id, claim_id)


def supersede_fact(old_fact_id: int, new_fact_id: int) -> None:
    db_supersede_fact(old_fact_id, new_fact_id)


def get_dismissed_knowledge(topic_id: int) -> list[dict]:
    from .db import get_dismissed_knowledge as db_get_dismissed_knowledge

    return db_get_dismissed_knowledge(topic_id)


def insert_web_query_cache(
    topic_id: int,
    query_text: str,
    result_ids: list[int],
    embedding: list[float],
) -> int | None:
    from .db import insert_web_query_cache as db_insert_web_query_cache

    return db_insert_web_query_cache(topic_id, query_text, result_ids, embedding)


def search_web_queries_semantic(
    topic_id: int,
    query_embedding: list[float],
    top_k: int = 5,
    max_age_days: int = 30,
) -> list[dict]:
    from .db import search_web_queries_semantic as db_search_web_queries_semantic

    return db_search_web_queries_semantic(
        topic_id, query_embedding, top_k=top_k, max_age_days=max_age_days
    )


# ---------------------------------------------------------------------------
# Phase F.1: TopicConfig wrappers
# ---------------------------------------------------------------------------


def set_topic_config(topic_id: int, key: str, value: str) -> None:
    db_set_topic_config(topic_id, key, value)


def get_topic_config(topic_id: int, key: str) -> str | None:
    return db_get_topic_config(topic_id, key)


def get_all_topic_config(topic_id: int) -> dict[str, str]:
    return db_get_all_topic_config(topic_id)


# ---------------------------------------------------------------------------
# Phase F.2: HITL wrappers
# ---------------------------------------------------------------------------


def pause_topic(topic_id: int, stage: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Topic SET status = 'Paused', paused_at_stage = ? WHERE id = ?",
            (stage, topic_id),
        )


def resume_topic(topic_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Topic SET status = 'Running', paused_at_stage = NULL WHERE id = ?",
            (topic_id,),
        )


def inject_knowledge(
    topic_id: int,
    injection_type: str,
    content: str,
    subtopic_id: int | None = None,
) -> int:
    return db_insert_user_injection(topic_id, injection_type, content, subtopic_id)


def get_pending_injections(topic_id: int) -> list[dict]:
    return db_get_pending_injections(topic_id)


def mark_injection_processed(injection_id: int) -> None:
    db_mark_injection_processed(injection_id)


# ---------------------------------------------------------------------------
# Phase F.3: Report wrappers
# ---------------------------------------------------------------------------


def save_report(topic_id: int, report_json: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Topic SET report_json = ? WHERE id = ?",
            (report_json, topic_id),
        )


def get_report(topic_id: int) -> str | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT report_json FROM Topic WHERE id = ?", (topic_id,)
        ).fetchone()
        return row["report_json"] if row and row["report_json"] else None


# ---------------------------------------------------------------------------
# Phase F.5: Topic queue wrappers
# ---------------------------------------------------------------------------


def enqueue_topic(summary: str, detail: str, config: dict | None = None) -> int:
    from . import topic_config

    # Validate config before creating topic
    if config and isinstance(config, dict):
        for key, value in config.items():
            ok, err = topic_config.validate_config_value(key, str(value))
            if not ok:
                raise ValueError(err)
    with get_db() as conn:
        # Atomic queue position assignment — subquery prevents race condition
        cursor = conn.execute(
            "INSERT INTO Topic (summary, detail, status, queue_position, queued_at) "
            "VALUES (?, ?, 'Queued', "
            "(SELECT COALESCE(MAX(queue_position), 0) + 1 FROM Topic WHERE status = 'Queued'), "
            "CURRENT_TIMESTAMP)",
            (summary, detail),
        )
        topic_id = cursor.lastrowid
        if config and isinstance(config, dict):
            for key, value in config.items():
                conn.execute(
                    "INSERT OR REPLACE INTO TopicConfig (topic_id, config_key, config_value) VALUES (?, ?, ?)",
                    (topic_id, key, str(value)),
                )
    analytics.capture(
        f"topic_{topic_id}",
        "topic_queued",
        {"summary_length": len(summary), "has_config": bool(config)},
    )
    return topic_id


def dequeue_topic(topic_id: int) -> None:
    db_dequeue_topic(topic_id)


def get_topic_queue() -> list[dict]:
    return db_get_topic_queue()


def reorder_queue(topic_ids: list[int]) -> None:
    db_reorder_queue(topic_ids)


def get_next_queued_topic() -> dict | None:
    return db_get_next_queued_topic()
