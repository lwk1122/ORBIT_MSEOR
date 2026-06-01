"""VIKI — Watchdog + Autonomous Repair System (Batch Mode).

Two layers:
1. Watchdog (pure SQL) — scans DB for anomalies, produces issue list
2. VIKI (LLM-powered) — batches same-type issues, one LLM call per batch

Trigger: every 2 rounds OR 20+ open issues (whichever comes first).
VikiTracker table prevents repeated checks (max 3 LLM attempts per issue).
"""

import json
import logging
import re
import sqlite3
from collections import defaultdict

from . import api
from . import topic_config
from .db import (
    get_db,
    increment_viki_check,
    mark_viki_resolved,
    upsert_viki_issue,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 20
TRIGGER_THRESHOLD = 20


def _control_provider(topic_id: int | None) -> str:
    try:
        return topic_config.get_provider_profile_for(topic_id or 0, "control_provider")
    except sqlite3.OperationalError:
        raise
    except Exception as exc:
        logger.debug("[VIKI] Control provider lookup failed: %s", exc)
        return "minimax"


def _batch_topic_id(issues: list[dict]) -> int:
    if not issues:
        return 0
    return int(issues[0].get("topic_id") or 0)


# ---------------------------------------------------------------------------
# Watchdog: Pure SQL checks (zero LLM cost)
# ---------------------------------------------------------------------------


def run_watchdog(topic_id: int) -> list[dict]:
    """Scan DB for data quality issues. Pure SQL, no LLM. Returns issue list."""
    issues: list[dict] = []
    issues += _check_claim_conflicts(topic_id)
    issues += _check_missing_spo(topic_id)
    issues += _check_ledger_anomalies(topic_id)
    issues += _check_ledger_missing_stat_type(topic_id)
    issues += _check_orphan_claim_refs(topic_id)
    issues += _check_stale_pending(topic_id)
    issues += _check_contradictory_facts(topic_id)
    issues += _check_bad_ce_summary(topic_id)
    issues += _check_solver_claim_evidence(topic_id)
    return _filter_new_issues(issues, topic_id)


def get_open_issue_count(topic_id: int) -> int:
    """Count open VikiTracker issues for trigger check (scoped to topic)."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT COUNT(*) FROM VikiTracker
               WHERE topic_id = ? AND status = 'open' AND check_count < max_checks""",
            (topic_id,),
        ).fetchone()
        return row[0] if row else 0


def _check_claim_conflicts(topic_id: int) -> list[dict]:
    """Find active claim-claim conflicts with no resolution."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT ke.source_id, ke.target_id,
                      c1.claim_score AS score_a, c2.claim_score AS score_b
               FROM KnowledgeEdge ke
               JOIN Claim c1 ON ke.source_id = c1.id AND c1.superseded_by IS NULL
               JOIN Claim c2 ON ke.target_id = c2.id AND c2.superseded_by IS NULL
               WHERE ke.topic_id = ? AND ke.source_type = 'claim'
                 AND ke.target_type = 'claim' AND ke.relation = 'conflicts_with'
                 AND ke.is_active = 1""",
            (topic_id,),
        ).fetchall()
    return [
        {
            "target_table": "Claim",
            "target_id": r["source_id"],
            "issue_type": "claim_conflict",
            "source_id": r["source_id"],
            "target_claim_id": r["target_id"],
            "score_a": r["score_a"],
            "score_b": r["score_b"],
        }
        for r in rows
    ]


def _check_missing_spo(topic_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id FROM Fact WHERE topic_id = ? AND subject IS NULL AND superseded_by IS NULL",
            (topic_id,),
        ).fetchall()
    return [
        {"target_table": "Fact", "target_id": r["id"], "issue_type": "missing_spo"}
        for r in rows
    ]


def _check_ledger_anomalies(topic_id: int) -> list[dict]:
    issues = []
    with get_db() as conn:
        rows = conn.execute(
            """SELECT l.id FROM Ledger l
               JOIN LedgerAttribute a ON l.attribute_id = a.id
               WHERE l.topic_id = ? AND a.canonical_name = 'F1 Score'
                 AND (l.value_stat_type IS NULL OR l.value_stat_type IN ('point', 'mean_std'))
                 AND (l.value_numeric_min > 1.0 OR l.value_numeric_min < 0.0)""",
            (topic_id,),
        ).fetchall()
        for r in rows:
            issues.append(
                {
                    "target_table": "Ledger",
                    "target_id": r["id"],
                    "issue_type": "ledger_anomaly",
                }
            )
        rows = conn.execute(
            """SELECT l.id FROM Ledger l
               JOIN LedgerAttribute a ON l.attribute_id = a.id
               WHERE l.topic_id = ? AND a.canonical_name = 'Training Time'
                 AND l.value_numeric_min < 0""",
            (topic_id,),
        ).fetchall()
        for r in rows:
            issues.append(
                {
                    "target_table": "Ledger",
                    "target_id": r["id"],
                    "issue_type": "ledger_anomaly",
                }
            )
    return issues


def _check_ledger_missing_stat_type(topic_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id FROM Ledger WHERE topic_id = ? AND value_stat_type IS NULL",
            (topic_id,),
        ).fetchall()
    return [
        {
            "target_table": "Ledger",
            "target_id": r["id"],
            "issue_type": "ledger_missing_stat_type",
        }
        for r in rows
    ]


def _check_orphan_claim_refs(topic_id: int) -> list[dict]:
    issues = []
    with get_db() as conn:
        claims = conn.execute(
            "SELECT id, support_fact_ids_json FROM Claim WHERE topic_id = ? AND superseded_by IS NULL",
            (topic_id,),
        ).fetchall()
        # Collect all referenced fact IDs across all claims
        all_fids: set[int] = set()
        claim_fids: list[tuple] = []  # (claim_row, fid_list)
        for c in claims:
            try:
                fids = json.loads(c["support_fact_ids_json"] or "[]")
                int_fids = [int(f) for f in fids if f is not None]
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if int_fids:
                claim_fids.append((c, int_fids))
                all_fids.update(int_fids)
        if not all_fids:
            return []
        # Batch query: find all active (non-superseded) fact IDs
        placeholders = ",".join("?" * len(all_fids))
        active_rows = conn.execute(
            f"SELECT id FROM Fact WHERE id IN ({placeholders}) AND superseded_by IS NULL",
            list(all_fids),
        ).fetchall()
        active_ids = {r["id"] for r in active_rows}
        for c, fids in claim_fids:
            if any(fid not in active_ids for fid in fids):
                issues.append(
                    {
                        "target_table": "Claim",
                        "target_id": c["id"],
                        "issue_type": "orphan_ref",
                    }
                )
    return issues


def _check_stale_pending(topic_id: int) -> list[dict]:
    with get_db() as conn:
        # Per-subtopic max round: a pending entry is stale when its subtopic
        # has advanced past its TTL, not when *any* subtopic has.
        rows = conn.execute(
            """SELECT lp.id FROM LedgerPending lp
               JOIN (
                   SELECT subtopic_id, MAX(round_number) AS max_rn
                   FROM Message WHERE topic_id = ? AND msg_type = 'standard'
                   GROUP BY subtopic_id
               ) m ON lp.subtopic_id = m.subtopic_id
               WHERE lp.topic_id = ? AND lp.ttl_expires_round IS NOT NULL
                 AND lp.ttl_expires_round <= m.max_rn""",
            (topic_id, topic_id),
        ).fetchall()
    return [
        {
            "target_table": "LedgerPending",
            "target_id": r["id"],
            "issue_type": "stale_pending",
        }
        for r in rows
    ]


def _check_contradictory_facts(topic_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT f1.id AS id1, f2.id AS id2
               FROM Fact f1
               JOIN Fact f2 ON f1.subject = f2.subject AND f1.predicate = f2.predicate
                 AND f1.id < f2.id AND f1.topic_id = f2.topic_id
               WHERE f1.topic_id = ? AND f1.superseded_by IS NULL AND f2.superseded_by IS NULL
                 AND f1.subject IS NOT NULL AND f1.object_json != f2.object_json
                 AND NOT EXISTS (
                     SELECT 1 FROM KnowledgeEdge ke
                     WHERE ke.source_id = f1.id AND ke.source_type = 'fact'
                       AND ke.target_id = f2.id AND ke.target_type = 'fact'
                       AND ke.relation = 'conflicts_with'
                 )
               LIMIT 50""",
            (topic_id,),
        ).fetchall()
    return [
        {
            "target_table": "Fact",
            "target_id": r["id1"],
            "issue_type": "fact_contradiction",
            "other_fact_id": r["id2"],
        }
        for r in rows
    ]


def _check_bad_ce_summary(topic_id: int) -> list[dict]:
    """Find CodeEvidence with polluted hypothesis or missing/bad summary."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id FROM CodeEvidence
               WHERE origin_topic_id = ?
                 AND (hypothesis LIKE 'import %' OR hypothesis LIKE 'from %'
                      OR hypothesis LIKE 'def %' OR hypothesis LIKE 'class %'
                      OR hypothesis LIKE '%' || X'0A' || 'import %'
                      OR hypothesis LIKE '%' || X'0A' || 'from %'
                      OR hypothesis LIKE '%' || X'0A' || 'def %'
                      OR hypothesis LIKE '%' || X'0A' || 'class %'
                      OR LENGTH(hypothesis) > 1000
                      OR summary IS NULL OR TRIM(summary) = '')""",
            (topic_id,),
        ).fetchall()
    return [
        {
            "target_table": "CodeEvidence",
            "target_id": r["id"],
            "issue_type": "bad_ce_summary",
        }
        for r in rows
    ]


def _solver_run_ids(text: str) -> list[int]:
    ids: list[int] = []
    for match in re.finditer(r"SolverRun\s+(\d+)", text or "", flags=re.IGNORECASE):
        try:
            ids.append(int(match.group(1)))
        except ValueError:
            continue
    return ids


def _check_solver_claim_evidence(topic_id: int) -> list[dict]:
    """Find optimization_result claims that lost solver/code evidence grounding."""
    issues: list[dict] = []
    with get_db() as conn:
        rows = conn.execute(
            """SELECT 'Claim' AS target_table, id, content AS text,
                      inference_logic, conclusion
               FROM Claim
               WHERE topic_id = ? AND claim_type = 'optimization_result'
                 AND superseded_by IS NULL
               UNION ALL
               SELECT 'ClaimCandidate' AS target_table, id, candidate_text AS text,
                      inference_logic, conclusion
               FROM ClaimCandidate
               WHERE topic_id = ? AND claim_type = 'optimization_result'
                 AND status = 'pending'""",
            (topic_id, topic_id),
        ).fetchall()
        for row in rows:
            text = " ".join(
                str(row[key] or "") for key in ("text", "inference_logic", "conclusion")
            )
            solver_ids = _solver_run_ids(text)
            has_code_ref = bool(re.search(r"\[E\d+\]", text))
            if not solver_ids and not has_code_ref:
                issues.append(
                    {
                        "target_table": row["target_table"],
                        "target_id": row["id"],
                        "issue_type": "solver_claim_missing_evidence",
                    }
                )
                continue
            if not solver_ids:
                continue
            placeholders = ",".join("?" for _ in solver_ids)
            solver_rows = conn.execute(
                f"SELECT id, status FROM SolverRun WHERE id IN ({placeholders})",
                solver_ids,
            ).fetchall()
            statuses = {solver["id"]: solver["status"] for solver in solver_rows}
            if any(solver_id not in statuses for solver_id in solver_ids):
                issues.append(
                    {
                        "target_table": row["target_table"],
                        "target_id": row["id"],
                        "issue_type": "solver_claim_orphan_run",
                    }
                )
                continue
            says_optimal = "optimal" in text.lower()
            if says_optimal and any(status != "optimal" for status in statuses.values()):
                issues.append(
                    {
                        "target_table": row["target_table"],
                        "target_id": row["id"],
                        "issue_type": "solver_claim_status_mismatch",
                    }
                )
                continue
            stale_rows = conn.execute(
                f"""
                SELECT md.id
                FROM SolverRun sr
                JOIN ModelDiagnostic md
                  ON (
                    md.solver_run_id = sr.id
                    OR (
                      md.solver_run_id IS NULL
                      AND md.artifact_id = sr.artifact_id
                    )
                  )
                WHERE sr.id IN ({placeholders})
                  AND md.status = 'open'
                  AND md.severity = 'error'
                  AND md.diagnostic_type IN (
                    'linked_component_inactive',
                    'linked_component_changed',
                    'linked_component_missing'
                  )
                LIMIT 1
                """,
                solver_ids,
            ).fetchone()
            if stale_rows:
                issues.append(
                    {
                        "target_table": row["target_table"],
                        "target_id": row["id"],
                        "issue_type": "solver_claim_stale_evidence",
                    }
                )
    return issues


def _filter_new_issues(issues: list[dict], topic_id: int | None = None) -> list[dict]:
    if not issues:
        return []
    # Batch-fetch existing tracker rows (scoped to topic_id if available)
    tracker_map: dict[tuple, dict] = {}
    with get_db() as conn:
        if topic_id is not None:
            rows = conn.execute(
                "SELECT id, target_table, target_id, issue_type, topic_id, status, check_count, max_checks "
                "FROM VikiTracker WHERE topic_id = ?",
                (topic_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, target_table, target_id, issue_type, topic_id, status, check_count, max_checks "
                "FROM VikiTracker"
            ).fetchall()
        for r in rows:
            key = (r["target_table"], r["target_id"], r["issue_type"], r["topic_id"])
            tracker_map[key] = dict(r)
    filtered = []
    for issue in issues:
        issue["topic_id"] = topic_id
        key = (issue["target_table"], issue["target_id"], issue["issue_type"], topic_id)
        existing = tracker_map.get(key)
        if existing:
            if existing["status"] in ("resolved", "wont_fix"):
                continue
            if existing["check_count"] >= existing["max_checks"]:
                continue
            issue["tracker_id"] = existing["id"]
            issue["check_count"] = existing["check_count"]
        else:
            tracker = upsert_viki_issue(
                issue["target_table"],
                issue["target_id"],
                issue["issue_type"],
                topic_id=topic_id,
            )
            issue["tracker_id"] = tracker.get("id")
            issue["check_count"] = 0
        filtered.append(issue)
    return filtered


# ---------------------------------------------------------------------------
# VIKI: Batch processing
# ---------------------------------------------------------------------------


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# SQL-only handlers (no LLM, process directly)
_SQL_HANDLERS = {}
# LLM batch handlers (one call per batch)
_BATCH_HANDLERS = {}


def _sql_handler(issue_type):
    def decorator(fn):
        _SQL_HANDLERS[issue_type] = fn
        return fn

    return decorator


def _batch_handler(issue_type):
    def decorator(fn):
        _BATCH_HANDLERS[issue_type] = fn
        return fn

    return decorator


def _auto_wont_fix_maxed(topic_id: int | None = None) -> None:
    """Transition open issues that hit max_checks to wont_fix (atomic, single UPDATE)."""
    with get_db() as conn:
        if topic_id is not None:
            cursor = conn.execute(
                "UPDATE VikiTracker SET status = 'wont_fix', resolution = 'max_checks_reached', "
                "last_checked_at = CURRENT_TIMESTAMP "
                "WHERE status = 'open' AND check_count >= max_checks AND topic_id = ?",
                (topic_id,),
            )
        else:
            cursor = conn.execute(
                "UPDATE VikiTracker SET status = 'wont_fix', resolution = 'max_checks_reached', "
                "last_checked_at = CURRENT_TIMESTAMP "
                "WHERE status = 'open' AND check_count >= max_checks",
            )
        if cursor.rowcount:
            logger.info("[VIKI] Auto wont_fix %d maxed-out issues", cursor.rowcount)


async def process_issues_background(topic_id: int, issues: list[dict]) -> None:
    """Process watchdog issues in batches. Fire-and-forget safe."""
    try:
        from .minimax_client import is_daemon_channel

        is_daemon_channel.set(True)

        # Group by issue_type
        by_type: dict[str, list[dict]] = defaultdict(list)
        for issue in issues:
            by_type[issue["issue_type"]].append(issue)

        # SQL-only types: process immediately, no LLM
        for issue_type, handler in _SQL_HANDLERS.items():
            for issue in by_type.pop(issue_type, []):
                try:
                    resolution = handler(issue)
                    if resolution and issue.get("tracker_id"):
                        mark_viki_resolved(issue["tracker_id"], resolution)
                except Exception as exc:
                    logger.warning(
                        "[VIKI] SQL handler failed for %s/%s: %s",
                        issue_type,
                        issue.get("target_id"),
                        exc,
                    )
                    if issue.get("tracker_id"):
                        increment_viki_check(issue["tracker_id"])

        # LLM types: batch by type, max BATCH_SIZE per chunk
        for issue_type, batch in by_type.items():
            handler = _BATCH_HANDLERS.get(issue_type)
            if not handler:
                logger.warning(
                    "[VIKI] No batch handler for %s (%d issues)", issue_type, len(batch)
                )
                continue
            for chunk in _chunks(batch, BATCH_SIZE):
                try:
                    results = await handler(chunk)
                    for r in results:
                        tid = r.get("tracker_id")
                        if not tid:
                            continue
                        if r.get("resolution"):
                            mark_viki_resolved(tid, r["resolution"])
                            logger.info(
                                "[VIKI] Resolved %s/%s: %s",
                                issue_type,
                                r.get("target_id"),
                                r["resolution"],
                            )
                        else:
                            increment_viki_check(tid)
                except Exception as exc:
                    logger.warning(
                        "[VIKI] Batch handler failed for %s (%d issues): %s",
                        issue_type,
                        len(chunk),
                        exc,
                    )
                    # Increment check_count for failed batch so we don't retry forever
                    for issue in chunk:
                        if issue.get("tracker_id"):
                            increment_viki_check(issue["tracker_id"])

        # Cleanup: transition any issues that hit max_checks to wont_fix
        _auto_wont_fix_maxed(topic_id)

    except Exception as exc:
        logger.error("[VIKI] Background processing failed: %s", exc)


# ---------------------------------------------------------------------------
# SQL-only handlers (no LLM cost)
# ---------------------------------------------------------------------------


@_sql_handler("stale_pending")
def _handle_stale_pending(issue: dict) -> str | None:
    pending_id = issue["target_id"]
    with get_db() as conn:
        conn.execute("DELETE FROM LedgerPending WHERE id = ?", (pending_id,))
    return "deleted"


@_sql_handler("orphan_ref")
def _handle_orphan_ref(issue: dict) -> str | None:
    claim_id = issue["target_id"]
    with get_db() as conn:
        row = conn.execute(
            "SELECT support_fact_ids_json FROM Claim WHERE id = ?", (claim_id,)
        ).fetchone()
        if not row:
            return "claim_not_found"
        try:
            fids = json.loads(row["support_fact_ids_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            return "bad_json"
        try:
            int_fids = [int(f) for f in fids if f is not None]
        except (ValueError, TypeError):
            return "bad_fids"
        if not int_fids:
            return "no_orphans_found"
        placeholders = ",".join("?" * len(int_fids))
        active_rows = conn.execute(
            f"SELECT id FROM Fact WHERE id IN ({placeholders}) AND superseded_by IS NULL",
            int_fids,
        ).fetchall()
        active_ids = {r["id"] for r in active_rows}
        valid_fids = [f for f in int_fids if f in active_ids]
        if len(valid_fids) == len(int_fids):
            return "no_orphans_found"
        if not valid_fids:
            conn.execute(
                "UPDATE Claim SET status = 'unsupported' WHERE id = ?", (claim_id,)
            )
            return "claim_unsupported"
        conn.execute(
            "UPDATE Claim SET support_fact_ids_json = ? WHERE id = ?",
            (json.dumps(valid_fids), claim_id),
        )
    return f"cleaned_{len(int_fids) - len(valid_fids)}_refs"


@_sql_handler("fact_contradiction")
def _handle_fact_contradiction(issue: dict) -> str | None:
    fact_id = issue["target_id"]
    other_id = issue.get("other_fact_id")
    if not other_id:
        return "missing_other_id"
    topic_id = _get_topic_for_fact(fact_id)
    if not topic_id:
        return "fact_deleted"
    api.insert_knowledge_edge(
        topic_id,
        fact_id,
        "fact",
        other_id,
        "fact",
        "conflicts_with",
        created_by="viki",
    )
    return f"edge_F{fact_id}_F{other_id}"


@_sql_handler("ledger_anomaly")
def _handle_ledger_anomaly(issue: dict) -> str | None:
    ledger_id = issue["target_id"]
    with get_db() as conn:
        conn.execute(
            "UPDATE Ledger SET review_status = 'anomalous' WHERE id = ?", (ledger_id,)
        )
    return "flagged_anomalous"


@_sql_handler("bad_ce_summary")
def _handle_bad_ce_summary(issue: dict) -> str | None:
    """Regenerate summary for CodeEvidence with polluted hypothesis or bad summary."""
    from .code_sandbox import _build_summary, _clean_hypothesis

    ce_id = issue["target_id"]
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, hypothesis, success, iterations, execution_time_s, "
            "exit_code, stderr, stdout FROM CodeEvidence WHERE id = ?",
            (ce_id,),
        ).fetchone()
        if not row:
            return "ce_not_found"
        new_summary = _build_summary(
            hypothesis=row["hypothesis"],
            success=bool(row["success"]),
            iterations=row["iterations"] or 0,
            execution_time_s=row["execution_time_s"] or 0.0,
            phase=None,
            exit_code=row["exit_code"],
            stderr=row["stderr"] or "",
            stdout=row["stdout"] or "",
        )
        # Also clean the hypothesis in DB if it contains code
        clean_hyp = _clean_hypothesis(row["hypothesis"])
        if clean_hyp != row["hypothesis"]:
            conn.execute(
                "UPDATE CodeEvidence SET hypothesis = ?, summary = ? WHERE id = ?",
                (clean_hyp, new_summary, ce_id),
            )
            return "cleaned_hypothesis_and_summary"
        else:
            conn.execute(
                "UPDATE CodeEvidence SET summary = ? WHERE id = ?",
                (new_summary, ce_id),
            )
            return "regenerated_summary"


def _quarantine_solver_claim_issue(issue: dict, reason: str) -> str | None:
    target_table = issue.get("target_table")
    target_id = int(issue["target_id"])
    note = f"VIKI: {reason}"
    with get_db() as conn:
        if target_table == "Claim":
            row = conn.execute("SELECT id FROM Claim WHERE id = ?", (target_id,)).fetchone()
            if not row:
                return "claim_not_found"
            conn.execute(
                """
                UPDATE Claim
                SET status = 'contested'
                WHERE id = ?
                  AND status NOT IN ('retired', 'superseded')
                """,
                (target_id,),
            )
            return f"claim_contested_{reason}"
        if target_table == "ClaimCandidate":
            row = conn.execute(
                "SELECT id FROM ClaimCandidate WHERE id = ?", (target_id,)
            ).fetchone()
            if not row:
                return "candidate_not_found"
            conn.execute(
                """
                UPDATE ClaimCandidate
                SET status = 'rejected',
                    review_note = ?,
                    reviewed_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'pending'
                """,
                (note, target_id),
            )
            return f"candidate_rejected_{reason}"
    return "unsupported_target_table"


@_sql_handler("solver_claim_missing_evidence")
def _handle_solver_claim_missing_evidence(issue: dict) -> str | None:
    return _quarantine_solver_claim_issue(issue, "solver_claim_missing_evidence")


@_sql_handler("solver_claim_orphan_run")
def _handle_solver_claim_orphan_run(issue: dict) -> str | None:
    return _quarantine_solver_claim_issue(issue, "solver_claim_orphan_run")


@_sql_handler("solver_claim_status_mismatch")
def _handle_solver_claim_status_mismatch(issue: dict) -> str | None:
    return _quarantine_solver_claim_issue(issue, "solver_claim_status_mismatch")


@_sql_handler("solver_claim_stale_evidence")
def _handle_solver_claim_stale_evidence(issue: dict) -> str | None:
    return _quarantine_solver_claim_issue(issue, "solver_claim_stale_evidence")


# ---------------------------------------------------------------------------
# LLM batch handlers
# ---------------------------------------------------------------------------


@_batch_handler("missing_spo")
async def _handle_batch_missing_spo(issues: list[dict]) -> list[dict]:
    """Batch S/P/O extraction for multiple facts in one LLM call."""
    # Fetch all fact contents + source context in one query
    fact_data: dict[int, dict] = {}
    target_ids = [issue["target_id"] for issue in issues]
    if target_ids:
        with get_db() as conn:
            placeholders = ",".join("?" * len(target_ids))
            rows = conn.execute(
                f"SELECT id, content, source_excerpt, source_refs_json, source_kind "
                f"FROM Fact WHERE id IN ({placeholders})",
                target_ids,
            ).fetchall()
            for row in rows:
                fact_data[row["id"]] = {
                    "content": row["content"],
                    "source_excerpt": row["source_excerpt"],
                    "source_refs_json": row["source_refs_json"],
                    "source_kind": row["source_kind"],
                }

    if not fact_data:
        return [
            {
                "tracker_id": i.get("tracker_id"),
                "target_id": i["target_id"],
                "resolution": "fact_not_found",
            }
            for i in issues
        ]

    # Build batch prompt with source context
    fact_lines = []
    for idx, (fid, info) in enumerate(fact_data.items(), 1):
        line = f"{idx}. [F{fid}] {info['content']}"
        if info.get("source_excerpt"):
            line += f"\n   Source context ({info.get('source_kind', 'unknown')}): {info['source_excerpt']}"
        fact_lines.append(line)

    from .broker import DEFAULT_MAX_TOKENS, call_text
    from .json_utils import extract_json_any

    prompt = (
        "Extract structured subject/predicate/object for each fact below.\n\n"
        "Facts:\n" + "\n".join(fact_lines) + "\n\n"
        "Output a JSON array with one element per fact:\n"
        '[{"fact_id": <id>, "subject": "...", "predicate": "...", '
        '"object": {"type": "string|quantity", "value": "..."}, '
        '"qualifiers": [{"key": "...", "value": "..."}], '
        '"attribution": {"claimed_by": "...", "evidence_id": "..."}}, ...]\n'
        'If a fact is too vague: {"fact_id": <id>, "subject": null}'
    )
    result = await call_text(
        prompt,
        provider=_control_provider(_batch_topic_id(issues)),
        strategy="direct",
        temperature=0.2,
        max_tokens=DEFAULT_MAX_TOKENS,
        require_json=True,
        fallback_role="skynet",
    )

    parsed = extract_json_any(result) if result else None
    if not isinstance(parsed, list):
        parsed = []

    # Map results back — LLM may return "F283", "[F283]", or 283
    result_map = {}
    for item in parsed:
        if isinstance(item, dict) and item.get("fact_id") is not None:
            try:
                raw_fid = str(item["fact_id"]).strip()
                # Only strip F/f prefix (this handler deals with facts only)
                if raw_fid.startswith(("F", "f")):
                    raw_fid = raw_fid[1:]
                raw_fid = raw_fid.strip("[]# ")
                result_map[int(raw_fid)] = item
            except (ValueError, TypeError):
                continue

    results = []
    for issue in issues:
        fid = issue["target_id"]
        spo = result_map.get(fid)
        if spo and spo.get("subject"):
            obj = spo.get("object")
            quals = spo.get("qualifiers")
            attr = spo.get("attribution")
            api.update_fact_structured_columns(
                fid,
                subject=spo["subject"],
                predicate=spo.get("predicate"),
                object_json=json.dumps(obj, ensure_ascii=False) if obj else None,
                qualifiers_json=(
                    json.dumps(quals, ensure_ascii=False) if quals else None
                ),
                attribution_json=json.dumps(attr, ensure_ascii=False) if attr else None,
            )
            results.append(
                {
                    "tracker_id": issue.get("tracker_id"),
                    "target_id": fid,
                    "resolution": "backfilled",
                }
            )
        else:
            results.append(
                {
                    "tracker_id": issue.get("tracker_id"),
                    "target_id": fid,
                    "resolution": None,
                }
            )
    return results


@_batch_handler("claim_conflict")
async def _handle_batch_claim_conflicts(issues: list[dict]) -> list[dict]:
    """Batch claim conflict resolution: fast-track where possible, LLM for close calls."""
    results = []
    need_llm = []

    # Batch-fetch all claim IDs referenced by issues
    all_claim_ids: set[int] = set()
    for issue in issues:
        if issue.get("source_id"):
            all_claim_ids.add(issue["source_id"])
        if issue.get("target_claim_id"):
            all_claim_ids.add(issue["target_claim_id"])
    claim_map: dict[int, dict] = {}
    if all_claim_ids:
        with get_db() as conn:
            placeholders = ",".join("?" * len(all_claim_ids))
            rows = conn.execute(
                f"SELECT * FROM Claim WHERE id IN ({placeholders})",
                list(all_claim_ids),
            ).fetchall()
            for r in rows:
                claim_map[r["id"]] = dict(r)

    for issue in issues:
        source_id = issue.get("source_id")
        target_id = issue.get("target_claim_id")
        if not source_id or not target_id:
            results.append(
                {
                    "tracker_id": issue.get("tracker_id"),
                    "target_id": issue["target_id"],
                    "resolution": "missing_ids",
                }
            )
            continue

        ca = claim_map.get(source_id)
        cb = claim_map.get(target_id)

        if not ca or not cb:
            results.append(
                {
                    "tracker_id": issue.get("tracker_id"),
                    "target_id": issue["target_id"],
                    "resolution": "claim_missing",
                }
            )
            continue
        if ca["superseded_by"] is not None or cb["superseded_by"] is not None:
            results.append(
                {
                    "tracker_id": issue.get("tracker_id"),
                    "target_id": issue["target_id"],
                    "resolution": "already_resolved",
                }
            )
            continue

        a_score = ca["claim_score"] or 0
        b_score = cb["claim_score"] or 0
        a_facts = _count_active_support_facts(ca)
        b_facts = _count_active_support_facts(cb)

        # Fast-track: clear winner
        if a_score - b_score >= 2 and a_facts >= b_facts:
            api.update_claim_superseded(target_id, source_id)
            results.append(
                {
                    "tracker_id": issue.get("tracker_id"),
                    "target_id": issue["target_id"],
                    "resolution": f"superseded_C{target_id}_by_C{source_id}",
                }
            )
        elif b_score - a_score >= 2 and b_facts >= a_facts:
            api.update_claim_superseded(source_id, target_id)
            results.append(
                {
                    "tracker_id": issue.get("tracker_id"),
                    "target_id": issue["target_id"],
                    "resolution": f"superseded_C{source_id}_by_C{target_id}",
                }
            )
        else:
            need_llm.append((issue, ca, cb))

    # LLM batch for close calls
    if need_llm:
        lines = []
        for idx, (issue, ca, cb) in enumerate(need_llm, 1):
            line = (
                f"{idx}. C{ca['id']} (score={ca.get('claim_score')}) vs C{cb['id']} (score={cb.get('claim_score')}):\n"
                f"   A: {ca['content']}\n"
                f"   B: {cb['content']}"
            )
            # Add support facts context
            for label, claim in [("A", ca), ("B", cb)]:
                rationale = claim.get("rationale_short")
                if rationale:
                    line += f"\n   {label} rationale: {rationale}"
            lines.append(line)

        from .broker import DEFAULT_MAX_TOKENS, call_text
        from .json_utils import extract_json_any

        prompt = (
            "For each pair of conflicting claims, decide which should supersede the other.\n\n"
            + "\n\n".join(lines)
            + "\n\n"
            'Output JSON array: [{"pair": 1, "winner": "A" or "B", "reason": "1 sentence"}, ...]\n'
            'If can\'t decide: {"pair": 1, "winner": null}'
        )
        result = await call_text(
            prompt,
            provider=_control_provider(_batch_topic_id(issues)),
            strategy="direct",
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS,
            require_json=True,
            fallback_role="skynet",
        )
        parsed = extract_json_any(result) if result else None
        if not isinstance(parsed, list):
            parsed = []

        decision_map = {}
        for item in parsed:
            if isinstance(item, dict) and item.get("pair"):
                decision_map[item["pair"]] = item.get("winner")

        for idx, (issue, ca, cb) in enumerate(need_llm, 1):
            winner = (decision_map.get(idx) or "").upper()
            if winner == "A":
                api.update_claim_superseded(cb["id"], ca["id"])
                results.append(
                    {
                        "tracker_id": issue.get("tracker_id"),
                        "target_id": issue["target_id"],
                        "resolution": f"llm_C{cb['id']}_by_C{ca['id']}",
                    }
                )
            elif winner == "B":
                api.update_claim_superseded(ca["id"], cb["id"])
                results.append(
                    {
                        "tracker_id": issue.get("tracker_id"),
                        "target_id": issue["target_id"],
                        "resolution": f"llm_C{ca['id']}_by_C{cb['id']}",
                    }
                )
            else:
                results.append(
                    {
                        "tracker_id": issue.get("tracker_id"),
                        "target_id": issue["target_id"],
                        "resolution": None,
                    }
                )

    return results


@_batch_handler("ledger_missing_stat_type")
async def _handle_batch_ledger_stat_type(issues: list[dict]) -> list[dict]:
    """Rule-based first, then LLM batch for remainder."""
    results = []
    need_llm = []

    # Rule-based inference — batch-fetch all ledger rows in one query
    target_ids = [issue["target_id"] for issue in issues]
    row_map: dict[int, dict] = {}
    if target_ids:
        with get_db() as conn:
            placeholders = ",".join("?" * len(target_ids))
            rows = conn.execute(
                f"""SELECT l.id, l.value, l.value_numeric_min, l.value_numeric_max,
                          l.source_ref, a.canonical_name AS attr_name
                   FROM Ledger l
                   JOIN LedgerAttribute a ON l.attribute_id = a.id
                   WHERE l.id IN ({placeholders})""",
                target_ids,
            ).fetchall()
            for r in rows:
                row_map[r["id"]] = dict(r)
    with get_db() as conn:
        for issue in issues:
            row = row_map.get(issue["target_id"])
            if not row:
                results.append(
                    {
                        "tracker_id": issue.get("tracker_id"),
                        "target_id": issue["target_id"],
                        "resolution": "not_found",
                    }
                )
                continue

            inferred = _infer_stat_type_by_rule(row)
            if inferred:
                conn.execute(
                    "UPDATE Ledger SET value_stat_type = ? WHERE id = ?",
                    (inferred, row["id"]),
                )
                results.append(
                    {
                        "tracker_id": issue.get("tracker_id"),
                        "target_id": issue["target_id"],
                        "resolution": f"rule_{inferred}",
                    }
                )
            else:
                need_llm.append((issue, dict(row)))

    # LLM batch for ambiguous cases
    if need_llm:
        lines = []
        for idx, (issue, row) in enumerate(need_llm, 1):
            src = row.get("source_ref", "")
            lines.append(
                f"{idx}. L{row['id']}: {row['attr_name']} = {row['value']} (source: {src})"
            )

        from .broker import DEFAULT_MAX_TOKENS, call_text
        from .json_utils import extract_json_any

        prompt = (
            "Classify the statistical type of each data point.\n\n"
            + "\n".join(lines)
            + "\n\n"
            'Output JSON array: [{"item": 1, "stat_type": "point|mean_std|delta|ratio|percentage|p_value|ci|correlation|rank|se"}, ...]\n'
            'If unclear: {"item": 1, "stat_type": "point"}'
        )
        result = await call_text(
            prompt,
            provider=_control_provider(_batch_topic_id(issues)),
            strategy="direct",
            temperature=0.2,
            max_tokens=DEFAULT_MAX_TOKENS,
            require_json=True,
            fallback_role="skynet",
        )
        parsed = extract_json_any(result) if result else None
        if not isinstance(parsed, list):
            parsed = []

        type_map = {}
        for item in parsed:
            if isinstance(item, dict) and item.get("item"):
                type_map[item["item"]] = item.get("stat_type", "point")

        with get_db() as conn:
            for idx, (issue, row) in enumerate(need_llm, 1):
                st = type_map.get(idx, "point")
                if st not in _VALID_STAT_TYPES:
                    st = "point"
                conn.execute(
                    "UPDATE Ledger SET value_stat_type = ? WHERE id = ?",
                    (st, row["id"]),
                )
                results.append(
                    {
                        "tracker_id": issue.get("tracker_id"),
                        "target_id": issue["target_id"],
                        "resolution": f"llm_{st}",
                    }
                )

    return results


# ---------------------------------------------------------------------------
# Rule-based stat_type inference
# ---------------------------------------------------------------------------

_VALID_STAT_TYPES = {
    "point",
    "mean_std",
    "delta",
    "ratio",
    "percentage",
    "p_value",
    "ci",
    "correlation",
    "rank",
    "se",
}

_POINT_ATTRIBUTES = {
    "F1 Score",
    "Accuracy",
    "AUC-ROC",
    "R-squared",
    "Training Time",
    "Inference Time",
    "Dataset Size",
    "Feature Dimensionality",
    "Class Count",
    "Noise Level",
    "Learning Rate",
    "Network Depth",
    "Network Width",
    "Number of Trees",
    "Mean Squared Error",
    "Sample Count",
}


def _infer_stat_type_by_rule(row: dict) -> str | None:
    """Try to infer stat_type without LLM. Returns stat_type or None if ambiguous."""
    attr = row.get("attr_name", "")
    vmin = row.get("value_numeric_min")

    # Negative values on metrics that should be 0-1 → likely delta (check BEFORE point attributes)
    if attr in ("F1 Score", "Accuracy") and vmin is not None and vmin < 0:
        return "delta"

    # Known point-estimate attributes — always return 'point' regardless of numeric presence
    if attr in _POINT_ATTRIBUTES:
        return "point"

    # Attribute name hints (only for non-point attributes)
    attr_lower = attr.lower()
    if "delta" in attr_lower or "gap" in attr_lower or "difference" in attr_lower:
        return "delta"
    if "ratio" in attr_lower:
        return "ratio"
    # Note: "rate" check removed — "Learning Rate" is in _POINT_ATTRIBUTES and returns above

    return None  # ambiguous, need LLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_active_support_facts(claim: dict) -> int:
    try:
        fids = json.loads(claim.get("support_fact_ids_json") or "[]")
        int_fids = [int(f) for f in fids if f is not None]
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0
    if not int_fids:
        return 0
    with get_db() as conn:
        placeholders = ",".join("?" * len(int_fids))
        row = conn.execute(
            f"SELECT COUNT(*) FROM Fact WHERE id IN ({placeholders}) AND superseded_by IS NULL",
            int_fids,
        ).fetchone()
        return row[0] if row else 0


def _get_topic_for_fact(fact_id: int) -> int | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT topic_id FROM Fact WHERE id = ?", (fact_id,)
        ).fetchone()
        return row["topic_id"] if row else None
