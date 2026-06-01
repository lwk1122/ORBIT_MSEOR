"""JTMS (Justification-based Truth Maintenance System) sweep.

Recomputes node states across the 4-layer knowledge graph:
  L0 (WebEvidence/CodeEvidence) → L1 (Ledger) → L2 (Fact) → L3 (Claim)

Called after librarian batches and at round boundaries.
"""

import logging
from collections import defaultdict

from . import db
from .evidence_parser import score_domain

logger = logging.getLogger(__name__)

CONTESTED_ROUNDS_TO_RETIRE = 3


def _is_node_active(conn, node_id: int, node_type: str) -> bool:
    """Check whether a knowledge node is in an active (usable) state.

    Uses the provided connection to see uncommitted changes from the same transaction.
    """
    if node_type == "fact":
        row = conn.execute(
            "SELECT review_status FROM Fact WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return False
        status = row["review_status"]
        return status not in ("retired", "superseded", "contested") if status else True
    elif node_type == "ledger":
        row = conn.execute(
            "SELECT review_status FROM Ledger WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return False
        status = row["review_status"]
        return status not in ("retired", "superseded", "contested") if status else True
    elif node_type == "code_evidence":
        row = conn.execute(
            "SELECT success FROM CodeEvidence WHERE id = ?", (node_id,)
        ).fetchone()
        if not row:
            return False
        if not bool(row["success"]):
            return False
        stale_solver_row = conn.execute(
            """
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
            WHERE sr.code_evidence_id = ?
              AND md.status = 'open'
              AND md.severity = 'error'
              AND md.diagnostic_type IN (
                'linked_component_inactive',
                'linked_component_changed',
                'linked_component_missing'
              )
            LIMIT 1
            """,
            (node_id,),
        ).fetchone()
        return stale_solver_row is None
    elif node_type == "web_evidence":
        row = conn.execute(
            "SELECT id FROM WebEvidence WHERE id = ?", (node_id,)
        ).fetchone()
        return row is not None
    elif node_type == "tool_trace":
        row = conn.execute(
            "SELECT id FROM ToolTrace WHERE id = ?", (node_id,)
        ).fetchone()
        return row is not None
    return True


def _get_domain_score_for_node(conn, node_id: int, node_type: str) -> float:
    """Get the domain score for a node's L0 source.

    Uses the provided connection for transactional consistency.
    """
    if node_type == "ledger":
        row = conn.execute(
            "SELECT domain_score FROM Ledger WHERE id = ?", (node_id,)
        ).fetchone()
        return (
            float(row["domain_score"])
            if row and row["domain_score"] is not None
            else 0.5
        )
    elif node_type == "fact":
        # For facts, look up derived_from edges to web_evidence, take max domain score
        rows = conn.execute(
            """SELECT we.source_domain FROM KnowledgeEdge ke
               JOIN WebEvidence we ON we.id = ke.source_id
               WHERE ke.target_id = ? AND ke.target_type = 'fact'
                 AND ke.source_type = 'web_evidence'
                 AND ke.relation = 'derived_from'
                 AND ke.is_active = 1""",
            (node_id,),
        ).fetchall()
        if not rows:
            return 0.5
        return max(score_domain(r["source_domain"] or "") for r in rows)
    return 0.5


def _any_valid_justification_group(conn, support_edges) -> bool:
    """Check if any justification group has all sources active (OR-of-AND)."""
    groups: dict[str, list] = defaultdict(list)
    for e in support_edges:
        groups[e["justification_group"]].append(e)
    for group_edges in groups.values():
        if all(
            _is_node_active(conn, e["source_id"], e["source_type"]) for e in group_edges
        ):
            return True
    return False


def _set_ledger_status(
    conn, ledger_id: int, status: str, current_round: int | None = None
) -> None:
    """Update ledger review_status and optionally set contested_since_round."""
    if current_round is not None:
        conn.execute(
            "UPDATE Ledger SET review_status = ?, contested_since_round = ? WHERE id = ? AND (review_status IS NULL OR review_status != ?)",
            (status, current_round, ledger_id, status),
        )
    else:
        conn.execute(
            "UPDATE Ledger SET review_status = ? WHERE id = ? AND (review_status IS NULL OR review_status != ?)",
            (status, ledger_id, status),
        )


def _set_fact_status(
    conn, fact_id: int, status: str, current_round: int | None = None
) -> None:
    """Update fact review_status and optionally set contested_since_round."""
    if current_round is not None:
        conn.execute(
            "UPDATE Fact SET review_status = ?, contested_since_round = ? WHERE id = ? AND (review_status IS NULL OR review_status != ?)",
            (status, current_round, fact_id, status),
        )
    else:
        conn.execute(
            "UPDATE Fact SET review_status = ? WHERE id = ? AND (review_status IS NULL OR review_status != ?)",
            (status, fact_id, status),
        )


def _set_claim_status(
    conn, claim_id: int, status: str, current_round: int | None = None
) -> None:
    """Update claim status and optionally set contested_since_round."""
    if current_round is not None:
        conn.execute(
            "UPDATE Claim SET status = ?, contested_since_round = ? WHERE id = ? AND status != ?",
            (status, current_round, claim_id, status),
        )
    else:
        conn.execute(
            "UPDATE Claim SET status = ? WHERE id = ? AND status != ?",
            (status, claim_id, status),
        )


def jtms_sweep(topic_id: int, current_round: int) -> list[dict]:
    """Recompute all node states based on edge validity.

    Sweep order:
      Phase 1a/1b: Resolve L1/L2 conflicts (contest losers by domain_score)
      Phase 1c/1d: Recover L1/L2 nodes whose conflicts resolved
      Phase 2: Retire stale contested nodes (>= CONTESTED_ROUNDS_TO_RETIRE rounds)
      Phase 3: Cascade to L3 Claims (contest claims with no valid justification groups)
      Phase 4: Recover contested Claims that regained valid support
    Returns list of state changes for logging.
    """
    changes: list[dict] = []

    with db.get_db() as conn:
        # Track processed conflict pairs to avoid double-processing (A→B and B→A)
        processed_conflict_pairs: set[tuple[str, int, int]] = set()

        # ---- Phase 1a: Resolve Ledger conflicts ----
        ledger_conflicts = conn.execute(
            """SELECT * FROM KnowledgeEdge
               WHERE topic_id = ? AND relation = 'conflicts_with'
                 AND source_type = 'ledger' AND target_type = 'ledger'
                 AND is_active = 1""",
            (topic_id,),
        ).fetchall()

        for edge in ledger_conflicts:
            src_id, tgt_id = edge["source_id"], edge["target_id"]
            pair_key = ("ledger", min(src_id, tgt_id), max(src_id, tgt_id))
            if pair_key in processed_conflict_pairs:
                continue
            processed_conflict_pairs.add(pair_key)

            src_row = conn.execute(
                "SELECT domain_score, review_status FROM Ledger WHERE id = ?", (src_id,)
            ).fetchone()
            tgt_row = conn.execute(
                "SELECT domain_score, review_status FROM Ledger WHERE id = ?", (tgt_id,)
            ).fetchone()
            if not src_row or not tgt_row:
                continue
            # Skip if either is already retired or superseded
            if (src_row["review_status"] or "") in ("retired", "superseded"):
                continue
            if (tgt_row["review_status"] or "") in ("retired", "superseded"):
                continue

            src_score = float(src_row["domain_score"] or 0.5)
            tgt_score = float(tgt_row["domain_score"] or 0.5)
            diff = src_score - tgt_score

            if diff > 0.1:
                # target (lower score) becomes contested
                _set_ledger_status(conn, tgt_id, "contested", current_round)
                changes.append(
                    {
                        "type": "ledger",
                        "id": tgt_id,
                        "new_status": "contested",
                        "reason": "conflict_lower_score",
                    }
                )
            elif diff < -0.1:
                # source (lower score) becomes contested
                _set_ledger_status(conn, src_id, "contested", current_round)
                changes.append(
                    {
                        "type": "ledger",
                        "id": src_id,
                        "new_status": "contested",
                        "reason": "conflict_lower_score",
                    }
                )
            # else: tie (diff within 0.1), both stay as-is

        # ---- Phase 1b: Resolve Fact conflicts ----
        fact_conflicts = conn.execute(
            """SELECT * FROM KnowledgeEdge
               WHERE topic_id = ? AND relation = 'conflicts_with'
                 AND source_type = 'fact' AND target_type = 'fact'
                 AND is_active = 1""",
            (topic_id,),
        ).fetchall()

        for edge in fact_conflicts:
            src_id, tgt_id = edge["source_id"], edge["target_id"]
            pair_key = ("fact", min(src_id, tgt_id), max(src_id, tgt_id))
            if pair_key in processed_conflict_pairs:
                continue
            processed_conflict_pairs.add(pair_key)

            src_row = conn.execute(
                "SELECT review_status FROM Fact WHERE id = ?", (src_id,)
            ).fetchone()
            tgt_row = conn.execute(
                "SELECT review_status FROM Fact WHERE id = ?", (tgt_id,)
            ).fetchone()
            if not src_row or not tgt_row:
                continue
            if (src_row["review_status"] or "") in ("retired", "superseded"):
                continue
            if (tgt_row["review_status"] or "") in ("retired", "superseded"):
                continue

            src_score = _get_domain_score_for_node(conn, src_id, "fact")
            tgt_score = _get_domain_score_for_node(conn, tgt_id, "fact")
            diff = src_score - tgt_score

            if diff > 0.1:
                _set_fact_status(conn, tgt_id, "contested", current_round)
                changes.append(
                    {
                        "type": "fact",
                        "id": tgt_id,
                        "new_status": "contested",
                        "reason": "conflict_lower_score",
                    }
                )
            elif diff < -0.1:
                _set_fact_status(conn, src_id, "contested", current_round)
                changes.append(
                    {
                        "type": "fact",
                        "id": src_id,
                        "new_status": "contested",
                        "reason": "conflict_lower_score",
                    }
                )

        # Check facts whose supporting Ledger entries are all contested/retired
        active_facts = conn.execute(
            """SELECT id, review_status FROM Fact
               WHERE topic_id = ? AND (review_status IS NULL OR review_status NOT IN ('retired', 'superseded'))""",
            (topic_id,),
        ).fetchall()

        for fact in active_facts:
            fact_id = fact["id"]
            # Get all support edges pointing to this fact
            support_edges = conn.execute(
                """SELECT source_id, source_type FROM KnowledgeEdge
                   WHERE topic_id = ? AND target_id = ? AND target_type = 'fact'
                     AND relation IN ('supports', 'derived_from')
                     AND is_active = 1""",
                (topic_id, fact_id),
            ).fetchall()

            if not support_edges:
                continue  # No tracked support — don't contest (may have untracked provenance)

            any_active_support = any(
                _is_node_active(conn, e["source_id"], e["source_type"])
                for e in support_edges
            )
            if not any_active_support and fact["review_status"] != "contested":
                _set_fact_status(conn, fact_id, "contested", current_round)
                changes.append(
                    {
                        "type": "fact",
                        "id": fact_id,
                        "new_status": "contested",
                        "reason": "all_support_inactive",
                    }
                )

        # ---- Phase 1c: Recover contested Ledger entries ----
        # Sort by domain_score descending so higher-scored entries recover first
        contested_ledgers = conn.execute(
            """SELECT id, domain_score FROM Ledger
               WHERE topic_id = ? AND review_status = 'contested'
               ORDER BY COALESCE(domain_score, 0.5) DESC""",
            (topic_id,),
        ).fetchall()
        recovered_ledger_ids: set[int] = set()
        for led in contested_ledgers:
            led_id = led["id"]
            if led_id in recovered_ledger_ids:
                continue
            my_score = float(led["domain_score"] or 0.5)
            # Check if any conflict counterpart is active AND has higher score
            conflict_edges = conn.execute(
                """SELECT source_id, target_id FROM KnowledgeEdge
                   WHERE topic_id = ? AND relation = 'conflicts_with'
                     AND source_type = 'ledger' AND target_type = 'ledger'
                     AND (source_id = ? OR target_id = ?) AND is_active = 1""",
                (topic_id, led_id, led_id),
            ).fetchall()
            blocked = False
            for ce in conflict_edges:
                other_id = (
                    ce["target_id"] if ce["source_id"] == led_id else ce["source_id"]
                )
                if other_id in recovered_ledger_ids:
                    # Other side already recovered in this sweep — we're the loser
                    blocked = True
                    break
                other_row = conn.execute(
                    "SELECT review_status, domain_score FROM Ledger WHERE id = ?",
                    (other_id,),
                ).fetchone()
                if not other_row:
                    continue
                other_status = other_row["review_status"] or ""
                if other_status in ("retired", "superseded"):
                    continue  # inactive, doesn't block
                # Active or contested counterpart with higher score blocks recovery
                other_score = float(other_row["domain_score"] or 0.5)
                if other_score - my_score > 0.1:
                    blocked = True
                    break
            if not blocked:
                conn.execute(
                    "UPDATE Ledger SET review_status = NULL, contested_since_round = NULL WHERE id = ?",
                    (led_id,),
                )
                recovered_ledger_ids.add(led_id)
                changes.append(
                    {
                        "type": "ledger",
                        "id": led_id,
                        "new_status": "active",
                        "reason": "recovered",
                    }
                )

        # ---- Phase 1d: Recover contested Facts ----
        contested_facts_raw = conn.execute(
            """SELECT id, review_status FROM Fact
               WHERE topic_id = ? AND review_status = 'contested'""",
            (topic_id,),
        ).fetchall()
        # Sort by domain_score descending so higher-scored facts recover first
        contested_facts = sorted(
            contested_facts_raw,
            key=lambda f: _get_domain_score_for_node(conn, f["id"], "fact"),
            reverse=True,
        )
        recovered_fact_ids: set[int] = set()
        for fact in contested_facts:
            fact_id = fact["id"]
            if fact_id in recovered_fact_ids:
                continue
            support_edges = conn.execute(
                """SELECT source_id, source_type FROM KnowledgeEdge
                   WHERE topic_id = ? AND target_id = ? AND target_type = 'fact'
                     AND relation IN ('supports', 'derived_from')
                     AND is_active = 1""",
                (topic_id, fact_id),
            ).fetchall()
            if support_edges and any(
                _is_node_active(conn, e["source_id"], e["source_type"])
                for e in support_edges
            ):
                # Check no active/recovered conflict winner with higher score
                conflict_edges = conn.execute(
                    """SELECT source_id, target_id FROM KnowledgeEdge
                       WHERE topic_id = ? AND relation = 'conflicts_with'
                         AND source_type = 'fact' AND target_type = 'fact'
                         AND (source_id = ? OR target_id = ?) AND is_active = 1""",
                    (topic_id, fact_id, fact_id),
                ).fetchall()
                blocked = False
                my_score = _get_domain_score_for_node(conn, fact_id, "fact")
                for ce in conflict_edges:
                    other_id = (
                        ce["target_id"]
                        if ce["source_id"] == fact_id
                        else ce["source_id"]
                    )
                    # If other side already recovered this sweep, we're the loser
                    if other_id in recovered_fact_ids:
                        blocked = True
                        break
                    other_row = conn.execute(
                        "SELECT review_status FROM Fact WHERE id = ?", (other_id,)
                    ).fetchone()
                    if not other_row:
                        continue
                    other_status = other_row["review_status"] or ""
                    if other_status in ("retired", "superseded", "contested"):
                        continue  # inactive or also contested, doesn't block
                    other_score = _get_domain_score_for_node(conn, other_id, "fact")
                    if other_score - my_score > 0.1:
                        blocked = True
                        break
                if not blocked:
                    conn.execute(
                        "UPDATE Fact SET review_status = NULL, contested_since_round = NULL WHERE id = ?",
                        (fact_id,),
                    )
                    recovered_fact_ids.add(fact_id)
                    changes.append(
                        {
                            "type": "fact",
                            "id": fact_id,
                            "new_status": "active",
                            "reason": "recovered",
                        }
                    )

        # ---- Phase 2: Retire stale contested nodes ----
        # Run before claim cascade so newly retired facts are visible to Phase 3.
        # Ledger entries
        stale_ledgers = conn.execute(
            """SELECT id FROM Ledger
               WHERE topic_id = ? AND review_status = 'contested'
                 AND contested_since_round IS NOT NULL
                 AND (? - contested_since_round) >= ?""",
            (topic_id, current_round, CONTESTED_ROUNDS_TO_RETIRE),
        ).fetchall()
        for led in stale_ledgers:
            conn.execute(
                "UPDATE Ledger SET review_status = 'retired', contested_since_round = NULL WHERE id = ?",
                (led["id"],),
            )
            changes.append(
                {
                    "type": "ledger",
                    "id": led["id"],
                    "new_status": "retired",
                    "reason": "stale_contested",
                }
            )

        # Facts
        stale_facts = conn.execute(
            """SELECT id FROM Fact
               WHERE topic_id = ? AND review_status = 'contested'
                 AND contested_since_round IS NOT NULL
                 AND (? - contested_since_round) >= ?""",
            (topic_id, current_round, CONTESTED_ROUNDS_TO_RETIRE),
        ).fetchall()
        for fact in stale_facts:
            conn.execute(
                "UPDATE Fact SET review_status = 'retired', contested_since_round = NULL WHERE id = ?",
                (fact["id"],),
            )
            changes.append(
                {
                    "type": "fact",
                    "id": fact["id"],
                    "new_status": "retired",
                    "reason": "stale_contested",
                }
            )

        # Claims
        stale_claims = conn.execute(
            """SELECT id FROM Claim
               WHERE topic_id = ? AND status = 'contested'
                 AND contested_since_round IS NOT NULL
                 AND (? - contested_since_round) >= ?""",
            (topic_id, current_round, CONTESTED_ROUNDS_TO_RETIRE),
        ).fetchall()
        for claim in stale_claims:
            conn.execute(
                "UPDATE Claim SET status = 'retired', contested_since_round = NULL WHERE id = ?",
                (claim["id"],),
            )
            changes.append(
                {
                    "type": "claim",
                    "id": claim["id"],
                    "new_status": "retired",
                    "reason": "stale_contested",
                }
            )

        # ---- Phase 3: Cascade to Claims ----
        active_claims = conn.execute(
            """SELECT id, status, contested_since_round FROM Claim
               WHERE topic_id = ? AND status NOT IN ('retired', 'superseded')
                 AND superseded_by IS NULL""",
            (topic_id,),
        ).fetchall()

        for claim in active_claims:
            claim_id = claim["id"]
            # Get justification groups
            support_edges = conn.execute(
                """SELECT * FROM KnowledgeEdge
                   WHERE topic_id = ? AND target_id = ? AND target_type = 'claim'
                     AND relation = 'supports'
                     AND is_active = 1""",
                (topic_id, claim_id),
            ).fetchall()

            if not support_edges:
                continue  # No tracked justification — don't contest

            any_valid = _any_valid_justification_group(conn, support_edges)

            if not any_valid and (claim["status"] or "") not in (
                "contested",
                "retired",
                "superseded",
            ):
                _set_claim_status(conn, claim_id, "contested", current_round)
                changes.append(
                    {
                        "type": "claim",
                        "id": claim_id,
                        "new_status": "contested",
                        "reason": "all_groups_failed",
                    }
                )

        # ---- Phase 4: Recover contested claims ----
        contested_claims = conn.execute(
            """SELECT id, status, contested_since_round FROM Claim
               WHERE topic_id = ? AND status = 'contested'
                 AND superseded_by IS NULL""",
            (topic_id,),
        ).fetchall()

        for claim in contested_claims:
            claim_id = claim["id"]
            support_edges = conn.execute(
                """SELECT * FROM KnowledgeEdge
                   WHERE topic_id = ? AND target_id = ? AND target_type = 'claim'
                     AND relation = 'supports'
                     AND is_active = 1""",
                (topic_id, claim_id),
            ).fetchall()

            if not support_edges:
                continue

            any_valid = _any_valid_justification_group(conn, support_edges)

            if any_valid:
                conn.execute(
                    "UPDATE Claim SET status = 'active', contested_since_round = NULL WHERE id = ?",
                    (claim_id,),
                )
                changes.append(
                    {
                        "type": "claim",
                        "id": claim_id,
                        "new_status": "active",
                        "reason": "recovered",
                    }
                )

    return changes
