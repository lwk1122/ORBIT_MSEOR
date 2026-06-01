import contextlib
import hashlib
import json
import logging
import os
import re
import sqlite3
import struct
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

MAX_CONTENT_LEN = 32000
MAX_SUMMARY_LEN = 500
MAX_CHUNK_TEXT_LEN = 20000


def _truncate(text: str | None, max_len: int) -> str | None:
    if text is None:
        return None
    return text[:max_len] if len(text) > max_len else text


def get_db_path() -> str:
    env_path = os.environ.get("ORBIT_DB_PATH")
    if env_path:
        return env_path
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    if os.environ.get("TESTING") == "1":
        return os.path.join(base_dir, "test_orbit.db")
    return os.path.join(base_dir, "orbit.db")


_sqlite_vec_available = True  # set to False if extension fails to load


@contextlib.contextmanager
def get_db():
    global _sqlite_vec_available
    conn = sqlite3.connect(get_db_path(), timeout=10.0)

    # Load sqlite-vec
    if _sqlite_vec_available:
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as e:
            _sqlite_vec_available = False
            logger.warning(
                "sqlite-vec extension unavailable — semantic search disabled: %s", e
            )

    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def serialize_f32(vector: List[float]) -> bytes:
    """Serializes a list of floats into a format sqlite-vec expects (blob of f32s)."""
    return struct.pack(f"{len(vector)}f", *vector)


_VALID_TABLES = frozenset(
    {
        "Topic",
        "Plan",
        "Subtopic",
        "Message",
        "FactCandidate",
        "Fact",
        "ClaimCandidate",
        "Claim",
        "WebEvidence",
        "VoteRecord",
        "LedgerEntity",
        "LedgerEntityAlias",
        "LedgerAttribute",
        "LedgerAttributeAlias",
        "Ledger",
        "LedgerPending",
        "LedgerEdge",
        "CodeEvidence",
        "ApiEvidence",
        "ToolTrace",
        "KnowledgeEdge",
        "TopicConfig",
        "UserInjection",
        "VikiTracker",
        "CorpusDocument",
        "CorpusChunk",
        "CorpusIngestRun",
        "OptimizationProblem",
        "OptimizationComponent",
        "OptimizationModelIR",
        "OptimizationArtifact",
        "SolverRun",
        "ModelDiagnostic",
        "ModelingExperience",
    }
)
_VALID_COLUMN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if table_name not in _VALID_TABLES:
        raise ValueError(f"Invalid table name: {table_name!r}")
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_column(
    conn: sqlite3.Connection, table_name: str, column_name: str, column_def: str
) -> None:
    if table_name not in _VALID_TABLES:
        raise ValueError(f"Invalid table name: {table_name!r}")
    if not _VALID_COLUMN_RE.match(column_name):
        raise ValueError(f"Invalid column name: {column_name!r}")
    if column_name in _table_columns(conn, table_name):
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")


def _insert_fact_fts(
    conn: sqlite3.Connection, fact_id: int, topic_id: int, content: str, source: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO facts_fts(rowid, content, topic_id, source) VALUES (?, ?, ?, ?)",
        (fact_id, content, str(topic_id), source),
    )


def _insert_message_fts(
    conn: sqlite3.Connection,
    msg_id: int,
    topic_id: int,
    sender: str,
    content: str,
    msg_type: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO messages_fts(rowid, content, topic_id, msg_type, sender) VALUES (?, ?, ?, ?, ?)",
        (msg_id, content, str(topic_id), msg_type, sender),
    )


def _insert_claim_fts(
    conn: sqlite3.Connection, claim_id: int, topic_id: int, content: str
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO claims_fts(rowid, content, topic_id) VALUES (?, ?, ?)",
        (claim_id, content, str(topic_id)),
    )


def _insert_web_evidence_fts(
    conn: sqlite3.Connection,
    web_id: int,
    origin_topic_id: int,
    source_domain: str,
    content: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO web_evidence_fts(rowid, content, origin_topic_id, source_domain) VALUES (?, ?, ?, ?)",
        (web_id, content, str(origin_topic_id), source_domain),
    )


def _insert_corpus_chunk_fts(
    conn: sqlite3.Connection,
    chunk_id: int,
    document_id: int,
    topic_id: int | None,
    doc_type: str | None,
    access_scope: str | None,
    content: str,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO corpus_chunks_fts(rowid, content, document_id, topic_id, doc_type, access_scope) VALUES (?, ?, ?, ?, ?, ?)",
        (
            chunk_id,
            content,
            str(document_id),
            str(topic_id) if topic_id is not None else "",
            doc_type or "",
            access_scope or "",
        ),
    )


def _backfill_fts(conn: sqlite3.Connection) -> None:
    fact_count = conn.execute("SELECT COUNT(*) FROM Fact").fetchone()[0]
    if fact_count:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO facts_fts(rowid, content, topic_id, source)
                SELECT
                    Fact.id,
                    CASE
                        WHEN Fact.summary IS NOT NULL AND Fact.summary != '' THEN Fact.summary || char(10) || char(10) || Fact.content
                        ELSE Fact.content
                    END,
                    CAST(Fact.topic_id AS TEXT), Fact.source
                FROM Fact
                """
            )
        except sqlite3.OperationalError as exc:
            logger.warning(
                "facts_fts backfill failed; attempting FTS rebuild fallback: %s", exc
            )
            try:
                conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError as rebuild_exc:
                logger.warning(
                    "facts_fts rebuild fallback also failed; continuing without legacy backfill: %s",
                    rebuild_exc,
                )

    message_count = conn.execute("SELECT COUNT(*) FROM Message").fetchone()[0]
    if not message_count:
        return

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO messages_fts(rowid, content, topic_id, msg_type, sender)
            SELECT 
                Message.id, 
                CASE 
                    WHEN Message.summary IS NOT NULL AND Message.summary != '' THEN Message.summary || char(10) || char(10) || Message.content 
                    ELSE Message.content 
                END, 
                CAST(Message.topic_id AS TEXT), Message.msg_type, Message.sender
            FROM Message
            """
        )
    except sqlite3.OperationalError as exc:
        logger.warning(
            "messages_fts backfill failed; attempting FTS rebuild fallback: %s", exc
        )
        try:
            conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")
        except sqlite3.OperationalError as rebuild_exc:
            logger.warning(
                "messages_fts rebuild fallback also failed; continuing without legacy backfill: %s",
                rebuild_exc,
            )

    claim_count = conn.execute("SELECT COUNT(*) FROM Claim").fetchone()[0]
    if claim_count:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO claims_fts(rowid, content, topic_id)
                SELECT Claim.id,
                    CASE
                        WHEN Claim.summary IS NOT NULL AND Claim.summary != '' THEN Claim.summary || char(10) || char(10) || Claim.content
                        ELSE Claim.content
                    END,
                    CAST(Claim.topic_id AS TEXT)
                FROM Claim
                WHERE Claim.superseded_by IS NULL
                """
            )
        except sqlite3.OperationalError as exc:
            logger.warning(
                "claims_fts backfill failed; attempting FTS rebuild fallback: %s", exc
            )
            try:
                conn.execute("INSERT INTO claims_fts(claims_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError as rebuild_exc:
                logger.warning(
                    "claims_fts rebuild fallback also failed; continuing without legacy backfill: %s",
                    rebuild_exc,
                )

    web_count = conn.execute("SELECT COUNT(*) FROM WebEvidence").fetchone()[0]
    if web_count:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO web_evidence_fts(rowid, content, origin_topic_id, source_domain)
                SELECT
                    WebEvidence.id,
                    TRIM(
                        COALESCE(WebEvidence.title, '') || char(10) || char(10) ||
                        COALESCE(WebEvidence.summary, '') || char(10) || char(10) ||
                        COALESCE(WebEvidence.snippet, '') || char(10) || char(10) ||
                        COALESCE(WebEvidence.query_text, '') || char(10) || char(10) ||
                        COALESCE(WebEvidence.source_domain, '')
                    ),
                    CAST(WebEvidence.origin_topic_id AS TEXT),
                    COALESCE(WebEvidence.source_domain, '')
                FROM WebEvidence
                """
            )
        except sqlite3.OperationalError as exc:
            logger.warning(
                "web_evidence_fts backfill failed; attempting FTS rebuild fallback: %s",
                exc,
            )
            try:
                conn.execute(
                    "INSERT INTO web_evidence_fts(web_evidence_fts) VALUES ('rebuild')"
                )
            except sqlite3.OperationalError as rebuild_exc:
                logger.warning(
                    "web_evidence_fts rebuild fallback also failed; continuing without legacy backfill: %s",
                    rebuild_exc,
                )

    try:
        corpus_count = conn.execute("SELECT COUNT(*) FROM CorpusChunk").fetchone()[0]
    except sqlite3.OperationalError:
        corpus_count = 0
    if corpus_count:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO corpus_chunks_fts(
                    rowid, content, document_id, topic_id, doc_type, access_scope
                )
                SELECT
                    CorpusChunk.id,
                    TRIM(
                        COALESCE(CorpusDocument.title, '') || char(10) ||
                        COALESCE(CorpusChunk.section_path, '') || char(10) ||
                        COALESCE(CorpusChunk.lexical_text, CorpusChunk.text, '') || char(10) ||
                        COALESCE(CorpusChunk.table_markdown, '')
                    ),
                    CAST(CorpusChunk.document_id AS TEXT),
                    COALESCE(CAST(CorpusDocument.topic_id AS TEXT), ''),
                    COALESCE(CorpusDocument.doc_type, ''),
                    COALESCE(CorpusDocument.access_scope, '')
                FROM CorpusChunk
                JOIN CorpusDocument ON CorpusDocument.id = CorpusChunk.document_id
                """
            )
        except sqlite3.OperationalError as exc:
            logger.warning(
                "corpus_chunks_fts backfill failed; attempting FTS rebuild fallback: %s",
                exc,
            )
            try:
                conn.execute("INSERT INTO corpus_chunks_fts(corpus_chunks_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError as rebuild_exc:
                logger.warning(
                    "corpus_chunks_fts rebuild fallback also failed; continuing without corpus backfill: %s",
                    rebuild_exc,
                )


_FTS5_RESERVED = frozenset({"AND", "OR", "NOT", "NEAR"})


def _build_fts_query(query_text: str) -> Optional[str]:
    tokens = re.findall(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+", query_text or "")
    if not tokens:
        return None
    safe_tokens = [
        f'"{token.replace(chr(34), chr(34)+chr(34))}"'
        for token in tokens
        if token.upper() not in _FTS5_RESERVED
    ]
    if not safe_tokens:
        return None
    return " OR ".join(safe_tokens)


def _migrate_knowledge_edge_check(conn: sqlite3.Connection) -> None:
    """Migrate KnowledgeEdge table to add 'refutes' and 'qualifies' to CHECK constraint.

    SQLite can't ALTER CHECK constraints, so we rebuild the table.
    Only runs if the new relations aren't already allowed.
    """
    # Check if CHECK constraint already allows 'refutes' by inspecting schema DDL
    schema_row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='KnowledgeEdge'"
    ).fetchone()
    if schema_row and "'refutes'" in (schema_row[0] or ""):
        return  # Already migrated
    if not schema_row:
        return  # Table doesn't exist yet (will be created by init_db)

    # Fallback: try a SAVEPOINT-protected probe insert (no orphan risk)
    try:
        conn.execute("SAVEPOINT ke_probe")
        conn.execute(
            """INSERT INTO KnowledgeEdge (topic_id, source_id, source_type, target_id, target_type, relation)
            VALUES (0, 0, 'fact', 0, 'fact', 'refutes')"""
        )
        # Probe succeeded — CHECK allows 'refutes'. Rollback the probe row.
        conn.execute("ROLLBACK TO ke_probe")
        conn.execute("RELEASE ke_probe")
        return
    except sqlite3.IntegrityError:
        # CHECK blocks 'refutes' — need migration. Release the savepoint.
        try:
            conn.execute("ROLLBACK TO ke_probe")
            conn.execute("RELEASE ke_probe")
        except Exception:
            pass

    logger.info(
        "[db] Migrating KnowledgeEdge table to add refutes/qualifies to CHECK constraint..."
    )

    # Must use autocommit for PRAGMA to work in Python sqlite3
    old_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("PRAGMA foreign_keys=OFF;")
        conn.execute("BEGIN TRANSACTION;")
        conn.execute(
            """CREATE TABLE KnowledgeEdge_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER NOT NULL,
            source_id INTEGER NOT NULL,
            source_type TEXT NOT NULL CHECK (source_type IN (
                'web_evidence', 'code_evidence', 'tool_trace',
                'ledger', 'fact', 'claim'
            )),
            target_id INTEGER NOT NULL,
            target_type TEXT NOT NULL CHECK (target_type IN (
                'web_evidence', 'code_evidence', 'tool_trace',
                'ledger', 'fact', 'claim'
            )),
            relation TEXT NOT NULL CHECK (relation IN (
                'supports', 'conflicts_with', 'subsumes',
                'supersedes', 'derived_from', 'same_source',
                'refutes', 'qualifies'
            )),
            justification_group TEXT NOT NULL DEFAULT 'default',
            confidence REAL CHECK (confidence BETWEEN 0.0 AND 1.0),
            created_by TEXT,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            is_active INTEGER NOT NULL DEFAULT 1,
            UNIQUE(source_id, source_type, target_id, target_type, relation, justification_group),
            FOREIGN KEY(topic_id) REFERENCES Topic(id)
        )"""
        )
        conn.execute("INSERT INTO KnowledgeEdge_new SELECT * FROM KnowledgeEdge;")
        conn.execute("DROP TABLE KnowledgeEdge;")
        conn.execute("ALTER TABLE KnowledgeEdge_new RENAME TO KnowledgeEdge;")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ke_source ON KnowledgeEdge(topic_id, source_id, source_type) WHERE is_active = 1;"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ke_target ON KnowledgeEdge(topic_id, target_id, target_type) WHERE is_active = 1;"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ke_relation ON KnowledgeEdge(topic_id, relation) WHERE is_active = 1;"
        )
        conn.execute("COMMIT;")
        logger.info("[db] KnowledgeEdge migration complete.")
    except Exception as exc:
        logger.error("[db] KnowledgeEdge migration failed: %s", exc)
        try:
            conn.execute("ROLLBACK;")
        except Exception:
            pass
    finally:
        try:
            conn.execute("PRAGMA foreign_keys=ON;")
        except Exception:
            pass
        conn.isolation_level = old_isolation


def _migrate_vec_tables_if_dim_mismatch(conn: sqlite3.Connection) -> None:
    """Drop and recreate vec_facts/vec_messages if their dimension doesn't match EMBEDDING_DIM."""
    from .embedding import EMBEDDING_DIM

    for table, pk_col in [
        ("vec_facts", "fact_id"),
        ("vec_messages", "msg_id"),
        ("vec_corpus_chunks", "chunk_id"),
    ]:
        try:
            row = conn.execute(f"SELECT embedding FROM {table} LIMIT 1").fetchone()
        except sqlite3.OperationalError:
            continue  # table doesn't exist yet
        if row is None:
            continue  # table is empty, nothing to migrate
        blob = row[0]
        if isinstance(blob, bytes):
            existing_dim = len(blob) // 4  # f32 = 4 bytes each
            if existing_dim != EMBEDDING_DIM:
                logger.info(
                    "[init_db] %s dim mismatch: %d vs %d, rebuilding",
                    table,
                    existing_dim,
                    EMBEDDING_DIM,
                )
                conn.execute(f"DROP TABLE IF EXISTS {table}")
                conn.execute(
                    f"CREATE VIRTUAL TABLE {table} USING vec0("
                    f"{pk_col} INTEGER PRIMARY KEY, embedding float[{EMBEDDING_DIM}])"
                )


def _ensure_vec_tables(conn: sqlite3.Connection) -> None:
    """Create sqlite-vec tables only when the extension is available."""
    global _sqlite_vec_available
    if not _sqlite_vec_available:
        return
    try:
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
                fact_id INTEGER PRIMARY KEY,
                embedding float[768]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
                msg_id INTEGER PRIMARY KEY,
                embedding float[768]
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS vec_corpus_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[768]
            );
            """
        )
    except sqlite3.OperationalError as exc:
        _sqlite_vec_available = False
        logger.warning(
            "sqlite-vec tables unavailable — semantic search disabled: %s", exc
        )


def init_db():
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS Topic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                detail TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Closed', -- Closed, Started, Running
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS Plan (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                current_index INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS Subtopic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                detail TEXT NOT NULL,
                start_msg_id INTEGER,
                conclusion TEXT,
                status TEXT NOT NULL DEFAULT 'Open', -- Open, Closed
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS Message (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                msg_type TEXT NOT NULL DEFAULT 'standard', -- standard, summary
                confidence_score REAL,
                round_number INTEGER,
                turn_kind TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id)
            );

            CREATE TABLE IF NOT EXISTS FactCandidate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                writer_msg_id INTEGER,
                candidate_text TEXT NOT NULL,
                fact_stage TEXT NOT NULL DEFAULT 'synthesized',
                candidate_type TEXT NOT NULL DEFAULT 'sourced_claim',
                source_kind TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                reviewed_text TEXT,
                review_note TEXT,
                evidence_note TEXT,
                source_refs_json TEXT,
                source_excerpt TEXT,
                verification_status TEXT,
                confidence_score REAL,
                round_number INTEGER,
                reviewer TEXT,
                accepted_fact_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                reviewed_at DATETIME,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(writer_msg_id) REFERENCES Message(id),
                FOREIGN KEY(accepted_fact_id) REFERENCES Fact(id)
            );
            
            CREATE TABLE IF NOT EXISTS Fact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                fact_stage TEXT NOT NULL DEFAULT 'synthesized',
                fact_type TEXT NOT NULL DEFAULT 'sourced_claim',
                verification_status TEXT,
                source_kind TEXT,
                source_refs_json TEXT,
                source_excerpt TEXT,
                candidate_id INTEGER,
                review_status TEXT,
                evidence_note TEXT,
                confidence_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(candidate_id) REFERENCES FactCandidate(id)
            );

            CREATE TABLE IF NOT EXISTS ClaimCandidate (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                clerk_msg_id INTEGER,
                candidate_text TEXT NOT NULL,
                summary TEXT,
                support_fact_ids_json TEXT,
                rationale_short TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                review_note TEXT,
                reviewed_text TEXT,
                claim_score REAL,
                accepted_claim_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                reviewed_at DATETIME,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(clerk_msg_id) REFERENCES Message(id)
            );

            CREATE TABLE IF NOT EXISTS Claim (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                content TEXT NOT NULL,
                summary TEXT,
                support_fact_ids_json TEXT,
                rationale_short TEXT,
                claim_score REAL,
                status TEXT NOT NULL DEFAULT 'active',
                candidate_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(candidate_id) REFERENCES ClaimCandidate(id)
            );

            CREATE TABLE IF NOT EXISTS WebEvidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin_topic_id INTEGER NOT NULL,
                origin_subtopic_id INTEGER,
                query_text TEXT NOT NULL,
                title TEXT,
                snippet TEXT,
                url TEXT,
                source_domain TEXT,
                result_rank INTEGER,
                search_provider TEXT,
                search_role TEXT,
                summary TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(origin_topic_id) REFERENCES Topic(id),
                FOREIGN KEY(origin_subtopic_id) REFERENCES Subtopic(id)
            );

            CREATE TABLE IF NOT EXISTS VoteRecord (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                round_number INTEGER,
                vote_kind TEXT NOT NULL,
                subject TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                voter TEXT NOT NULL,
                parsed_ok INTEGER NOT NULL DEFAULT 0,
                decision TEXT,
                reason TEXT,
                raw_response TEXT NOT NULL,
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id)
            );

            CREATE TABLE IF NOT EXISTS LedgerEntity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                canonical_name TEXT NOT NULL,
                entity_type TEXT,
                last_mentioned_round INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic_id, canonical_name),
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS LedgerEntityAlias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL,
                alias_text TEXT NOT NULL,
                confirmed INTEGER NOT NULL DEFAULT 0,
                match_count INTEGER NOT NULL DEFAULT 1,
                UNIQUE(entity_id, alias_text),
                FOREIGN KEY(entity_id) REFERENCES LedgerEntity(id)
            );

            CREATE TABLE IF NOT EXISTS LedgerAttribute (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                canonical_name TEXT NOT NULL,
                value_type TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic_id, canonical_name),
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS LedgerAttributeAlias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attribute_id INTEGER NOT NULL,
                alias_text TEXT NOT NULL,
                confirmed INTEGER NOT NULL DEFAULT 0,
                match_count INTEGER NOT NULL DEFAULT 1,
                UNIQUE(attribute_id, alias_text),
                FOREIGN KEY(attribute_id) REFERENCES LedgerAttribute(id)
            );

            CREATE TABLE IF NOT EXISTS Ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                entity_id INTEGER NOT NULL,
                attribute_id INTEGER NOT NULL,
                value TEXT NOT NULL,
                value_numeric_min REAL,
                value_numeric_max REAL,
                unit TEXT,
                normalized_timeframe TEXT NOT NULL DEFAULT '',
                entry_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'accepted',
                source_ref TEXT NOT NULL,
                source_domain TEXT,
                domain_score REAL,
                decontextualized TEXT,
                created_by TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic_id, entity_id, attribute_id, normalized_timeframe, source_ref),
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id),
                FOREIGN KEY(entity_id) REFERENCES LedgerEntity(id),
                FOREIGN KEY(attribute_id) REFERENCES LedgerAttribute(id)
            );

            CREATE TABLE IF NOT EXISTS LedgerPending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                raw_text TEXT NOT NULL,
                source_ref TEXT,
                extracted_numbers TEXT,
                missing_fields TEXT,
                created_round INTEGER,
                ttl_expires_round INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS LedgerEdge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                from_entry_id INTEGER NOT NULL,
                to_entry_id INTEGER NOT NULL,
                edge_type TEXT NOT NULL,
                created_by TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(from_entry_id, to_entry_id, edge_type),
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(from_entry_id) REFERENCES Ledger(id),
                FOREIGN KEY(to_entry_id) REFERENCES Ledger(id)
            );

            CREATE TABLE IF NOT EXISTS KnowledgeEdge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                source_id INTEGER NOT NULL,
                source_type TEXT NOT NULL CHECK (source_type IN (
                    'web_evidence', 'code_evidence', 'tool_trace',
                    'ledger', 'fact', 'claim'
                )),
                target_id INTEGER NOT NULL,
                target_type TEXT NOT NULL CHECK (target_type IN (
                    'web_evidence', 'code_evidence', 'tool_trace',
                    'ledger', 'fact', 'claim'
                )),
                relation TEXT NOT NULL CHECK (relation IN (
                    'supports', 'conflicts_with', 'subsumes',
                    'supersedes', 'derived_from', 'same_source',
                    'refutes', 'qualifies'
                )),
                justification_group TEXT NOT NULL DEFAULT 'default',
                confidence REAL CHECK (confidence BETWEEN 0.0 AND 1.0),
                created_by TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                is_active INTEGER NOT NULL DEFAULT 1,
                UNIQUE(source_id, source_type, target_id, target_type, relation, justification_group),
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE INDEX IF NOT EXISTS idx_ke_source ON KnowledgeEdge(topic_id, source_id, source_type) WHERE is_active = 1;
            CREATE INDEX IF NOT EXISTS idx_ke_target ON KnowledgeEdge(topic_id, target_id, target_type) WHERE is_active = 1;
            CREATE INDEX IF NOT EXISTS idx_ke_relation ON KnowledgeEdge(topic_id, relation) WHERE is_active = 1;

            CREATE TABLE IF NOT EXISTS ToolTrace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                tool_type TEXT NOT NULL,
                query TEXT,
                result_count INTEGER,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata_json TEXT,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS CodeEvidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin_topic_id INTEGER NOT NULL,
                origin_subtopic_id INTEGER,
                hypothesis TEXT NOT NULL,
                source_code TEXT NOT NULL,
                stdout TEXT,
                stderr TEXT,
                exit_code INTEGER NOT NULL DEFAULT -1,
                execution_time_s REAL,
                iterations INTEGER NOT NULL DEFAULT 1,
                success INTEGER NOT NULL DEFAULT 0,
                requesting_role TEXT,
                summary TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(origin_topic_id) REFERENCES Topic(id),
                FOREIGN KEY(origin_subtopic_id) REFERENCES Subtopic(id)
            );

            CREATE TABLE IF NOT EXISTS ApiEvidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin_topic_id INTEGER NOT NULL,
                origin_subtopic_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                provider TEXT,
                requested_provider TEXT,
                model TEXT,
                requesting_role TEXT,
                planner_reason TEXT,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(origin_topic_id) REFERENCES Topic(id),
                FOREIGN KEY(origin_subtopic_id) REFERENCES Subtopic(id)
            );

            CREATE INDEX IF NOT EXISTS idx_api_evidence_topic_id
                ON ApiEvidence(origin_topic_id, id DESC);

            CREATE TABLE IF NOT EXISTS CorpusDocument (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                title TEXT NOT NULL,
                doc_type TEXT NOT NULL DEFAULT 'text',
                author TEXT,
                source_path TEXT,
                source_url TEXT,
                source_hash TEXT NOT NULL,
                access_scope TEXT NOT NULL DEFAULT 'topic',
                parser_version TEXT NOT NULL DEFAULT 'text-v1',
                index_status TEXT NOT NULL DEFAULT 'pending',
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic_id, source_hash),
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS CorpusIngestRun (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                topic_id INTEGER,
                source_path TEXT,
                parser_version TEXT NOT NULL DEFAULT 'text-v1',
                status TEXT NOT NULL DEFAULT 'running',
                error TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                FOREIGN KEY(document_id) REFERENCES CorpusDocument(id),
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS CorpusChunk (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                parent_chunk_id INTEGER,
                chunk_index INTEGER NOT NULL,
                granularity TEXT NOT NULL DEFAULT 'paragraph',
                section_path TEXT,
                page_start INTEGER,
                page_end INTEGER,
                position_start INTEGER,
                position_end INTEGER,
                text TEXT NOT NULL,
                table_markdown TEXT,
                lexical_text TEXT,
                token_count INTEGER,
                checksum TEXT NOT NULL,
                freshness_timestamp TEXT,
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_id, chunk_index),
                FOREIGN KEY(document_id) REFERENCES CorpusDocument(id),
                FOREIGN KEY(parent_chunk_id) REFERENCES CorpusChunk(id)
            );

            CREATE INDEX IF NOT EXISTS idx_corpus_document_topic
                ON CorpusDocument(topic_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_corpus_chunk_document
                ON CorpusChunk(document_id, chunk_index);

            CREATE TABLE IF NOT EXISTS OptimizationProblem (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                title TEXT NOT NULL,
                source_text TEXT NOT NULL,
                problem_class TEXT,
                domain_context TEXT,
                stakeholder TEXT,
                time_horizon TEXT,
                status TEXT NOT NULL DEFAULT 'candidate',
                source_refs_json TEXT,
                metadata_json TEXT,
                created_by TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(subtopic_id) REFERENCES Subtopic(id)
            );

            CREATE TABLE IF NOT EXISTS OptimizationComponent (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                component_type TEXT NOT NULL,
                natural_text TEXT NOT NULL,
                formal_text TEXT,
                symbol TEXT,
                unit TEXT,
                domain TEXT,
                source_refs_json TEXT,
                review_status TEXT NOT NULL DEFAULT 'candidate',
                validation_notes TEXT,
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(problem_id) REFERENCES OptimizationProblem(id),
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS OptimizationModelIR (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                ir_json TEXT NOT NULL,
                ir_hash TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                validation_notes TEXT,
                linked_component_ids_json TEXT,
                component_fingerprints_json TEXT,
                generator_role TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(problem_id) REFERENCES OptimizationProblem(id),
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE TABLE IF NOT EXISTS OptimizationArtifact (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                artifact_type TEXT NOT NULL,
                model_language TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                parser_status TEXT NOT NULL DEFAULT 'pending',
                parser_notes TEXT,
                linked_component_ids_json TEXT,
                component_fingerprints_json TEXT,
                generator_role TEXT,
                source_artifact_id INTEGER,
                repair_status TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(problem_id) REFERENCES OptimizationProblem(id),
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(source_artifact_id) REFERENCES OptimizationArtifact(id)
            );

            CREATE TABLE IF NOT EXISTS SolverRun (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id INTEGER NOT NULL,
                problem_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                solver_backend TEXT NOT NULL,
                status TEXT NOT NULL,
                objective_value REAL,
                variable_values_json TEXT,
                stdout TEXT,
                stderr TEXT,
                error_trace TEXT,
                elapsed_time_s REAL,
                code_evidence_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(artifact_id) REFERENCES OptimizationArtifact(id),
                FOREIGN KEY(problem_id) REFERENCES OptimizationProblem(id),
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(code_evidence_id) REFERENCES CodeEvidence(id)
            );

            CREATE TABLE IF NOT EXISTS ModelDiagnostic (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                component_id INTEGER,
                artifact_id INTEGER,
                solver_run_id INTEGER,
                diagnostic_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'warning',
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                source_refs_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME,
                FOREIGN KEY(problem_id) REFERENCES OptimizationProblem(id),
                FOREIGN KEY(topic_id) REFERENCES Topic(id),
                FOREIGN KEY(component_id) REFERENCES OptimizationComponent(id),
                FOREIGN KEY(artifact_id) REFERENCES OptimizationArtifact(id),
                FOREIGN KEY(solver_run_id) REFERENCES SolverRun(id)
            );

            CREATE TABLE IF NOT EXISTS ModelingExperience (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                family TEXT NOT NULL,
                structure_key TEXT NOT NULL,
                content TEXT NOT NULL,
                applies_when_json TEXT,
                rejects_when_json TEXT,
                source_refs_json TEXT,
                status TEXT NOT NULL DEFAULT 'candidate',
                validation_summary_json TEXT,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            );

            CREATE INDEX IF NOT EXISTS idx_opt_problem_topic
                ON OptimizationProblem(topic_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_opt_component_problem
                ON OptimizationComponent(problem_id, component_type);
            CREATE INDEX IF NOT EXISTS idx_opt_model_ir_problem
                ON OptimizationModelIR(problem_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_opt_artifact_problem
                ON OptimizationArtifact(problem_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_solver_run_problem
                ON SolverRun(problem_id, id DESC);
            CREATE INDEX IF NOT EXISTS idx_model_diagnostic_problem
                ON ModelDiagnostic(problem_id, status);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_modeling_experience_family_key
                ON ModelingExperience(family, structure_key);
            CREATE INDEX IF NOT EXISTS idx_modeling_experience_family_status
                ON ModelingExperience(family, status);

            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                content,
                topic_id UNINDEXED,
                source UNINDEXED
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                topic_id UNINDEXED,
                msg_type UNINDEXED,
                sender UNINDEXED
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS claims_fts USING fts5(
                content,
                topic_id UNINDEXED
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS web_evidence_fts USING fts5(
                content,
                origin_topic_id UNINDEXED,
                source_domain UNINDEXED
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS corpus_chunks_fts USING fts5(
                content,
                document_id UNINDEXED,
                topic_id UNINDEXED,
                doc_type UNINDEXED,
                access_scope UNINDEXED
            );
        """
        )
        _ensure_vec_tables(conn)
        _ensure_column(conn, "Plan", "current_index", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "Subtopic", "start_msg_id", "INTEGER")
        _ensure_column(conn, "Subtopic", "conclusion", "TEXT")
        _ensure_column(conn, "Subtopic", "status", "TEXT NOT NULL DEFAULT 'Open'")
        _ensure_column(conn, "Subtopic", "locked_scope", "TEXT")
        _ensure_column(conn, "Message", "confidence_score", "REAL")
        _ensure_column(conn, "Message", "round_number", "INTEGER")
        _ensure_column(conn, "Message", "turn_kind", "TEXT")
        _ensure_column(conn, "Fact", "candidate_id", "INTEGER")
        _ensure_column(conn, "Fact", "subtopic_id", "INTEGER")
        _ensure_column(
            conn, "Fact", "fact_stage", "TEXT NOT NULL DEFAULT 'synthesized'"
        )
        _ensure_column(
            conn, "Fact", "fact_type", "TEXT NOT NULL DEFAULT 'sourced_claim'"
        )
        _ensure_column(conn, "Fact", "verification_status", "TEXT")
        _ensure_column(conn, "Fact", "source_kind", "TEXT")
        _ensure_column(conn, "Fact", "source_refs_json", "TEXT")
        _ensure_column(conn, "Fact", "source_excerpt", "TEXT")
        _ensure_column(conn, "Topic", "conclusion", "TEXT")
        _ensure_column(conn, "Fact", "review_status", "TEXT")
        _ensure_column(conn, "Fact", "summary", "TEXT")
        _ensure_column(conn, "Claim", "summary", "TEXT")
        _ensure_column(conn, "Message", "summary", "TEXT")
        _ensure_column(conn, "FactCandidate", "summary", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "summary", "TEXT")
        _ensure_column(conn, "WebEvidence", "summary", "TEXT")
        _ensure_column(conn, "FactCandidate", "evidence_note", "TEXT")
        _ensure_column(conn, "Fact", "confidence_score", "REAL")
        _ensure_column(
            conn, "FactCandidate", "fact_stage", "TEXT NOT NULL DEFAULT 'synthesized'"
        )
        _ensure_column(
            conn,
            "FactCandidate",
            "candidate_type",
            "TEXT NOT NULL DEFAULT 'sourced_claim'",
        )
        # Phase D: FactCandidate source_kind
        _ensure_column(conn, "FactCandidate", "source_kind", "TEXT")
        _ensure_column(conn, "FactCandidate", "source_refs_json", "TEXT")
        _ensure_column(conn, "FactCandidate", "source_excerpt", "TEXT")
        _ensure_column(conn, "FactCandidate", "verification_status", "TEXT")
        _ensure_column(conn, "FactCandidate", "round_number", "INTEGER")
        _ensure_column(
            conn, "WebEvidence", "ledger_processed", "INTEGER NOT NULL DEFAULT 0"
        )
        _ensure_column(conn, "Claim", "superseded_by", "INTEGER")
        _ensure_column(conn, "CodeEvidence", "parent_evidence_id", "INTEGER")
        _ensure_column(
            conn, "CodeEvidence", "review_count", "INTEGER NOT NULL DEFAULT 0"
        )
        _ensure_column(conn, "ApiEvidence", "requested_provider", "TEXT")
        _ensure_column(
            conn, "ApiEvidence", "fallback_used", "INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_api_evidence_topic_id
               ON ApiEvidence(origin_topic_id, id DESC)"""
        )
        _ensure_column(conn, "Ledger", "valid_from", "TEXT")
        _ensure_column(conn, "Ledger", "valid_to", "TEXT")
        # Phase A: Structured Fact columns (Wikidata-style)
        _ensure_column(conn, "Fact", "subject", "TEXT")
        _ensure_column(conn, "Fact", "predicate", "TEXT")
        _ensure_column(conn, "Fact", "object_json", "TEXT")
        _ensure_column(conn, "Fact", "qualifiers_json", "TEXT")
        _ensure_column(conn, "Fact", "attribution_json", "TEXT")
        # Phase A.2: FactCandidate structured columns
        _ensure_column(conn, "FactCandidate", "subject", "TEXT")
        _ensure_column(conn, "FactCandidate", "predicate", "TEXT")
        _ensure_column(conn, "FactCandidate", "object_json", "TEXT")
        _ensure_column(conn, "FactCandidate", "qualifiers_json", "TEXT")
        _ensure_column(conn, "FactCandidate", "attribution_json", "TEXT")
        # Phase A.2: WebEvidence snippet dedup
        _ensure_column(conn, "WebEvidence", "snippet_hash", "TEXT")
        # Phase A.2: URL dedup index
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_web_evidence_topic_url "
            "ON WebEvidence(origin_topic_id, url)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_web_evidence_topic_snippet_hash "
            "ON WebEvidence(origin_topic_id, snippet_hash)"
        )
        # Phase A: Structured Claim columns
        _ensure_column(conn, "Claim", "subject", "TEXT")
        _ensure_column(conn, "Claim", "predicate", "TEXT")
        _ensure_column(conn, "Claim", "object_json", "TEXT")
        _ensure_column(conn, "Claim", "qualifiers_json", "TEXT")
        _ensure_column(conn, "Claim", "polarity", "TEXT")
        # Phase C: Knowledge lifecycle columns
        _ensure_column(conn, "Fact", "superseded_by", "INTEGER")
        _ensure_column(conn, "Fact", "contested_since_round", "INTEGER")
        _ensure_column(conn, "Claim", "contested_since_round", "INTEGER")
        _ensure_column(conn, "Ledger", "contested_since_round", "INTEGER")
        _ensure_column(conn, "Ledger", "review_status", "TEXT")
        _ensure_column(conn, "Ledger", "superseded_by", "INTEGER")
        # Phase E: WebQueryCache for semantic web query dedup
        conn.execute(
            """CREATE TABLE IF NOT EXISTS WebQueryCache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                query_text TEXT NOT NULL,
                result_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )"""
        )
        try:
            conn.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS vec_web_queries USING vec0(
                    query_id INTEGER PRIMARY KEY,
                    embedding float[768]
                )"""
            )
        except Exception as exc:
            logger.debug("vec_web_queries creation skipped: %s", exc)
        # Phase F.1: TopicConfig table
        conn.execute(
            """CREATE TABLE IF NOT EXISTS TopicConfig (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                config_key TEXT NOT NULL,
                config_value TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(topic_id, config_key),
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            )"""
        )
        # Phase F.2: HITL columns and UserInjection table
        _ensure_column(conn, "Topic", "paused_at_stage", "TEXT")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS UserInjection (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER NOT NULL,
                subtopic_id INTEGER,
                injection_type TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(topic_id) REFERENCES Topic(id)
            )"""
        )
        # Phase F.3: Report storage
        _ensure_column(conn, "Topic", "report_json", "TEXT")
        # Phase F.5: Queue columns
        _ensure_column(conn, "Topic", "queue_position", "INTEGER")
        _ensure_column(conn, "Topic", "queued_at", "DATETIME")
        # Phase G: Ledger rich schema — statistical quantity columns
        _ensure_column(conn, "Ledger", "value_mean", "REAL")
        _ensure_column(conn, "Ledger", "value_std", "REAL")
        _ensure_column(conn, "Ledger", "value_ci_lower", "REAL")
        _ensure_column(conn, "Ledger", "value_ci_upper", "REAL")
        _ensure_column(conn, "Ledger", "value_ci_level", "REAL")
        _ensure_column(conn, "Ledger", "value_p", "REAL")
        _ensure_column(conn, "Ledger", "value_n", "INTEGER")
        _ensure_column(conn, "Ledger", "value_stat_type", "TEXT")
        _ensure_column(conn, "Ledger", "baseline_entity_id", "INTEGER")
        _ensure_column(conn, "Ledger", "split", "TEXT")
        _ensure_column(conn, "Ledger", "config_json", "TEXT")
        _ensure_column(conn, "OptimizationArtifact", "source_artifact_id", "INTEGER")
        _ensure_column(conn, "OptimizationArtifact", "repair_status", "TEXT")
        _ensure_column(
            conn, "OptimizationArtifact", "component_fingerprints_json", "TEXT"
        )
        # VIKI: Autonomous repair tracking
        conn.execute(
            """CREATE TABLE IF NOT EXISTS VikiTracker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_table TEXT NOT NULL,
                target_id INTEGER NOT NULL,
                issue_type TEXT NOT NULL,
                check_count INTEGER NOT NULL DEFAULT 0,
                max_checks INTEGER NOT NULL DEFAULT 3,
                last_checked_at DATETIME,
                status TEXT NOT NULL DEFAULT 'open',
                resolution TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(target_table, target_id, issue_type)
            )"""
        )
        _ensure_column(conn, "VikiTracker", "topic_id", "INTEGER")
        # G.4: Claim V2 — structured claims with scope, falsification, reasoning
        _ensure_column(conn, "Claim", "claim_type", "TEXT")
        _ensure_column(conn, "Claim", "scope_tags", "TEXT")
        _ensure_column(conn, "Claim", "scope_context", "TEXT")
        _ensure_column(conn, "Claim", "falsification_criteria", "TEXT")
        _ensure_column(conn, "Claim", "inference_logic", "TEXT")
        _ensure_column(conn, "Claim", "conclusion", "TEXT")
        _ensure_column(conn, "Claim", "evidence_strength", "REAL")
        _ensure_column(conn, "Claim", "scope_breadth", "REAL")
        _ensure_column(conn, "Claim", "submitted_by", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "claim_type", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "scope_tags", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "scope_context", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "falsification_criteria", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "inference_logic", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "conclusion", "TEXT")
        _ensure_column(conn, "ClaimCandidate", "evidence_strength", "REAL")
        _ensure_column(conn, "ClaimCandidate", "scope_breadth", "REAL")
        _ensure_column(conn, "ClaimCandidate", "submitted_by", "TEXT")
        _ensure_column(
            conn, "Message", "has_formal_claim", "INTEGER NOT NULL DEFAULT 0"
        )
        # G.4: KnowledgeEdge migration — add refutes + qualifies to CHECK
        _migrate_knowledge_edge_check(conn)
        # Phase B.1: Migrate vec tables from 384-dim to 768-dim if needed
        if _sqlite_vec_available:
            _migrate_vec_tables_if_dim_mismatch(conn)
        _backfill_fts(conn)
        _backfill_knowledge_edges(conn)
        _backfill_code_and_fact_edges(conn)


def _backfill_knowledge_edges(conn):
    """Backfill KnowledgeEdge from existing JSON arrays and LedgerEdge. Idempotent."""
    # Skip if already backfilled (any edges exist)
    count = conn.execute("SELECT COUNT(*) FROM KnowledgeEdge").fetchone()[0]
    if count > 0:
        return

    # 1. Claim.support_fact_ids_json -> supports edges (fact->claim)
    claims = conn.execute(
        "SELECT id, topic_id, support_fact_ids_json FROM Claim WHERE support_fact_ids_json IS NOT NULL"
    ).fetchall()
    for claim in claims:
        try:
            fact_ids = json.loads(claim["support_fact_ids_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        for fid in fact_ids:
            try:
                fid = int(fid)
            except (TypeError, ValueError):
                continue
            conn.execute(
                """INSERT OR IGNORE INTO KnowledgeEdge
                    (topic_id, source_id, source_type, target_id, target_type, relation, justification_group)
                    VALUES (?, ?, 'fact', ?, 'claim', 'supports', 'default')""",
                (claim["topic_id"], fid, claim["id"]),
            )

    # 2. Fact.source_refs_json -> derived_from edges (web_evidence->fact)
    facts = conn.execute(
        "SELECT id, topic_id, source_refs_json FROM Fact WHERE source_refs_json IS NOT NULL"
    ).fetchall()
    for fact in facts:
        try:
            refs = json.loads(fact["source_refs_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        for ref in refs:
            m = re.match(r"W(\d+)", str(ref))
            if m:
                conn.execute(
                    """INSERT OR IGNORE INTO KnowledgeEdge
                        (topic_id, source_id, source_type, target_id, target_type, relation)
                        VALUES (?, ?, 'web_evidence', ?, 'fact', 'derived_from')""",
                    (fact["topic_id"], int(m.group(1)), fact["id"]),
                )

    # 3. Ledger.source_ref -> derived_from edges (web_evidence/code_evidence->ledger)
    ledgers = conn.execute(
        "SELECT id, topic_id, source_ref FROM Ledger WHERE source_ref IS NOT NULL"
    ).fetchall()
    for led in ledgers:
        m = re.match(r"\[?W(\d+)\]?", str(led["source_ref"]))
        if m:
            conn.execute(
                """INSERT OR IGNORE INTO KnowledgeEdge
                    (topic_id, source_id, source_type, target_id, target_type, relation)
                    VALUES (?, ?, 'web_evidence', ?, 'ledger', 'derived_from')""",
                (led["topic_id"], int(m.group(1)), led["id"]),
            )

    # 4. LedgerEdge -> KnowledgeEdge (filter to valid KnowledgeEdge relations)
    _valid_ke_relations = {
        "supports",
        "conflicts_with",
        "subsumes",
        "supersedes",
        "derived_from",
        "same_source",
    }
    ledger_edges = conn.execute(
        "SELECT topic_id, from_entry_id, to_entry_id, edge_type FROM LedgerEdge"
    ).fetchall()
    for le in ledger_edges:
        if le["edge_type"] not in _valid_ke_relations:
            continue
        conn.execute(
            """INSERT OR IGNORE INTO KnowledgeEdge
                (topic_id, source_id, source_type, target_id, target_type, relation)
                VALUES (?, ?, 'ledger', ?, 'ledger', ?)""",
            (le["topic_id"], le["from_entry_id"], le["to_entry_id"], le["edge_type"]),
        )


def _backfill_code_and_fact_edges(conn):
    """Backfill E-ref and F-ref KnowledgeEdges from Fact.source_refs_json. Runs unconditionally with INSERT OR IGNORE."""
    facts = conn.execute(
        "SELECT id, topic_id, source_refs_json FROM Fact WHERE source_refs_json IS NOT NULL"
    ).fetchall()
    for fact in facts:
        try:
            refs = json.loads(fact["source_refs_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            continue
        for ref in refs:
            m = re.match(r"\[?E(\d+)\]?", str(ref))
            if m:
                conn.execute(
                    """INSERT OR IGNORE INTO KnowledgeEdge
                        (topic_id, source_id, source_type, target_id, target_type, relation)
                        VALUES (?, ?, 'code_evidence', ?, 'fact', 'derived_from')""",
                    (fact["topic_id"], int(m.group(1)), fact["id"]),
                )
            else:
                m = re.match(r"\[?F(\d+)\]?", str(ref))
                if m:
                    src_fid = int(m.group(1))
                    if src_fid != fact["id"]:
                        conn.execute(
                            """INSERT OR IGNORE INTO KnowledgeEdge
                                (topic_id, source_id, source_type, target_id, target_type, relation)
                                VALUES (?, ?, 'fact', ?, 'fact', 'derived_from')""",
                            (fact["topic_id"], src_fid, fact["id"]),
                        )


def _insert_fact_row(
    conn: sqlite3.Connection,
    topic_id: int,
    subtopic_id: Optional[int],
    content: str,
    source: str,
    fact_stage: str = "synthesized",
    fact_type: str = "sourced_claim",
    verification_status: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_refs_json: Optional[str] = None,
    source_excerpt: Optional[str] = None,
    candidate_id: Optional[int] = None,
    review_status: Optional[str] = None,
    evidence_note: Optional[str] = None,
    confidence_score: Optional[float] = None,
    summary: Optional[str] = None,
) -> int:
    content = _truncate(content, MAX_CONTENT_LEN)
    summary = _truncate(summary, MAX_SUMMARY_LEN)
    cursor = conn.execute(
        """
        INSERT INTO Fact (
            topic_id,
            subtopic_id,
            content,
            summary,
            source,
            fact_stage,
            fact_type,
            verification_status,
            source_kind,
            source_refs_json,
            source_excerpt,
            candidate_id,
            review_status,
            evidence_note,
            confidence_score
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            topic_id,
            subtopic_id,
            content,
            summary,
            source,
            fact_stage,
            fact_type,
            verification_status,
            source_kind,
            source_refs_json,
            source_excerpt,
            candidate_id,
            review_status,
            evidence_note,
            confidence_score,
        ),
    )
    fact_id = cursor.lastrowid
    fts_content = content
    if summary:
        fts_content = f"{summary}\n\n{content}"
    _insert_fact_fts(conn, fact_id, topic_id, fts_content, source)
    return fact_id


def insert_fact(
    topic_id: int,
    content: str,
    source: str,
    subtopic_id: Optional[int] = None,
    fact_stage: str = "synthesized",
    fact_type: str = "sourced_claim",
    verification_status: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_refs_json: Optional[str] = None,
    source_excerpt: Optional[str] = None,
    candidate_id: Optional[int] = None,
    review_status: Optional[str] = None,
    evidence_note: Optional[str] = None,
    confidence_score: Optional[float] = None,
    summary: Optional[str] = None,
) -> int:
    with get_db() as conn:
        return _insert_fact_row(
            conn,
            topic_id,
            subtopic_id,
            content,
            source,
            fact_stage=fact_stage,
            fact_type=fact_type,
            verification_status=verification_status,
            source_kind=source_kind,
            source_refs_json=source_refs_json,
            source_excerpt=source_excerpt,
            candidate_id=candidate_id,
            review_status=review_status,
            evidence_note=evidence_note,
            confidence_score=confidence_score,
            summary=summary,
        )


def insert_fact_with_embedding(
    topic_id: int,
    content: str,
    source: str,
    embedding: List[float],
    subtopic_id: Optional[int] = None,
    fact_stage: str = "synthesized",
    fact_type: str = "sourced_claim",
    verification_status: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_refs_json: Optional[str] = None,
    source_excerpt: Optional[str] = None,
    candidate_id: Optional[int] = None,
    review_status: Optional[str] = None,
    evidence_note: Optional[str] = None,
    confidence_score: Optional[float] = None,
    summary: Optional[str] = None,
) -> int:
    """Insert a fact and its corresponding embedding into the database."""
    global _sqlite_vec_available
    with get_db() as conn:
        fact_id = _insert_fact_row(
            conn,
            topic_id,
            subtopic_id,
            content,
            source,
            fact_stage=fact_stage,
            fact_type=fact_type,
            verification_status=verification_status,
            source_kind=source_kind,
            source_refs_json=source_refs_json,
            source_excerpt=source_excerpt,
            candidate_id=candidate_id,
            review_status=review_status,
            evidence_note=evidence_note,
            confidence_score=confidence_score,
            summary=summary,
        )
        if _sqlite_vec_available:
            try:
                conn.execute(
                    "INSERT INTO vec_facts(fact_id, embedding) VALUES (?, ?)",
                    (fact_id, serialize_f32(embedding)),
                )
            except sqlite3.OperationalError as exc:
                _sqlite_vec_available = False
                logger.debug("fact semantic index unavailable: %s", exc)
        return fact_id


def update_fact_summary_and_embedding(
    fact_id: int, summary: str, embedding: List[float]
) -> None:
    global _sqlite_vec_available
    summary = _truncate(summary, MAX_SUMMARY_LEN)
    with get_db() as conn:
        conn.execute("UPDATE Fact SET summary = ? WHERE id = ?", (summary, fact_id))
        row = conn.execute(
            "SELECT content, topic_id, source FROM Fact WHERE id = ?", (fact_id,)
        ).fetchone()
        if not row:
            logger.warning(
                "[update_fact_summary_and_embedding] Fact %s not found; skipping.",
                fact_id,
            )
            return
        fts_content = f"{summary}\n\n{row['content']}" if summary else row["content"]
        conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
        conn.execute(
            "INSERT INTO facts_fts(rowid, content, topic_id, source) VALUES (?, ?, ?, ?)",
            (fact_id, fts_content, str(row["topic_id"]), row["source"]),
        )
        if _sqlite_vec_available:
            try:
                conn.execute("DELETE FROM vec_facts WHERE fact_id = ?", (fact_id,))
                conn.execute(
                    "INSERT INTO vec_facts(fact_id, embedding) VALUES (?, ?)",
                    (fact_id, serialize_f32(embedding)),
                )
            except sqlite3.OperationalError as exc:
                _sqlite_vec_available = False
                logger.debug("fact semantic index unavailable: %s", exc)


def update_claim_summary(claim_id: int, summary: str) -> None:
    summary = _truncate(summary, MAX_SUMMARY_LEN)
    with get_db() as conn:
        conn.execute("UPDATE Claim SET summary = ? WHERE id = ?", (summary, claim_id))
        row = conn.execute(
            "SELECT content, topic_id FROM Claim WHERE id = ?", (claim_id,)
        ).fetchone()
        if row:
            fts_content = (
                f"{summary}\n\n{row['content']}" if summary else row["content"]
            )
            conn.execute("DELETE FROM claims_fts WHERE rowid = ?", (claim_id,))
            conn.execute(
                "INSERT INTO claims_fts(rowid, content, topic_id) VALUES (?, ?, ?)",
                (claim_id, fts_content, str(row["topic_id"])),
            )


def create_fact_candidate(
    topic_id: int,
    subtopic_id: int,
    writer_msg_id: Optional[int],
    candidate_text: str,
    summary: Optional[str] = None,
    fact_stage: str = "synthesized",
    candidate_type: str = "sourced_claim",
    source_kind: Optional[str] = None,
    evidence_note: Optional[str] = None,
    source_refs_json: Optional[str] = None,
    source_excerpt: Optional[str] = None,
    verification_status: Optional[str] = None,
    round_number: Optional[int] = None,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_json: Optional[str] = None,
    qualifiers_json: Optional[str] = None,
    attribution_json: Optional[str] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO FactCandidate (
                topic_id,
                subtopic_id,
                writer_msg_id,
                candidate_text,
                summary,
                fact_stage,
                candidate_type,
                source_kind,
                evidence_note,
                source_refs_json,
                source_excerpt,
                verification_status,
                round_number,
                subject,
                predicate,
                object_json,
                qualifiers_json,
                attribution_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                writer_msg_id,
                candidate_text,
                summary,
                fact_stage,
                candidate_type,
                source_kind,
                evidence_note,
                source_refs_json,
                source_excerpt,
                verification_status,
                round_number,
                subject,
                predicate,
                object_json,
                qualifiers_json,
                attribution_json,
            ),
        )
        return cursor.lastrowid


def create_claim_candidate(
    topic_id: int,
    subtopic_id: int,
    clerk_msg_id: Optional[int],
    candidate_text: str,
    summary: Optional[str] = None,
    support_fact_ids_json: Optional[str] = None,
    rationale_short: Optional[str] = None,
    # G.4: Structured claim fields
    claim_type: Optional[str] = None,
    scope_tags: Optional[str] = None,
    scope_context: Optional[str] = None,
    falsification_criteria: Optional[str] = None,
    inference_logic: Optional[str] = None,
    conclusion: Optional[str] = None,
    evidence_strength: Optional[float] = None,
    scope_breadth: Optional[float] = None,
    submitted_by: Optional[str] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ClaimCandidate (
                topic_id, subtopic_id, clerk_msg_id, candidate_text,
                summary, support_fact_ids_json, rationale_short,
                claim_type, scope_tags, scope_context,
                falsification_criteria, inference_logic, conclusion,
                evidence_strength, scope_breadth, submitted_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                clerk_msg_id,
                candidate_text,
                summary,
                support_fact_ids_json,
                rationale_short,
                claim_type,
                scope_tags,
                scope_context,
                falsification_criteria,
                inference_logic,
                conclusion,
                evidence_strength,
                scope_breadth,
                submitted_by,
            ),
        )
        return cursor.lastrowid


def get_fact_candidates(
    topic_id: int,
    subtopic_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with get_db() as conn:
        clauses = ["topic_id = ?"]
        params: list[Any] = [topic_id]
        if subtopic_id is not None:
            clauses.append("subtopic_id = ?")
            params.append(subtopic_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        rows = conn.execute(
            f"SELECT * FROM FactCandidate WHERE {' AND '.join(clauses)} ORDER BY id ASC",
            params,
        ).fetchall()
        return [dict(row) for row in rows]


def get_claim_candidates(
    topic_id: int,
    subtopic_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with get_db() as conn:
        clauses = ["topic_id = ?"]
        params: list[Any] = [topic_id]
        if subtopic_id is not None:
            clauses.append("subtopic_id = ?")
            params.append(subtopic_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        try:
            rows = conn.execute(
                f"SELECT * FROM ClaimCandidate WHERE {' AND '.join(clauses)} ORDER BY id ASC",
                params,
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return []
            raise
        return [dict(row) for row in rows]


def fact_candidate_exists(
    topic_id: int,
    candidate_text: str,
    statuses: Optional[Iterable[str]] = None,
) -> bool:
    with get_db() as conn:
        params: list[Any] = [topic_id, candidate_text]
        query = """
            SELECT 1
            FROM FactCandidate
            WHERE topic_id = ? AND candidate_text = ?
        """
        status_list = [status for status in (statuses or []) if status]
        if status_list:
            placeholders = ", ".join("?" for _ in status_list)
            query += f" AND status IN ({placeholders})"
            params.extend(status_list)
        query += " LIMIT 1"
        row = conn.execute(query, params).fetchone()
        return row is not None


def claim_candidate_exists(
    topic_id: int,
    candidate_text: str,
    statuses: Optional[Iterable[str]] = None,
) -> bool:
    with get_db() as conn:
        params: list[Any] = [topic_id, candidate_text]
        query = """
            SELECT 1
            FROM ClaimCandidate
            WHERE topic_id = ? AND candidate_text = ?
        """
        status_list = [status for status in (statuses or []) if status]
        if status_list:
            placeholders = ", ".join("?" for _ in status_list)
            query += f" AND status IN ({placeholders})"
            params.extend(status_list)
        query += " LIMIT 1"
        try:
            row = conn.execute(query, params).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return False
            raise
        return row is not None


def update_fact_candidate_review(
    candidate_id: int,
    status: str,
    reviewed_text: Optional[str] = None,
    review_note: Optional[str] = None,
    evidence_note: Optional[str] = None,
    confidence_score: Optional[float] = None,
    reviewer: Optional[str] = None,
    accepted_fact_id: Optional[int] = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE FactCandidate
            SET status = ?,
                reviewed_text = ?,
                review_note = ?,
                evidence_note = ?,
                confidence_score = ?,
                reviewer = ?,
                accepted_fact_id = ?,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                status,
                reviewed_text,
                review_note,
                evidence_note,
                confidence_score,
                reviewer,
                accepted_fact_id,
                candidate_id,
            ),
        )


def update_claim_candidate_review(
    candidate_id: int,
    status: str,
    reviewed_text: Optional[str] = None,
    review_note: Optional[str] = None,
    claim_score: Optional[float] = None,
    accepted_claim_id: Optional[int] = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE ClaimCandidate
            SET status = ?,
                reviewed_text = ?,
                review_note = ?,
                claim_score = ?,
                accepted_claim_id = ?,
                reviewed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                status,
                reviewed_text,
                review_note,
                claim_score,
                accepted_claim_id,
                candidate_id,
            ),
        )


def fact_exists(topic_id: int, content: str, source: Optional[str] = None) -> bool:
    with get_db() as conn:
        params: list[Any] = [topic_id, content]
        query = """
            SELECT 1
            FROM Fact
            WHERE topic_id = ? AND content = ?
        """
        if source is not None:
            query += " AND source = ?"
            params.append(source)
        query += " LIMIT 1"
        row = conn.execute(query, params).fetchone()
        return row is not None


def get_fact_by_content(topic_id: int, content: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM Fact
            WHERE topic_id = ? AND content = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (topic_id, content),
        ).fetchone()
        return dict(row) if row else None


def get_claim_by_content(topic_id: int, content: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM Claim WHERE topic_id = ? AND content = ? "
            "AND superseded_by IS NULL ORDER BY id DESC LIMIT 1",
            (topic_id, content),
        ).fetchone()
        return dict(row) if row else None


def get_claims_by_support_fact_set(
    topic_id: int,
    sorted_fact_ids_json: str,
) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM Claim WHERE topic_id = ? AND support_fact_ids_json = ? "
            "AND superseded_by IS NULL ORDER BY COALESCE(claim_score, -1) DESC, id DESC",
            (topic_id, sorted_fact_ids_json),
        ).fetchall()
        return [dict(row) for row in rows]


def merge_fact_source_ref(fact_id: int, new_refs: list[str]) -> None:
    """Append source_refs to an existing fact without duplicates."""
    import json as _json

    with get_db() as conn:
        # get_db() context manager already handles transaction
        row = conn.execute(
            "SELECT source_refs_json FROM Fact WHERE id = ?", (fact_id,)
        ).fetchone()
        if not row:
            return
        existing_raw = row["source_refs_json"] or "[]"
        try:
            existing = [x for x in _json.loads(existing_raw) if isinstance(x, str)]
        except (ValueError, TypeError):
            existing = []
        merged = list(dict.fromkeys(existing + new_refs))  # preserve order, dedup
        conn.execute(
            "UPDATE Fact SET source_refs_json = ? WHERE id = ?",
            (_json.dumps(merged, ensure_ascii=True), fact_id),
        )


def merge_claim_support_facts(claim_id: int, new_fact_ids: list[int]) -> None:
    """Merge additional support_fact_ids into an existing claim."""
    import json as _json

    with get_db() as conn:
        # get_db() context manager already handles transaction
        row = conn.execute(
            "SELECT support_fact_ids_json FROM Claim WHERE id = ?", (claim_id,)
        ).fetchone()
        if not row:
            return
        existing_raw = row["support_fact_ids_json"] or "[]"
        try:
            existing = [
                int(x) for x in _json.loads(existing_raw) if isinstance(x, (int, float))
            ]
        except (ValueError, TypeError):
            existing = []
        merged = sorted(set(existing) | set(new_fact_ids))
        conn.execute(
            "UPDATE Claim SET support_fact_ids_json = ? WHERE id = ?",
            (_json.dumps(merged, ensure_ascii=True), claim_id),
        )


def update_claim_superseded(claim_id: int, superseded_by_id: int) -> None:
    if claim_id == superseded_by_id:
        raise ValueError(f"Cannot supersede claim {claim_id} by itself")
    with get_db() as conn:
        conn.execute(
            "UPDATE Claim SET superseded_by = ?, status = 'superseded' WHERE id = ?",
            (superseded_by_id, claim_id),
        )


def get_facts_by_ids(topic_id: int, fact_ids: Iterable[int]) -> List[Dict[str, Any]]:
    ids = [int(fact_id) for fact_id in fact_ids]
    if not ids:
        return []
    placeholders = ", ".join("?" for _ in ids)
    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM Fact
            WHERE topic_id = ? AND id IN ({placeholders})
            ORDER BY id ASC
            """,
            [topic_id, *ids],
        ).fetchall()
        return [dict(row) for row in rows]


def insert_corpus_document(
    *,
    title: str,
    source_hash: str,
    topic_id: int | None = None,
    doc_type: str = "text",
    author: str | None = None,
    source_path: str | None = None,
    source_url: str | None = None,
    access_scope: str = "topic",
    parser_version: str = "text-v1",
    index_status: str = "pending",
    metadata_json: str | None = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO CorpusDocument (
                topic_id, title, doc_type, author, source_path, source_url,
                source_hash, access_scope, parser_version, index_status,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id, source_hash) DO UPDATE SET
                title = excluded.title,
                doc_type = excluded.doc_type,
                author = excluded.author,
                source_path = excluded.source_path,
                source_url = excluded.source_url,
                access_scope = excluded.access_scope,
                parser_version = excluded.parser_version,
                index_status = excluded.index_status,
                metadata_json = excluded.metadata_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                topic_id,
                title,
                doc_type,
                author,
                source_path,
                source_url,
                source_hash,
                access_scope,
                parser_version,
                index_status,
                metadata_json,
            ),
        )
        if cursor.lastrowid:
            return cursor.lastrowid
        row = conn.execute(
            "SELECT id FROM CorpusDocument WHERE topic_id IS ? AND source_hash = ?",
            (topic_id, source_hash),
        ).fetchone()
        if not row:
            raise RuntimeError("CorpusDocument upsert did not return an id")
        return int(row["id"])


def get_corpus_document(document_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM CorpusDocument WHERE id = ?", (document_id,)
        ).fetchone()
        return dict(row) if row else None


def list_corpus_documents(
    topic_id: int | None = None,
    *,
    include_global: bool = True,
    access_scope: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    if topic_id is not None:
        if include_global:
            clauses.append("(topic_id = ? OR topic_id IS NULL)")
        else:
            clauses.append("topic_id = ?")
        params.append(topic_id)
    if access_scope is not None:
        clauses.append("access_scope = ?")
        params.append(access_scope)
    query = "SELECT * FROM CorpusDocument"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def create_corpus_ingest_run(
    *,
    document_id: int | None = None,
    topic_id: int | None = None,
    source_path: str | None = None,
    parser_version: str = "text-v1",
    status: str = "running",
    metadata_json: str | None = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO CorpusIngestRun (
                document_id, topic_id, source_path, parser_version, status,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                topic_id,
                source_path,
                parser_version,
                status,
                metadata_json,
            ),
        )
        return int(cursor.lastrowid)


def update_corpus_ingest_run(
    run_id: int,
    *,
    document_id: int | None = None,
    status: str | None = None,
    error: str | None = None,
    chunk_count: int | None = None,
) -> None:
    assignments: list[str] = []
    params: list[Any] = []
    if document_id is not None:
        assignments.append("document_id = ?")
        params.append(document_id)
    if status is not None:
        assignments.append("status = ?")
        params.append(status)
    if error is not None:
        assignments.append("error = ?")
        params.append(error)
    if chunk_count is not None:
        assignments.append("chunk_count = ?")
        params.append(chunk_count)
    if status in {"completed", "failed"}:
        assignments.append("completed_at = CURRENT_TIMESTAMP")
    if not assignments:
        return
    params.append(run_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE CorpusIngestRun SET {', '.join(assignments)} WHERE id = ?",
            params,
        )


def _insert_corpus_chunk_row(
    conn: sqlite3.Connection,
    *,
    document_id: int,
    chunk_index: int,
    text: str,
    parent_chunk_id: int | None = None,
    granularity: str = "paragraph",
    section_path: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    position_start: int | None = None,
    position_end: int | None = None,
    table_markdown: str | None = None,
    lexical_text: str | None = None,
    token_count: int | None = None,
    checksum: str | None = None,
    freshness_timestamp: str | None = None,
    metadata_json: str | None = None,
) -> int:
    text = _truncate(text, MAX_CHUNK_TEXT_LEN) or ""
    table_markdown = _truncate(table_markdown, MAX_CHUNK_TEXT_LEN)
    lexical_text = _truncate(lexical_text, MAX_CHUNK_TEXT_LEN)
    checksum = checksum or hashlib.sha256(text.encode("utf-8")).hexdigest()
    cursor = conn.execute(
        """
        INSERT INTO CorpusChunk (
            document_id, parent_chunk_id, chunk_index, granularity, section_path,
            page_start, page_end, position_start, position_end, text,
            table_markdown, lexical_text, token_count, checksum,
            freshness_timestamp, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id, chunk_index) DO UPDATE SET
            parent_chunk_id = excluded.parent_chunk_id,
            granularity = excluded.granularity,
            section_path = excluded.section_path,
            page_start = excluded.page_start,
            page_end = excluded.page_end,
            position_start = excluded.position_start,
            position_end = excluded.position_end,
            text = excluded.text,
            table_markdown = excluded.table_markdown,
            lexical_text = excluded.lexical_text,
            token_count = excluded.token_count,
            checksum = excluded.checksum,
            freshness_timestamp = excluded.freshness_timestamp,
            metadata_json = excluded.metadata_json
        """,
        (
            document_id,
            parent_chunk_id,
            chunk_index,
            granularity,
            section_path,
            page_start,
            page_end,
            position_start,
            position_end,
            text,
            table_markdown,
            lexical_text,
            token_count,
            checksum,
            freshness_timestamp,
            metadata_json,
        ),
    )
    if cursor.lastrowid:
        chunk_id = int(cursor.lastrowid)
    else:
        row = conn.execute(
            "SELECT id FROM CorpusChunk WHERE document_id = ? AND chunk_index = ?",
            (document_id, chunk_index),
        ).fetchone()
        if not row:
            raise RuntimeError("CorpusChunk upsert did not return an id")
        chunk_id = int(row["id"])
    doc = conn.execute(
        "SELECT topic_id, doc_type, access_scope, title FROM CorpusDocument WHERE id = ?",
        (document_id,),
    ).fetchone()
    fts_content = "\n".join(
        part
        for part in (
            doc["title"] if doc else "",
            section_path or "",
            lexical_text or text,
            table_markdown or "",
        )
        if part
    )
    _insert_corpus_chunk_fts(
        conn,
        chunk_id,
        document_id,
        doc["topic_id"] if doc else None,
        doc["doc_type"] if doc else None,
        doc["access_scope"] if doc else None,
        fts_content,
    )
    return chunk_id


def insert_corpus_chunk(
    *,
    document_id: int,
    chunk_index: int,
    text: str,
    parent_chunk_id: int | None = None,
    granularity: str = "paragraph",
    section_path: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    position_start: int | None = None,
    position_end: int | None = None,
    table_markdown: str | None = None,
    lexical_text: str | None = None,
    token_count: int | None = None,
    checksum: str | None = None,
    freshness_timestamp: str | None = None,
    metadata_json: str | None = None,
) -> int:
    with get_db() as conn:
        return _insert_corpus_chunk_row(
            conn,
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            parent_chunk_id=parent_chunk_id,
            granularity=granularity,
            section_path=section_path,
            page_start=page_start,
            page_end=page_end,
            position_start=position_start,
            position_end=position_end,
            table_markdown=table_markdown,
            lexical_text=lexical_text,
            token_count=token_count,
            checksum=checksum,
            freshness_timestamp=freshness_timestamp,
            metadata_json=metadata_json,
        )


def insert_corpus_chunk_with_embedding(
    *,
    document_id: int,
    chunk_index: int,
    text: str,
    embedding: list[float] | None,
    **kwargs: Any,
) -> int:
    global _sqlite_vec_available
    with get_db() as conn:
        chunk_id = _insert_corpus_chunk_row(
            conn,
            document_id=document_id,
            chunk_index=chunk_index,
            text=text,
            **kwargs,
        )
        if embedding is not None and _sqlite_vec_available:
            try:
                conn.execute(
                    "DELETE FROM vec_corpus_chunks WHERE chunk_id = ?", (chunk_id,)
                )
                conn.execute(
                    "INSERT INTO vec_corpus_chunks(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_f32(embedding)),
                )
            except sqlite3.OperationalError as exc:
                _sqlite_vec_available = False
                logger.debug("corpus semantic index unavailable: %s", exc)
        return chunk_id


def get_corpus_chunks_for_document(document_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM CorpusChunk WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def reindex_corpus_document(document_id: int) -> int:
    """Rebuild lexical FTS rows for an existing corpus document."""
    with get_db() as conn:
        doc = conn.execute(
            "SELECT * FROM CorpusDocument WHERE id = ?", (document_id,)
        ).fetchone()
        if not doc:
            return 0
        chunks = conn.execute(
            "SELECT * FROM CorpusChunk WHERE document_id = ? ORDER BY chunk_index ASC",
            (document_id,),
        ).fetchall()
        for chunk in chunks:
            conn.execute("DELETE FROM corpus_chunks_fts WHERE rowid = ?", (chunk["id"],))
            fts_content = "\n".join(
                part
                for part in (
                    doc["title"],
                    chunk["section_path"],
                    chunk["lexical_text"] or chunk["text"],
                    chunk["table_markdown"],
                )
                if part
            )
            _insert_corpus_chunk_fts(
                conn,
                int(chunk["id"]),
                int(document_id),
                doc["topic_id"],
                doc["doc_type"],
                doc["access_scope"],
                fts_content,
            )
        conn.execute(
            """
            UPDATE CorpusDocument
            SET index_status = 'indexed',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (document_id,),
        )
        return len(chunks)


def _corpus_topic_filter(
    topic_id: int | None,
    *,
    include_global: bool = True,
    table_alias: str = "CorpusDocument",
) -> tuple[str, list[Any]]:
    if topic_id is None:
        return "", []
    if include_global:
        return f" AND ({table_alias}.topic_id = ? OR {table_alias}.topic_id IS NULL)", [
            topic_id
        ]
    return f" AND {table_alias}.topic_id = ?", [topic_id]


def search_corpus_chunks(
    topic_id: int | None,
    query_embedding: list[float],
    top_k: int = 8,
    *,
    include_global: bool = True,
) -> list[dict]:
    if not query_embedding:
        return []
    filter_sql, filter_params = _corpus_topic_filter(
        topic_id, include_global=include_global
    )
    query = (
        """
        SELECT
            CorpusChunk.*,
            CorpusDocument.topic_id,
            CorpusDocument.title AS document_title,
            CorpusDocument.doc_type,
            CorpusDocument.source_path,
            CorpusDocument.source_url,
            CorpusDocument.access_scope,
            CorpusChunk.text AS content,
            vec_distance_L2(vec_corpus_chunks.embedding, ?) AS distance
        FROM vec_corpus_chunks
        JOIN CorpusChunk ON CorpusChunk.id = vec_corpus_chunks.chunk_id
        JOIN CorpusDocument ON CorpusDocument.id = CorpusChunk.document_id
        WHERE CorpusDocument.index_status = 'indexed'
        """
        + filter_sql
        + """
        ORDER BY distance
        LIMIT ?
        """
    )
    params = [serialize_f32(query_embedding), *filter_params, top_k]
    with get_db() as conn:
        try:
            rows = conn.execute(query, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("corpus semantic search unavailable: %s", exc)
            return []
        return [dict(row) for row in rows]


def search_corpus_chunks_lexical(
    topic_id: int | None,
    query_text: str,
    top_k: int = 8,
    *,
    include_global: bool = True,
) -> list[dict]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []
    filter_sql, filter_params = _corpus_topic_filter(
        topic_id, include_global=include_global
    )
    query = (
        """
        SELECT
            CorpusChunk.*,
            CorpusDocument.topic_id,
            CorpusDocument.title AS document_title,
            CorpusDocument.doc_type,
            CorpusDocument.source_path,
            CorpusDocument.source_url,
            CorpusDocument.access_scope,
            CorpusChunk.text AS content,
            bm25(corpus_chunks_fts) AS lexical_score
        FROM corpus_chunks_fts
        JOIN CorpusChunk ON CorpusChunk.id = corpus_chunks_fts.rowid
        JOIN CorpusDocument ON CorpusDocument.id = CorpusChunk.document_id
        WHERE corpus_chunks_fts MATCH ?
          AND CorpusDocument.index_status = 'indexed'
        """
        + filter_sql
        + """
        ORDER BY lexical_score
        LIMIT ?
        """
    )
    params = [match_query, *filter_params, top_k]
    with get_db() as conn:
        try:
            rows = conn.execute(query, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("corpus lexical search unavailable: %s", exc)
            return []
        return [dict(row) for row in rows]


def get_corpus_neighbor_chunks(chunk_id: int, window: int = 1) -> list[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT document_id, chunk_index FROM CorpusChunk WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if not row:
            return []
        rows = conn.execute(
            """
            SELECT
                CorpusChunk.*,
                CorpusDocument.topic_id,
                CorpusDocument.title AS document_title,
                CorpusDocument.doc_type,
                CorpusDocument.source_path,
                CorpusDocument.source_url,
                CorpusDocument.access_scope,
                CorpusChunk.text AS content
            FROM CorpusChunk
            JOIN CorpusDocument ON CorpusDocument.id = CorpusChunk.document_id
            WHERE CorpusChunk.document_id = ?
              AND CorpusChunk.chunk_index BETWEEN ? AND ?
            ORDER BY CorpusChunk.chunk_index ASC
            """,
            (
                row["document_id"],
                row["chunk_index"] - int(window),
                row["chunk_index"] + int(window),
            ),
        ).fetchall()
        return [dict(item) for item in rows]


def insert_optimization_problem(
    *,
    topic_id: int,
    title: str,
    source_text: str,
    subtopic_id: int | None = None,
    problem_class: str | None = None,
    domain_context: str | None = None,
    stakeholder: str | None = None,
    time_horizon: str | None = None,
    status: str = "candidate",
    source_refs_json: str | None = None,
    metadata_json: str | None = None,
    created_by: str | None = None,
) -> int:
    source_text = _truncate(source_text, MAX_CONTENT_LEN) or ""
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO OptimizationProblem (
                topic_id, subtopic_id, title, source_text, problem_class,
                domain_context, stakeholder, time_horizon, status,
                source_refs_json, metadata_json, created_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                title,
                source_text,
                problem_class,
                domain_context,
                stakeholder,
                time_horizon,
                status,
                source_refs_json,
                metadata_json,
                created_by,
            ),
        )
        return int(cursor.lastrowid)


def get_optimization_problem(problem_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM OptimizationProblem WHERE id = ?", (problem_id,)
        ).fetchone()
        return dict(row) if row else None


def list_optimization_problems(topic_id: int, limit: int | None = None) -> list[dict]:
    params: list[Any] = [topic_id]
    query = "SELECT * FROM OptimizationProblem WHERE topic_id = ? ORDER BY id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def insert_optimization_component(
    *,
    problem_id: int,
    topic_id: int,
    component_type: str,
    natural_text: str,
    formal_text: str | None = None,
    symbol: str | None = None,
    unit: str | None = None,
    domain: str | None = None,
    source_refs_json: str | None = None,
    review_status: str = "candidate",
    validation_notes: str | None = None,
    metadata_json: str | None = None,
) -> int:
    natural_text = _truncate(natural_text, MAX_CONTENT_LEN) or ""
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO OptimizationComponent (
                problem_id, topic_id, component_type, natural_text, formal_text,
                symbol, unit, domain, source_refs_json, review_status,
                validation_notes, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                problem_id,
                topic_id,
                component_type,
                natural_text,
                formal_text,
                symbol,
                unit,
                domain,
                source_refs_json,
                review_status,
                validation_notes,
                metadata_json,
            ),
        )
        return int(cursor.lastrowid)


def get_optimization_components(
    problem_id: int,
    *,
    component_type: str | None = None,
    review_status: str | None = None,
) -> list[dict]:
    clauses = ["problem_id = ?"]
    params: list[Any] = [problem_id]
    if component_type is not None:
        clauses.append("component_type = ?")
        params.append(component_type)
    if review_status is not None:
        clauses.append("review_status = ?")
        params.append(review_status)
    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM OptimizationComponent
            WHERE {' AND '.join(clauses)}
            ORDER BY id ASC
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]


def update_optimization_component_review(
    component_id: int,
    *,
    review_status: str,
    validation_notes: str | None = None,
) -> bool:
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE OptimizationComponent
            SET review_status = ?,
                validation_notes = COALESCE(?, validation_notes),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (review_status, validation_notes, component_id),
        )
        return cursor.rowcount > 0


def insert_optimization_model_ir(
    *,
    problem_id: int,
    topic_id: int,
    ir_json: str,
    status: str = "candidate",
    validation_notes: str | None = None,
    linked_component_ids_json: str | None = None,
    component_fingerprints_json: str | None = None,
    generator_role: str | None = None,
) -> int:
    ir_json = _truncate(ir_json, MAX_CONTENT_LEN) or "{}"
    ir_hash = hashlib.sha256(ir_json.encode("utf-8")).hexdigest()
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO OptimizationModelIR (
                problem_id, topic_id, ir_json, ir_hash, status, validation_notes,
                linked_component_ids_json, component_fingerprints_json, generator_role
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                problem_id,
                topic_id,
                ir_json,
                ir_hash,
                status,
                validation_notes,
                linked_component_ids_json,
                component_fingerprints_json,
                generator_role,
            ),
        )
        return int(cursor.lastrowid)


def get_optimization_model_irs(problem_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM OptimizationModelIR WHERE problem_id = ? ORDER BY id DESC",
            (problem_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def insert_optimization_artifact(
    *,
    problem_id: int,
    topic_id: int,
    artifact_type: str,
    model_language: str,
    content: str,
    parser_status: str = "pending",
    parser_notes: str | None = None,
    linked_component_ids_json: str | None = None,
    component_fingerprints_json: str | None = None,
    generator_role: str | None = None,
    source_artifact_id: int | None = None,
    repair_status: str | None = None,
) -> int:
    content = _truncate(content, MAX_CONTENT_LEN) or ""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO OptimizationArtifact (
                problem_id, topic_id, artifact_type, model_language, content,
                content_hash, parser_status, parser_notes,
                linked_component_ids_json, component_fingerprints_json,
                generator_role, source_artifact_id, repair_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                problem_id,
                topic_id,
                artifact_type,
                model_language,
                content,
                content_hash,
                parser_status,
                parser_notes,
                linked_component_ids_json,
                component_fingerprints_json,
                generator_role,
                source_artifact_id,
                repair_status,
            ),
        )
        return int(cursor.lastrowid)


def get_optimization_artifacts(problem_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM OptimizationArtifact WHERE problem_id = ? ORDER BY id DESC",
            (problem_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def insert_solver_run(
    *,
    artifact_id: int,
    problem_id: int,
    topic_id: int,
    solver_backend: str,
    status: str,
    objective_value: float | None = None,
    variable_values_json: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    error_trace: str | None = None,
    elapsed_time_s: float | None = None,
    code_evidence_id: int | None = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO SolverRun (
                artifact_id, problem_id, topic_id, solver_backend, status,
                objective_value, variable_values_json, stdout, stderr,
                error_trace, elapsed_time_s, code_evidence_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                problem_id,
                topic_id,
                solver_backend,
                status,
                objective_value,
                variable_values_json,
                stdout,
                stderr,
                error_trace,
                elapsed_time_s,
                code_evidence_id,
            ),
        )
        return int(cursor.lastrowid)


def get_solver_runs(problem_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM SolverRun WHERE problem_id = ? ORDER BY id DESC",
            (problem_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def insert_model_diagnostic(
    *,
    problem_id: int,
    topic_id: int,
    diagnostic_type: str,
    message: str,
    severity: str = "warning",
    component_id: int | None = None,
    artifact_id: int | None = None,
    solver_run_id: int | None = None,
    status: str = "open",
    source_refs_json: str | None = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ModelDiagnostic (
                problem_id, topic_id, component_id, artifact_id, solver_run_id,
                diagnostic_type, severity, message, status, source_refs_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                problem_id,
                topic_id,
                component_id,
                artifact_id,
                solver_run_id,
                diagnostic_type,
                severity,
                message,
                status,
                source_refs_json,
            ),
        )
        return int(cursor.lastrowid)


def get_model_diagnostics(
    problem_id: int, *, status: str | None = None
) -> list[dict]:
    params: list[Any] = [problem_id]
    query = "SELECT * FROM ModelDiagnostic WHERE problem_id = ?"
    if status is not None:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY id ASC"
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def update_model_diagnostic_status(
    diagnostic_id: int,
    *,
    status: str,
    resolution: str | None = None,
) -> bool:
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE ModelDiagnostic
            SET status = ?,
                resolved_at = CASE
                    WHEN ? IN ('resolved', 'dismissed', 'closed')
                    THEN CURRENT_TIMESTAMP
                    ELSE resolved_at
                END,
                message = CASE
                    WHEN ? IS NULL OR ? = '' THEN message
                    ELSE message || char(10) || 'Resolution: ' || ?
                END
            WHERE id = ?
            """,
            (status, status, resolution, resolution, resolution, diagnostic_id),
        )
        return cursor.rowcount > 0


def upsert_modeling_experience(
    *,
    family: str,
    structure_key: str,
    content: str,
    topic_id: int | None = None,
    applies_when_json: str | None = None,
    rejects_when_json: str | None = None,
    source_refs_json: str | None = None,
    status: str = "candidate",
    validation_summary_json: str | None = None,
) -> int:
    content = _truncate(content, MAX_CONTENT_LEN) or ""
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ModelingExperience (
                topic_id, family, structure_key, content, applies_when_json,
                rejects_when_json, source_refs_json, status,
                validation_summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(family, structure_key) DO UPDATE SET
                topic_id = COALESCE(excluded.topic_id, ModelingExperience.topic_id),
                content = excluded.content,
                applies_when_json = COALESCE(
                    excluded.applies_when_json,
                    ModelingExperience.applies_when_json
                ),
                rejects_when_json = COALESCE(
                    excluded.rejects_when_json,
                    ModelingExperience.rejects_when_json
                ),
                source_refs_json = COALESCE(
                    excluded.source_refs_json,
                    ModelingExperience.source_refs_json
                ),
                status = excluded.status,
                validation_summary_json = COALESCE(
                    excluded.validation_summary_json,
                    ModelingExperience.validation_summary_json
                ),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (
                topic_id,
                family,
                structure_key,
                content,
                applies_when_json,
                rejects_when_json,
                source_refs_json,
                status,
                validation_summary_json,
            ),
        )
        return int(cursor.fetchone()["id"])


def list_modeling_experiences(
    *,
    family: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    if family is not None:
        clauses.append("family = ?")
        params.append(family)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    query = "SELECT * FROM ModelingExperience"
    if clauses:
        query += f" WHERE {' AND '.join(clauses)}"
    query += " ORDER BY updated_at DESC, id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def record_modeling_experience_event(
    experience_id: int,
    *,
    outcome: str,
    validation_summary_json: str | None = None,
) -> bool:
    normalized = outcome.strip().lower()
    if normalized not in {"success", "failure"}:
        raise ValueError("outcome must be 'success' or 'failure'")
    success_increment = 1 if normalized == "success" else 0
    failure_increment = 1 if normalized == "failure" else 0
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE ModelingExperience
            SET success_count = success_count + ?,
                failure_count = failure_count + ?,
                validation_summary_json = COALESCE(?, validation_summary_json),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                success_increment,
                failure_increment,
                validation_summary_json,
                experience_id,
            ),
        )
        return cursor.rowcount > 0


def insert_claim(
    topic_id: int,
    subtopic_id: Optional[int],
    content: str,
    summary: Optional[str] = None,
    support_fact_ids_json: Optional[str] = None,
    rationale_short: Optional[str] = None,
    claim_score: Optional[float] = None,
    status: str = "active",
    candidate_id: Optional[int] = None,
    # G.4: Structured claim fields
    claim_type: Optional[str] = None,
    scope_tags: Optional[str] = None,
    scope_context: Optional[str] = None,
    falsification_criteria: Optional[str] = None,
    inference_logic: Optional[str] = None,
    conclusion: Optional[str] = None,
    evidence_strength: Optional[float] = None,
    scope_breadth: Optional[float] = None,
    submitted_by: Optional[str] = None,
) -> int:
    content = _truncate(content, MAX_CONTENT_LEN)
    summary = _truncate(summary, MAX_SUMMARY_LEN)
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO Claim (
                topic_id, subtopic_id, content, summary,
                support_fact_ids_json, rationale_short, claim_score,
                status, candidate_id,
                claim_type, scope_tags, scope_context,
                falsification_criteria, inference_logic, conclusion,
                evidence_strength, scope_breadth, submitted_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                content,
                summary,
                support_fact_ids_json,
                rationale_short,
                claim_score,
                status,
                candidate_id,
                claim_type,
                scope_tags,
                scope_context,
                falsification_criteria,
                inference_logic,
                conclusion,
                evidence_strength,
                scope_breadth,
                submitted_by,
            ),
        )
        claim_id = cursor.lastrowid
        fts_content = content
        if summary:
            fts_content = f"{summary}\n\n{content}"
        _insert_claim_fts(conn, claim_id, topic_id, fts_content)
        return claim_id


def insert_claim_and_supersede(
    topic_id: int,
    subtopic_id: Optional[int],
    content: str,
    *,
    supersede_claim_id: int,
    summary: Optional[str] = None,
    support_fact_ids_json: Optional[str] = None,
    rationale_short: Optional[str] = None,
    claim_score: Optional[float] = None,
    status: str = "active",
    candidate_id: Optional[int] = None,
    # G.4: Structured claim fields
    claim_type: Optional[str] = None,
    scope_tags: Optional[str] = None,
    scope_context: Optional[str] = None,
    falsification_criteria: Optional[str] = None,
    inference_logic: Optional[str] = None,
    conclusion: Optional[str] = None,
    evidence_strength: Optional[float] = None,
    scope_breadth: Optional[float] = None,
    submitted_by: Optional[str] = None,
) -> int:
    """Atomically insert a new claim and supersede the old one in a single transaction."""
    content = _truncate(content, MAX_CONTENT_LEN)
    summary = _truncate(summary, MAX_SUMMARY_LEN)
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO Claim (
                topic_id, subtopic_id, content, summary,
                support_fact_ids_json, rationale_short,
                claim_score, status, candidate_id,
                claim_type, scope_tags, scope_context,
                falsification_criteria, inference_logic, conclusion,
                evidence_strength, scope_breadth, submitted_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                content,
                summary,
                support_fact_ids_json,
                rationale_short,
                claim_score,
                status,
                candidate_id,
                claim_type,
                scope_tags,
                scope_context,
                falsification_criteria,
                inference_logic,
                conclusion,
                evidence_strength,
                scope_breadth,
                submitted_by,
            ),
        )
        new_id = cursor.lastrowid
        fts_content = content
        if summary:
            fts_content = f"{summary}\n\n{content}"
        _insert_claim_fts(conn, new_id, topic_id, fts_content)
        conn.execute(
            "UPDATE Claim SET superseded_by = ?, status = 'superseded' WHERE id = ?",
            (new_id, supersede_claim_id),
        )
        return new_id


def _snippet_hash(snippet: str) -> Optional[str]:
    """SHA-256 of normalized snippet for dedup."""
    if not snippet or not snippet.strip():
        return None
    return hashlib.sha256(snippet.strip().encode("utf-8")).hexdigest()


def _find_web_evidence_by_snippet_hash(
    conn: sqlite3.Connection, origin_topic_id: int, sh: str
) -> Optional[int]:
    """Return existing WebEvidence ID if snippet hash matches within topic."""
    if not sh:
        return None
    row = conn.execute(
        "SELECT id FROM WebEvidence WHERE origin_topic_id = ? AND snippet_hash = ? LIMIT 1",
        (origin_topic_id, sh),
    ).fetchone()
    return row[0] if row else None


_LOW_VALUE_DOMAINS = {
    "linkedin.com",
    "youtube.com",
    "pinterest.com",
    "quora.com",
    "reddit.com",
}


def insert_web_evidence(
    origin_topic_id: int,
    origin_subtopic_id: Optional[int],
    query_text: str,
    title: str,
    snippet: str,
    url: str,
    source_domain: str,
    result_rank: int,
    search_provider: str,
    search_role: str,
    summary: Optional[str] = None,
) -> int | None:
    normalized_domain = (source_domain or "").lower().removeprefix("www.")
    if normalized_domain in _LOW_VALUE_DOMAINS:
        logger.info("[WE-4] Blocked low-value domain: %s", source_domain)
        return None
    with get_db() as conn:
        # URL dedup: return existing ID if same URL already stored for topic
        # Merge snippets if the new one is different from the existing one
        if url:
            existing = conn.execute(
                "SELECT id, snippet FROM WebEvidence WHERE origin_topic_id = ? AND url = ? LIMIT 1",
                (origin_topic_id, url),
            ).fetchone()
            if existing:
                existing_id = existing[0]
                existing_snippet = existing[1] or ""
                if (
                    snippet
                    and snippet.strip()
                    and snippet.strip() not in existing_snippet
                ):
                    merged = existing_snippet + "\n\n" + snippet.strip()
                    conn.execute(
                        "UPDATE WebEvidence SET snippet = ? WHERE id = ?",
                        (merged[:4000], existing_id),
                    )
                return existing_id

        # Snippet dedup: same content from different URL
        sh = _snippet_hash(snippet)
        if sh:
            existing_by_snippet = _find_web_evidence_by_snippet_hash(
                conn, origin_topic_id, sh
            )
            if existing_by_snippet is not None:
                return existing_by_snippet

        cursor = conn.execute(
            """
            INSERT INTO WebEvidence (
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
                summary,
                snippet_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
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
                summary,
                sh,
            ),
        )
        web_id = cursor.lastrowid
        content = "\n\n".join(
            part.strip()
            for part in (
                title or "",
                summary or "",
                snippet or "",
                query_text or "",
                source_domain or "",
            )
            if isinstance(part, str) and part.strip()
        )
        _insert_web_evidence_fts(
            conn, web_id, origin_topic_id, source_domain or "", content
        )
        return web_id


def clone_web_evidence_to_topic(
    source_rows: list[dict],
    target_topic_id: int,
    target_subtopic_id: int | None = None,
) -> dict[int, int]:
    """Clone cross-topic WebEvidence rows into target topic.

    Returns {old_id: new_id}. Deduplicates by URL and snippet_hash.
    Preserves original ``fetched_at`` to avoid TTL refresh.
    """
    id_map: dict[int, int] = {}
    with get_db() as conn:
        for row in source_rows:
            old_id = row.get("id")
            if old_id is None:
                continue
            url = row.get("url") or ""
            snippet = row.get("snippet") or ""
            title = row.get("title") or ""
            summary = row.get("summary")
            source_domain = row.get("source_domain") or ""

            # URL dedup
            if url:
                existing = conn.execute(
                    "SELECT id, snippet FROM WebEvidence WHERE origin_topic_id = ? AND url = ? LIMIT 1",
                    (target_topic_id, url),
                ).fetchone()
                if existing:
                    existing_id = existing[0]
                    existing_snippet = existing[1] or ""
                    stripped_snippet = snippet.strip()
                    if stripped_snippet and stripped_snippet not in existing_snippet:
                        merged = existing_snippet + "\n\n" + stripped_snippet
                        conn.execute(
                            "UPDATE WebEvidence SET snippet = ? WHERE id = ?",
                            (merged[:4000], existing_id),
                        )
                    id_map[old_id] = existing_id
                    continue

            # Snippet hash dedup
            sh = _snippet_hash(snippet)
            if sh:
                existing_by_snippet = _find_web_evidence_by_snippet_hash(
                    conn, target_topic_id, sh
                )
                if existing_by_snippet is not None:
                    id_map[old_id] = existing_by_snippet
                    continue

            # Insert clone with preserved fetched_at
            fetched_at = row.get("fetched_at")
            cursor = conn.execute(
                """
                INSERT INTO WebEvidence (
                    origin_topic_id, origin_subtopic_id, query_text, title,
                    snippet, url, source_domain, result_rank, search_provider,
                    search_role, summary, snippet_hash, ledger_processed,
                    verified, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, COALESCE(?, CURRENT_TIMESTAMP))
                """,
                (
                    target_topic_id,
                    target_subtopic_id,
                    row.get("query_text") or "",
                    title,
                    snippet,
                    url,
                    source_domain,
                    row.get("result_rank"),
                    row.get("search_provider") or "",
                    row.get("search_role") or "",
                    summary,
                    sh,
                    fetched_at,
                ),
            )
            new_id = cursor.lastrowid
            if new_id is None:
                continue
            id_map[old_id] = new_id

            content = "\n\n".join(
                part.strip()
                for part in (
                    title,
                    summary or "",
                    snippet,
                    row.get("query_text") or "",
                    source_domain,
                )
                if isinstance(part, str) and part.strip()
            )
            _insert_web_evidence_fts(
                conn, new_id, target_topic_id, source_domain, content
            )

    return id_map


def insert_vote_record(
    topic_id: int,
    subtopic_id: Optional[int],
    round_number: Optional[int],
    vote_kind: str,
    subject: str,
    prompt_text: str,
    voter: str,
    parsed_ok: bool,
    decision: Optional[str],
    reason: Optional[str],
    raw_response: str,
    metadata_json: Optional[str] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO VoteRecord (
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
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                round_number,
                vote_kind,
                subject,
                prompt_text,
                voter,
                int(parsed_ok),
                decision,
                reason,
                raw_response,
                metadata_json,
            ),
        )
        return cursor.lastrowid


def get_vote_records(
    topic_id: int,
    *,
    subtopic_id: Optional[int] = None,
    vote_kind: Optional[str] = None,
    round_number: Optional[int] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    clauses = ["topic_id = ?"]
    params: list[Any] = [topic_id]
    if subtopic_id is not None:
        clauses.append("subtopic_id = ?")
        params.append(subtopic_id)
    if vote_kind is not None:
        clauses.append("vote_kind = ?")
        params.append(vote_kind)
    if round_number is not None:
        clauses.append("round_number = ?")
        params.append(round_number)

    query = f"SELECT * FROM VoteRecord WHERE {' AND '.join(clauses)} ORDER BY id ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def search_facts(
    topic_id: int, query_embedding: List[float], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Search for the most semantically similar facts using sqlite-vec."""
    with get_db() as conn:
        query = """
            SELECT Fact.*, vec_distance_L2(vec_facts.embedding, ?) as distance
            FROM vec_facts
            JOIN Fact ON Fact.id = vec_facts.fact_id
            WHERE Fact.topic_id = ? AND (Fact.review_status IS NULL OR Fact.review_status NOT IN ('superseded', 'retired'))
            ORDER BY distance
            LIMIT ?
        """
        try:
            rows = conn.execute(
                query, (serialize_f32(query_embedding), topic_id, top_k)
            ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("fact semantic search unavailable: %s", exc)
            return []
        return [dict(row) for row in rows]


def search_facts_lexical(
    topic_id: int, query_text: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT Fact.*, bm25(facts_fts) as lexical_score
            FROM facts_fts
            JOIN Fact ON Fact.id = facts_fts.rowid
            WHERE facts_fts MATCH ? AND Fact.topic_id = ?
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, topic_id, top_k),
        ).fetchall()
        return [dict(row) for row in rows]


def search_claims_lexical(
    topic_id: int, query_text: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT Claim.*, bm25(claims_fts) as lexical_score
            FROM claims_fts
            JOIN Claim ON Claim.id = claims_fts.rowid
            WHERE claims_fts MATCH ? AND Claim.topic_id = ?
              AND Claim.superseded_by IS NULL
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, topic_id, top_k),
        ).fetchall()
        return [dict(row) for row in rows]


def search_web_evidence_same_topic(
    topic_id: int,
    query_text: str,
    top_k: int = 5,
    max_age_days: int = 30,
) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                WebEvidence.*,
                TRIM(
                    COALESCE(WebEvidence.title, '') || ' ' ||
                    COALESCE(WebEvidence.snippet, '') || ' ' ||
                    COALESCE(WebEvidence.query_text, '') || ' ' ||
                    COALESCE(WebEvidence.source_domain, '')
                ) AS content,
                bm25(web_evidence_fts) AS lexical_score
            FROM web_evidence_fts
            JOIN WebEvidence ON WebEvidence.id = web_evidence_fts.rowid
            WHERE web_evidence_fts MATCH ?
              AND WebEvidence.origin_topic_id = ?
              AND WebEvidence.fetched_at >= datetime('now', ?)
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, topic_id, f"-{int(max_age_days)} days", top_k),
        ).fetchall()
        return [dict(row) for row in rows]


def search_web_evidence_cross_topic(
    topic_id: int,
    query_text: str,
    top_k: int = 5,
    max_age_days: int = 30,
) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                WebEvidence.*,
                TRIM(
                    COALESCE(WebEvidence.title, '') || ' ' ||
                    COALESCE(WebEvidence.snippet, '') || ' ' ||
                    COALESCE(WebEvidence.query_text, '') || ' ' ||
                    COALESCE(WebEvidence.source_domain, '')
                ) AS content,
                bm25(web_evidence_fts) AS lexical_score
            FROM web_evidence_fts
            JOIN WebEvidence ON WebEvidence.id = web_evidence_fts.rowid
            WHERE web_evidence_fts MATCH ?
              AND WebEvidence.origin_topic_id != ?
              AND WebEvidence.fetched_at >= datetime('now', ?)
            ORDER BY lexical_score
            LIMIT ?
            """,
            (match_query, topic_id, f"-{int(max_age_days)} days", top_k),
        ).fetchall()
        return [dict(row) for row in rows]


def insert_message_with_embedding(
    topic_id: int,
    subtopic_id: int,
    sender: str,
    content: str,
    msg_type: str,
    embedding: List[float] = None,
    confidence_score: Optional[float] = None,
    round_number: Optional[int] = None,
    turn_kind: Optional[str] = None,
    summary: Optional[str] = None,
) -> int:
    """Insert a message and its embedding."""
    global _sqlite_vec_available
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO Message (topic_id, subtopic_id, sender, content, summary, msg_type, confidence_score, round_number, turn_kind)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                sender,
                content,
                summary,
                msg_type,
                confidence_score,
                round_number,
                turn_kind,
            ),
        )
        msg_id = cursor.lastrowid
        # For FTS, we combine content and summary
        fts_content = content
        if summary:
            fts_content = f"{summary}\n\n{content}"

        _insert_message_fts(conn, msg_id, topic_id, sender, fts_content, msg_type)

        if embedding is not None and _sqlite_vec_available:
            try:
                conn.execute(
                    "INSERT INTO vec_messages(msg_id, embedding) VALUES (?, ?)",
                    (msg_id, serialize_f32(embedding)),
                )
            except sqlite3.OperationalError as exc:
                _sqlite_vec_available = False
                logger.debug("message semantic index unavailable: %s", exc)
        return msg_id


def get_messages_since(
    topic_id: int, subtopic_id: int, since_id: int, msg_type: str = "standard"
) -> list[dict]:
    """Return messages newer than *since_id* in chronological order."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, topic_id, subtopic_id, sender, content, msg_type, "
            "round_number, turn_kind, confidence_score, has_formal_claim "
            "FROM Message WHERE topic_id = ? AND subtopic_id = ? AND id > ? AND msg_type = ? ORDER BY id ASC",
            (topic_id, subtopic_id, since_id, msg_type),
        ).fetchall()
        return [dict(r) for r in rows]


def get_max_round_number(topic_id: int, subtopic_id: int) -> int:
    """Return the highest round_number for standard messages, or 0 if none."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT MAX(round_number) AS max_rn FROM Message "
            "WHERE topic_id = ? AND subtopic_id = ? AND msg_type = 'standard'",
            (topic_id, subtopic_id),
        ).fetchone()
        return row["max_rn"] or 0


def _exclude_clause(
    column_name: str, values: Optional[Iterable[int]]
) -> tuple[str, list[int]]:
    items = [int(v) for v in values or []]
    if not items:
        return "", []
    placeholders = ", ".join("?" for _ in items)
    return f" AND {column_name} NOT IN ({placeholders})", items


def search_messages(
    topic_id: int,
    query_embedding: List[float],
    msg_type: str = None,
    top_k: int = 5,
    exclude_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    """Search for semantically similar messages (e.g. summaries)."""
    with get_db() as conn:
        exclude_sql, exclude_params = _exclude_clause("Message.id", exclude_ids)
        if msg_type:
            query = (
                """
                SELECT Message.*, vec_distance_L2(vec_messages.embedding, ?) as distance
                FROM vec_messages
                JOIN Message ON Message.id = vec_messages.msg_id
                WHERE Message.topic_id = ? AND Message.msg_type = ?
            """
                + exclude_sql
                + """
                ORDER BY distance
                LIMIT ?
            """
            )
            params = [
                serialize_f32(query_embedding),
                topic_id,
                msg_type,
                *exclude_params,
                top_k,
            ]
        else:
            query = (
                """
                SELECT Message.*, vec_distance_L2(vec_messages.embedding, ?) as distance
                FROM vec_messages
                JOIN Message ON Message.id = vec_messages.msg_id
                WHERE Message.topic_id = ?
            """
                + exclude_sql
                + """
                ORDER BY distance
                LIMIT ?
            """
            )
            params = [serialize_f32(query_embedding), topic_id, *exclude_params, top_k]

        try:
            rows = conn.execute(query, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("message semantic search unavailable: %s", exc)
            return []
        return [dict(row) for row in rows]


def search_messages_lexical(
    topic_id: int,
    query_text: str,
    msg_type: str = None,
    top_k: int = 5,
    exclude_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    match_query = _build_fts_query(query_text)
    if not match_query:
        return []

    with get_db() as conn:
        exclude_sql, exclude_params = _exclude_clause("Message.id", exclude_ids)
        if msg_type:
            query = (
                """
                SELECT Message.*, bm25(messages_fts) as lexical_score
                FROM messages_fts
                JOIN Message ON Message.id = messages_fts.rowid
                WHERE messages_fts MATCH ? AND Message.topic_id = ? AND Message.msg_type = ?
            """
                + exclude_sql
                + """
                ORDER BY lexical_score
                LIMIT ?
            """
            )
            params = [match_query, topic_id, msg_type, *exclude_params, top_k]
        else:
            query = (
                """
                SELECT Message.*, bm25(messages_fts) as lexical_score
                FROM messages_fts
                JOIN Message ON Message.id = messages_fts.rowid
                WHERE messages_fts MATCH ? AND Message.topic_id = ?
            """
                + exclude_sql
                + """
                ORDER BY lexical_score
                LIMIT ?
            """
            )
            params = [match_query, topic_id, *exclude_params, top_k]

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def update_plan_cursor(plan_id: int, current_index: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Plan SET current_index = ? WHERE id = ?", (current_index, plan_id)
        )


def update_subtopic_start_msg(subtopic_id: int, start_msg_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Subtopic SET start_msg_id = ? WHERE id = ?",
            (start_msg_id, subtopic_id),
        )


def update_subtopic_locked_scope(subtopic_id: int, locked_scope: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Subtopic SET locked_scope = ? WHERE id = ?",
            (locked_scope, subtopic_id),
        )


def close_subtopic(subtopic_id: int, conclusion: str) -> None:
    conclusion = _truncate(conclusion, MAX_CONTENT_LEN)
    with get_db() as conn:
        conn.execute(
            "UPDATE Subtopic SET status = 'Closed', conclusion = ? WHERE id = ?",
            (conclusion, subtopic_id),
        )


def get_open_subtopic(topic_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM Subtopic
            WHERE topic_id = ? AND status = 'Open'
            ORDER BY id DESC
            LIMIT 1
            """,
            (topic_id,),
        ).fetchone()
        return dict(row) if row else None


def get_web_evidence_for_topic(topic_id: int) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT id, query_text, title, snippet, source_domain, url FROM WebEvidence WHERE origin_topic_id = ? ORDER BY id DESC",
            (topic_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_unprocessed_web_evidence(
    topic_id: int, limit: int = 20
) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, origin_topic_id, origin_subtopic_id, query_text, title, "
            "snippet, url, source_domain, result_rank "
            "FROM WebEvidence "
            "WHERE origin_topic_id = ? AND ledger_processed = 0 "
            "ORDER BY id ASC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_web_evidence_ledger_processed(web_ids: List[int]) -> None:
    if not web_ids:
        return
    with get_db() as conn:
        placeholders = ",".join("?" * len(web_ids))
        conn.execute(
            f"UPDATE WebEvidence SET ledger_processed = 1 WHERE id IN ({placeholders})",
            web_ids,
        )


def get_web_evidence_count(topic_id: int) -> int:
    """Return total web evidence count for a topic."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM WebEvidence WHERE origin_topic_id = ?",
            (topic_id,),
        ).fetchone()
        return row[0] if row else 0


def web_evidence_ids_exist(topic_id: int, web_ids: list[int]) -> set[int]:
    """Return the subset of web_ids that exist in WebEvidence for the topic."""
    if not web_ids:
        return set()
    placeholders = ",".join("?" for _ in web_ids)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id FROM WebEvidence WHERE origin_topic_id = ? AND id IN ({placeholders})",
            [topic_id] + list(web_ids),
        ).fetchall()
        return {row[0] for row in rows}


def insert_tool_trace(
    topic_id: int,
    tool_type: str,
    query: Optional[str] = None,
    result_count: Optional[int] = None,
    metadata_json: Optional[str] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO ToolTrace (topic_id, tool_type, query, result_count, metadata_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (topic_id, tool_type, query, result_count, metadata_json),
        )
        return cursor.lastrowid


def update_fact_structured_columns(
    fact_id: int,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_json: Optional[str] = None,
    qualifiers_json: Optional[str] = None,
    attribution_json: Optional[str] = None,
) -> None:
    """Update the Wikidata-style structured columns on a Fact row.

    Only columns with non-None values are updated; existing values are preserved.
    """
    updates: list[str] = []
    params: list[Any] = []
    for col, val in [
        ("subject", subject),
        ("predicate", predicate),
        ("object_json", object_json),
        ("qualifiers_json", qualifiers_json),
        ("attribution_json", attribution_json),
    ]:
        if val is not None:
            updates.append(f"{col} = ?")
            params.append(val)
    if not updates:
        return
    params.append(fact_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE Fact SET {', '.join(updates)} WHERE id = ?",
            params,
        )


def update_claim_structured_columns(
    claim_id: int,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_json: Optional[str] = None,
    qualifiers_json: Optional[str] = None,
    polarity: Optional[str] = None,
) -> None:
    """Update the Wikidata-style structured columns on a Claim row.

    Only columns with non-None values are updated; existing values are preserved.
    """
    updates: list[str] = []
    params: list[Any] = []
    for col, val in [
        ("subject", subject),
        ("predicate", predicate),
        ("object_json", object_json),
        ("qualifiers_json", qualifiers_json),
        ("polarity", polarity),
    ]:
        if val is not None:
            updates.append(f"{col} = ?")
            params.append(val)
    if not updates:
        return
    params.append(claim_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE Claim SET {', '.join(updates)} WHERE id = ?",
            params,
        )


def delete_ledger_pending(pending_id: int) -> int:
    """Delete a pending entry. Returns number of rows deleted (0 or 1)."""
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM LedgerPending WHERE id = ?", (pending_id,))
        return cursor.rowcount


def insert_code_evidence(
    origin_topic_id: int,
    origin_subtopic_id: Optional[int],
    hypothesis: str,
    source_code: str,
    stdout: Optional[str],
    stderr: Optional[str],
    exit_code: int,
    execution_time_s: Optional[float],
    iterations: int,
    success: bool,
    requesting_role: Optional[str] = None,
    summary: Optional[str] = None,
    parent_evidence_id: Optional[int] = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO CodeEvidence (
                origin_topic_id, origin_subtopic_id, hypothesis, source_code,
                stdout, stderr, exit_code, execution_time_s, iterations,
                success, requesting_role, summary, parent_evidence_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                origin_topic_id,
                origin_subtopic_id,
                _truncate(hypothesis, MAX_CONTENT_LEN),
                _truncate(source_code, MAX_CONTENT_LEN),
                _truncate(stdout, MAX_CONTENT_LEN),
                _truncate(stderr, MAX_CONTENT_LEN),
                exit_code,
                execution_time_s,
                iterations,
                int(success),
                requesting_role,
                _truncate(summary, MAX_SUMMARY_LEN),
                parent_evidence_id,
            ),
        )
        return cursor.lastrowid


def get_code_evidence_for_topic(topic_id: int) -> List[Dict[str, Any]]:
    """Return code evidence for a topic. Excludes source_code for efficiency;
    use get_code_evidence_by_id() for full details."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, hypothesis, stdout, stderr, exit_code, "
            "execution_time_s, iterations, success, requesting_role, summary, "
            "parent_evidence_id, created_at "
            "FROM CodeEvidence WHERE origin_topic_id = ? ORDER BY id DESC",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_code_evidence_for_topic_full(topic_id: int) -> List[Dict[str, Any]]:
    """Return code evidence for a topic including source_code (for dashboard)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, hypothesis, source_code, stdout, stderr, exit_code, "
            "execution_time_s, iterations, success, requesting_role, summary, "
            "parent_evidence_id, review_count, created_at "
            "FROM CodeEvidence WHERE origin_topic_id = ? ORDER BY id DESC",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_code_evidence_by_id(evidence_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM CodeEvidence WHERE id = ?",
            (evidence_id,),
        ).fetchone()
        return dict(row) if row else None


def increment_code_evidence_review_count(evidence_id: int) -> None:
    """Bump review_count when a review finds no issues (confirms original)."""
    with get_db() as conn:
        conn.execute(
            "UPDATE CodeEvidence SET review_count = review_count + 1 WHERE id = ?",
            (evidence_id,),
        )


def reset_code_evidence_review_count(evidence_id: int) -> None:
    """Reset review_count when a review finds a real problem (contradicts original)."""
    with get_db() as conn:
        conn.execute(
            "UPDATE CodeEvidence SET review_count = 0 WHERE id = ?",
            (evidence_id,),
        )


def insert_api_evidence(
    origin_topic_id: int,
    origin_subtopic_id: Optional[int],
    question: str,
    answer: str,
    provider: Optional[str] = None,
    requested_provider: Optional[str] = None,
    model: Optional[str] = None,
    requesting_role: Optional[str] = None,
    planner_reason: Optional[str] = None,
    fallback_used: bool = False,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO ApiEvidence (
                origin_topic_id, origin_subtopic_id, question, answer,
                provider, requested_provider, model, requesting_role,
                planner_reason, fallback_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                origin_topic_id,
                origin_subtopic_id,
                _truncate(question, MAX_CONTENT_LEN),
                _truncate(answer, MAX_CONTENT_LEN),
                provider,
                requested_provider,
                model,
                requesting_role,
                _truncate(planner_reason, MAX_SUMMARY_LEN),
                int(fallback_used),
            ),
        )
        return cursor.lastrowid


def get_api_evidence_for_topic(
    topic_id: int, limit: int = 10
) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, question, answer, provider, requested_provider, model,
                      requesting_role, planner_reason, fallback_used, created_at
               FROM ApiEvidence
               WHERE origin_topic_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (topic_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_api_evidence_by_id(evidence_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM ApiEvidence WHERE id = ?",
            (evidence_id,),
        ).fetchone()
        return dict(row) if row else None


def supersede_facts(fact_ids: List[int]) -> None:
    # TODO: Need a more stable, graph-based way to handle fact invalidation and cascading claim invalidation
    if not fact_ids:
        return
    with get_db() as conn:
        placeholders = ",".join("?" * len(fact_ids))
        conn.execute(
            f"UPDATE Fact SET review_status = 'superseded' WHERE id IN ({placeholders})",
            fact_ids,
        )


def supersede_fact(old_fact_id: int, new_fact_id: int) -> None:
    """Mark a single fact as superseded by another."""
    with get_db() as conn:
        conn.execute(
            "UPDATE Fact SET review_status = 'superseded', superseded_by = ? WHERE id = ?",
            (new_fact_id, old_fact_id),
        )


def update_topic_conclusion(topic_id: int, conclusion: str) -> None:
    conclusion = _truncate(conclusion, MAX_CONTENT_LEN)
    with get_db() as conn:
        conn.execute(
            "UPDATE Topic SET conclusion = ? WHERE id = ?", (conclusion, topic_id)
        )


# ---------------------------------------------------------------------------
# Ledger CRUD
# ---------------------------------------------------------------------------


def create_ledger_entity(
    topic_id: int, canonical_name: str, entity_type: Optional[str] = None
) -> int:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO LedgerEntity (topic_id, canonical_name, entity_type) VALUES (?, ?, ?)
               ON CONFLICT(topic_id, canonical_name) DO UPDATE SET
                   entity_type = COALESCE(excluded.entity_type, entity_type)""",
            (topic_id, canonical_name, entity_type),
        )
        row = conn.execute(
            "SELECT id FROM LedgerEntity WHERE topic_id = ? AND canonical_name = ?",
            (topic_id, canonical_name),
        ).fetchone()
        return row["id"]


def get_ledger_entities(topic_id: int) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM LedgerEntity WHERE topic_id = ? ORDER BY id",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_ledger_entity(entity_id: int, conn=None) -> Optional[Dict[str, Any]]:
    def _run(c):
        row = c.execute(
            "SELECT * FROM LedgerEntity WHERE id = ?", (entity_id,)
        ).fetchone()
        return dict(row) if row else None

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def update_ledger_entity_last_mentioned(entity_id: int, round_number: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE LedgerEntity SET last_mentioned_round = ? WHERE id = ?",
            (round_number, entity_id),
        )


def create_ledger_entity_alias(
    entity_id: int,
    alias_text: str,
    confirmed: bool = False,
    match_count: int = 1,
) -> int:
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO LedgerEntityAlias (entity_id, alias_text, confirmed, match_count) VALUES (?, ?, ?, ?)",
            (entity_id, alias_text, int(confirmed), match_count),
        )
        row = conn.execute(
            "SELECT id FROM LedgerEntityAlias WHERE entity_id = ? AND alias_text = ?",
            (entity_id, alias_text),
        ).fetchone()
        return row["id"]


def lookup_entity_alias(alias_text: str, topic_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT a.id AS alias_id, a.entity_id, a.confirmed, a.match_count
            FROM LedgerEntityAlias a
            JOIN LedgerEntity e ON e.id = a.entity_id
            WHERE a.alias_text = ? AND e.topic_id = ?
            """,
            (alias_text, topic_id),
        ).fetchone()
        return dict(row) if row else None


def increment_entity_alias_match(alias_id: int) -> int:
    with get_db() as conn:
        conn.execute(
            "UPDATE LedgerEntityAlias SET match_count = match_count + 1 WHERE id = ?",
            (alias_id,),
        )
        row = conn.execute(
            "SELECT match_count FROM LedgerEntityAlias WHERE id = ?", (alias_id,)
        ).fetchone()
        return row["match_count"]


def confirm_entity_alias(alias_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE LedgerEntityAlias SET confirmed = 1 WHERE id = ?", (alias_id,)
        )


def create_ledger_attribute(
    topic_id: int, canonical_name: str, value_type: Optional[str] = None
) -> int:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO LedgerAttribute (topic_id, canonical_name, value_type) VALUES (?, ?, ?)
               ON CONFLICT(topic_id, canonical_name) DO UPDATE SET
                   value_type = COALESCE(excluded.value_type, value_type)""",
            (topic_id, canonical_name, value_type),
        )
        row = conn.execute(
            "SELECT id FROM LedgerAttribute WHERE topic_id = ? AND canonical_name = ?",
            (topic_id, canonical_name),
        ).fetchone()
        return row["id"]


def get_ledger_attributes(topic_id: int) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM LedgerAttribute WHERE topic_id = ? ORDER BY id",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def create_ledger_attribute_alias(
    attribute_id: int,
    alias_text: str,
    confirmed: bool = False,
    match_count: int = 1,
) -> int:
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO LedgerAttributeAlias (attribute_id, alias_text, confirmed, match_count) VALUES (?, ?, ?, ?)",
            (attribute_id, alias_text, int(confirmed), match_count),
        )
        row = conn.execute(
            "SELECT id FROM LedgerAttributeAlias WHERE attribute_id = ? AND alias_text = ?",
            (attribute_id, alias_text),
        ).fetchone()
        return row["id"]


def lookup_attribute_alias(alias_text: str, topic_id: int) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT a.id AS alias_id, a.attribute_id, a.confirmed, a.match_count
            FROM LedgerAttributeAlias a
            JOIN LedgerAttribute attr ON attr.id = a.attribute_id
            WHERE a.alias_text = ? AND attr.topic_id = ?
            """,
            (alias_text, topic_id),
        ).fetchone()
        return dict(row) if row else None


def increment_attribute_alias_match(alias_id: int) -> int:
    with get_db() as conn:
        conn.execute(
            "UPDATE LedgerAttributeAlias SET match_count = match_count + 1 WHERE id = ?",
            (alias_id,),
        )
        row = conn.execute(
            "SELECT match_count FROM LedgerAttributeAlias WHERE id = ?", (alias_id,)
        ).fetchone()
        return row["match_count"]


def confirm_attribute_alias(alias_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE LedgerAttributeAlias SET confirmed = 1 WHERE id = ?", (alias_id,)
        )


def create_entity_with_aliases_batch(
    topic_id: int,
    canonical_name: str,
    entity_type: Optional[str],
    aliases: list[str],
    confirmed: bool = False,
) -> int:
    """Create entity + all aliases in a single transaction."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO LedgerEntity (topic_id, canonical_name, entity_type) VALUES (?, ?, ?)
               ON CONFLICT(topic_id, canonical_name) DO UPDATE SET
                   entity_type = COALESCE(excluded.entity_type, entity_type)""",
            (topic_id, canonical_name, entity_type),
        )
        row = conn.execute(
            "SELECT id FROM LedgerEntity WHERE topic_id = ? AND canonical_name = ?",
            (topic_id, canonical_name),
        ).fetchone()
        entity_id = row["id"]
        # Canonical name always confirmed
        conn.execute(
            "INSERT OR IGNORE INTO LedgerEntityAlias (entity_id, alias_text, confirmed, match_count) VALUES (?, ?, 1, 1)",
            (entity_id, canonical_name.lower()),
        )
        for alias in aliases:
            alias_lower = alias.strip().lower()
            if alias_lower and alias_lower != canonical_name.lower():
                conn.execute(
                    "INSERT OR IGNORE INTO LedgerEntityAlias (entity_id, alias_text, confirmed, match_count) VALUES (?, ?, ?, 1)",
                    (entity_id, alias_lower, int(confirmed)),
                )
        return entity_id


def create_attribute_with_aliases_batch(
    topic_id: int,
    canonical_name: str,
    value_type: Optional[str],
    aliases: list[str],
    confirmed: bool = False,
) -> int:
    """Create attribute + all aliases in a single transaction."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO LedgerAttribute (topic_id, canonical_name, value_type) VALUES (?, ?, ?)
               ON CONFLICT(topic_id, canonical_name) DO UPDATE SET
                   value_type = COALESCE(excluded.value_type, value_type)""",
            (topic_id, canonical_name, value_type),
        )
        row = conn.execute(
            "SELECT id FROM LedgerAttribute WHERE topic_id = ? AND canonical_name = ?",
            (topic_id, canonical_name),
        ).fetchone()
        attr_id = row["id"]
        conn.execute(
            "INSERT OR IGNORE INTO LedgerAttributeAlias (attribute_id, alias_text, confirmed, match_count) VALUES (?, ?, 1, 1)",
            (attr_id, canonical_name.lower()),
        )
        for alias in aliases:
            alias_lower = alias.strip().lower()
            if alias_lower and alias_lower != canonical_name.lower():
                conn.execute(
                    "INSERT OR IGNORE INTO LedgerAttributeAlias (attribute_id, alias_text, confirmed, match_count) VALUES (?, ?, ?, 1)",
                    (attr_id, alias_lower, int(confirmed)),
                )
        return attr_id


def resolve_entity_alias_atomic(
    alias_text: str,
    topic_id: int,
    confirmation_threshold: int,
    round_number: Optional[int] = None,
    conn=None,
) -> Optional[int]:
    """Lookup + increment + optional confirm in a single transaction. Returns entity_id or None."""

    def _run(c):
        row = c.execute(
            """SELECT a.id AS alias_id, a.entity_id, a.confirmed, a.match_count
               FROM LedgerEntityAlias a
               JOIN LedgerEntity e ON e.id = a.entity_id
               WHERE a.alias_text = ? COLLATE NOCASE AND e.topic_id = ?""",
            (alias_text, topic_id),
        ).fetchone()
        if row is None:
            return None
        entity_id = row["entity_id"]
        if not row["confirmed"]:
            c.execute(
                "UPDATE LedgerEntityAlias SET match_count = match_count + 1 WHERE id = ?",
                (row["alias_id"],),
            )
            new_count = c.execute(
                "SELECT match_count FROM LedgerEntityAlias WHERE id = ?",
                (row["alias_id"],),
            ).fetchone()["match_count"]
            if new_count >= confirmation_threshold:
                c.execute(
                    "UPDATE LedgerEntityAlias SET confirmed = 1 WHERE id = ?",
                    (row["alias_id"],),
                )
        if round_number is not None:
            c.execute(
                "UPDATE LedgerEntity SET last_mentioned_round = ? WHERE id = ?",
                (round_number, entity_id),
            )
        return entity_id

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def resolve_attribute_alias_atomic(
    alias_text: str, topic_id: int, confirmation_threshold: int, conn=None
) -> Optional[int]:
    """Lookup + increment + optional confirm in a single transaction. Returns attribute_id or None."""

    def _run(c):
        row = c.execute(
            """SELECT a.id AS alias_id, a.attribute_id, a.confirmed, a.match_count
               FROM LedgerAttributeAlias a
               JOIN LedgerAttribute attr ON attr.id = a.attribute_id
               WHERE a.alias_text = ? COLLATE NOCASE AND attr.topic_id = ?""",
            (alias_text, topic_id),
        ).fetchone()
        if row is None:
            return None
        attr_id = row["attribute_id"]
        if not row["confirmed"]:
            c.execute(
                "UPDATE LedgerAttributeAlias SET match_count = match_count + 1 WHERE id = ?",
                (row["alias_id"],),
            )
            new_count = c.execute(
                "SELECT match_count FROM LedgerAttributeAlias WHERE id = ?",
                (row["alias_id"],),
            ).fetchone()["match_count"]
            if new_count >= confirmation_threshold:
                c.execute(
                    "UPDATE LedgerAttributeAlias SET confirmed = 1 WHERE id = ?",
                    (row["alias_id"],),
                )
        return attr_id

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def upsert_ledger_entry(
    topic_id: int,
    subtopic_id: Optional[int],
    entity_id: int,
    attribute_id: int,
    value: str,
    value_numeric_min: Optional[float],
    value_numeric_max: Optional[float],
    unit: Optional[str],
    normalized_timeframe: str,
    entry_type: str,
    source_ref: str,
    source_domain: Optional[str] = None,
    domain_score: Optional[float] = None,
    decontextualized: Optional[str] = None,
    created_by: Optional[str] = None,
    status: str = "accepted",
    valid_from: Optional[str] = None,
    valid_to: Optional[str] = None,
    # Phase G: rich statistical columns
    value_mean: Optional[float] = None,
    value_std: Optional[float] = None,
    value_ci_lower: Optional[float] = None,
    value_ci_upper: Optional[float] = None,
    value_ci_level: Optional[float] = None,
    value_p: Optional[float] = None,
    value_n: Optional[int] = None,
    value_stat_type: Optional[str] = None,
    baseline_entity_id: Optional[int] = None,
    split: Optional[str] = None,
    config_json: Optional[str] = None,
    conn=None,
) -> tuple[int, bool]:
    # was_inserted is best-effort; concurrent writes may make it inaccurate (cosmetic only)
    def _run(c):
        existing = c.execute(
            """SELECT id FROM Ledger
            WHERE topic_id = ? AND entity_id = ? AND attribute_id = ?
                AND normalized_timeframe = ? AND source_ref = ?""",
            (topic_id, entity_id, attribute_id, normalized_timeframe, source_ref),
        ).fetchone()
        cursor = c.execute(
            """
            INSERT INTO Ledger (
                topic_id, subtopic_id, entity_id, attribute_id,
                value, value_numeric_min, value_numeric_max, unit,
                normalized_timeframe, entry_type, status, source_ref,
                source_domain, domain_score, decontextualized, created_by,
                valid_from, valid_to,
                value_mean, value_std, value_ci_lower, value_ci_upper,
                value_ci_level, value_p, value_n, value_stat_type,
                baseline_entity_id, split, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id, entity_id, attribute_id, normalized_timeframe, source_ref)
            DO UPDATE SET value = excluded.value,
                value_numeric_min = excluded.value_numeric_min,
                value_numeric_max = excluded.value_numeric_max,
                unit = excluded.unit,
                status = excluded.status,
                decontextualized = excluded.decontextualized,
                valid_from = excluded.valid_from,
                valid_to = excluded.valid_to,
                domain_score = excluded.domain_score,
                source_domain = excluded.source_domain,
                value_mean = excluded.value_mean,
                value_std = excluded.value_std,
                value_ci_lower = excluded.value_ci_lower,
                value_ci_upper = excluded.value_ci_upper,
                value_ci_level = excluded.value_ci_level,
                value_p = excluded.value_p,
                value_n = excluded.value_n,
                value_stat_type = excluded.value_stat_type,
                baseline_entity_id = excluded.baseline_entity_id,
                split = excluded.split,
                config_json = excluded.config_json
            RETURNING id
            """,
            (
                topic_id,
                subtopic_id,
                entity_id,
                attribute_id,
                value,
                value_numeric_min,
                value_numeric_max,
                unit,
                normalized_timeframe,
                entry_type,
                status,
                source_ref,
                source_domain,
                domain_score,
                decontextualized,
                created_by,
                valid_from,
                valid_to,
                value_mean,
                value_std,
                value_ci_lower,
                value_ci_upper,
                value_ci_level,
                value_p,
                value_n,
                value_stat_type,
                baseline_entity_id,
                split,
                config_json,
            ),
        )
        row = cursor.fetchone()
        return row["id"], existing is None

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def find_value_duplicate_ledger_entry(
    topic_id: int,
    entity_id: int,
    attribute_id: int,
    normalized_timeframe: str,
    value_numeric_min: Optional[float],
    value_numeric_max: Optional[float],
    conn=None,
) -> Optional[Dict[str, Any]]:
    """Find existing entry with same semantic key and matching numeric range.

    Only deduplicates when at least one of min/max is non-None (actual numbers).
    Returns first matching row as dict or None.
    """

    def _run(c):
        if value_numeric_min is None and value_numeric_max is None:
            return None
        row = c.execute(
            """SELECT id, source_ref FROM Ledger
            WHERE topic_id = ? AND entity_id = ? AND attribute_id = ?
                AND normalized_timeframe = ?
                AND value_numeric_min IS ? AND value_numeric_max IS ?
            LIMIT 1""",
            (
                topic_id,
                entity_id,
                attribute_id,
                normalized_timeframe,
                value_numeric_min,
                value_numeric_max,
            ),
        ).fetchone()
        return dict(row) if row else None

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def merge_ledger_source_ref(ledger_id: int, new_source_ref: str, conn=None) -> None:
    """Append new citation markers to existing entry's source_ref if not already present."""

    def _run(c):
        row = c.execute(
            "SELECT source_ref FROM Ledger WHERE id = ?", (ledger_id,)
        ).fetchone()
        if not row:
            return
        old_ref = row["source_ref"] or ""
        existing_markers = set(old_ref.split())
        new_markers = [m for m in new_source_ref.split() if m not in existing_markers]
        if not new_markers:
            return
        updated_ref = f"{old_ref} {' '.join(new_markers)}".strip()
        c.execute(
            "UPDATE Ledger SET source_ref = ? WHERE id = ?", (updated_ref, ledger_id)
        )

    if conn is not None:
        _run(conn)
    else:
        with get_db() as c:
            _run(c)


def get_ledger_entries(
    topic_id: int,
    subtopic_id: Optional[int] = None,
    entity_id: Optional[int] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    clauses = ["topic_id = ?"]
    params: list[Any] = [topic_id]
    if subtopic_id is not None:
        clauses.append("subtopic_id = ?")
        params.append(subtopic_id)
    if entity_id is not None:
        clauses.append("entity_id = ?")
        params.append(entity_id)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    query = f"SELECT * FROM Ledger WHERE {' AND '.join(clauses)} ORDER BY id"
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def update_ledger_entry_status(entry_id: int, status: str) -> None:
    with get_db() as conn:
        conn.execute("UPDATE Ledger SET status = ? WHERE id = ?", (status, entry_id))


def create_ledger_pending(
    topic_id: int,
    subtopic_id: Optional[int],
    raw_text: str,
    source_ref: Optional[str],
    extracted_numbers: Optional[str],
    missing_fields: Optional[str],
    created_round: Optional[int],
    ttl_expires_round: Optional[int],
    conn=None,
) -> int:
    def _run(c):
        cursor = c.execute(
            """
            INSERT INTO LedgerPending (
                topic_id, subtopic_id, raw_text, source_ref,
                extracted_numbers, missing_fields, created_round, ttl_expires_round
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic_id,
                subtopic_id,
                raw_text,
                source_ref,
                extracted_numbers,
                missing_fields,
                created_round,
                ttl_expires_round,
            ),
        )
        return cursor.lastrowid

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def get_active_ledger_pending(
    topic_id: int, current_round: int
) -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM LedgerPending WHERE topic_id = ? AND (ttl_expires_round IS NULL OR ttl_expires_round >= ?) ORDER BY id",
            (topic_id, current_round),
        ).fetchall()
        return [dict(r) for r in rows]


def expire_ledger_pending(current_round: int) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM LedgerPending WHERE ttl_expires_round IS NOT NULL AND ttl_expires_round < ?",
            (current_round,),
        )
        return cursor.rowcount


def get_ledger_attribute(attribute_id: int, conn=None) -> Optional[Dict[str, Any]]:
    def _run(c):
        row = c.execute(
            "SELECT * FROM LedgerAttribute WHERE id = ?", (attribute_id,)
        ).fetchone()
        return dict(row) if row else None

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def get_ledger_entries_with_names(topic_id: int) -> List[Dict[str, Any]]:
    """JOIN Ledger with LedgerEntity + LedgerAttribute to get names in one query."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT L.*, E.canonical_name AS entity_name, A.canonical_name AS attribute_name
            FROM Ledger L
            JOIN LedgerEntity E ON L.entity_id = E.id
            JOIN LedgerAttribute A ON L.attribute_id = A.id
            WHERE L.topic_id = ?
            ORDER BY E.canonical_name, A.canonical_name, L.normalized_timeframe""",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_contested_ledger_pairs(topic_id: int) -> List[Dict[str, Any]]:
    """Find entries where same (entity, attribute) with overlapping time has multiple different values.

    Uses interval overlap (valid_from/valid_to) when available, falling back to
    normalized_timeframe string equality for legacy entries.
    """
    from .ledger import timeframe_to_interval, intervals_overlap as _intervals_overlap

    with get_db() as conn:
        rows = conn.execute(
            """SELECT L.*, E.canonical_name AS entity_name, A.canonical_name AS attribute_name
            FROM Ledger L
            JOIN LedgerEntity E ON L.entity_id = E.id
            JOIN LedgerAttribute A ON L.attribute_id = A.id
            WHERE L.topic_id = ? AND value != ''
            ORDER BY L.entity_id, L.attribute_id, L.id""",
            (topic_id,),
        ).fetchall()

    # Group by (entity_id, attribute_id)
    grouped: dict[tuple[int, int], list[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["entity_id"], r["attribute_id"])
        grouped.setdefault(key, []).append(dict(r))

    def _value_fingerprint(entry: Dict[str, Any]) -> str:
        if (
            entry.get("value_numeric_min") is not None
            or entry.get("value_numeric_max") is not None
        ):
            return (
                f"{entry.get('value_numeric_min')}|{entry.get('value_numeric_max')}|"
                f"{entry.get('unit', '')}"
            )
        return entry.get("value", "").lower().strip()

    def _get_interval(entry: Dict[str, Any]) -> tuple:
        vf = entry.get("valid_from")
        vt = entry.get("valid_to")
        if vf or vt:
            return (vf, vt)
        tf = entry.get("normalized_timeframe", "")
        if tf:
            return timeframe_to_interval(tf)
        return (None, None)

    result = []
    for (eid, aid), entries in grouped.items():
        if len(entries) < 2:
            continue
        # Find pairs with overlapping time and distinct values
        contested_ids: set[int] = set()
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                ei, ej = entries[i], entries[j]
                int_i = _get_interval(ei)
                int_j = _get_interval(ej)
                # Skip if both have empty/no timeframe (legacy behavior)
                if (
                    int_i == (None, None)
                    and int_j == (None, None)
                    and not ei.get("valid_from")
                    and not ej.get("valid_from")
                    and not ei.get("normalized_timeframe")
                    and not ej.get("normalized_timeframe")
                ):
                    continue
                if not _intervals_overlap(int_i, int_j):
                    continue
                if _value_fingerprint(ei) != _value_fingerprint(ej):
                    contested_ids.add(ei["id"])
                    contested_ids.add(ej["id"])
        if contested_ids:
            contested_entries = [e for e in entries if e["id"] in contested_ids]
            result.append(
                {
                    "entity_name": contested_entries[0]["entity_name"],
                    "attribute_name": contested_entries[0]["attribute_name"],
                    "timeframe": contested_entries[0].get("normalized_timeframe", ""),
                    "entries": contested_entries,
                }
            )
    return result


def ledger_entry_exists(entry_id: int) -> bool:
    """Fast existence check for a Ledger entry."""
    with get_db() as conn:
        row = conn.execute("SELECT 1 FROM Ledger WHERE id = ?", (entry_id,)).fetchone()
        return row is not None


# ---------------------------------------------------------------------------
# LedgerEdge CRUD
# ---------------------------------------------------------------------------


_SYMMETRIC_EDGE_TYPES = frozenset({"conflicts_with", "supports"})


def create_ledger_edge(
    topic_id: int,
    from_entry_id: int,
    to_entry_id: int,
    edge_type: str,
    created_by: Optional[str] = None,
    conn=None,
) -> Optional[int]:
    """Insert a new edge. Returns edge_id or None if already exists (UNIQUE constraint).

    Symmetric edge types (conflicts_with, supports) are canonicalized so
    from_entry_id < to_entry_id, preventing duplicate reverse pairs.
    Raises ValueError on self-edges.
    """
    if from_entry_id == to_entry_id:
        raise ValueError("Self-edges are not allowed")
    # Canonical ordering for symmetric edge types
    if edge_type in _SYMMETRIC_EDGE_TYPES and from_entry_id > to_entry_id:
        from_entry_id, to_entry_id = to_entry_id, from_entry_id

    def _run(c):
        try:
            cursor = c.execute(
                """INSERT INTO LedgerEdge
                   (topic_id, from_entry_id, to_entry_id, edge_type, created_by)
                   VALUES (?, ?, ?, ?, ?)""",
                (topic_id, from_entry_id, to_entry_id, edge_type, created_by),
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            if "UNIQUE" in str(e):
                return None
            raise

    if conn is not None:
        return _run(conn)
    with get_db() as c:
        return _run(c)


def get_ledger_edges(
    topic_id: int,
    entry_id: Optional[int] = None,
    edge_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query edges by topic, optionally filtered by entry_id (either direction) and edge_type."""
    clauses = ["topic_id = ?"]
    params: list[Any] = [topic_id]
    if entry_id is not None:
        clauses.append("(from_entry_id = ? OR to_entry_id = ?)")
        params.extend([entry_id, entry_id])
    if edge_type is not None:
        clauses.append("edge_type = ?")
        params.append(edge_type)
    sql = f"SELECT * FROM LedgerEdge WHERE {' AND '.join(clauses)} ORDER BY id"
    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def bulk_create_ledger_edges(
    topic_id: int,
    edges: List[tuple[int, int]],
    edge_type: str,
    created_by: Optional[str] = None,
) -> int:
    """Insert multiple edges in a single transaction. Returns count of newly created edges.

    Each edge is a (from_entry_id, to_entry_id) pair. Symmetric edge types are
    canonicalized. Duplicates and FK violations are silently skipped.
    """
    if not edges:
        return 0
    new_count = 0
    with get_db() as conn:
        for from_id, to_id in edges:
            if from_id == to_id:
                continue
            if edge_type in _SYMMETRIC_EDGE_TYPES and from_id > to_id:
                from_id, to_id = to_id, from_id
            try:
                cursor = conn.execute(
                    """INSERT INTO LedgerEdge
                       (topic_id, from_entry_id, to_entry_id, edge_type, created_by)
                       VALUES (?, ?, ?, ?, ?)""",
                    (topic_id, from_id, to_id, edge_type, created_by),
                )
                if cursor.rowcount > 0:
                    new_count += 1
            except sqlite3.IntegrityError:
                pass  # duplicate or FK violation
    return new_count


def delete_ledger_edge(edge_id: int, topic_id: int) -> bool:
    """Delete a single edge by id, scoped to topic_id. Returns True if deleted."""
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM LedgerEdge WHERE id = ? AND topic_id = ?",
            (edge_id, topic_id),
        )
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# KnowledgeEdge CRUD
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
    """Insert a KnowledgeEdge. Returns edge id, or None if duplicate."""
    with get_db() as conn:
        try:
            cursor = conn.execute(
                """INSERT INTO KnowledgeEdge
                    (topic_id, source_id, source_type, target_id, target_type,
                     relation, justification_group, confidence, created_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    topic_id,
                    source_id,
                    source_type,
                    target_id,
                    target_type,
                    relation,
                    justification_group,
                    confidence,
                    created_by,
                ),
            )
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None


def get_knowledge_edges(
    topic_id: int,
    source_id: int | None = None,
    source_type: str | None = None,
    target_id: int | None = None,
    target_type: str | None = None,
    relation: str | None = None,
    active_only: bool = True,
) -> list[dict]:
    """Query edges with optional filters."""
    clauses = ["topic_id = ?"]
    params: list = [topic_id]
    if source_id is not None:
        clauses.append("source_id = ?")
        params.append(source_id)
    if source_type is not None:
        clauses.append("source_type = ?")
        params.append(source_type)
    if target_id is not None:
        clauses.append("target_id = ?")
        params.append(target_id)
    if target_type is not None:
        clauses.append("target_type = ?")
        params.append(target_type)
    if relation is not None:
        clauses.append("relation = ?")
        params.append(relation)
    if active_only:
        clauses.append("is_active = 1")
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM KnowledgeEdge WHERE {' AND '.join(clauses)} ORDER BY id",
            params,
        ).fetchall()
        return [dict(r) for r in rows]


def deactivate_knowledge_edge(edge_id: int) -> None:
    """Soft-delete: set is_active=0."""
    with get_db() as conn:
        conn.execute("UPDATE KnowledgeEdge SET is_active = 0 WHERE id = ?", (edge_id,))


def get_active_conflicts(topic_id: int, node_type: str) -> list[dict]:
    """Get all active conflicts_with edges for a given node type."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM KnowledgeEdge
               WHERE topic_id = ? AND relation = 'conflicts_with'
                 AND source_type = ? AND target_type = ?
                 AND is_active = 1
               ORDER BY id""",
            (topic_id, node_type, node_type),
        ).fetchall()
        return [dict(r) for r in rows]


def get_claim_justification_groups(
    topic_id: int, claim_id: int
) -> dict[str, list[dict]]:
    """Return {group_name: [source_nodes]} for a claim's supports edges."""
    edges = get_knowledge_edges(
        topic_id,
        target_id=claim_id,
        target_type="claim",
        relation="supports",
        active_only=True,
    )
    groups: dict[str, list[dict]] = {}
    for e in edges:
        groups.setdefault(e["justification_group"], []).append(e)
    return groups


def get_dismissed_knowledge(topic_id: int) -> list[dict]:
    """Return facts/claims with status refuted/retired/superseded."""
    results: list[dict] = []
    try:
        with get_db() as conn:
            fact_rows = conn.execute(
                """SELECT id, content, summary, review_status, superseded_by
                   FROM Fact
                   WHERE topic_id = ? AND review_status IN ('refuted', 'retired', 'superseded')
                   ORDER BY id DESC""",
                (topic_id,),
            ).fetchall()
            for r in fact_rows:
                results.append(
                    {
                        "type": "fact",
                        "id": r["id"],
                        "summary": r["summary"],
                        "content": r["content"],
                        "status": r["review_status"],
                        "superseded_by": r["superseded_by"],
                    }
                )
            claim_rows = conn.execute(
                """SELECT id, content, summary, status, superseded_by
                   FROM Claim
                   WHERE topic_id = ? AND status IN ('retired', 'superseded')
                   ORDER BY id DESC""",
                (topic_id,),
            ).fetchall()
            for r in claim_rows:
                results.append(
                    {
                        "type": "claim",
                        "id": r["id"],
                        "summary": r["summary"],
                        "content": r["content"],
                        "status": r["status"],
                        "superseded_by": r["superseded_by"],
                    }
                )
    except Exception as exc:
        logger.debug(
            "[dismissed] Failed to fetch dismissed knowledge for topic %s: %s",
            topic_id,
            exc,
        )
    return results


# ---------------------------------------------------------------------------
# WebQueryCache (WE-1: semantic query dedup)
# ---------------------------------------------------------------------------


def insert_web_query_cache(
    topic_id: int,
    query_text: str,
    result_ids: list[int],
    embedding: list[float],
) -> int | None:
    """Cache a web query's embedding + result IDs for semantic dedup."""
    with get_db() as conn:
        cursor = conn.execute(
            """INSERT INTO WebQueryCache (topic_id, query_text, result_ids_json)
               VALUES (?, ?, ?)""",
            (topic_id, query_text, json.dumps(result_ids)),
        )
        cache_id = cursor.lastrowid
        try:
            conn.execute(
                "INSERT INTO vec_web_queries (query_id, embedding) VALUES (?, ?)",
                (cache_id, serialize_f32(embedding)),
            )
        except Exception as exc:
            logger.debug("vec_web_queries insert failed: %s", exc)
        return cache_id


def search_web_queries_semantic(
    topic_id: int,
    query_embedding: list[float],
    top_k: int = 5,
    max_age_days: int = 30,
) -> list[dict]:
    """Find semantically similar cached web queries, return matched WebEvidence rows."""
    with get_db() as conn:
        try:
            vec_rows = conn.execute(
                """SELECT query_id, distance
                   FROM vec_web_queries
                   WHERE embedding MATCH ? AND k = ?""",
                (serialize_f32(query_embedding), top_k * 2),
            ).fetchall()
        except Exception:
            return []

        if not vec_rows:
            return []

        # Collect valid query_ids in batch (avoid N+1)
        valid_query_ids = [vr["query_id"] for vr in vec_rows if vr["distance"] <= 1.0]
        if not valid_query_ids:
            return []

        ph = ",".join("?" * len(valid_query_ids))
        cache_rows = conn.execute(
            f"""SELECT id, topic_id, query_text, result_ids_json, created_at
               FROM WebQueryCache
               WHERE id IN ({ph}) AND topic_id = ?
                 AND created_at >= strftime('%Y-%m-%dT%H:%M:%fZ',
                     'now', '-' || ? || ' days')""",
            (*valid_query_ids, topic_id, max_age_days),
        ).fetchall()

        # Collect all web evidence IDs from all matching cache rows
        all_web_ids: list[int] = []
        for cr in cache_rows:
            try:
                raw_ids = json.loads(cr["result_ids_json"] or "[]")
                all_web_ids.extend(
                    int(x)
                    for x in raw_ids
                    if x is not None and isinstance(x, (int, float))
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        if not all_web_ids:
            return []

        # Dedup and batch-fetch web evidence
        unique_ids = list(dict.fromkeys(all_web_ids))
        ph2 = ",".join("?" * len(unique_ids))
        web_rows = conn.execute(
            f"SELECT * FROM WebEvidence WHERE id IN ({ph2})",
            unique_ids,
        ).fetchall()
        return [dict(wr) for wr in web_rows]


# ---------------------------------------------------------------------------
# Phase F.1: TopicConfig CRUD
# ---------------------------------------------------------------------------


def set_topic_config(topic_id: int, key: str, value: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO TopicConfig (topic_id, config_key, config_value) VALUES (?, ?, ?)",
            (topic_id, key, value),
        )


def get_topic_config(topic_id: int, key: str) -> Optional[str]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT config_value FROM TopicConfig WHERE topic_id = ? AND config_key = ?",
            (topic_id, key),
        ).fetchone()
        return row["config_value"] if row else None


def get_all_topic_config(topic_id: int) -> Dict[str, str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT config_key, config_value FROM TopicConfig WHERE topic_id = ?",
            (topic_id,),
        ).fetchall()
        return {row["config_key"]: row["config_value"] for row in rows}


# ---------------------------------------------------------------------------
# Phase F.2: UserInjection CRUD
# ---------------------------------------------------------------------------


def insert_user_injection(
    topic_id: int,
    injection_type: str,
    content: str,
    subtopic_id: int | None = None,
) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO UserInjection (topic_id, subtopic_id, injection_type, content) VALUES (?, ?, ?, ?)",
            (topic_id, subtopic_id, injection_type, content),
        )
        return cursor.lastrowid


def get_pending_injections(topic_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM UserInjection WHERE topic_id = ? AND status = 'pending' ORDER BY id ASC",
            (topic_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_injection_processed(injection_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE UserInjection SET status = 'processed' WHERE id = ?",
            (injection_id,),
        )


# ---------------------------------------------------------------------------
# Phase F.5: Topic queue helpers
# ---------------------------------------------------------------------------


def get_next_queued_topic() -> Optional[Dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM Topic WHERE status = 'Queued' AND queue_position IS NOT NULL ORDER BY queue_position ASC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def start_next_queued_topic() -> Optional[Dict]:
    """Atomically pop the next queued topic and set it to Started."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM Topic WHERE status = 'Queued' AND queue_position IS NOT NULL "
            "ORDER BY queue_position ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        conn.execute(
            "UPDATE Topic SET status = 'Started', queue_position = NULL, queued_at = NULL WHERE id = ?",
            (row["id"],),
        )
        updated = conn.execute(
            "SELECT * FROM Topic WHERE id = ?", (row["id"],)
        ).fetchone()
        return dict(updated) if updated else dict(row)


def dequeue_topic(topic_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE Topic SET queue_position = NULL, queued_at = NULL, "
            "status = CASE WHEN status = 'Queued' THEN 'Closed' ELSE status END "
            "WHERE id = ?",
            (topic_id,),
        )


def get_topic_queue() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM Topic WHERE status = 'Queued' AND queue_position IS NOT NULL ORDER BY queue_position ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def reorder_queue(topic_ids: list[int]) -> None:
    with get_db() as conn:
        for pos, tid in enumerate(topic_ids, 1):
            conn.execute(
                "UPDATE Topic SET queue_position = ? WHERE id = ? AND status = 'Queued'",
                (pos, tid),
            )


# ---------------------------------------------------------------------------
# VIKI: Autonomous repair tracker
# ---------------------------------------------------------------------------


def upsert_viki_issue(
    target_table: str,
    target_id: int,
    issue_type: str,
    topic_id: int | None = None,
    **extra,
) -> dict:
    """Insert or return existing VikiTracker issue. Returns the row as dict."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO VikiTracker (target_table, target_id, issue_type, topic_id)
            VALUES (?, ?, ?, ?)""",
            (target_table, target_id, issue_type, topic_id),
        )
        if topic_id is not None:
            row = conn.execute(
                "SELECT * FROM VikiTracker WHERE target_table = ? AND target_id = ? AND issue_type = ? AND topic_id = ?",
                (target_table, target_id, issue_type, topic_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM VikiTracker WHERE target_table = ? AND target_id = ? AND issue_type = ?",
                (target_table, target_id, issue_type),
            ).fetchone()
        return dict(row) if row else {}


def increment_viki_check(tracker_id: int) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE VikiTracker SET check_count = check_count + 1, last_checked_at = CURRENT_TIMESTAMP WHERE id = ?",
            (tracker_id,),
        )


def mark_viki_resolved(tracker_id: int, resolution: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE VikiTracker SET status = 'resolved', resolution = ?, last_checked_at = CURRENT_TIMESTAMP WHERE id = ?",
            (resolution, tracker_id),
        )


def mark_viki_wont_fix(tracker_id: int, resolution: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE VikiTracker SET status = 'wont_fix', resolution = ?, last_checked_at = CURRENT_TIMESTAMP WHERE id = ?",
            (resolution, tracker_id),
        )


def get_open_viki_issues(issue_type: str | None = None) -> list[dict]:
    with get_db() as conn:
        if issue_type:
            rows = conn.execute(
                "SELECT * FROM VikiTracker WHERE status = 'open' AND check_count < max_checks AND issue_type = ? ORDER BY id",
                (issue_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM VikiTracker WHERE status = 'open' AND check_count < max_checks ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]
