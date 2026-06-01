import os
import sys
from types import SimpleNamespace

import pytest

from orbit_or import api
from orbit_or.corpus import (
    LayoutBlock,
    build_chunk_drafts,
    build_layout_blocks_from_text,
    build_layout_chunk_drafts,
    ingest_layout_document,
    ingest_text_document,
    parse_pdf_layout,
)
from orbit_or.db import get_db, get_db_path, init_db
from orbit_or.rag import build_query_rag_context


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


def test_build_chunk_drafts_preserves_markdown_section_path():
    text = "# Operations\n\nCapacity is 10 units.\n\n## Demand\n\nDemand must be met."

    chunks = build_chunk_drafts(text, max_chars=200)

    assert len(chunks) == 2
    assert chunks[0].section_path == "Operations"
    assert chunks[1].section_path == "Operations / Demand"
    assert "Capacity" in chunks[0].text
    assert chunks[1].token_count > 0


def test_ingest_text_document_indexes_chunks_for_lexical_search():
    topic_id = api.create_topic("MSE topic", "optimize service capacity")

    result = ingest_text_document(
        topic_id=topic_id,
        title="Capacity memo",
        doc_type="markdown",
        text="# Capacity\n\nThe warehouse capacity constraint is 10 pallets.",
    )

    assert result["chunk_count"] == 1
    docs = api.list_corpus_documents(topic_id)
    assert docs[0]["title"] == "Capacity memo"

    hits = api.search_corpus_chunks_hybrid(
        topic_id, "warehouse capacity pallets", query_embedding=None, top_k=3
    )
    assert len(hits) == 1
    assert hits[0]["document_title"] == "Capacity memo"
    assert "capacity constraint" in hits[0]["content"]


def test_reindex_corpus_document_rebuilds_lexical_rows():
    topic_id = api.create_topic("MSE topic", "optimize service capacity")
    result = ingest_text_document(
        topic_id=topic_id,
        title="Capacity memo",
        doc_type="markdown",
        text="# Capacity\n\nThe warehouse capacity constraint is 10 pallets.",
    )
    chunk_id = result["chunk_ids"][0]
    with get_db() as conn:
        conn.execute("DELETE FROM corpus_chunks_fts WHERE rowid = ?", (chunk_id,))

    missing = api.search_corpus_chunks_hybrid(
        topic_id, "warehouse capacity pallets", query_embedding=None, top_k=3
    )
    assert missing == []

    rebuilt_count = api.reindex_corpus_document(result["document_id"])

    assert rebuilt_count == 1
    hits = api.search_corpus_chunks_hybrid(
        topic_id, "warehouse capacity pallets", query_embedding=None, top_k=3
    )
    assert hits[0]["id"] == chunk_id


def test_layout_blocks_detect_tables_and_preserve_page_metadata():
    page_text = (
        "Capacity assumptions\n\n"
        "Site  Capacity  Cost\n"
        "A     10        5\n"
        "B     20        8\n\n"
        "All units are pallets per day."
    )

    blocks = build_layout_blocks_from_text(
        page_text, page_number=2, section_path="Page 2"
    )
    drafts = build_layout_chunk_drafts(blocks, max_chars=500)

    assert [block.block_type for block in blocks] == ["text", "table", "text"]
    table_draft = [draft for draft in drafts if draft.granularity == "table"][0]
    assert table_draft.page_start == 2
    assert table_draft.section_path == "Page 2"
    assert "| Site | Capacity | Cost |" in table_draft.table_markdown


def test_ingest_layout_document_indexes_table_markdown():
    topic_id = api.create_topic("MSE topic", "optimize service capacity")
    table = "| Site | Capacity | Cost |\n| --- | --- | --- |\n| A | 10 | 5 |"

    result = ingest_layout_document(
        topic_id=topic_id,
        title="Capacity table",
        doc_type="pdf",
        blocks=[
            LayoutBlock(
                text=table,
                page_number=3,
                section_path="Appendix A",
                block_type="table",
                table_markdown=table,
            )
        ],
    )

    chunks = api.get_corpus_chunks_for_document(result["document_id"])
    assert chunks[0]["page_start"] == 3
    assert chunks[0]["granularity"] == "table"
    assert chunks[0]["table_markdown"] == table

    hits = api.search_corpus_chunks_hybrid(
        topic_id, "site capacity cost", query_embedding=None, top_k=3
    )
    assert hits[0]["page_start"] == 3
    assert "Capacity table" == hits[0]["document_title"]


def test_pdfplumber_adapter_preserves_extracted_tables(monkeypatch, tmp_path):
    class FakePage:
        def extract_text(self):
            return "Capacity assumptions"

        def extract_tables(self):
            return [[["Site", "Capacity"], ["A", "10"], ["B", "20"]]]

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_pdfplumber = SimpleNamespace(open=lambda _path: FakePdf())
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)
    pdf_path = tmp_path / "capacity.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    blocks, parser_used = parse_pdf_layout(pdf_path, parser="pdfplumber")

    assert parser_used == "pdfplumber"
    assert [block.block_type for block in blocks] == ["text", "table"]
    table = blocks[1]
    assert table.page_number == 1
    assert table.section_path == "Page 1 / Table 1"
    assert "| Site | Capacity |" in table.table_markdown


@pytest.mark.asyncio
async def test_query_rag_context_includes_corpus_neighbors():
    topic_id = api.create_topic("MSE topic", "optimize service capacity")
    ingest_text_document(
        topic_id=topic_id,
        title="Operations memo",
        doc_type="markdown",
        text=(
            "# Capacity\n\nWarehouse capacity is 10 pallets.\n\n"
            "# Demand\n\nDemand from customers must be fully satisfied."
        ),
    )

    rag_text, degraded = await build_query_rag_context(topic_id, "customers demand")

    assert degraded is False
    assert "[Private Corpus Chunks]" in rag_text
    assert "Warehouse capacity is 10 pallets" in rag_text
    assert "Demand from customers" in rag_text
