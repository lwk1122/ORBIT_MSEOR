"""Private corpus ingestion helpers.

This module intentionally starts with plain text and markdown. PDF and
layout-aware parsers should plug into the same database API later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from . import api
from .embedding import aget_embedding

DEFAULT_CHUNK_MAX_CHARS = 2400
DEFAULT_CHUNK_OVERLAP_CHARS = 240
MIN_CHUNK_CHARS = 120

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    text: str
    section_path: str
    granularity: str
    position_start: int
    position_end: int
    token_count: int
    checksum: str
    page_start: int | None = None
    page_end: int | None = None
    table_markdown: str | None = None


@dataclass(frozen=True)
class LayoutBlock:
    text: str
    page_number: int | None = None
    section_path: str = ""
    block_type: str = "text"
    table_markdown: str | None = None
    position_start: int = 0
    position_end: int = 0


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    words = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    return max(1, len(words))


def _split_large_block(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + MIN_CHUNK_CHARS:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def _is_markdown_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    if "|" not in lines[0] or "|" not in lines[1]:
        return False
    separator = lines[1].strip().strip("|")
    cells = [cell.strip() for cell in separator.split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def _split_aligned_row(line: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"\t+|\s{2,}", line.strip()) if cell.strip()]


def _is_aligned_table(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    rows = [_split_aligned_row(line) for line in lines if line.strip()]
    if len(rows) < 2 or any(len(row) < 2 for row in rows):
        return False
    most_common_len = max({len(row) for row in rows}, key=[len(row) for row in rows].count)
    return sum(1 for row in rows if len(row) == most_common_len) >= 2


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [
        [str(cell or "").strip().replace("\n", " ") for cell in row]
        + [""] * (width - len(row))
        for row in rows
    ]
    header = normalized[0]
    separator = ["---"] * width
    body = normalized[1:]
    all_rows = [header, separator, *body]
    return "\n".join("| " + " | ".join(row) + " |" for row in all_rows)


def _clean_table_rows(rows: list[list[Any]]) -> list[list[str]]:
    cleaned: list[list[str]] = []
    for row in rows or []:
        cells = [str(cell or "").strip() for cell in row]
        if any(cells):
            cleaned.append(cells)
    return cleaned


def _normalize_table_block(lines: list[str]) -> str:
    cleaned = [line.strip() for line in lines if line.strip()]
    if _is_markdown_table(cleaned):
        return "\n".join(cleaned)
    rows = [_split_aligned_row(line) for line in cleaned]
    return _rows_to_markdown(rows)


def build_layout_blocks_from_text(
    text: str,
    *,
    page_number: int | None = None,
    section_path: str = "",
) -> list[LayoutBlock]:
    """Split page text into text/table blocks with conservative table detection."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return []
    blocks: list[LayoutBlock] = []
    offset = 0
    current: list[tuple[str, int, int]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        lines = [item[0] for item in current]
        start = current[0][1]
        end = current[-1][2]
        paragraph = "\n".join(lines).strip()
        is_table = _is_markdown_table(lines) or _is_aligned_table(lines)
        table_markdown = _normalize_table_block(lines) if is_table else None
        blocks.append(
            LayoutBlock(
                text=table_markdown or paragraph,
                page_number=page_number,
                section_path=section_path,
                block_type="table" if is_table else "text",
                table_markdown=table_markdown,
                position_start=start,
                position_end=end,
            )
        )
        current = []

    for line in raw.split("\n"):
        start = offset
        end = start + len(line)
        offset = end + 1
        if not line.strip():
            flush()
            continue
        current.append((line, start, end))
    flush()
    return blocks


def build_layout_chunk_drafts(
    blocks: list[LayoutBlock],
    *,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[ChunkDraft]:
    """Build chunks from layout blocks while preserving page/table metadata."""
    drafts: list[ChunkDraft] = []
    for block in blocks:
        source_text = (block.table_markdown or block.text or "").strip()
        if not source_text:
            continue
        granularity = "table" if block.block_type == "table" else "section"
        local_offset = 0
        for part in _split_large_block(source_text, max_chars, overlap_chars):
            normalized = part.strip()
            if not normalized:
                continue
            pos_start = block.position_start + local_offset
            pos_end = pos_start + len(normalized)
            local_offset = max(0, pos_end - block.position_start - overlap_chars)
            drafts.append(
                ChunkDraft(
                    chunk_index=len(drafts),
                    text=normalized,
                    section_path=block.section_path,
                    granularity=granularity,
                    position_start=pos_start,
                    position_end=pos_end,
                    token_count=_approx_token_count(normalized),
                    checksum=stable_text_hash(normalized),
                    page_start=block.page_number,
                    page_end=block.page_number,
                    table_markdown=normalized if block.block_type == "table" else None,
                )
            )
    return drafts


def build_chunk_drafts(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[ChunkDraft]:
    """Build section-aware chunks from plain text or markdown."""
    if max_chars < MIN_CHUNK_CHARS:
        raise ValueError("max_chars is too small")
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []

    heading_stack: list[str] = []
    blocks: list[tuple[str, str, int, int]] = []
    current: list[str] = []
    current_start = 0
    offset = 0

    def flush(end_offset: int) -> None:
        nonlocal current, current_start
        block_text = "\n".join(part for part in current if part.strip()).strip()
        if block_text:
            section = " / ".join(heading_stack) if heading_stack else ""
            blocks.append((section, block_text, current_start, end_offset))
        current = []

    for line in raw.split("\n"):
        line_start = offset
        line_end = line_start + len(line)
        offset = line_end + 1
        heading = _HEADING_RE.match(line)
        if heading:
            flush(line_start)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_start = offset
            continue
        if not current:
            current_start = line_start
        current.append(line)
        if sum(len(part) + 1 for part in current) >= max_chars:
            flush(line_end)
            current_start = offset
    flush(len(raw))

    if not blocks:
        blocks = [("", raw, 0, len(raw))]

    drafts: list[ChunkDraft] = []
    for section, block_text, block_start, _block_end in blocks:
        local_offset = 0
        for part in _split_large_block(block_text, max_chars, overlap_chars):
            pos_start = block_start + local_offset
            pos_end = pos_start + len(part)
            local_offset = max(0, pos_end - block_start - overlap_chars)
            normalized = part.strip()
            if not normalized:
                continue
            drafts.append(
                ChunkDraft(
                    chunk_index=len(drafts),
                    text=normalized,
                    section_path=section,
                    granularity="section" if section else "paragraph",
                    position_start=pos_start,
                    position_end=pos_end,
                    token_count=_approx_token_count(normalized),
                    checksum=stable_text_hash(normalized),
                )
            )
    return drafts


def ingest_text_document(
    *,
    title: str,
    text: str,
    topic_id: int | None = None,
    doc_type: str = "text",
    author: str | None = None,
    source_path: str | None = None,
    source_url: str | None = None,
    access_scope: str = "topic",
    parser_version: str = "text-v1",
    metadata: dict[str, Any] | None = None,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> dict:
    """Ingest text without embeddings.

    Use `aingest_text_document` when embeddings are desired.
    """
    metadata_json = json.dumps(metadata or {}, ensure_ascii=True)
    run_id = api.create_corpus_ingest_run(
        topic_id=topic_id,
        source_path=source_path,
        parser_version=parser_version,
        metadata_json=metadata_json,
    )
    try:
        drafts = build_chunk_drafts(
            text, max_chars=max_chars, overlap_chars=overlap_chars
        )
        source_hash = stable_text_hash(
            "\n".join([title, source_path or "", source_url or "", text])
        )
        document_id = api.insert_corpus_document(
            topic_id=topic_id,
            title=title,
            doc_type=doc_type,
            author=author,
            source_path=source_path,
            source_url=source_url,
            source_hash=source_hash,
            access_scope=access_scope,
            parser_version=parser_version,
            index_status="indexed",
            metadata_json=metadata_json,
        )
        chunk_ids: list[int] = []
        for draft in drafts:
            chunk_ids.append(
                api.insert_corpus_chunk(
                    document_id=document_id,
                    chunk_index=draft.chunk_index,
                    text=draft.text,
                    granularity=draft.granularity,
                    section_path=draft.section_path,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    position_start=draft.position_start,
                    position_end=draft.position_end,
                    table_markdown=draft.table_markdown,
                    lexical_text=draft.text,
                    token_count=draft.token_count,
                    checksum=draft.checksum,
                    freshness_timestamp=_utcnow(),
                )
            )
        api.update_corpus_ingest_run(
            run_id,
            document_id=document_id,
            status="completed",
            chunk_count=len(chunk_ids),
        )
        return {
            "document_id": document_id,
            "ingest_run_id": run_id,
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
        }
    except Exception as exc:
        api.update_corpus_ingest_run(run_id, status="failed", error=str(exc))
        raise


def ingest_layout_document(
    *,
    title: str,
    blocks: list[LayoutBlock],
    topic_id: int | None = None,
    doc_type: str = "document",
    author: str | None = None,
    source_path: str | None = None,
    source_url: str | None = None,
    access_scope: str = "topic",
    parser_version: str = "layout-v1",
    metadata: dict[str, Any] | None = None,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> dict:
    """Ingest pre-parsed layout blocks with page and table metadata."""
    metadata_payload = {
        **(metadata or {}),
        "layout_block_count": len(blocks),
        "table_block_count": sum(1 for block in blocks if block.block_type == "table"),
    }
    metadata_json = json.dumps(metadata_payload, ensure_ascii=True)
    run_id = api.create_corpus_ingest_run(
        topic_id=topic_id,
        source_path=source_path,
        parser_version=parser_version,
        metadata_json=metadata_json,
    )
    try:
        drafts = build_layout_chunk_drafts(
            blocks, max_chars=max_chars, overlap_chars=overlap_chars
        )
        source_text = "\n".join(block.table_markdown or block.text for block in blocks)
        source_hash = stable_text_hash(
            "\n".join([title, source_path or "", source_url or "", source_text])
        )
        document_id = api.insert_corpus_document(
            topic_id=topic_id,
            title=title,
            doc_type=doc_type,
            author=author,
            source_path=source_path,
            source_url=source_url,
            source_hash=source_hash,
            access_scope=access_scope,
            parser_version=parser_version,
            index_status="indexed",
            metadata_json=metadata_json,
        )
        chunk_ids: list[int] = []
        for draft in drafts:
            chunk_ids.append(
                api.insert_corpus_chunk(
                    document_id=document_id,
                    chunk_index=draft.chunk_index,
                    text=draft.text,
                    granularity=draft.granularity,
                    section_path=draft.section_path,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    position_start=draft.position_start,
                    position_end=draft.position_end,
                    table_markdown=draft.table_markdown,
                    lexical_text="\n".join(
                        part
                        for part in (draft.section_path, draft.text, draft.table_markdown)
                        if part
                    ),
                    token_count=draft.token_count,
                    checksum=draft.checksum,
                    freshness_timestamp=_utcnow(),
                )
            )
        api.update_corpus_ingest_run(
            run_id,
            document_id=document_id,
            status="completed",
            chunk_count=len(chunk_ids),
        )
        return {
            "document_id": document_id,
            "ingest_run_id": run_id,
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
        }
    except Exception as exc:
        api.update_corpus_ingest_run(run_id, status="failed", error=str(exc))
        raise


async def aingest_text_document(*, embed: bool = True, **kwargs: Any) -> dict:
    """Async text ingestion with optional embedding creation."""
    if not embed:
        return ingest_text_document(**kwargs)

    title = kwargs["title"]
    text = kwargs["text"]
    topic_id = kwargs.get("topic_id")
    doc_type = kwargs.get("doc_type", "text")
    author = kwargs.get("author")
    source_path = kwargs.get("source_path")
    source_url = kwargs.get("source_url")
    access_scope = kwargs.get("access_scope", "topic")
    parser_version = kwargs.get("parser_version", "text-v1")
    metadata = kwargs.get("metadata")
    max_chars = kwargs.get("max_chars", DEFAULT_CHUNK_MAX_CHARS)
    overlap_chars = kwargs.get("overlap_chars", DEFAULT_CHUNK_OVERLAP_CHARS)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=True)

    run_id = api.create_corpus_ingest_run(
        topic_id=topic_id,
        source_path=source_path,
        parser_version=parser_version,
        metadata_json=metadata_json,
    )
    try:
        drafts = build_chunk_drafts(
            text, max_chars=max_chars, overlap_chars=overlap_chars
        )
        source_hash = stable_text_hash(
            "\n".join([title, source_path or "", source_url or "", text])
        )
        document_id = api.insert_corpus_document(
            topic_id=topic_id,
            title=title,
            doc_type=doc_type,
            author=author,
            source_path=source_path,
            source_url=source_url,
            source_hash=source_hash,
            access_scope=access_scope,
            parser_version=parser_version,
            index_status="indexed",
            metadata_json=metadata_json,
        )
        chunk_ids: list[int] = []
        for draft in drafts:
            embedding = await aget_embedding(draft.text)
            chunk_ids.append(
                api.insert_corpus_chunk_with_embedding(
                    document_id=document_id,
                    chunk_index=draft.chunk_index,
                    text=draft.text,
                    embedding=embedding,
                    granularity=draft.granularity,
                    section_path=draft.section_path,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    position_start=draft.position_start,
                    position_end=draft.position_end,
                    table_markdown=draft.table_markdown,
                    lexical_text=draft.text,
                    token_count=draft.token_count,
                    checksum=draft.checksum,
                    freshness_timestamp=_utcnow(),
                )
            )
        api.update_corpus_ingest_run(
            run_id,
            document_id=document_id,
            status="completed",
            chunk_count=len(chunk_ids),
        )
        return {
            "document_id": document_id,
            "ingest_run_id": run_id,
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
        }
    except Exception as exc:
        api.update_corpus_ingest_run(run_id, status="failed", error=str(exc))
        raise


def ingest_text_file(path: str | Path, **kwargs: Any) -> dict:
    path_obj = Path(path)
    text = path_obj.read_text(encoding=kwargs.pop("encoding", "utf-8"))
    doc_type = kwargs.pop(
        "doc_type", "markdown" if path_obj.suffix.lower() == ".md" else "text"
    )
    return ingest_text_document(
        title=kwargs.pop("title", path_obj.name),
        text=text,
        source_path=str(path_obj),
        doc_type=doc_type,
        **kwargs,
    )


def extract_pdf_text(path: str | Path) -> str:
    """Extract text from a PDF if pypdf is installed."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF ingestion requires pypdf. Install it or convert the PDF to markdown/text first."
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"# Page {index}\n\n{page_text.strip()}")
    return "\n\n".join(pages).strip()


def _extract_pdf_layout_blocks_pypdf(path: str | Path) -> list[LayoutBlock]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF layout ingestion requires pypdf. Install it or convert the PDF to markdown/text first."
        ) from exc

    reader = PdfReader(str(path))
    blocks: list[LayoutBlock] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        section = f"Page {index}"
        blocks.extend(
            build_layout_blocks_from_text(
                page_text,
                page_number=index,
                section_path=section,
            )
        )
    return blocks


def _extract_pdf_layout_blocks_pdfplumber(path: str | Path) -> list[LayoutBlock]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is not installed") from exc

    blocks: list[LayoutBlock] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            section = f"Page {index}"
            page_text = page.extract_text() or ""
            if page_text.strip():
                blocks.extend(
                    build_layout_blocks_from_text(
                        page_text,
                        page_number=index,
                        section_path=section,
                    )
                )
            for table_index, table in enumerate(page.extract_tables() or [], start=1):
                rows = _clean_table_rows(table)
                table_markdown = _rows_to_markdown(rows)
                if not table_markdown:
                    continue
                blocks.append(
                    LayoutBlock(
                        text=table_markdown,
                        page_number=index,
                        section_path=f"{section} / Table {table_index}",
                        block_type="table",
                        table_markdown=table_markdown,
                    )
                )
    return blocks


def available_pdf_layout_parsers() -> list[str]:
    """Return installed PDF layout parser adapters in preferred order."""
    parsers = []
    if importlib.util.find_spec("pdfplumber") is not None:
        parsers.append("pdfplumber")
    if importlib.util.find_spec("pypdf") is not None:
        parsers.append("pypdf")
    return parsers


def parse_pdf_layout(
    path: str | Path, *, parser: str = "auto"
) -> tuple[list[LayoutBlock], str]:
    """Extract PDF layout blocks and return `(blocks, parser_used)`.

    `pdfplumber` is preferred when available because it can expose tables
    directly. `pypdf` remains the dependency-light fallback.
    """
    parser = (parser or "auto").strip().lower()
    if parser not in {"auto", "pdfplumber", "pypdf"}:
        raise ValueError("parser must be 'auto', 'pdfplumber', or 'pypdf'")
    if parser in {"auto", "pdfplumber"}:
        try:
            blocks = _extract_pdf_layout_blocks_pdfplumber(path)
            if blocks or parser == "pdfplumber":
                return blocks, "pdfplumber"
        except RuntimeError:
            if parser == "pdfplumber":
                raise
    blocks = _extract_pdf_layout_blocks_pypdf(path)
    return blocks, "pypdf"


def extract_pdf_layout_blocks(
    path: str | Path, *, parser: str = "auto"
) -> list[LayoutBlock]:
    """Extract page-scoped text/table layout blocks from a PDF."""
    blocks, _parser_used = parse_pdf_layout(path, parser=parser)
    return blocks


def ingest_document_file(path: str | Path, **kwargs: Any) -> dict:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".pdf":
        pdf_parser = kwargs.pop("pdf_parser", "auto")
        blocks, parser_used = parse_pdf_layout(path_obj, parser=pdf_parser)
        return ingest_layout_document(
            title=kwargs.pop("title", path_obj.name),
            blocks=blocks,
            source_path=str(path_obj),
            doc_type=kwargs.pop("doc_type", "pdf"),
            parser_version=kwargs.pop(
                "parser_version", f"pdf-{parser_used}-layout-v2"
            ),
            **kwargs,
        )
    return ingest_text_file(path_obj, **kwargs)
