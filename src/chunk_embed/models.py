from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ChunkData:
    text: str
    chunk_type: str
    heading_context: list[str]
    heading_level: int | None
    page_number: int | None
    source_line_start: int
    source_line_end: int


@dataclass(frozen=True)
class ChunksInput:
    total_chunks: int
    mode: str
    chunks: list[ChunkData]


@dataclass(frozen=True)
class DocumentInfo:
    id: int
    source_path: str
    mode: str
    total_chunks: int
    created_at: datetime


@dataclass(frozen=True)
class ChunkSummary:
    chunk_type: str
    count: int
    total_chars: int


ALL_CHUNK_TYPES: frozenset[str] = frozenset({
    "paragraph", "heading", "list_item", "block_quote",
    "definition_item", "theorem",
    "code_block", "math_block", "table",
})

TEXTUAL_TYPES: frozenset[str] = frozenset({
    "paragraph", "heading", "list_item", "block_quote",
    "definition_item", "theorem",
})


@dataclass(frozen=True)
class SearchResult:
    similarity: float
    source_path: str
    text: str
    chunk_type: str
    heading_context: list[str]
    heading_level: int | None
    page_number: int | None
    source_line_start: int
    source_line_end: int
