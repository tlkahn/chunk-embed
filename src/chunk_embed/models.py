from dataclasses import dataclass


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
