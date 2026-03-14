import json

from chunk_embed.models import ChunkData, ChunksInput


class ParseError(Exception):
    pass


def parse_chunks(raw: str) -> ChunksInput:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}") from e

    for key in ("total_chunks", "mode", "chunks"):
        if key not in data:
            raise ParseError(f"Missing required key: '{key}'")

    chunks = [
        ChunkData(
            text=c["text"],
            chunk_type=c["chunk_type"],
            heading_context=c.get("heading_context", []),
            heading_level=c.get("heading_level"),
            page_number=c.get("page_number"),
            source_line_start=c["source_line_start"],
            source_line_end=c["source_line_end"],
        )
        for c in data["chunks"]
    ]

    if data["total_chunks"] != len(chunks):
        raise ParseError(
            f"Chunk count mismatch: total_chunks={data['total_chunks']}, "
            f"actual={len(chunks)}"
        )

    return ChunksInput(
        total_chunks=data["total_chunks"],
        mode=data["mode"],
        chunks=chunks,
    )
