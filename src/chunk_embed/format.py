from __future__ import annotations

import json

from chunk_embed.models import SearchResult

_TRUNCATE_LEN = 200


def format_results_human(results: list[SearchResult]) -> str:
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        lines = f"lines {r.source_line_start}\u2013{r.source_line_end}"
        meta = f"    {r.chunk_type}  {lines}"
        if r.page_number is not None:
            meta += f"  page {r.page_number}"

        text_preview = r.text
        if len(text_preview) > _TRUNCATE_LEN:
            text_preview = text_preview[:_TRUNCATE_LEN] + "..."

        block = f"[{i}] {r.similarity:.4f}  {r.source_path}\n{meta}"
        if r.heading_context:
            block += f"\n    {' > '.join(r.heading_context)}"
        block += f"\n    {text_preview}"
        parts.append(block)

    return "\n\n".join(parts)


def format_results_json(results: list[SearchResult]) -> str:
    data = [
        {
            "rank": i,
            "similarity": r.similarity,
            "source_path": r.source_path,
            "chunk_type": r.chunk_type,
            "heading_context": r.heading_context,
            "page_number": r.page_number,
            "source_line_start": r.source_line_start,
            "source_line_end": r.source_line_end,
            "text": r.text,
        }
        for i, r in enumerate(results, 1)
    ]
    return json.dumps(data, indent=2, ensure_ascii=False)
