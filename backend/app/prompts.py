"""System prompts and prompt builders.

Separated from `llm.py` so prompt wording can be tweaked without
touching client code, and so tests can inspect prompts directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from app.store import RetrievedChunk

FINAL_ANSWER_SYSTEM_INSTRUCTION = (
    "You are pc_agent, the user's personal browser-history assistant. "
    "Answer the user's question using ONLY the provided memories from their "
    "own browsing. If the memories don't contain enough information, say so "
    "honestly rather than guessing. Always include citations: copy the "
    "URL/ts/snippet of every memory you actually used. Be concise."
)


AGENTIC_SYSTEM_INSTRUCTION = (
    "You are pc_agent, the user's personal browser-history assistant. "
    "You have three tools:\n"
    "  1. search_memory(query): search the user's local browsing history.\n"
    "  2. visit_page(url, wait_for_selector?): open a tab in the background\n"
    "     and read the page. Use ONLY when the user asks you to go check\n"
    "     something live (e.g. 'check what X replied').\n"
    "  3. extract_from_page(url, what, css_hint?): read a known page with a\n"
    "     targeted extraction. Use this for SPAs (Gmail, LinkedIn) where\n"
    "     Readability returns garbage; pass a CSS selector for the region.\n\n"
    "Guidance:\n"
    "  - Always start with search_memory to see what the user already has.\n"
    "  - Only use visit_page / extract_from_page when memory alone is\n"
    "    insufficient AND the user is clearly asking you to fetch.\n"
    "  - When you have enough information, write a short final answer that\n"
    "    names the URLs and timestamps you used. The system will then\n"
    "    convert your answer into a structured response with citations."
)


def build_final_answer_prompt(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    """Format the retrieved chunks into a prompt for the final-answer call."""
    if not retrieved:
        body = "(no memories matched the question)"
    else:
        body = "\n\n".join(format_chunk(i, c) for i, c in enumerate(retrieved, start=1))
    return (
        f"# Question\n{question}\n\n"
        f"# Memories from the user's browsing\n{body}\n\n"
        "# Task\nAnswer the question using only the memories above. "
        "Include citations for the memories you used."
    )


def format_chunk(idx: int, chunk: RetrievedChunk) -> str:
    iso = datetime.fromtimestamp(chunk.ts / 1000, tz=UTC).isoformat()
    title = chunk.title or chunk.url
    return (
        f"[{idx}] {title}\n"
        f"    url: {chunk.url}\n"
        f"    ts:  {chunk.ts}  ({iso})\n"
        f"    text: {chunk.text}"
    )


def format_search_results(chunks: Sequence[RetrievedChunk]) -> str:
    """Compact representation for feeding back into the agentic loop as a
    function_response to `search_memory`."""
    if not chunks:
        return "(no matches)"
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        title = c.title or c.url
        text = c.text[:600] + ("..." if len(c.text) > 600 else "")
        lines.append(
            f"[{i}] chunk_id={c.chunk_id} url={c.url} ts={c.ts}\n"
            f"    title: {title}\n"
            f"    text:  {text}"
        )
    return "\n\n".join(lines)


FINAL_ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "ts": {"type": "integer"},
                    "snippet": {"type": "string"},
                },
                "required": ["url", "ts", "snippet"],
            },
        },
    },
    "required": ["answer", "citations"],
}
