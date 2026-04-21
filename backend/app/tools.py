"""Gemini function-calling tool declarations.

These live outside `llm.py` so they can be inspected/tested without pulling
in the full LLM client, and so adding a new tool only touches this file.
"""

from __future__ import annotations

from enum import StrEnum

from google.genai import types as genai_types


class ToolName(StrEnum):
    SEARCH_MEMORY = "search_memory"
    VISIT_PAGE = "visit_page"
    EXTRACT_FROM_PAGE = "extract_from_page"


SEARCH_MEMORY = genai_types.FunctionDeclaration(
    name=ToolName.SEARCH_MEMORY.value,
    description=(
        "Search the user's local browsing history (pages, selections, form "
        "inputs they've sent). Returns the top relevant snippets."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "query": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="A natural-language search query.",
            ),
        },
        required=["query"],
    ),
)


VISIT_PAGE = genai_types.FunctionDeclaration(
    name=ToolName.VISIT_PAGE.value,
    description=(
        "Open a URL in a background tab and return the page's main content. "
        "Uses Readability to extract article-style text. Best for "
        "article/blog/docs pages."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "url": genai_types.Schema(type=genai_types.Type.STRING),
            "wait_for_selector": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="Optional CSS selector to wait for before extracting.",
            ),
        },
        required=["url"],
    ),
)


EXTRACT_FROM_PAGE = genai_types.FunctionDeclaration(
    name=ToolName.EXTRACT_FROM_PAGE.value,
    description=(
        "Open a URL and extract content from a specific CSS-targeted region. "
        "Use this for SPAs (Gmail, LinkedIn, Slack web) where the main "
        "content lives in dynamic regions and Readability returns junk."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "url": genai_types.Schema(type=genai_types.Type.STRING),
            "what": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="What to look for (free-text, included in the result).",
            ),
            "css_hint": genai_types.Schema(
                type=genai_types.Type.STRING,
                description="A CSS selector for the region of interest.",
            ),
        },
        required=["url", "what"],
    ),
)


ALL_DECLARATIONS: list[genai_types.FunctionDeclaration] = [
    SEARCH_MEMORY,
    VISIT_PAGE,
    EXTRACT_FROM_PAGE,
]


def all_tools() -> list[genai_types.Tool]:
    """One Tool grouping all declarations (what the Gemini API expects)."""
    return [genai_types.Tool(function_declarations=ALL_DECLARATIONS)]
