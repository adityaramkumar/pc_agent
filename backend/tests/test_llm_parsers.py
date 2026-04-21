"""Tests for the pure parsing helpers in `app.llm`.

These don't hit the Gemini API; they verify we can turn the SDK's
Content responses into our dataclasses, and that malformed JSON in the
final-answer response gracefully degrades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.llm import parse_agentic_response, parse_final_answer

# --- Stub shapes matching what google-genai returns ----------------------


@dataclass
class FakeFunctionCall:
    name: str
    args: dict[str, Any]


@dataclass
class FakePart:
    text: str | None = None
    function_call: FakeFunctionCall | None = None


@dataclass
class FakeContent:
    parts: list[FakePart]
    role: str = "model"


@dataclass
class FakeCandidate:
    content: FakeContent


@dataclass
class FakeAgenticResponse:
    candidates: list[FakeCandidate]


@dataclass
class FakeFinalResponse:
    text: str | None


# --- parse_agentic_response ----------------------------------------------


def test_parse_agentic_empty_response() -> None:
    result = parse_agentic_response(FakeAgenticResponse(candidates=[]))
    assert result.function_call is None
    assert result.text is None
    assert result.raw_content is None


def test_parse_agentic_text_only() -> None:
    response = FakeAgenticResponse(
        candidates=[
            FakeCandidate(
                content=FakeContent(parts=[FakePart(text="hello"), FakePart(text=" world")])
            )
        ]
    )
    result = parse_agentic_response(response)
    assert result.function_call is None
    assert result.text == "hello world"
    assert result.raw_content is not None


def test_parse_agentic_function_call() -> None:
    response = FakeAgenticResponse(
        candidates=[
            FakeCandidate(
                content=FakeContent(
                    parts=[
                        FakePart(
                            function_call=FakeFunctionCall(
                                name="search_memory",
                                args={"query": "pricing"},
                            )
                        )
                    ]
                )
            )
        ]
    )
    result = parse_agentic_response(response)
    assert result.function_call is not None
    assert result.function_call.name == "search_memory"
    assert result.function_call.args == {"query": "pricing"}
    assert result.text is None


def test_parse_agentic_function_call_takes_priority_over_text() -> None:
    """If the model mixed text and a function call, we pick the call."""
    response = FakeAgenticResponse(
        candidates=[
            FakeCandidate(
                content=FakeContent(
                    parts=[
                        FakePart(
                            function_call=FakeFunctionCall(name="visit_page", args={"url": "u"})
                        ),
                        FakePart(text="some trailing text"),
                    ]
                )
            )
        ]
    )
    result = parse_agentic_response(response)
    assert result.function_call is not None
    assert result.function_call.name == "visit_page"


# --- parse_final_answer --------------------------------------------------


def test_parse_final_answer_valid_json() -> None:
    payload = (
        '{"answer": "the answer is 42",'
        ' "citations": ['
        '  {"url": "https://ex.com", "ts": 123, "snippet": "a quote"}'
        " ]}"
    )
    result = parse_final_answer(FakeFinalResponse(text=payload))
    assert result.answer == "the answer is 42"
    assert len(result.citations) == 1
    assert result.citations[0].url == "https://ex.com"
    assert result.citations[0].ts == 123
    assert result.citations[0].snippet == "a quote"


def test_parse_final_answer_empty_citations() -> None:
    result = parse_final_answer(FakeFinalResponse(text='{"answer": "hi", "citations": []}'))
    assert result.answer == "hi"
    assert result.citations == []


def test_parse_final_answer_malformed_json_returns_raw() -> None:
    result = parse_final_answer(FakeFinalResponse(text="not json"))
    assert result.answer == "not json"
    assert result.citations == []


def test_parse_final_answer_missing_text() -> None:
    result = parse_final_answer(FakeFinalResponse(text=None))
    assert "no response" in result.answer.lower()
    assert result.citations == []


def test_parse_final_answer_ignores_non_dict_citations() -> None:
    payload = '{"answer": "x", "citations": ["bad", {"url": "u", "ts": 1, "snippet": "s"}]}'
    result = parse_final_answer(FakeFinalResponse(text=payload))
    assert len(result.citations) == 1
    assert result.citations[0].url == "u"
