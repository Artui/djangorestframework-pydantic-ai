"""Reading a tool retry out of a run's messages, whichever shape it is in.

pydantic-ai below 2.60 records a tool's ``ModelRetry``, arguments that failed
validation, or a call to a tool that does not exist as a ``retry-prompt`` part.
From 2.60 that part is deprecated and no longer emitted: the retry is the call's
own ``ToolReturnPart``, marked ``outcome="retried"``. The lock holds 2.54 on
Python 3.10, which 2.55 and later do not support, and the floor job holds the
declared floor, so the suite meets both shapes and every test reading a retry
reads it here.

Matched on ``part_kind`` rather than by importing ``RetryPromptPart``, which
pydantic-ai 3 removes, and ``outcome`` is read through ``getattr`` because a
pydantic-ai old enough to sit at the floor may not carry the field at all.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic_ai.messages import ModelMessage


def is_tool_retry(part: Any) -> bool:
    """A part asking the model to make a tool call again."""
    if part.part_kind == "retry-prompt":
        return part.tool_name is not None
    return part.part_kind == "tool-return" and getattr(part, "outcome", None) == "retried"


def is_tool_answer(part: Any) -> bool:
    """A tool return that is not a retry: the call settled, one way or another."""
    return part.part_kind == "tool-return" and not is_tool_retry(part)


def tool_retries(messages: Iterable[ModelMessage]) -> list[Any]:
    """Every tool retry in ``messages``, in order."""
    return [part for message in messages for part in message.parts if is_tool_retry(part)]


def tool_answers(messages: Iterable[ModelMessage]) -> list[Any]:
    """Every tool return in ``messages`` that is not a retry, in order."""
    return [part for message in messages for part in message.parts if is_tool_answer(part)]
