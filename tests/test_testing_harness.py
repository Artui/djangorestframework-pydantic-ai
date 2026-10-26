"""The packaged test doubles, tested against a real run loop.

A harness shipped in the wheel is API: it is what a consumer's suite will
depend on, so its own contract needs pinning. In particular the branch order
inside ``tool_calling_model`` -- retry before tool return -- is the part that
reads as arbitrary and is not. A run that actually retries proves it against
the installed pydantic-ai; a hand-built history proves it for the retry shape
that pydantic-ai does not write.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from rest_framework import serializers
from rest_framework.permissions import AllowAny
from rest_framework_services import ServiceSpec

from rest_framework_pydantic_ai import AgentDeps, SpecToolset
from rest_framework_pydantic_ai.testing import (
    instruction_capturing_model,
    streaming_tool_calling_model,
    tool_calling_model,
)


def _toolset(service, name="t"):
    # ``atomic=False`` because none of these services touch the ORM: the point
    # here is the run loop, not dispatch.
    return SpecToolset(
        {name: ServiceSpec(service=service, permission_classes=[AllowAny], atomic=False)}
    )


async def test_tool_calling_model_drives_one_call_and_stops():
    seen = {}

    def ping(user):
        """Ping."""
        seen["user"] = user
        return {"ok": True}

    agent = Agent(tool_calling_model("t"), deps_type=AgentDeps, toolsets=[_toolset(ping)])

    result = await agent.run("go", deps=AgentDeps(user="alice"))

    assert result.output == "done"
    assert seen["user"] == "alice"


async def test_final_text_is_the_run_output():
    def ping(user):
        """Ping."""
        return {}

    agent = Agent(
        tool_calling_model("t", final_text="all set"),
        deps_type=AgentDeps,
        toolsets=[_toolset(ping)],
    )

    assert (await agent.run("go", deps=AgentDeps(user="u"))).output == "all set"


class _ModeInputSerializer(serializers.Serializer):
    mode = serializers.CharField()


async def test_retry_args_default_to_repeating_the_first_arguments():
    """Documented as fine for *did the retry arrive* and wrong for recovery.

    Pinned because the default is the trap: a spec that rejected these arguments
    once rejects them again, so a caller who wanted recovery and omitted
    ``retry_args`` gets a run that exhausts its retries instead of settling.

    The recovery path -- distinct ``retry_args``, run completes -- is covered by
    ``test_agent_run_recovers_from_model_retry`` in the toolset suite, which is
    where the retry contract itself belongs.
    """
    calls: list[str] = []

    def always_retries(data, user):
        """Always asks for another attempt."""
        calls.append(data["mode"])
        raise ModelRetry("never satisfied")

    toolset = SpecToolset(
        {
            "t": ServiceSpec(
                service=always_retries,
                input_serializer=_ModeInputSerializer,
                permission_classes=[AllowAny],
                atomic=False,
            )
        }
    )
    agent = Agent(tool_calling_model("t", {"mode": "bad"}), deps_type=AgentDeps, toolsets=[toolset])

    with pytest.raises(UnexpectedModelBehavior):
        await agent.run("go", deps=AgentDeps(user="u"))

    assert calls == ["bad", "bad"], "the default repeated the first arguments"


async def test_instruction_capturing_model_records_what_the_toolset_contributed():
    def ping(user):
        """Ping."""
        return {}

    captured: dict[str, Any] = {}
    agent = Agent(
        instruction_capturing_model(captured), deps_type=AgentDeps, toolsets=[_toolset(ping)]
    )

    result = await agent.run("go", deps=AgentDeps(user="u"))

    assert result.output == "done"
    # The toolset's conventions arrive on the request, not on the toolset.
    assert captured["instructions"] is not None


async def test_instruction_capturing_model_honours_final_text():
    captured: dict[str, Any] = {}
    agent = Agent(instruction_capturing_model(captured, final_text="ok"), deps_type=AgentDeps)

    assert (await agent.run("go", deps=AgentDeps(user="u"))).output == "ok"


def _ping(user: Any) -> Any:
    """Ping."""
    return {"ok": True}


async def test_the_streaming_double_serves_a_streamed_run() -> None:
    """The reason this double exists: the other one cannot serve one at all.

    ``FunctionModel`` built with ``function=`` alone raises on a streamed
    request, and AG-UI always streams -- so a consumer testing what its browser
    receives got a run that died before the toolset was touched, reported
    through the transport's redaction as "The run failed", naming nothing.
    """
    agent = Agent(
        streaming_tool_calling_model("t"),
        deps_type=AgentDeps,
        toolsets=[_toolset(_ping)],
    )

    async with agent.run_stream("go", deps=AgentDeps(user="alice")) as run:
        output = await run.get_output()

    assert output == "done"


async def test_the_streaming_double_still_serves_a_plain_run() -> None:
    """Both functions are supplied, so one double covers either transport.

    A test should not have to know whether the thing under it streams.
    """
    agent = Agent(
        streaming_tool_calling_model("t"),
        deps_type=AgentDeps,
        toolsets=[_toolset(_ping)],
    )

    result = await agent.run("go", deps=AgentDeps(user="alice"))

    assert result.output == "done"


async def test_the_streaming_double_retries_in_the_same_order() -> None:
    """The branch order is shared with the non-streamed double, not copied.

    A retry has to be read before a tool return, because from pydantic-ai 2.60
    a retried call is a tool return marked ``outcome="retried"``, and reading
    them the other way round ends the run on the failed attempt. This asserts
    the streamed path has the same order -- the reason the decision lives in
    one function.
    """
    attempts: list[str] = []

    def picky(data, user):
        """Refuses the first attempt and accepts the second."""
        attempts.append(data["mode"])
        if len(attempts) == 1:
            raise ModelRetry("try again")
        return {"mode": data["mode"]}

    toolset = SpecToolset(
        {
            "t": ServiceSpec(
                service=picky,
                input_serializer=_ModeInputSerializer,
                permission_classes=[AllowAny],
                atomic=False,
            )
        }
    )
    agent = Agent(
        streaming_tool_calling_model("t", {"mode": "first"}, retry_args={"mode": "second"}),
        deps_type=AgentDeps,
        toolsets=[toolset],
    )

    async with agent.run_stream("go", deps=AgentDeps(user="u")) as run:
        await run.get_output()

    assert attempts == ["first", "second"], "the streamed double retried out of order"


def _legacy_retry_prompt() -> RetryPromptPart:
    # Deprecated from pydantic-ai 2.60, which no longer emits it. Built here
    # because a consumer's own history, or a pydantic-ai below 2.60, still has it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RetryPromptPart(content="try again", tool_name="t", tool_call_id="c1")


def _retried_tool_return() -> ToolReturnPart:
    # What pydantic-ai 2.60 and later records for a tool's ``ModelRetry`` or a
    # call whose arguments failed validation: the call's own return, marked.
    # A dataclass does not enforce its ``Literal`` fields, so this builds on a
    # pydantic-ai that predates the ``"retried"`` value too.
    return ToolReturnPart(tool_name="t", content="try again", tool_call_id="c1", outcome="retried")


@pytest.mark.parametrize(
    "retry",
    [_legacy_retry_prompt, _retried_tool_return],
    ids=["retry-prompt", "retried-tool-return"],
)
@pytest.mark.parametrize("double", [tool_calling_model, streaming_tool_calling_model])
def test_a_retry_in_either_shape_gets_the_retry_args(retry: Any, double: Any) -> None:
    """Whichever shape the installed pydantic-ai writes a retry in.

    Driven with a hand-built history rather than a run, so the shape under test
    does not depend on which pydantic-ai is installed: a run on 2.60 or later
    only ever produces the second, and one below it only the first. Each case
    holds one arm of the double's retry check.
    """
    model = double("t", {"mode": "first"}, retry_args={"mode": "second"})
    history = [
        ModelRequest(parts=[UserPromptPart("go")]),
        ModelResponse(parts=[ToolCallPart(tool_name="t", args={"mode": "first"})]),
        ModelRequest(parts=[retry()]),
    ]

    (part,) = model.function(history, None).parts

    assert isinstance(part, ToolCallPart), f"the double ended the run on a retry: {part!r}"
    assert part.args == {"mode": "second"}


def test_a_successful_tool_return_still_ends_the_run() -> None:
    """The other side of the retry check: an ordinary return is not a retry."""
    model = tool_calling_model("t", {"mode": "first"}, final_text="finished")
    history = [
        ModelRequest(parts=[UserPromptPart("go")]),
        ModelResponse(parts=[ToolCallPart(tool_name="t", args={"mode": "first"})]),
        ModelRequest(parts=[ToolReturnPart(tool_name="t", content="ok", tool_call_id="c1")]),
    ]

    (part,) = model.function(history, None).parts

    assert isinstance(part, TextPart)
    assert part.content == "finished"
