"""The packaged test doubles, tested against a real run loop.

A harness shipped in the wheel is API: it is what a consumer's suite will
depend on, so its own contract needs pinning. In particular the branch order
inside ``tool_calling_model`` -- retry prompt before tool return -- is the part
that reads as arbitrary and is not, and the only way to prove it is a run that
actually retries.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior
from rest_framework import serializers
from rest_framework.permissions import AllowAny
from rest_framework_services import ServiceSpec

from rest_framework_pydantic_ai import AgentDeps, SpecToolset
from rest_framework_pydantic_ai.testing import instruction_capturing_model, tool_calling_model


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
