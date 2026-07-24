from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from rest_framework_services import (
    SelectorKind,
    SelectorSpec,
    ServiceSpec,
)

from rest_framework_pydantic_ai import AgentDeps, QueryParam, SpecCapability, SpecToolset, UrlKwarg
from rest_framework_pydantic_ai.spec_toolset import _BASE_INSTRUCTIONS, _LIST_INSTRUCTION
from tests.testapp.models import Widget
from tests.testapp.serializers import WidgetSerializer

# --- specs under test --------------------------------------------------------


def list_widgets(user):
    """List widgets owned by the acting user."""
    return Widget.objects.filter(owner=user)


def get_widget(user, pk):
    """Fetch a single widget by primary key."""
    return Widget.objects.filter(owner=user, pk=pk)


def ping(user):
    """Ping."""
    return {"ok": True}


def list_spec(**kwargs):
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=WidgetSerializer,
        **kwargs,
    )


def retrieve_spec(**kwargs):
    return SelectorSpec(
        kind=SelectorKind.RETRIEVE,
        selector=get_widget,
        output_serializer=WidgetSerializer,
        **kwargs,
    )


def ping_spec(**kwargs):
    return ServiceSpec(service=ping, atomic=False, **kwargs)


# --- get_toolset -------------------------------------------------------------


def test_get_toolset_returns_a_spec_toolset_with_the_tools():
    cap = SpecCapability({"list": list_spec(), "get": retrieve_spec()})
    toolset = cap.get_toolset()
    assert isinstance(toolset, SpecToolset)
    assert set(toolset._specs) == {"list", "get"}


def test_id_defaults_and_forwards():
    assert SpecCapability({"go": ping_spec()}).id == "drf-specs"
    cap = SpecCapability({"go": ping_spec()}, id="orders")
    assert cap.id == "orders"
    assert cap.get_toolset().id == "orders"


# --- instructions delegation -------------------------------------------------
#
# The conventions live on the toolset's ``get_instructions`` (see
# ``test_spec_toolset``); the capability deliberately does *not* re-emit them —
# Pydantic-AI collects the owned toolset's instructions, so overriding here would
# duplicate them in the prompt (see the agent-run guard below).


def test_capability_does_not_emit_its_own_instructions():
    # Inherits ``AbstractCapability.get_instructions`` → ``None`` (delegates).
    assert SpecCapability({"list": list_spec()}).get_instructions() is None


async def test_instructions_override_forwards_to_the_toolset():
    cap = SpecCapability({"list": list_spec()}, instructions="just do it")
    assert await cap.get_toolset().get_instructions(None) == "just do it"


# --- defer_loading -----------------------------------------------------------


def test_defer_loading_defaults_off_and_is_settable():
    assert SpecCapability({"go": ping_spec()}).defer_loading is False
    cap = SpecCapability({"go": ping_spec()}, defer_loading=True)
    assert cap.defer_loading is True
    # A stable ``id`` is present, so an Agent accepts the deferred capability.
    Agent(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("ok")])), capabilities=[cap])


# --- from_toolset ------------------------------------------------------------


def test_from_toolset_wraps_a_prebuilt_toolset():
    toolset = SpecToolset({"list": list_spec()}, id="orders", query_params=[QueryParam("query")])
    cap = SpecCapability.from_toolset(toolset)
    assert cap.get_toolset() is toolset
    assert cap.id == "orders"


def test_from_toolset_honours_defer_loading():
    toolset = SpecToolset({"go": ping_spec()})
    cap = SpecCapability.from_toolset(toolset, defer_loading=True)
    assert cap.defer_loading is True


async def test_url_kwargs_forward_to_the_built_toolset():
    cap = SpecCapability({"list": list_spec()}, url_kwargs=[UrlKwarg("parent_pk")])
    tools = await cap.get_toolset().get_tools(None)
    assert "parent_pk" in tools["list"].tool_def.parameters_json_schema["properties"]


# --- agent-run integration ---------------------------------------------------
#
# Drive a real ``Agent`` run loop (FunctionModel — no provider, no network)
# through the *capability* path: the wrapped toolset's tools execute in-process,
# and its instructions reach the model's request exactly once.


async def test_capability_executes_its_tool_in_process():
    seen = {}

    def tool(user):
        """Run it."""
        seen["user"] = user
        return {"ok": True}

    def model_fn(messages, info):
        if any(part.part_kind == "tool-return" for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(tool_name="run", args={})])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AgentDeps,
        capabilities=[SpecCapability({"run": ServiceSpec(service=tool, atomic=False)})],
    )
    result = await agent.run("go", deps=AgentDeps(user="alice"))
    assert result.output == "done"
    assert seen["user"] == "alice"


async def test_capability_instructions_reach_the_model_exactly_once():
    captured = {}

    def model_fn(messages, info):
        captured["instructions"] = messages[-1].instructions
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AgentDeps,
        capabilities=[SpecCapability({"list": list_spec()})],
    )
    result = await agent.run("go", deps=AgentDeps(user="alice"))
    assert result.output == "done"
    instr = captured["instructions"]
    assert instr is not None
    assert _LIST_INSTRUCTION in instr
    # The conventions come from the toolset only — not doubled by the capability.
    assert instr.count(_BASE_INSTRUCTIONS) == 1
