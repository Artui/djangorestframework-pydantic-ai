from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from rest_framework_services import (
    SelectorKind,
    SelectorSpec,
    ServiceSpec,
)

from rest_framework_pydantic_ai import AgentDeps, QueryParam, SpecCapability, SpecToolset
from rest_framework_pydantic_ai.spec_capability import (
    _BASE_INSTRUCTIONS,
    _LIST_INSTRUCTION,
    _derive_instructions,
    _is_list_selector,
)
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


# --- get_instructions --------------------------------------------------------


def test_instructions_always_carry_the_error_contract():
    instr = SpecCapability({"go": ping_spec()}).get_instructions()
    assert instr is not None
    assert _BASE_INSTRUCTIONS in instr
    assert '{"error"' in instr
    assert "unknown arguments are rejected" in instr


def test_pagination_line_present_only_with_a_list_selector():
    with_list = SpecCapability({"list": list_spec()}).get_instructions()
    assert _LIST_INSTRUCTION in with_list

    # A retrieve selector is a SelectorSpec but not LIST — no pagination line.
    retrieve_only = SpecCapability({"get": retrieve_spec()}).get_instructions()
    assert _LIST_INSTRUCTION not in retrieve_only

    # A service spec is not a SelectorSpec at all — no pagination line.
    service_only = SpecCapability({"go": ping_spec()}).get_instructions()
    assert _LIST_INSTRUCTION not in service_only


def test_read_shaping_line_lists_declared_query_params_sorted():
    cap = SpecCapability(
        {"list": list_spec()},
        query_params=[QueryParam("query"), QueryParam("fields")],
    )
    instr = cap.get_instructions()
    assert instr is not None
    assert "read-shaping parameters (`fields`, `query`)" in instr


def test_read_shaping_line_absent_without_query_params():
    instr = SpecCapability({"list": list_spec()}).get_instructions()
    assert "read-shaping parameters" not in instr


def test_instructions_override_wins_verbatim():
    cap = SpecCapability({"list": list_spec()}, instructions="just do it")
    assert cap.get_instructions() == "just do it"


# --- defer_loading -----------------------------------------------------------


def test_defer_loading_defaults_off_and_is_settable():
    assert SpecCapability({"go": ping_spec()}).defer_loading is False
    cap = SpecCapability({"go": ping_spec()}, defer_loading=True)
    assert cap.defer_loading is True
    # A stable ``id`` is present, so an Agent accepts the deferred capability.
    Agent(FunctionModel(lambda m, i: ModelResponse(parts=[TextPart("ok")])), capabilities=[cap])


# --- from_toolset ------------------------------------------------------------


def test_from_toolset_wraps_a_prebuilt_toolset_with_matching_behaviour():
    toolset = SpecToolset({"list": list_spec()}, id="orders", query_params=[QueryParam("query")])
    cap = SpecCapability.from_toolset(toolset)
    assert cap.get_toolset() is toolset
    assert cap.id == "orders"
    # Same specs / query params in → same derived instructions as the direct ctor.
    assert (
        cap.get_instructions()
        == SpecCapability(
            {"list": list_spec()}, id="orders", query_params=[QueryParam("query")]
        ).get_instructions()
    )


def test_from_toolset_honours_instructions_override_and_defer_loading():
    toolset = SpecToolset({"go": ping_spec()})
    cap = SpecCapability.from_toolset(toolset, instructions="x", defer_loading=True)
    assert cap.get_instructions() == "x"
    assert cap.defer_loading is True


# --- helpers -----------------------------------------------------------------


def test_is_list_selector():
    assert _is_list_selector(list_spec()) is True
    assert _is_list_selector(retrieve_spec()) is False
    assert _is_list_selector(ping_spec()) is False


def test_derive_instructions_matches_the_public_surface():
    toolset = SpecToolset({"list": list_spec()})
    assert _derive_instructions(toolset) == SpecCapability({"list": list_spec()}).get_instructions()


# --- agent-run integration ---------------------------------------------------
#
# Drive a real ``Agent`` run loop (FunctionModel — no provider, no network)
# through the *capability* path: the wrapped toolset's tools execute in-process,
# and the capability's instructions reach the model's request.


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


async def test_capability_instructions_reach_the_model():
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
    assert captured["instructions"] is not None
    assert "business-rule failure" in captured["instructions"]
    assert _LIST_INSTRUCTION in captured["instructions"]
