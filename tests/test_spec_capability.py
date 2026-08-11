from __future__ import annotations

import inspect

import pytest
from django.core.exceptions import ImproperlyConfigured
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from rest_framework.permissions import AllowAny
from rest_framework_services import (
    SelectorKind,
    SelectorSpec,
    ServiceSpec,
)

from rest_framework_pydantic_ai import AgentDeps, QueryParam, SpecCapability, SpecToolset, UrlKwarg
from rest_framework_pydantic_ai.spec_toolset import (
    _BASE_INSTRUCTIONS,
    _LIST_INSTRUCTION,
    UnguardedSpecWarning,
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
    kwargs.setdefault("permission_classes", [AllowAny])
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=WidgetSerializer,
        **kwargs,
    )


def retrieve_spec(**kwargs):
    kwargs.setdefault("permission_classes", [AllowAny])
    return SelectorSpec(
        kind=SelectorKind.RETRIEVE,
        selector=get_widget,
        output_serializer=WidgetSerializer,
        **kwargs,
    )


def ping_spec(**kwargs):
    kwargs.setdefault("permission_classes", [AllowAny])
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


# --- forwarding --------------------------------------------------------------
#
# The capability re-declares the toolset's constructor rather than taking
# ``**kwargs``, which buys type checking and completion at the cost of a
# signature that can drift. These two tests are what pays that cost: a knob
# added to ``SpecToolset`` and forgotten here is not a missing feature, it is an
# *unreachable* one — silently so, for every consumer that composes through the
# capability rather than attaching the toolset directly. Both of the ecosystem's
# own wrappers do exactly that.


def _params(func):
    return inspect.signature(func).parameters


def test_the_capability_accepts_every_keyword_the_toolset_does():
    toolset, capability = _params(SpecToolset.__init__), _params(SpecCapability.__init__)

    unreachable = sorted(set(toolset) - set(capability))
    assert not unreachable, f"SpecToolset keywords with no way in: {unreachable}"
    # And nothing extra beyond the one keyword that is genuinely the
    # capability's own — an addition here that the toolset knows nothing about
    # would be accepted and then quietly dropped on the floor.
    assert set(capability) - set(toolset) == {"defer_loading"}


def test_a_shared_keyword_means_the_same_thing_on_both():
    """Same name is not enough — same default and same type, or it is a trap.

    A default that drifts is the worse half: ``SpecCapability(specs)`` and
    ``SpecToolset(specs)`` would disagree about a security posture with nothing
    at either call site to show it.
    """
    toolset, capability = _params(SpecToolset.__init__), _params(SpecCapability.__init__)

    drifted = {
        name: ((param.default, param.annotation), (mirror.default, mirror.annotation))
        for name, param in toolset.items()
        if (mirror := capability[name]).default != param.default
        or mirror.annotation != param.annotation
    }
    assert not drifted, f"same keyword, different meaning: {drifted}"


def test_an_unguarded_spec_is_refused_through_the_capability_too():
    """The check the capability path could not reach before 0.13.1.

    Worth asserting on the wrapper and not only the toolset: the refusal is the
    entire security content of ``require_permissions``, and it travelled through
    a constructor that had never forwarded it.
    """
    with pytest.raises(ImproperlyConfigured, match="no permission_classes"):
        SpecCapability({"list": list_spec(permission_classes=None)})


def test_the_migration_escape_hatch_is_reachable_from_the_capability():
    """``require_permissions=False`` — documented, and until now unavailable here.

    A consumer with a large registry to migrate had no way to downgrade to the
    warning short of dropping to ``from_toolset``, which the docstring never
    told them to do.
    """
    with pytest.warns(UnguardedSpecWarning):
        cap = SpecCapability(
            {"list": list_spec(permission_classes=None)}, require_permissions=False
        )

    assert cap.get_toolset() is not None


async def test_a_bound_forwards_all_the_way_to_what_the_model_is_told():
    """One end-to-end forwarding case, on the arguments the model actually sees.

    ``max_page_size`` and ``ordering_fields`` both change the advertised schema,
    so this asserts the value survived the hop rather than that an attribute was
    assigned somewhere. ``ordering_fields`` is deprecated in favour of a
    ``filter_set``'s own ``OrderingFilter``, and its warning has to survive the
    hop too — a deprecation a consumer only hears when they attach the toolset
    directly is one the consumers composing through a capability never hear.
    """
    with pytest.deprecated_call():
        cap = SpecCapability(
            {"list": list_spec()},
            max_page_size=50,
            ordering_fields=["name", "created_at"],
        )
    tools = await cap.get_toolset().get_tools(None)
    schema = tools["list"].tool_def.parameters_json_schema["properties"]

    assert schema["limit"]["maximum"] == 50
    assert set(schema["ordering"]["enum"]) == {"name", "-name", "created_at", "-created_at"}


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
        capabilities=[
            SpecCapability(
                {"run": ServiceSpec(service=tool, permission_classes=[AllowAny], atomic=False)}
            )
        ],
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
