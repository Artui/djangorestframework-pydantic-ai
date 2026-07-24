from __future__ import annotations

from types import SimpleNamespace

import django_filters
import pytest
from django.contrib.auth.models import User
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from rest_framework import serializers
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import BasePermission
from rest_framework_services import (
    SelectorKind,
    SelectorSpec,
    ServiceError,
    ServiceSpec,
    ServiceValidationError,
    UnknownArguments,
)
from typing_extensions import TypedDict, Unpack

from rest_framework_pydantic_ai import AgentDeps, QueryParam, SpecToolset, UrlKwarg
from rest_framework_pydantic_ai.spec_toolset import (
    _BASE_INSTRUCTIONS,
    _LIST_INSTRUCTION,
    _call_spec,
    _derive_instructions,
    _is_list_selector,
    _output_extras,
    _paginate,
    _split_order,
)
from tests.testapp.models import Widget
from tests.testapp.serializers import WidgetInputSerializer, WidgetSerializer

# --- specs under test --------------------------------------------------------


def list_widgets(user):
    """List widgets owned by the acting user."""
    return Widget.objects.filter(owner=user)


def get_widget(user, pk):
    """Fetch a single widget by primary key."""
    return Widget.objects.filter(owner=user, pk=pk)


def create_widget(data, user):
    """Create a widget for the acting user."""
    return Widget.objects.create(owner=user, **data)


def boom():
    """Always fails with a business error."""
    raise ServiceError("nope")


def reject():
    """Always rejects its input."""
    raise ServiceValidationError("bad input")


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


def create_spec(**kwargs):
    return ServiceSpec(
        service=create_widget,
        input_serializer=WidgetInputSerializer,
        output_selector_spec=SelectorSpec(
            kind=SelectorKind.RETRIEVE, output_serializer=WidgetSerializer
        ),
        **kwargs,
    )


class DenyAll(BasePermission):
    def has_permission(self, request, view):
        return False


def ctx_for(user):
    return SimpleNamespace(deps=AgentDeps(user=user))


# --- get_instructions --------------------------------------------------------
#
# ``get_instructions`` derives the conventions block from the specs; it ignores
# ``ctx`` (no per-run state), so the tests pass ``None``.


async def test_instructions_carry_the_error_contract():
    instr = await SpecToolset({"go": create_spec()}).get_instructions(None)
    assert instr is not None
    assert _BASE_INSTRUCTIONS in instr
    assert '{"error"' in instr
    assert "unknown arguments are rejected" in instr


async def test_pagination_line_present_only_with_a_list_selector():
    with_list = await SpecToolset({"list": list_spec()}).get_instructions(None)
    assert _LIST_INSTRUCTION in with_list

    # A retrieve selector is a SelectorSpec but not LIST — no pagination line.
    retrieve_only = await SpecToolset({"get": retrieve_spec()}).get_instructions(None)
    assert _LIST_INSTRUCTION not in retrieve_only

    # A service spec is not a SelectorSpec at all — no pagination line.
    service_only = await SpecToolset({"go": create_spec()}).get_instructions(None)
    assert _LIST_INSTRUCTION not in service_only


async def test_read_shaping_line_lists_declared_query_params_sorted():
    toolset = SpecToolset(
        {"list": list_spec()},
        query_params=[QueryParam("query"), QueryParam("fields")],
    )
    instr = await toolset.get_instructions(None)
    assert instr is not None
    assert "read-shaping parameters (`fields`, `query`)" in instr


async def test_read_shaping_line_absent_without_query_params():
    instr = await SpecToolset({"list": list_spec()}).get_instructions(None)
    assert "read-shaping parameters" not in instr


async def test_instructions_override_wins_verbatim():
    toolset = SpecToolset({"list": list_spec()}, instructions="just do it")
    assert await toolset.get_instructions(None) == "just do it"


def test_is_list_selector():
    assert _is_list_selector(list_spec()) is True
    assert _is_list_selector(retrieve_spec()) is False
    assert _is_list_selector(create_spec()) is False


def test_derive_instructions_matches_get_instructions_input():
    toolset = SpecToolset({"list": list_spec()}, query_params=[QueryParam("query")])
    # The public ``get_instructions`` derives from exactly these two mappings.
    assert _derive_instructions(toolset._specs, toolset._tool_query_params) == "\n".join(
        [
            _BASE_INSTRUCTIONS,
            _LIST_INSTRUCTION,
            "- Some tools accept read-shaping parameters (`query`) that adjust the shape "
            "of the returned data without filtering it.",
        ]
    )


# --- get_tools ---------------------------------------------------------------


async def test_get_tools_builds_function_tools():
    toolset = SpecToolset({"list_widgets": list_spec(), "create_widget": create_spec()})
    tools = await toolset.get_tools(None)
    assert set(tools) == {"list_widgets", "create_widget"}
    # "function" is the in-process kind — the run loop invokes call_tool rather
    # than deferring the call to the client.
    assert all(tool.tool_def.kind == "function" for tool in tools.values())


async def test_get_tools_description_from_docstring():
    toolset = SpecToolset({"list_widgets": list_spec()})
    tools = await toolset.get_tools(None)
    assert tools["list_widgets"].tool_def.description == "List widgets owned by the acting user."


async def test_list_selector_tool_advertises_pagination_args():
    toolset = SpecToolset({"list_widgets": list_spec()})
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert {"page", "limit", "order"} <= set(props)


async def test_annotations_mark_read_vs_write():
    toolset = SpecToolset({"list_widgets": list_spec(), "create_widget": create_spec()})
    tools = await toolset.get_tools(None)
    assert tools["list_widgets"].tool_def.metadata == {"annotations": {"readOnlyHint": True}}
    assert tools["create_widget"].tool_def.metadata == {"annotations": {"readOnlyHint": False}}


async def test_selector_without_callable_has_no_description():
    spec = SelectorSpec(kind=SelectorKind.LIST, selector=None, output_serializer=WidgetSerializer)
    toolset = SpecToolset({"empty": spec})
    tools = await toolset.get_tools(None)
    assert tools["empty"].tool_def.description is None


async def test_id_property():
    assert SpecToolset({"list_widgets": list_spec()}).id == "drf-specs"
    assert SpecToolset({"list_widgets": list_spec()}, id="custom").id == "custom"


async def test_max_retries_default_and_override():
    tools = await SpecToolset({"list_widgets": list_spec()}).get_tools(None)
    assert tools["list_widgets"].max_retries == 1
    tools = await SpecToolset({"list_widgets": list_spec()}, max_retries=3).get_tools(None)
    assert tools["list_widgets"].max_retries == 3


# --- call_tool (async wrapper + identity) ------------------------------------


async def test_call_tool_uses_default_deps_user():
    seen = {}

    def ping(user):
        """Ping."""
        seen["user"] = user
        return {"ok": True}

    toolset = SpecToolset({"ping": ServiceSpec(service=ping, atomic=False)})
    result = await toolset.call_tool("ping", {}, ctx_for("alice"), None)
    assert result == {"ok": True}
    assert seen["user"] == "alice"


async def test_call_tool_honours_custom_get_user():
    seen = {}

    def ping(user):
        """Ping."""
        seen["user"] = user
        return {"ok": True}

    toolset = SpecToolset(
        {"ping": ServiceSpec(service=ping, atomic=False)}, get_user=lambda ctx: ctx.principal
    )
    await toolset.call_tool("ping", {}, SimpleNamespace(principal="bob"), None)
    assert seen["user"] == "bob"


# --- selector dispatch -------------------------------------------------------


@pytest.mark.django_db
def test_list_selector_renders_owned_widgets():
    user = User.objects.create(username="u")
    other = User.objects.create(username="o")
    Widget.objects.create(name="a", price=1, owner=user)
    Widget.objects.create(name="b", price=2, owner=other)
    result = _call_spec(list_spec(), user, {})
    assert [w["name"] for w in result] == ["a"]


@pytest.mark.django_db
def test_list_selector_orders_and_limits():
    user = User.objects.create(username="u")
    for name, price in [("a", 3), ("b", 1), ("c", 2)]:
        Widget.objects.create(name=name, price=price, owner=user)
    result = _call_spec(list_spec(), user, {"order": "price", "limit": 2})
    assert [w["name"] for w in result] == ["b", "c"]


@pytest.mark.django_db
def test_list_selector_second_page():
    user = User.objects.create(username="u")
    for name, price in [("a", 1), ("b", 2), ("c", 3)]:
        Widget.objects.create(name=name, price=price, owner=user)
    result = _call_spec(list_spec(), user, {"order": "price", "page": 2, "limit": 2})
    assert [w["name"] for w in result] == ["c"]


@pytest.mark.django_db
def test_bad_order_field_becomes_model_retry():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    with pytest.raises(ModelRetry):
        _call_spec(list_spec(), user, {"order": "nope"})


@pytest.mark.django_db
def test_retrieve_selector_found():
    user = User.objects.create(username="u")
    widget = Widget.objects.create(name="a", price=1, owner=user)
    result = _call_spec(retrieve_spec(), user, {"pk": widget.pk})
    assert result["name"] == "a"


@pytest.mark.django_db
def test_retrieve_selector_not_found_is_error_payload():
    user = User.objects.create(username="u")
    result = _call_spec(retrieve_spec(), user, {"pk": 999})
    assert result == {"error": "not found"}


# --- service dispatch --------------------------------------------------------


@pytest.mark.django_db
def test_create_service_renders_output():
    user = User.objects.create(username="u")
    result = _call_spec(create_spec(), user, {"name": "z", "price": 5})
    assert result["name"] == "z"
    assert Widget.objects.filter(name="z", owner=user).exists()


@pytest.mark.django_db
def test_create_service_validation_error_is_model_retry():
    user = User.objects.create(username="u")
    with pytest.raises(ModelRetry):
        _call_spec(create_spec(), user, {"name": "z", "price": -1})


def test_service_error_is_returned_as_payload():
    result = _call_spec(ServiceSpec(service=boom, atomic=False), object(), {})
    assert result == {"error": "nope"}


def test_service_validation_error_is_model_retry():
    with pytest.raises(ModelRetry):
        _call_spec(ServiceSpec(service=reject, atomic=False), object(), {})


# --- permissions -------------------------------------------------------------


def test_denied_permission_raises():
    with pytest.raises(PermissionDenied):
        _call_spec(list_spec(permission_classes=[DenyAll]), object(), {})


# --- pure helpers ------------------------------------------------------------


def test_split_order_variants():
    assert _split_order(None) == []
    assert _split_order("") == []
    assert _split_order("name") == ["name"]
    assert _split_order(" -price , name ") == ["-price", "name"]
    assert _split_order("a,,b") == ["a", "b"]


def test_paginate_variants():
    assert _paginate([1, 2, 3], None, None) == [1, 2, 3]
    assert _paginate([1, 2, 3, 4], None, 2) == [1, 2]
    assert _paginate([1, 2, 3, 4], 1, 2) == [1, 2]
    assert _paginate([1, 2, 3, 4], 2, 2) == [3, 4]


def test_output_extras_branches():
    sentinel = object()
    assert _output_extras(list_spec(), sentinel, many=True) == {"page": sentinel}
    assert _output_extras(create_spec(), sentinel, many=False) == {"result": sentinel}
    assert _output_extras(retrieve_spec(), sentinel, many=False) == {"instance": sentinel}


# --- tool-name validation ----------------------------------------------------


@pytest.mark.parametrize("bad", ["has space", "bang!", "", "x" * 65])
def test_invalid_tool_name_raises_at_construction(bad):
    with pytest.raises(ValueError, match="tool names"):
        SpecToolset({bad: list_spec()})


def test_valid_tool_names_are_accepted():
    # letters, digits, underscore, hyphen, up to 64 chars — no error.
    SpecToolset({"list_widgets-v2": list_spec()})


# --- pagination arg validation -----------------------------------------------


@pytest.mark.django_db
def test_string_limit_is_coerced_to_int():
    user = User.objects.create(username="u")
    Widget.objects.create(owner=user, name="a", price=1)
    Widget.objects.create(owner=user, name="b", price=2)
    result = _call_spec(list_spec(), user, {"limit": "1"})
    assert len(result) == 1


@pytest.mark.parametrize("value", ["abc", "2.5", "-1", 0, -3, 2.0])
def test_non_positive_int_limit_is_model_retry(value):
    with pytest.raises(ModelRetry, match="positive integer"):
        _call_spec(list_spec(), object(), {"limit": value})


def test_bool_page_is_model_retry():
    # ``True`` is an ``int`` subclass but never a valid count.
    with pytest.raises(ModelRetry, match="positive integer"):
        _call_spec(list_spec(), object(), {"page": True})


def test_non_string_order_is_model_retry():
    with pytest.raises(ModelRetry, match="order"):
        _call_spec(list_spec(), object(), {"order": ["price"]})


# --- unknown-arguments knob --------------------------------------------------


@pytest.mark.django_db
def test_unknown_argument_rejected_by_default():
    user = User.objects.create(username="u")
    with pytest.raises(ModelRetry, match="bogus"):
        _call_spec(create_spec(), user, {"name": "z", "price": 5, "bogus": 1})


@pytest.mark.django_db
def test_unknown_argument_ignored_when_configured():
    user = User.objects.create(username="u")
    result = _call_spec(
        create_spec(),
        user,
        {"name": "z", "price": 5, "bogus": 1},
        unknown_arguments=UnknownArguments.IGNORE,
    )
    assert result["name"] == "z"


@pytest.mark.django_db(transaction=True)
async def test_toolset_threads_the_unknown_arguments_knob():
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    toolset = SpecToolset(
        {"create_widget": create_spec()}, unknown_arguments=UnknownArguments.IGNORE
    )
    out = await toolset.call_tool(
        "create_widget", {"name": "z", "price": 5, "bogus": 1}, ctx_for(user), None
    )
    assert out["name"] == "z"


# --- object-level permission enforcement -------------------------------------


class IsOwner(BasePermission):
    def has_permission(self, request, view):
        return True

    def has_object_permission(self, request, view, obj):
        return obj.owner_id == getattr(request.user, "id", None)


def get_any_widget(pk):
    """Fetch a widget by primary key, regardless of owner."""
    return Widget.objects.filter(pk=pk)


def update_widget(instance, data):
    """Rename a widget in place."""
    instance.name = data["name"]
    instance.save(update_fields=["name"])
    return instance


@pytest.mark.django_db
def test_object_permission_denies_cross_user_retrieve():
    owner = User.objects.create(username="owner")
    other = User.objects.create(username="other")
    widget = Widget.objects.create(owner=owner, name="a", price=1)
    spec = SelectorSpec(
        kind=SelectorKind.RETRIEVE,
        selector=get_any_widget,
        output_serializer=WidgetSerializer,
        permission_classes=[IsOwner],
    )
    with pytest.raises(PermissionDenied):
        _call_spec(spec, other, {"pk": widget.pk})


@pytest.mark.django_db
def test_object_permission_allows_owner_retrieve():
    owner = User.objects.create(username="owner")
    widget = Widget.objects.create(owner=owner, name="a", price=1)
    spec = SelectorSpec(
        kind=SelectorKind.RETRIEVE,
        selector=get_any_widget,
        output_serializer=WidgetSerializer,
        permission_classes=[IsOwner],
    )
    assert _call_spec(spec, owner, {"pk": widget.pk})["name"] == "a"


@pytest.mark.django_db
def test_object_permission_denies_cross_user_mutation():
    owner = User.objects.create(username="owner")
    other = User.objects.create(username="other")
    widget = Widget.objects.create(owner=owner, name="a", price=1)
    spec = ServiceSpec(
        service=update_widget,
        input_serializer=WidgetInputSerializer,
        instance_selector_spec=SelectorSpec(kind=SelectorKind.RETRIEVE, selector=get_any_widget),
        permission_classes=[IsOwner],
        atomic=False,
    )
    # A denial aborts the run (PermissionDenied) — not a ModelRetry — before the
    # service mutates the row, exactly as it would over HTTP.
    with pytest.raises(PermissionDenied):
        _call_spec(spec, other, {"pk": widget.pk, "name": "hacked", "price": 9})
    widget.refresh_from_db()
    assert widget.name == "a"


# --- QueryParam registration (QP-2) ------------------------------------------


class _FieldsEchoSerializer(serializers.Serializer):
    """Reflects a read-shaping query param back, proving it reached the request.

    Stands in for a django-restql / custom serializer that branches on
    ``request.query_params`` — which needs the request in its context, wired by
    the spec's ``output_serializer_context`` provider (as a real consumer does).
    """

    def to_representation(self, instance):
        request = self.context["request"]
        return {"name": instance.name, "fields": request.query_params.get("fields")}


def _pass_request(request):
    return {"request": request}


def _echo_list_spec(**kwargs):
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=_FieldsEchoSerializer,
        output_serializer_context=_pass_request,
        **kwargs,
    )


async def test_toolset_wide_query_params_appear_in_every_tool_schema():
    toolset = SpecToolset(
        {"list_widgets": list_spec(), "get_widget": retrieve_spec()},
        query_params=[QueryParam("fields", description="restql field selection")],
    )
    tools = await toolset.get_tools(None)
    for name in ("list_widgets", "get_widget"):
        props = tools[name].tool_def.parameters_json_schema["properties"]
        assert props["fields"] == {"type": "string", "description": "restql field selection"}


async def test_per_tool_query_params_only_apply_to_that_tool():
    toolset = SpecToolset(
        {"list_widgets": list_spec(), "get_widget": retrieve_spec()},
        tool_query_params={"list_widgets": [QueryParam("expand", type="boolean")]},
    )
    tools = await toolset.get_tools(None)
    assert tools["list_widgets"].tool_def.parameters_json_schema["properties"]["expand"] == {
        "type": "boolean"
    }
    get_widget_props = tools["get_widget"].tool_def.parameters_json_schema.get("properties", {})
    assert "expand" not in get_widget_props


async def test_per_tool_query_param_overrides_toolset_wide_by_name():
    toolset = SpecToolset(
        {"list_widgets": list_spec()},
        query_params=[QueryParam("fields", description="wide")],
        tool_query_params={"list_widgets": [QueryParam("fields", description="specific")]},
    )
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert props["fields"]["description"] == "specific"


async def test_query_param_default_appears_in_schema():
    toolset = SpecToolset(
        {"list_widgets": list_spec()}, query_params=[QueryParam("fields", default="id")]
    )
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert props["fields"]["default"] == "id"


def test_reserved_query_param_name_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        SpecToolset({"list_widgets": list_spec()}, query_params=[QueryParam("order")])


def test_reserved_per_tool_query_param_name_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        SpecToolset(
            {"list_widgets": list_spec()},
            tool_query_params={"list_widgets": [QueryParam("limit")]},
        )


def test_unknown_per_tool_key_is_rejected():
    with pytest.raises(ValueError, match="unknown tool"):
        SpecToolset(
            {"list_widgets": list_spec()},
            tool_query_params={"nope": [QueryParam("fields")]},
        )


@pytest.mark.django_db
def test_query_param_reaches_the_serializer_via_request_query_params():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _call_spec(
        _echo_list_spec(), user, {"fields": "id,name"}, query_params=(QueryParam("fields"),)
    )
    assert result == [{"name": "a", "fields": "id,name"}]


@pytest.mark.django_db
def test_query_param_default_is_seeded_when_the_model_omits_it():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _call_spec(
        _echo_list_spec(), user, {}, query_params=(QueryParam("fields", default="id"),)
    )
    assert result == [{"name": "a", "fields": "id"}]


@pytest.mark.django_db
def test_query_param_omitted_without_default_seeds_nothing():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _call_spec(_echo_list_spec(), user, {}, query_params=(QueryParam("fields"),))
    assert result == [{"name": "a", "fields": None}]


@pytest.mark.django_db
def test_query_param_is_popped_before_dispatch_so_reject_ignores_it():
    # A closed-input list selector under REJECT: an undeclared arg would raise
    # ModelRetry. The query param must be popped before dispatch, so this passes.
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _call_spec(
        list_spec(),
        user,
        {"fields": "x"},
        query_params=(QueryParam("fields"),),
        unknown_arguments=UnknownArguments.REJECT,
    )
    assert [w["name"] for w in result] == ["a"]


# --- full agent-run integration ----------------------------------------------
#
# Drive a real ``Agent`` run loop (FunctionModel — no provider, no network) to
# pin the toolset's run-loop contract on the locked pydantic-ai: the tools
# execute in-process (``call_tool`` is invoked and the run completes, rather
# than the call being deferred to the client), and a ``ModelRetry`` is fed back
# to the model to self-correct instead of aborting the run.


def _tool_calling_model(tool_name: str, first_args: dict, retry_args: dict):
    """A model that calls ``tool_name``, corrects itself once if retried, then stops."""

    def model_fn(messages, info):
        last = messages[-1]
        if any(part.part_kind == "retry-prompt" for part in last.parts):
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=retry_args)])
        if any(part.part_kind == "tool-return" for part in last.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=first_args)])

    return FunctionModel(model_fn)


async def test_agent_run_executes_spec_tool_in_process():
    seen = {}

    def ping(user):
        """Ping."""
        seen["user"] = user
        return {"ok": True}

    toolset = SpecToolset({"ping": ServiceSpec(service=ping, atomic=False)})
    agent = Agent(_tool_calling_model("ping", {}, {}), deps_type=AgentDeps, toolsets=[toolset])
    result = await agent.run("go", deps=AgentDeps(user="alice"))
    assert result.output == "done"
    assert seen["user"] == "alice"


async def test_toolset_instructions_reach_the_model_when_attached_directly():
    # A plain Agent adding SpecToolset to ``toolsets=`` (no capability) gets the
    # conventions: Pydantic-AI collects the toolset's ``get_instructions``.
    captured = {}

    def model_fn(messages, info):
        captured["instructions"] = messages[-1].instructions
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(
        FunctionModel(model_fn),
        deps_type=AgentDeps,
        toolsets=[SpecToolset({"list": list_spec()})],
    )
    result = await agent.run("go", deps=AgentDeps(user="alice"))
    assert result.output == "done"
    assert captured["instructions"] is not None
    assert _BASE_INSTRUCTIONS in captured["instructions"]
    assert _LIST_INSTRUCTION in captured["instructions"]


class _ModeInputSerializer(serializers.Serializer):
    mode = serializers.CharField()


async def test_agent_run_recovers_from_model_retry():
    calls = []

    def flaky(data, user):
        """Fails validation on the bad mode, succeeds otherwise."""
        calls.append(data["mode"])
        if data["mode"] == "bad":
            raise ServiceValidationError("mode must not be 'bad'")
        return {"ok": True}

    toolset = SpecToolset(
        {"flaky": ServiceSpec(service=flaky, input_serializer=_ModeInputSerializer, atomic=False)}
    )
    agent = Agent(
        _tool_calling_model("flaky", {"mode": "bad"}, {"mode": "good"}),
        deps_type=AgentDeps,
        toolsets=[toolset],
    )
    result = await agent.run("go", deps=AgentDeps(user="alice"))
    assert result.output == "done"
    assert calls == ["bad", "good"]


# --- filter_set needs no QueryParam ------------------------------------------


class _WidgetFilterSet(django_filters.FilterSet):
    min_price = django_filters.NumberFilter(field_name="price", lookup_expr="gte")

    class Meta:
        model = Widget
        fields = []


def _filtered_list_spec():
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=WidgetSerializer,
        filter_set=_WidgetFilterSet,
    )


async def test_filter_set_fields_are_auto_exposed_as_tool_args():
    # A filter_set selector's fields land in the tool schema via
    # spec_to_json_schema — no QueryParam declaration needed.
    toolset = SpecToolset({"list_widgets": _filtered_list_spec()})
    tools = await toolset.get_tools(None)
    assert "min_price" in tools["list_widgets"].tool_def.parameters_json_schema["properties"]


@pytest.mark.django_db
def test_filter_set_filters_via_ordinary_params_not_query_params():
    user = User.objects.create(username="u")
    Widget.objects.create(name="cheap", price=1, owner=user)
    Widget.objects.create(name="pricey", price=10, owner=user)
    # The filter value is an ordinary tool arg: a filter_set selector's declared
    # set is open (REJECT doesn't flag it), and dispatch hands it to the FilterSet
    # as filter_data. No QueryParam involved.
    result = _call_spec(_filtered_list_spec(), user, {"min_price": "5"})
    assert [w["name"] for w in result] == ["pricey"]


# --- UrlKwarg registration ---------------------------------------------------


async def test_toolset_wide_url_kwargs_appear_in_every_tool_schema():
    toolset = SpecToolset(
        {"list_widgets": list_spec(), "get_widget": retrieve_spec()},
        url_kwargs=[UrlKwarg("project_pk", type="integer", description="owning project")],
    )
    tools = await toolset.get_tools(None)
    for name in ("list_widgets", "get_widget"):
        props = tools[name].tool_def.parameters_json_schema["properties"]
        assert props["project_pk"] == {"type": "integer", "description": "owning project"}


async def test_per_tool_url_kwargs_only_apply_to_that_tool():
    toolset = SpecToolset(
        {"list_widgets": list_spec(), "get_widget": retrieve_spec()},
        tool_url_kwargs={"list_widgets": [UrlKwarg("parent_pk")]},
    )
    tools = await toolset.get_tools(None)
    assert "parent_pk" in tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    get_widget_props = tools["get_widget"].tool_def.parameters_json_schema.get("properties", {})
    assert "parent_pk" not in get_widget_props


async def test_per_tool_url_kwarg_overrides_toolset_wide_by_name():
    toolset = SpecToolset(
        {"list_widgets": list_spec()},
        url_kwargs=[UrlKwarg("parent_pk", description="wide")],
        tool_url_kwargs={"list_widgets": [UrlKwarg("parent_pk", description="specific")]},
    )
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert props["parent_pk"]["description"] == "specific"


async def test_url_kwarg_default_appears_in_schema():
    toolset = SpecToolset(
        {"list_widgets": list_spec()}, url_kwargs=[UrlKwarg("parent_pk", default="1")]
    )
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert props["parent_pk"]["default"] == "1"


def test_reserved_url_kwarg_name_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        SpecToolset({"list_widgets": list_spec()}, url_kwargs=[UrlKwarg("order")])


def test_reserved_per_tool_url_kwarg_name_is_rejected():
    with pytest.raises(ValueError, match="reserved"):
        SpecToolset(
            {"list_widgets": list_spec()},
            tool_url_kwargs={"list_widgets": [UrlKwarg("limit")]},
        )


def test_unknown_per_tool_url_kwarg_key_is_rejected():
    with pytest.raises(ValueError, match="unknown tool"):
        SpecToolset(
            {"list_widgets": list_spec()},
            tool_url_kwargs={"nope": [UrlKwarg("parent_pk")]},
        )


def test_name_registered_as_both_query_param_and_url_kwarg_is_rejected():
    with pytest.raises(ValueError, match="two channels"):
        SpecToolset(
            {"list_widgets": list_spec()},
            query_params=[QueryParam("scope")],
            url_kwargs=[UrlKwarg("scope")],
        )


def _ceiling_from_project(view):
    """Scoping provider: derive a price ceiling from the URL's ``project_pk``.

    Stands in for the consumer's ``team_role`` fallback that reads
    ``view.kwargs["project_pk"]`` — a value that lives only on the transport.
    """
    pk = view.kwargs.get("project_pk")
    return {"ceiling": int(pk) if pk is not None else 0}


def list_under_ceiling(user, ceiling):
    """List the user's widgets priced at or below the project's ceiling."""
    return Widget.objects.filter(owner=user, price__lte=ceiling)


def _provider_scoped_spec():
    # ``project_pk`` is consumed by the provider off ``view.kwargs`` — the
    # selector never declares it, so it is a pure provider-read (not a spec input).
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_under_ceiling,
        output_serializer=WidgetSerializer,
        kwargs=_ceiling_from_project,
    )


@pytest.mark.django_db
def test_url_kwarg_reaches_a_scoping_provider_via_view_kwargs():
    user = User.objects.create(username="u")
    Widget.objects.create(name="cheap", price=5, owner=user)
    Widget.objects.create(name="dear", price=15, owner=user)
    result = _call_spec(
        _provider_scoped_spec(),
        user,
        {"project_pk": "10"},
        url_kwargs=(UrlKwarg("project_pk"),),
    )
    assert [w["name"] for w in result] == ["cheap"]


@pytest.mark.django_db
def test_url_kwarg_is_popped_before_dispatch_so_reject_ignores_it():
    # The selector's declared set is closed and does not include ``project_pk``;
    # left in params under REJECT it would raise. Popping it into ``kwargs=`` is
    # what makes the provider-read case work.
    user = User.objects.create(username="u")
    Widget.objects.create(name="cheap", price=5, owner=user)
    result = _call_spec(
        _provider_scoped_spec(),
        user,
        {"project_pk": "10"},
        url_kwargs=(UrlKwarg("project_pk"),),
        unknown_arguments=UnknownArguments.REJECT,
    )
    assert [w["name"] for w in result] == ["cheap"]


@pytest.mark.django_db
def test_url_kwarg_default_is_seeded_when_the_model_omits_it():
    user = User.objects.create(username="u")
    Widget.objects.create(name="cheap", price=5, owner=user)
    Widget.objects.create(name="dear", price=15, owner=user)
    result = _call_spec(
        _provider_scoped_spec(),
        user,
        {},
        url_kwargs=(UrlKwarg("project_pk", default="10"),),
    )
    assert [w["name"] for w in result] == ["cheap"]


@pytest.mark.django_db
def test_url_kwarg_omitted_without_default_seeds_nothing():
    user = User.objects.create(username="u")
    Widget.objects.create(name="cheap", price=5, owner=user)
    # No project_pk, no default → view.kwargs empty → provider ceiling 0 → nothing.
    result = _call_spec(_provider_scoped_spec(), user, {}, url_kwargs=(UrlKwarg("project_pk"),))
    assert result == []


class _ProjectExtras(TypedDict, total=False):
    project_pk: int


def list_in_project(user, **extras: Unpack[_ProjectExtras]):
    """List the user's widgets in the given project (price ceiling as proxy)."""
    pk = extras.get("project_pk")
    qs = Widget.objects.filter(owner=user)
    return qs.filter(price__lte=pk) if pk is not None else qs.none()


def _dual_declared_spec():
    return SelectorSpec(
        kind=SelectorKind.LIST, selector=list_in_project, output_serializer=WidgetSerializer
    )


async def test_dual_declared_url_kwarg_schema_wins_over_reflected_property():
    # ``project_pk`` is reflected by drf-services (the selector's Unpack extras)
    # *and* registered as a UrlKwarg; the explicit UrlKwarg schema wins the merge.
    toolset = SpecToolset(
        {"list_in_project": _dual_declared_spec()},
        url_kwargs=[UrlKwarg("project_pk", type="integer", description="the project")],
    )
    tools = await toolset.get_tools(None)
    props = tools["list_in_project"].tool_def.parameters_json_schema["properties"]
    assert props["project_pk"] == {"type": "integer", "description": "the project"}


@pytest.mark.django_db
def test_dual_declared_url_kwarg_delivers_to_the_selector_pool():
    user = User.objects.create(username="u")
    Widget.objects.create(name="cheap", price=5, owner=user)
    Widget.objects.create(name="dear", price=15, owner=user)
    # Popped from params into kwargs=, then drf-services' authoritative spread
    # delivers it to the selector pool where the Unpack extras read it.
    result = _call_spec(
        _dual_declared_spec(), user, {"project_pk": 10}, url_kwargs=(UrlKwarg("project_pk"),)
    )
    assert [w["name"] for w in result] == ["cheap"]
