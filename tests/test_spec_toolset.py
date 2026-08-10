from __future__ import annotations

from types import SimpleNamespace

import django_filters
import pytest
from django.contrib.auth.models import User
from django.core.exceptions import ImproperlyConfigured
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from rest_framework import serializers
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import AllowAny, BasePermission
from rest_framework_services import (
    AdditionalInputRequired,
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
    UndescribedToolWarning,
    UnguardedSpecWarning,
    _call_spec,
    _derive_instructions,
    _is_list_selector,
    _ordering_values,
    _output_extras,
    _paginate,
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
    # Guarded by default: ``require_permissions`` defaults to True, so an
    # unguarded spec is a construction error and every fixture would trip it.
    kwargs.setdefault("permission_classes", [AllowAny])
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=WidgetSerializer,
        **kwargs,
    )


def retrieve_spec(**kwargs):
    # Guarded by default: ``require_permissions`` defaults to True, so an
    # unguarded spec is a construction error and every fixture would trip it.
    kwargs.setdefault("permission_classes", [AllowAny])
    return SelectorSpec(
        kind=SelectorKind.RETRIEVE,
        selector=get_widget,
        output_serializer=WidgetSerializer,
        **kwargs,
    )


def create_spec(**kwargs):
    # Guarded by default: ``require_permissions`` defaults to True, so an
    # unguarded spec is a construction error and every fixture would trip it.
    kwargs.setdefault("permission_classes", [AllowAny])
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
    assert {"page", "limit"} <= set(props)
    # ``ordering`` is opt-in: nothing declared, nothing advertised.
    assert "ordering" not in props


async def test_annotations_mark_read_vs_write():
    toolset = SpecToolset({"list_widgets": list_spec(), "create_widget": create_spec()})
    tools = await toolset.get_tools(None)
    assert tools["list_widgets"].tool_def.metadata == {"annotations": {"readOnlyHint": True}}
    assert tools["create_widget"].tool_def.metadata == {"annotations": {"readOnlyHint": False}}


async def test_selector_without_callable_has_no_description():
    spec = SelectorSpec(
        kind=SelectorKind.LIST,
        selector=None,
        output_serializer=WidgetSerializer,
        permission_classes=[AllowAny],
    )
    with pytest.warns(UndescribedToolWarning):
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

    toolset = SpecToolset(
        {"ping": ServiceSpec(service=ping, permission_classes=[AllowAny], atomic=False)}
    )
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
        {"ping": ServiceSpec(service=ping, permission_classes=[AllowAny], atomic=False)},
        get_user=lambda ctx: ctx.principal,
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
    result = _call_spec(
        list_spec(), user, {"ordering": "price", "limit": 2}, ordering_fields=["price"]
    )
    assert [w["name"] for w in result] == ["b", "c"]


@pytest.mark.django_db
def test_list_selector_second_page():
    user = User.objects.create(username="u")
    for name, price in [("a", 1), ("b", 2), ("c", 3)]:
        Widget.objects.create(name=name, price=price, owner=user)
    result = _call_spec(
        list_spec(),
        user,
        {"ordering": "price", "page": 2, "limit": 2},
        ordering_fields=["price"],
    )
    assert [w["name"] for w in result] == ["c"]


@pytest.mark.django_db
def test_a_declared_field_that_is_not_a_column_becomes_a_retry():
    """The remaining way to reach a ``FieldError`` — and it is the author's fault.

    Enum validation stops a model from naming an arbitrary column, so what is
    left is ``ordering_fields=["nope"]``: a declaration that cannot be checked
    at construction, because knowing whether a name is a real column means
    asking a queryset that does not exist yet.
    """
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    with pytest.raises(ModelRetry, match="invalid ordering"):
        _call_spec(list_spec(), user, {"ordering": "nope"}, ordering_fields=["nope"])


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


# --- a service asking for input it was not given -----------------------------


def needs_confirmation(*, data=None, **_):
    """A service that discovers mid-run that it needs one more argument."""
    raise AdditionalInputRequired(
        "9412 rows match. Confirm to proceed.",
        schema={"confirmed": {"type": "boolean"}},
    )


def needs_something_unnamed(**_):
    raise AdditionalInputRequired("This needs something I cannot describe.")


def test_a_request_for_input_is_a_retry_not_a_dead_end():
    """⭐ A model *is* the thing that can answer, and ``ModelRetry`` is already
    the "here is what to fix, call me again" channel — so no elicitation surface
    or second result type is needed on this transport."""
    with pytest.raises(ModelRetry) as caught:
        _call_spec(ServiceSpec(service=needs_confirmation, atomic=False), object(), {})
    assert "9412 rows match" in str(caught.value)


def test_the_retry_names_the_arguments_to_add() -> None:
    """The keys of ``schema`` are input names, and the model is about to call the
    same tool again — so the names are the actionable part."""
    with pytest.raises(ModelRetry) as caught:
        _call_spec(ServiceSpec(service=needs_confirmation, atomic=False), object(), {})
    assert "`confirmed`" in str(caught.value)


def test_a_bare_message_still_retries() -> None:
    """``schema`` is optional upstream. A message alone is less actionable but
    still better as a retry than as a terminal error."""
    with pytest.raises(ModelRetry) as caught:
        _call_spec(ServiceSpec(service=needs_something_unnamed, atomic=False), object(), {})
    assert str(caught.value) == "This needs something I cannot describe."


def test_it_is_not_swallowed_by_the_generic_service_error_arm() -> None:
    """⚠ The ordering trap drf-services documents: ``AdditionalInputRequired``
    subclasses ``ServiceError``, so a handler for the parent catches it first
    unless the specific arm precedes it — which would report a request for input
    as a terminal failure."""
    result = _call_spec(ServiceSpec(service=boom, atomic=False), object(), {})
    assert result == {"error": "nope"}, "the generic arm must still work"
    with pytest.raises(ModelRetry):
        _call_spec(ServiceSpec(service=needs_confirmation, atomic=False), object(), {})


# --- permissions -------------------------------------------------------------


def test_denied_permission_raises():
    with pytest.raises(PermissionDenied):
        _call_spec(list_spec(permission_classes=[DenyAll]), object(), {})


# --- pure helpers ------------------------------------------------------------


def test_ordering_values_pairs_each_field_with_its_descending_form():
    # The same construction the MCP transport uses, so one registry exposed over
    # both surfaces advertises one vocabulary.
    assert _ordering_values([]) == []
    assert _ordering_values(["name"]) == ["name", "-name"]
    assert _ordering_values(["name", "price"]) == ["name", "-name", "price", "-price"]


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
    with pytest.raises(ModelRetry, match="ordering"):
        _call_spec(list_spec(), object(), {"ordering": ["price"]}, ordering_fields=["price"])


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
    # ``ImproperlyConfigured`` since 0.9.0: reserved-name checking moved to
    # drf-services' shared validator, so one exception type now covers every bad
    # declaration instead of ValueError-for-pagination / something-else-for-seeds.
    with pytest.raises(ImproperlyConfigured, match="reserved transport keys"):
        SpecToolset({"list_widgets": list_spec()}, query_params=[QueryParam("ordering")])


def test_reserved_per_tool_query_param_name_is_rejected():
    with pytest.raises(ImproperlyConfigured, match="reserved transport keys"):
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

    toolset = SpecToolset(
        {"ping": ServiceSpec(service=ping, permission_classes=[AllowAny], atomic=False)}
    )
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
        {
            "flaky": ServiceSpec(
                service=flaky,
                input_serializer=_ModeInputSerializer,
                permission_classes=[AllowAny],
                atomic=False,
            )
        }
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
        permission_classes=[AllowAny],
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
    with pytest.raises(ImproperlyConfigured, match="reserved transport keys"):
        SpecToolset({"list_widgets": list_spec()}, url_kwargs=[UrlKwarg("ordering")])


def test_reserved_per_tool_url_kwarg_name_is_rejected():
    with pytest.raises(ImproperlyConfigured, match="reserved transport keys"):
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
        kind=SelectorKind.LIST,
        selector=list_in_project,
        output_serializer=WidgetSerializer,
        permission_classes=[AllowAny],
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
def test_reflected_extras_key_never_reaches_view_kwargs():
    """The distinction consumers get wrong: reflected ≠ route capture.

    A key reflected from the selector's ``Unpack`` extras (``InputRequired`` or
    not) is delivered as a spec *param*. It never lands on ``view.kwargs``, so a
    scoping ``spec.kwargs`` provider reads ``None`` and silently mis-scopes.
    Registering the same name as a ``UrlKwarg`` is what routes it — and is a
    strict superset, since the authoritative spread still reaches the selector
    (covered by the two tests below).
    """
    seen = {}

    def scope_provider(view):
        seen["view_kwargs"] = dict(view.kwargs)
        return {}

    user = User.objects.create(username="u")
    Widget.objects.create(name="cheap", price=5, owner=user)
    spec = SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_in_project,
        output_serializer=WidgetSerializer,
        kwargs=scope_provider,
    )
    # Supplied as an ordinary argument, with no ``UrlKwarg`` registered.
    result = _call_spec(spec, user, {"project_pk": 10})
    assert [w["name"] for w in result] == ["cheap"]  # the selector did get it
    assert seen["view_kwargs"] == {}  # the provider did not


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


# --- required URL kwargs (drf-services 0.28) ---------------------------------


async def test_required_url_kwarg_is_advertised_as_required():
    toolset = SpecToolset(
        {"list_widgets": list_spec()},
        url_kwargs=[UrlKwarg("project_pk", type="integer", required=True)],
    )
    tools = await toolset.get_tools(None)
    schema = tools["list_widgets"].tool_def.parameters_json_schema
    assert schema["required"] == ["project_pk"]


async def test_per_tool_override_can_add_requiredness():
    # Per-tool wins by name, so an override may tighten a toolset-wide optional
    # declaration into a required one for a single tool.
    toolset = SpecToolset(
        {"list_widgets": list_spec(), "get_widget": retrieve_spec()},
        url_kwargs=[UrlKwarg("project_pk")],
        tool_url_kwargs={"list_widgets": [UrlKwarg("project_pk", required=True)]},
    )
    tools = await toolset.get_tools(None)
    assert tools["list_widgets"].tool_def.parameters_json_schema["required"] == ["project_pk"]
    assert "required" not in tools["get_widget"].tool_def.parameters_json_schema


def test_omitting_a_required_url_kwarg_raises_model_retry():
    # Schema ``required`` is a hint models routinely ignore, so the runtime has
    # to give the model a way back: ModelRetry naming the argument, not a crash.
    with pytest.raises(ModelRetry, match="project_pk"):
        _call_spec(list_spec(), None, {}, url_kwargs=[UrlKwarg("project_pk", required=True)])


@pytest.mark.django_db
def test_supplying_a_required_url_kwarg_dispatches_normally():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _call_spec(
        list_spec(),
        user,
        {"project_pk": "P1"},
        url_kwargs=[UrlKwarg("project_pk", required=True)],
    )
    assert [w["name"] for w in result] == ["a"]


def test_required_url_kwarg_with_a_default_is_rejected():
    with pytest.raises(ImproperlyConfigured, match="cannot also be required"):
        SpecToolset(
            {"list_widgets": list_spec()},
            url_kwargs=[UrlKwarg("project_pk", default="P1", required=True)],
        )


def test_a_pool_seed_name_is_now_rejected():
    # Previously only the pagination names were checked here, so ``user`` — a
    # dispatcher-controlled seed — passed registration in this toolset while
    # being rejected by the MCP transport. The shared validator closes that.
    with pytest.raises(ImproperlyConfigured, match="reserved transport keys"):
        SpecToolset({"list_widgets": list_spec()}, url_kwargs=[UrlKwarg("user")])


def test_the_lifted_types_are_the_sister_repo_types():
    from rest_framework_services.types.query_param import QueryParam as SharedQueryParam
    from rest_framework_services.types.url_kwarg import UrlKwarg as SharedUrlKwarg

    assert UrlKwarg is SharedUrlKwarg
    assert QueryParam is SharedQueryParam


# --- DRF's baseline serializer context ---------------------------------------


@pytest.mark.django_db
def test_render_supplies_the_request_in_the_serializer_context():
    """A serializer reading ``self.context["request"]`` renders as it does on HTTP.

    Off the HTTP path there is no view to call ``get_serializer_context()`` on,
    so drf-services synthesizes the baseline (``request`` / ``format`` /
    ``view``) from the offline pair. Without it this raised ``KeyError:
    'request'`` on a serializer that works behind a view.
    """

    class ContextReadingSerializer(serializers.ModelSerializer):
        owned_by = serializers.SerializerMethodField()

        class Meta:
            model = Widget
            fields = ("id", "name", "owned_by")

        def get_owned_by(self, _):
            return self.context["request"].user.username

    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    spec = SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=ContextReadingSerializer,
    )
    assert [w["owned_by"] for w in _call_spec(spec, user, {})] == ["u"]


@pytest.mark.django_db
def test_input_validation_supplies_the_request_in_the_serializer_context():
    class ContextReadingInput(serializers.Serializer):
        name = serializers.CharField()
        price = serializers.IntegerField()

        def validate_name(self, value):
            return f"{value}-{self.context['request'].user.username}"

    user = User.objects.create(username="u")
    spec = ServiceSpec(
        service=create_widget,
        input_serializer=ContextReadingInput,
        atomic=False,
    )
    _call_spec(spec, user, {"name": "a", "price": 1})
    assert Widget.objects.get().name == "a-u"


# --- host (absolute URLs off the HTTP path) ----------------------------------


class FileishSerializer(serializers.ModelSerializer):
    """Mirrors DRF's ``FileField.to_representation`` branch for a URL."""

    doc_url = serializers.SerializerMethodField()

    class Meta:
        model = Widget
        fields = ("id", "name", "doc_url")

    def get_doc_url(self, _):
        request = self.context.get("request", None)
        if request is not None:
            return request.build_absolute_uri("/media/doc.pdf")
        return "/media/doc.pdf"


def fileish_spec():
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=FileishSerializer,
        permission_classes=[AllowAny],
    )


@pytest.mark.django_db
def test_without_a_host_file_urls_are_relative():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    assert [w["doc_url"] for w in _call_spec(fileish_spec(), user, {})] == ["/media/doc.pdf"]


@pytest.mark.django_db
def test_host_makes_file_urls_absolute():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _call_spec(fileish_spec(), user, {}, host="https://files.example.com")
    assert [w["doc_url"] for w in result] == ["https://files.example.com/media/doc.pdf"]


@pytest.mark.django_db(transaction=True)
async def test_toolset_threads_its_host_into_the_call():
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    await sync_to_async(Widget.objects.create)(name="a", price=1, owner=user)
    toolset = SpecToolset({"list_widgets": fileish_spec()}, host="app.example.com:8000")
    tools = await toolset.get_tools(None)
    result = await toolset.call_tool("list_widgets", {}, ctx_for(user), tools["list_widgets"])
    assert [w["doc_url"] for w in result] == ["http://app.example.com:8000/media/doc.pdf"]


@pytest.mark.django_db(transaction=True)
async def test_toolset_without_a_host_still_renders():
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    await sync_to_async(Widget.objects.create)(name="a", price=1, owner=user)
    toolset = SpecToolset({"list_widgets": fileish_spec()})
    tools = await toolset.get_tools(None)
    result = await toolset.call_tool("list_widgets", {}, ctx_for(user), tools["list_widgets"])
    assert [w["doc_url"] for w in result] == ["/media/doc.pdf"]


# ----- registration-time honesty: permissions and descriptions -----


def _unguarded_list_spec():
    """Deliberately without ``permission_classes`` — the shape under test."""
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=WidgetSerializer,
    )


def test_an_unguarded_spec_is_refused_by_default():
    """Over HTTP ``None`` means *inherit*; off HTTP there is nothing to inherit.

    A spec guarded behind a viewset, with passing HTTP tests, is callable by
    whatever the model decides to call the moment it reaches a toolset. There is
    no signal anywhere, which is why this has to fire at construction.
    """
    with pytest.raises(ImproperlyConfigured) as excinfo:
        SpecToolset({"list_widgets": _unguarded_list_spec()})

    message = str(excinfo.value)
    assert "'list_widgets'" in message
    assert "require_permissions=False" in message


def test_every_unguarded_spec_is_named_at_once():
    """One construction, one list — not one error per fix-and-rerun cycle."""
    with pytest.raises(ImproperlyConfigured) as excinfo:
        SpecToolset({"a": _unguarded_list_spec(), "b": _unguarded_list_spec()})

    assert "'a', 'b'" in str(excinfo.value)


def test_require_permissions_false_downgrades_to_a_warning():
    with pytest.warns(UnguardedSpecWarning, match="list_widgets"):
        toolset = SpecToolset({"list_widgets": _unguarded_list_spec()}, require_permissions=False)

    assert toolset.id == "drf-specs"


def test_a_guarded_spec_says_nothing(recwarn):
    SpecToolset({"list_widgets": list_spec()})
    assert [w for w in recwarn.list if issubclass(w.category, UnguardedSpecWarning)] == []


async def test_a_description_override_reaches_the_tool_def():
    """The docstring an API developer reads is not the sentence a model needs."""
    toolset = SpecToolset(
        {"list_widgets": list_spec()},
        descriptions={"list_widgets": "Use when the user asks what they own."},
    )
    tools = await toolset.get_tools(None)
    assert tools["list_widgets"].tool_def.description == "Use when the user asks what they own."


def test_a_description_for_an_unknown_tool_is_a_typo():
    """Dropping it silently would leave the tool with the text it was meant to lose."""
    with pytest.raises(ValueError, match="unknown tool 'list_widget'"):
        SpecToolset({"list_widgets": list_spec()}, descriptions={"list_widget": "…"})


def test_a_tool_with_no_description_anywhere_warns():
    spec = SelectorSpec(
        kind=SelectorKind.LIST,
        selector=None,
        output_serializer=WidgetSerializer,
        permission_classes=[AllowAny],
    )
    with pytest.warns(UndescribedToolWarning, match="'empty'"):
        SpecToolset({"empty": spec})


def test_a_whitespace_only_description_counts_as_none():
    """Otherwise the override is a way to silence the warning without fixing it."""
    with pytest.warns(UndescribedToolWarning, match="'list_widgets'"):
        SpecToolset({"list_widgets": list_spec()}, descriptions={"list_widgets": "   "})


# ----- ordering: the MCP transport's vocabulary -----


async def test_declared_ordering_fields_become_an_enum():
    """An enum, not a free string — the only shape that says what may be sorted."""
    toolset = SpecToolset({"list_widgets": list_spec()}, ordering_fields=["name", "price"])
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert props["ordering"]["enum"] == ["name", "-name", "price", "-price"]


async def test_per_tool_ordering_fields_replace_the_toolset_wide_set():
    """Replace, not merge: the sort keys for two collections have nothing to do
    with each other, unlike a query param, which is usually cross-cutting."""
    toolset = SpecToolset(
        {"list_widgets": list_spec(), "other": list_spec()},
        ordering_fields=["name"],
        tool_ordering_fields={"other": ["price"]},
    )
    tools = await toolset.get_tools(None)
    assert tools["list_widgets"].tool_def.parameters_json_schema["properties"]["ordering"][
        "enum"
    ] == ["name", "-name"]
    assert tools["other"].tool_def.parameters_json_schema["properties"]["ordering"]["enum"] == [
        "price",
        "-price",
    ]


@pytest.mark.django_db
def test_a_value_outside_the_enum_is_a_retry_naming_the_options():
    """⚠ Deliberately unlike the MCP transport, which silently ignores it.

    Silently returning unsorted rows to something that asked for newest-first is
    the worst outcome available: the model cannot tell, and neither can the user
    reading its answer.
    """
    with pytest.raises(ModelRetry, match="`name`, `-name`"):
        _call_spec(list_spec(), object(), {"ordering": "price"}, ordering_fields=["name"])


@pytest.mark.django_db
def test_ordering_on_a_tool_that_declares_none_says_so():
    with pytest.raises(ModelRetry, match="does not accept an `ordering` argument"):
        _call_spec(list_spec(), object(), {"ordering": "name"})


async def test_the_ordering_instruction_appears_only_when_some_tool_declares_fields():
    without = await SpecToolset({"list_widgets": list_spec()}).get_instructions(None)
    assert "accept `ordering`" not in without

    with_fields = await SpecToolset(
        {"list_widgets": list_spec()}, ordering_fields=["name"]
    ).get_instructions(None)
    assert "accept `ordering`" in with_fields
