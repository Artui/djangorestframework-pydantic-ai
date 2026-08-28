from __future__ import annotations

import asyncio
import logging
import warnings
from types import SimpleNamespace
from typing import Any

import django_filters
import pytest
from django.contrib.auth.models import User
from django.core.exceptions import ImproperlyConfigured
from django.core.exceptions import ValidationError as DjangoValidationError
from django.test import RequestFactory
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from rest_framework import serializers
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import AllowAny, BasePermission
from rest_framework_services import (
    UNSET,
    AdditionalInputRequired,
    SelectorKind,
    SelectorSpec,
    ServiceError,
    ServiceSpec,
    ServiceValidationError,
    SpecRegistry,
    UnknownArguments,
    build_offline_context,
)
from typing_extensions import TypedDict, Unpack

from rest_framework_pydantic_ai import AgentDeps, QueryParam, SpecToolset, UrlKwarg
from rest_framework_pydantic_ai.spec_toolset import (
    _BASE_INSTRUCTIONS,
    _HANDLE_INSTRUCTION,
    _LIST_INSTRUCTION,
    UndescribedToolWarning,
    UnguardedSpecWarning,
    _default_get_http_request,
    _default_get_progress,
    _derive_instructions,
    _input_schema,
    _is_list_selector,
    _ordering_values,
    _output_extras,
    _paginate,
    _pop_query_params,
    _pop_url_kwargs,
    _spec_ordering_argument,
    _with_deadline,
)
from tests.testapp.models import Widget
from tests.testapp.serializers import (
    AgentWidgetSerializer,
    WidgetInputSerializer,
    WidgetSerializer,
)

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


def _dispatch(
    spec: Any,
    user: Any,
    args: dict[str, Any] | None = None,
    *,
    toolset_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Run one spec through the toolset's dispatch pipeline.

    Calls ``SpecToolset._call_spec`` rather than the module-level function, so
    these tests exercise the same seam ``call_tool`` goes through — including
    the ``build_context`` / ``translate_exception`` overrides bound to the run.
    """
    toolset_kwargs = toolset_kwargs or {}
    # A throwaway toolset per call; construction validation has its own
    # tests, so its warnings are noise here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        toolset = SpecToolset({"t": spec}, require_permissions=False, **toolset_kwargs)
    ctx = SimpleNamespace(deps=AgentDeps(user=user))
    return toolset._call_spec(spec, user, dict(args or {}), ctx=ctx, **kwargs)


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
    result = _dispatch(list_spec(), user, {})
    assert [w["name"] for w in result] == ["a"]


@pytest.mark.django_db
def test_list_selector_orders_and_limits():
    user = User.objects.create(username="u")
    for name, price in [("a", 3), ("b", 1), ("c", 2)]:
        Widget.objects.create(name=name, price=price, owner=user)
    result = _dispatch(
        list_spec(), user, {"ordering": "price", "limit": 2}, ordering_fields=["price"]
    )
    assert [w["name"] for w in result] == ["b", "c"]


@pytest.mark.django_db
def test_list_selector_second_page():
    user = User.objects.create(username="u")
    for name, price in [("a", 1), ("b", 2), ("c", 3)]:
        Widget.objects.create(name=name, price=price, owner=user)
    result = _dispatch(
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
        _dispatch(list_spec(), user, {"ordering": "nope"}, ordering_fields=["nope"])


@pytest.mark.django_db
def test_retrieve_selector_found():
    user = User.objects.create(username="u")
    widget = Widget.objects.create(name="a", price=1, owner=user)
    result = _dispatch(retrieve_spec(), user, {"pk": widget.pk})
    assert result["name"] == "a"


@pytest.mark.django_db
def test_retrieve_selector_not_found_is_error_payload():
    user = User.objects.create(username="u")
    result = _dispatch(retrieve_spec(), user, {"pk": 999})
    assert result == {"error": "not found"}


# --- service dispatch --------------------------------------------------------


@pytest.mark.django_db
def test_create_service_renders_output():
    user = User.objects.create(username="u")
    result = _dispatch(create_spec(), user, {"name": "z", "price": 5})
    assert result["name"] == "z"
    assert Widget.objects.filter(name="z", owner=user).exists()


@pytest.mark.django_db
def test_create_service_validation_error_is_model_retry():
    user = User.objects.create(username="u")
    with pytest.raises(ModelRetry):
        _dispatch(create_spec(), user, {"name": "z", "price": -1})


def test_service_error_is_returned_as_payload():
    result = _dispatch(ServiceSpec(service=boom, atomic=False), object(), {})
    assert result == {"error": "nope"}


def test_service_validation_error_is_model_retry():
    with pytest.raises(ModelRetry):
        _dispatch(ServiceSpec(service=reject, atomic=False), object(), {})


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
    """A model *is* the thing that can answer, and ``ModelRetry`` is already
    the "here is what to fix, call me again" channel — so no elicitation surface
    or second result type is needed on this transport."""
    with pytest.raises(ModelRetry) as caught:
        _dispatch(ServiceSpec(service=needs_confirmation, atomic=False), object(), {})
    assert "9412 rows match" in str(caught.value)


def test_the_retry_names_the_arguments_to_add() -> None:
    """The keys of ``schema`` are input names, and the model is about to call the
    same tool again — so the names are the actionable part."""
    with pytest.raises(ModelRetry) as caught:
        _dispatch(ServiceSpec(service=needs_confirmation, atomic=False), object(), {})
    assert "`confirmed`" in str(caught.value)


def test_a_bare_message_still_retries() -> None:
    """``schema`` is optional upstream. A message alone is less actionable but
    still better as a retry than as a terminal error."""
    with pytest.raises(ModelRetry) as caught:
        _dispatch(ServiceSpec(service=needs_something_unnamed, atomic=False), object(), {})
    assert str(caught.value) == "This needs something I cannot describe."


def test_it_is_not_swallowed_by_the_generic_service_error_arm() -> None:
    """The ordering trap drf-services documents: ``AdditionalInputRequired``
    subclasses ``ServiceError``, so a handler for the parent catches it first
    unless the specific arm precedes it — which would report a request for input
    as a terminal failure."""
    result = _dispatch(ServiceSpec(service=boom, atomic=False), object(), {})
    assert result == {"error": "nope"}, "the generic arm must still work"
    with pytest.raises(ModelRetry):
        _dispatch(ServiceSpec(service=needs_confirmation, atomic=False), object(), {})


# --- permissions -------------------------------------------------------------


def test_denied_permission_raises():
    with pytest.raises(PermissionDenied):
        _dispatch(list_spec(permission_classes=[DenyAll]), object(), {})


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
    result = _dispatch(list_spec(), user, {"limit": "1"})
    assert len(result) == 1


@pytest.mark.parametrize("value", ["abc", "2.5", "-1", 0, -3, 2.0])
def test_non_positive_int_limit_is_model_retry(value):
    with pytest.raises(ModelRetry, match="positive integer"):
        _dispatch(list_spec(), object(), {"limit": value})


def test_bool_page_is_model_retry():
    # ``True`` is an ``int`` subclass but never a valid count.
    with pytest.raises(ModelRetry, match="positive integer"):
        _dispatch(list_spec(), object(), {"page": True})


def test_non_string_order_is_model_retry():
    with pytest.raises(ModelRetry, match="ordering"):
        _dispatch(list_spec(), object(), {"ordering": ["price"]}, ordering_fields=["price"])


# --- unknown-arguments knob --------------------------------------------------


@pytest.mark.django_db
def test_unknown_argument_rejected_by_default():
    user = User.objects.create(username="u")
    with pytest.raises(ModelRetry, match="bogus"):
        _dispatch(create_spec(), user, {"name": "z", "price": 5, "bogus": 1})


@pytest.mark.django_db
def test_unknown_argument_ignored_when_configured():
    user = User.objects.create(username="u")
    result = _dispatch(
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
        _dispatch(spec, other, {"pk": widget.pk})


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
    assert _dispatch(spec, owner, {"pk": widget.pk})["name"] == "a"


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
        _dispatch(spec, other, {"pk": widget.pk, "name": "hacked", "price": 9})
    widget.refresh_from_db()
    assert widget.name == "a"


# --- QueryParam registration -------------------------------------------------


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
    result = _dispatch(
        _echo_list_spec(), user, {"fields": "id,name"}, query_params=(QueryParam("fields"),)
    )
    assert result == [{"name": "a", "fields": "id,name"}]


@pytest.mark.django_db
def test_query_param_default_is_seeded_when_the_model_omits_it():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _dispatch(
        _echo_list_spec(), user, {}, query_params=(QueryParam("fields", default="id"),)
    )
    assert result == [{"name": "a", "fields": "id"}]


@pytest.mark.django_db
def test_query_param_omitted_without_default_seeds_nothing():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _dispatch(_echo_list_spec(), user, {}, query_params=(QueryParam("fields"),))
    assert result == [{"name": "a", "fields": None}]


@pytest.mark.django_db
def test_query_param_is_popped_before_dispatch_so_reject_ignores_it():
    # A closed-input list selector under REJECT: an undeclared arg would raise
    # ModelRetry. The query param must be popped before dispatch, so this passes.
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _dispatch(
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
    result = _dispatch(_filtered_list_spec(), user, {"min_price": "5"})
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
    result = _dispatch(
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
    result = _dispatch(
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
    result = _dispatch(
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
    result = _dispatch(_provider_scoped_spec(), user, {}, url_kwargs=(UrlKwarg("project_pk"),))
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
    result = _dispatch(spec, user, {"project_pk": 10})
    assert [w["name"] for w in result] == ["cheap"]  # the selector did get it
    assert seen["view_kwargs"] == {}  # the provider did not


@pytest.mark.django_db
def test_dual_declared_url_kwarg_delivers_to_the_selector_pool():
    user = User.objects.create(username="u")
    Widget.objects.create(name="cheap", price=5, owner=user)
    Widget.objects.create(name="dear", price=15, owner=user)
    # Popped from params into kwargs=, then drf-services' authoritative spread
    # delivers it to the selector pool where the Unpack extras read it.
    result = _dispatch(
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
        _dispatch(list_spec(), None, {}, url_kwargs=[UrlKwarg("project_pk", required=True)])


@pytest.mark.django_db
def test_supplying_a_required_url_kwarg_dispatches_normally():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _dispatch(
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
    assert [w["owned_by"] for w in _dispatch(spec, user, {})] == ["u"]


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
    _dispatch(spec, user, {"name": "a", "price": 1})
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
    assert [w["doc_url"] for w in _dispatch(fileish_spec(), user, {})] == ["/media/doc.pdf"]


@pytest.mark.django_db
def test_host_makes_file_urls_absolute():
    user = User.objects.create(username="u")
    Widget.objects.create(name="a", price=1, owner=user)
    result = _dispatch(fileish_spec(), user, {}, host="https://files.example.com")
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


# ----- ordering: the deprecated ordering_fields vocabulary -----
#
# Unchanged behaviour, for the one case that still has no other route: a list
# selector with no ``filter_set``. Every construction that declares fields now
# also carries a ``DeprecationWarning``.


async def test_declared_ordering_fields_become_an_enum():
    """An enum, not a free string — the only shape that says what may be sorted."""
    with pytest.deprecated_call():
        toolset = SpecToolset({"list_widgets": list_spec()}, ordering_fields=["name", "price"])
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert props["ordering"]["enum"] == ["name", "-name", "price", "-price"]


async def test_per_tool_ordering_fields_replace_the_toolset_wide_set():
    """Replace, not merge: the sort keys for two collections have nothing to do
    with each other, unlike a query param, which is usually cross-cutting."""
    with pytest.deprecated_call():
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
    """Deliberately unlike the MCP transport, which silently ignores it.

    Silently returning unsorted rows to something that asked for newest-first is
    the worst outcome available: the model cannot tell, and neither can the user
    reading its answer.
    """
    with pytest.raises(ModelRetry, match="`name`, `-name`"):
        _dispatch(list_spec(), object(), {"ordering": "price"}, ordering_fields=["name"])


@pytest.mark.django_db
def test_ordering_on_a_tool_that_declares_none_says_so():
    with pytest.raises(ModelRetry, match="does not accept an `ordering` argument"):
        _dispatch(list_spec(), object(), {"ordering": "name"})


async def test_the_ordering_instruction_appears_only_when_some_tool_declares_fields():
    without = await SpecToolset({"list_widgets": list_spec()}).get_instructions(None)
    assert "accept `ordering`" not in without

    with pytest.deprecated_call():
        toolset = SpecToolset({"list_widgets": list_spec()}, ordering_fields=["name"])
    assert "accept `ordering`" in await toolset.get_instructions(None)


# ----- ordering: the filter_set owns it -----
#
# The defect these pin: a FilterSet carrying an ``OrderingFilter`` advertises
# ``ordering`` to the model all on its own — ``OrderingFilter`` subclasses
# ``ChoiceFilter``, which drf-services maps to an enum — with nothing declared on
# the toolset. The toolset used to strip that argument out of every list call and
# answer "this tool does not accept an `ordering` argument", so the schema and the
# dispatch contradicted each other and the value never reached the FilterSet.


class _OrderedWidgetFilterSet(django_filters.FilterSet):
    """Public sort names that are *not* the ORM paths they resolve to.

    ``cost`` / ``title`` map to ``price`` / ``name`` through the filter's own
    ``param_map`` — the reason a second ``ordering_fields`` vocabulary of raw ORM
    paths cannot be substituted for this one.
    """

    min_price = django_filters.NumberFilter(field_name="price", lookup_expr="gte")
    ordering = django_filters.OrderingFilter(fields=(("price", "cost"), ("name", "title")))

    class Meta:
        model = Widget
        fields = []


def _ordered_list_spec():
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=WidgetSerializer,
        filter_set=_OrderedWidgetFilterSet,
        permission_classes=[AllowAny],
    )


class _SortingWidgetFilterSet(django_filters.FilterSet):
    """The same sort, declared under a name that is not ``ordering``.

    Nothing requires the name. The deprecation warning suggests ``ordering``,
    and a project following it to the letter was fine, but the name is the
    project's to choose and django-filter has no opinion about it.
    """

    sorting = django_filters.OrderingFilter(fields=(("price", "cost"), ("name", "title")))

    class Meta:
        model = Widget
        fields = []


def _sorting_list_spec():
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=WidgetSerializer,
        filter_set=_SortingWidgetFilterSet,
        permission_classes=[AllowAny],
    )


def _three_widgets(user):
    """Insertion order a, b, c; price order b, c, a — so the two are tellable apart."""
    for name, price in [("a", 3), ("b", 1), ("c", 2)]:
        Widget.objects.create(name=name, price=price, owner=user)


def test_the_sort_argument_is_found_under_whatever_name_it_was_declared():
    """The name is the project's to choose, and nothing required ``ordering``.

    This used to test the literal string, so a FilterSet declaring ``sorting``
    read as "owns no ordering" -- which dropped the usage instruction and, more
    quietly, left the value in the callable's kwarg pool.
    """
    assert _spec_ordering_argument(_sorting_list_spec()) == "sorting"


def test_a_plain_filter_is_not_mistaken_for_a_sort():
    # min_price is a NumberFilter. Only OrderingFilter defines
    # ``get_ordering_value``, which is what makes the duck-type specific.
    assert _spec_ordering_argument(_filtered_list_spec()) is None


@pytest.mark.django_db
def test_a_renamed_sort_is_popped_out_of_the_callables_kwargs():
    """The half nobody had found, and the one with a consequence.

    With a filter named ``sorting`` the rows still came back correctly sorted --
    ``filter_data`` stayed defaulted to ``params`` and the FilterSet reads
    ``params`` either way. What was lost was the *pop*: the value stayed in the
    selector's kwarg pool, so a selector declaring ``**kwargs`` received a
    read-shaping argument it never asked for. Which is exactly the hazard
    ``_pop_filter_ordering``'s own docstring describes for the ``ordering`` case.
    """
    seen: dict[str, object] = {}

    def recording_selector(**kwargs):
        seen.update(kwargs)
        return Widget.objects.all()

    spec = SelectorSpec(
        kind=SelectorKind.LIST,
        selector=recording_selector,
        output_serializer=WidgetSerializer,
        filter_set=_SortingWidgetFilterSet,
        permission_classes=[AllowAny],
    )
    user = User.objects.create(username="u")
    _three_widgets(user)

    result = _dispatch(spec, user, {"sorting": "cost"})

    assert "sorting" not in seen, "the sort reached the selector as a surprise kwarg"
    assert [w["name"] for w in result] == ["b", "c", "a"]


@pytest.mark.django_db
def test_a_renamed_sort_still_orders_the_rows():
    user = User.objects.create(username="u")
    _three_widgets(user)

    result = _dispatch(_sorting_list_spec(), user, {"sorting": "cost"})

    assert [w["name"] for w in result] == ["b", "c", "a"]


def test_the_instruction_names_the_argument_that_actually_exists():
    """A hard-coded name in the prose is a second copy of the same assumption.

    An instruction naming an argument the schema does not carry is worse than no
    instruction: the model is told to send something that will be rejected.
    """
    instructions = _derive_instructions({"list_widgets": _sorting_list_spec()}, {})

    assert "`sorting`" in instructions
    assert "`ordering`" not in instructions


def test_the_instruction_names_every_live_sort_argument():
    # One toolset, two tools, two names. A model reads one block for all of them.
    instructions = _derive_instructions(
        {"a": _ordered_list_spec(), "b": _sorting_list_spec()},
        {},
    )

    assert "`ordering`" in instructions
    assert "`sorting`" in instructions


@pytest.mark.django_db
def test_filter_owned_ordering_actually_orders_the_rows():
    """The test that would have caught the original defect.

    No ``ordering_fields`` anywhere: the FilterSet advertises ``ordering``, so the
    value has to reach it and the rows have to come back sorted. Asserting the
    order rather than "no error" is the point — the failure being guarded against
    is a tool that answers cheerfully with rows in the wrong order.
    """
    user = User.objects.create(username="u")
    _three_widgets(user)
    result = _dispatch(_ordered_list_spec(), user, {"ordering": "cost"})
    assert [w["name"] for w in result] == ["b", "c", "a"]


@pytest.mark.django_db
def test_filter_owned_ordering_descends():
    user = User.objects.create(username="u")
    _three_widgets(user)
    result = _dispatch(_ordered_list_spec(), user, {"ordering": "-cost"})
    assert [w["name"] for w in result] == ["a", "c", "b"]


@pytest.mark.django_db
def test_filter_owned_ordering_composes_with_the_other_filters():
    """``filter_data`` *replaces* ``params`` as the filter source rather than
    adding to it, so routing ``ordering`` through it has to carry the rest of the
    filter args along — otherwise ordering a filtered list silently unfilters it.
    """
    user = User.objects.create(username="u")
    _three_widgets(user)
    result = _dispatch(_ordered_list_spec(), user, {"min_price": "2", "ordering": "cost"})
    assert [w["name"] for w in result] == ["c", "a"]


@pytest.mark.django_db
def test_filter_owned_tool_still_works_with_no_ordering_supplied():
    user = User.objects.create(username="u")
    _three_widgets(user)
    result = _dispatch(_ordered_list_spec(), user, {"min_price": "2"})
    assert sorted(w["name"] for w in result) == ["a", "c"]


@pytest.mark.django_db
def test_an_ordering_outside_the_filters_choices_is_a_retry():
    """The FilterSet validates it — including against the ORM path a declared
    ``ordering_fields`` would have used, which is not one of its public choices.
    drf-services raises that as a DRF ``ValidationError``, which is already the
    ``ModelRetry`` arm; the toolset adds no second validation of its own.
    """
    user = User.objects.create(username="u")
    _three_widgets(user)
    with pytest.raises(ModelRetry, match="ordering"):
        _dispatch(_ordered_list_spec(), user, {"ordering": "price"})


async def test_the_advertised_ordering_enum_is_the_filters_own_vocabulary():
    toolset = SpecToolset({"list_widgets": _ordered_list_spec()})
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert props["ordering"]["enum"] == ["cost", "-cost", "title", "-title"]


def test_spec_owns_ordering_reads_the_reflected_schema():
    assert _spec_ordering_argument(_ordered_list_spec()) == "ordering"
    assert _spec_ordering_argument(_filtered_list_spec()) is None
    assert _spec_ordering_argument(list_spec()) is None
    # Only a list selector can contest the argument: nothing else is ever offered
    # one, so a service with an `ordering` input field is not a clash.
    assert _spec_ordering_argument(retrieve_spec()) is None
    assert _spec_ordering_argument(create_spec()) is None


def test_declaring_ordering_fields_beside_a_filter_that_owns_ordering_is_refused():
    """Not resolved by preferring one — that *is* the defect, one level up."""
    with pytest.raises(ValueError, match="declare ordering twice"):
        SpecToolset({"list_widgets": _ordered_list_spec()}, ordering_fields=["price"])


def test_the_refusal_names_the_tool_both_channels_and_the_fix():
    with pytest.raises(ValueError) as excinfo:
        SpecToolset(
            {"plain": list_spec(), "list_widgets": _ordered_list_spec()},
            tool_ordering_fields={"list_widgets": ["price"]},
        )
    message = str(excinfo.value)
    assert "'list_widgets'" in message and "'plain'" not in message
    assert "ordering_fields / tool_ordering_fields" in message
    assert "filter_set" in message
    assert "Drop ordering_fields" in message


def test_ordering_fields_are_deprecated_in_favour_of_the_filter():
    with pytest.warns(DeprecationWarning, match="OrderingFilter") as record:
        SpecToolset({"list_widgets": list_spec()}, ordering_fields=["price"])
    assert "'list_widgets'" in str(record[0].message)


def test_tool_ordering_fields_are_deprecated_too():
    with pytest.warns(DeprecationWarning, match="OrderingFilter"):
        SpecToolset({"list_widgets": list_spec()}, tool_ordering_fields={"list_widgets": ["price"]})


def test_declaring_nothing_warns_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        SpecToolset({"list_widgets": list_spec()}, tool_ordering_fields={"list_widgets": []})


def test_the_schema_builder_never_overwrites_the_filters_enum():
    """Belt to the constructor's braces, at the one line that could overwrite it.

    The merge that builds ``properties`` puts the toolset's ``extra`` over the
    reflected schema, so a value written here under a name the filter already
    advertised replaces the filter's vocabulary with ORM paths the FilterSet
    would then reject.
    """
    props = _input_schema(_ordered_list_spec(), ordering_fields=["price"])["properties"]
    assert props["ordering"]["enum"] == ["cost", "-cost", "title", "-title"]


async def test_the_ordering_instruction_is_emitted_for_a_filter_owned_tool():
    """Gated on ``tool_ordering_fields`` alone, this tool — the one whose
    ``ordering`` is enum-valued and needs exactly one value picked from it — got
    no ordering guidance at all.
    """
    instructions = await SpecToolset({"list_widgets": _ordered_list_spec()}).get_instructions(None)
    assert "accept `ordering`" in instructions


# The guard on the strip this change narrows. drf-services spreads the *entire*
# pool into a selector declaring ``**kwargs`` ("if fn declares **kwargs, the
# entire pool is passed"), so any transport arg left in ``params`` lands in the
# callable's signature as an argument it never declared. That is why a
# filter-owned ``ordering`` is popped and routed through ``filter_data`` rather
# than simply left in place — and why ``page`` / ``limit`` must keep being popped.

_KWARGS_SEEN: list[dict[str, Any]] = []
_FILTER_DATA_SEEN: list[dict[str, Any]] = []


def list_widgets_greedy(user, **extras):
    """List widgets owned by the acting user, recording every extra argument."""
    _KWARGS_SEEN.append(dict(extras))
    return Widget.objects.filter(owner=user)


class _RecordingFilterSet(django_filters.FilterSet):
    """Records the filter data it was handed, so the split can be asserted on."""

    min_price = django_filters.NumberFilter(field_name="price", lookup_expr="gte")
    ordering = django_filters.OrderingFilter(fields=(("price", "cost"),))

    class Meta:
        model = Widget
        fields = []

    def __init__(self, data=None, queryset=None, *, request=None, prefix=None):
        _FILTER_DATA_SEEN.append(dict(data or {}))
        super().__init__(data=data, queryset=queryset, request=request, prefix=prefix)


@pytest.mark.django_db
def test_the_transport_args_never_reach_a_kwargs_selector():
    """``page`` / ``limit`` / ``ordering`` are the transport's, whoever consumes them.

    A selector declaring ``**kwargs`` receives the whole dispatch pool, so this is
    the test that catches a transport arg being left in ``params`` — including the
    filter-owned ``ordering``, which reaches the FilterSet through ``filter_data``
    precisely so that it does *not* reach the callable.
    """
    _KWARGS_SEEN.clear()
    _FILTER_DATA_SEEN.clear()
    user = User.objects.create(username="u")
    _three_widgets(user)
    spec = SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets_greedy,
        output_serializer=WidgetSerializer,
        filter_set=_RecordingFilterSet,
        permission_classes=[AllowAny],
    )
    result = _dispatch(spec, user, {"page": 1, "limit": 2, "ordering": "cost", "min_price": "1"})
    assert [w["name"] for w in result] == ["b", "c"]

    seen = _KWARGS_SEEN[-1]
    assert "page" not in seen
    assert "limit" not in seen
    assert "ordering" not in seen
    # The selector's own filter arg is untouched — only the transport's are taken.
    assert seen["min_price"] == "1"

    # The FilterSet gets the filter args plus ordering, and *not* the pagination
    # args: ``filter_data`` is built after ``page`` / ``limit`` have been popped,
    # so widening the ordering route did not widen what the FilterSet sees.
    assert _FILTER_DATA_SEEN[-1] == {"min_price": "1", "ordering": "cost"}


# A list selector with no ``filter_set`` whose *callable* declares ``ordering``:
# the schema advertises it for the same reason, so the same invariant applies —
# but the argument is the selector's own, and has to stay in ``params`` where the
# callable is given it rather than being routed to a FilterSet that isn't there.


def list_widgets_sorted(user, ordering: str = "name"):
    """List widgets owned by the acting user, in a given order."""
    return Widget.objects.filter(owner=user).order_by(ordering)


@pytest.mark.django_db
def test_an_ordering_the_selector_itself_declares_reaches_the_selector():
    user = User.objects.create(username="u")
    _three_widgets(user)
    spec = SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets_sorted,
        output_serializer=WidgetSerializer,
        permission_classes=[AllowAny],
    )
    assert _spec_ordering_argument(spec) == "ordering"
    result = _dispatch(spec, user, {"ordering": "price"})
    assert [w["name"] for w in result] == ["b", "c", "a"]


# ----- progress: accepted and forwarded, never constructed -----


class _RecordingProgress:
    """Stands in for whatever is driving the agent — an SSE frame, a task record."""

    def __init__(self):
        self.reports = []

    def __call__(self, progress, *, total=None, message=None, meta=None):
        self.reports.append((progress, total, message, meta))


def reporting_service(data, user, progress):
    """Creates a widget, reporting as it goes."""
    progress(1, total=1, message="Creating")
    return Widget.objects.create(owner=user, **data)


def reporting_spec():
    return ServiceSpec(
        service=reporting_service,
        input_serializer=WidgetInputSerializer,
        output_selector_spec=SelectorSpec(
            kind=SelectorKind.RETRIEVE, output_serializer=WidgetSerializer
        ),
        permission_classes=[AllowAny],
    )


@pytest.mark.django_db(transaction=True)
async def test_a_reporter_on_the_deps_reaches_the_service():
    """The caller supplies the sink; the toolset only carries it.

    A toolset that *constructed* one — "write progress to the logger" — would
    have picked a transport it does not own, which is the thing this package
    exists not to do.
    """
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    sink = _RecordingProgress()
    toolset = SpecToolset({"create_widget": reporting_spec()})
    tools = await toolset.get_tools(None)

    await toolset.call_tool(
        "create_widget",
        {"name": "a", "price": 1},
        SimpleNamespace(deps=AgentDeps(user=user, progress=sink)),
        tools["create_widget"],
    )
    assert sink.reports == [(1, 1, "Creating", None)]


@pytest.mark.django_db(transaction=True)
async def test_the_same_service_runs_with_no_reporter_supplied():
    """``None`` is the ordinary case: drf-services substitutes its no-op."""
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    toolset = SpecToolset({"create_widget": reporting_spec()})
    tools = await toolset.get_tools(None)

    result = await toolset.call_tool(
        "create_widget", {"name": "a", "price": 1}, ctx_for(user), tools["create_widget"]
    )
    assert result["name"] == "a"


async def test_a_deps_type_without_a_progress_field_is_fine():
    """A project's own deps class predates this field; a missing sink is normal."""
    assert _default_get_progress(SimpleNamespace(deps=SimpleNamespace(user=None))) is None
    assert _default_get_progress(SimpleNamespace()) is None


# ----- the logger -----


@pytest.mark.django_db(transaction=True)
async def test_a_call_is_timed_at_debug(caplog):
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    toolset = SpecToolset({"list_widgets": list_spec()})
    tools = await toolset.get_tools(None)

    with caplog.at_level(logging.DEBUG, logger="rest_framework_pydantic_ai"):
        await toolset.call_tool("list_widgets", {}, ctx_for(user), tools["list_widgets"])

    assert "Tool 'list_widgets' on toolset 'drf-specs' took" in caplog.text


@pytest.mark.django_db(transaction=True)
async def test_a_denial_is_logged_at_warning_and_still_raises(caplog):
    """The one failure with no other trace.

    A retry reaches the model and an ``{"error": …}`` reaches the answer, but a
    denial aborts the run and is absorbed by whatever is driving it — so over
    HTTP this is a `403` in the access log and here it was nothing at all.
    """
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    toolset = SpecToolset({"list_widgets": list_spec(permission_classes=[DenyAll])})
    tools = await toolset.get_tools(None)

    with (
        caplog.at_level(logging.WARNING, logger="rest_framework_pydantic_ai"),
        pytest.raises(PermissionDenied),
    ):
        await toolset.call_tool("list_widgets", {}, ctx_for(user), tools["list_widgets"])

    assert "Permission denied calling tool 'list_widgets'" in caplog.text


# ----- outbound bounds -----


@pytest.mark.django_db(transaction=True)
async def test_an_over_budget_result_fails_rather_than_truncating():
    """A partial payload looks complete.

    A list cut at the byte ceiling is indistinguishable from a list that had
    that many rows, so a model would answer confidently from data it does not
    know is missing. Refusing costs a turn; truncating costs correctness.
    """
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    for n in range(5):
        await sync_to_async(Widget.objects.create)(name=f"w{n}", price=n, owner=user)

    toolset = SpecToolset({"list_widgets": list_spec()}, max_result_bytes=20)
    tools = await toolset.get_tools(None)
    result = await toolset.call_tool("list_widgets", {}, ctx_for(user), tools["list_widgets"])

    assert "over the 20 byte ceiling" in result["error"]
    assert "not truncated" in result["error"]


@pytest.mark.django_db(transaction=True)
async def test_a_result_inside_the_ceiling_passes_through_untouched():
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    await sync_to_async(Widget.objects.create)(name="a", price=1, owner=user)

    toolset = SpecToolset({"list_widgets": list_spec()}, max_result_bytes=100_000)
    tools = await toolset.get_tools(None)
    result = await toolset.call_tool("list_widgets", {}, ctx_for(user), tools["list_widgets"])

    assert [w["name"] for w in result] == ["a"]


@pytest.mark.django_db(transaction=True)
async def test_a_per_tool_ceiling_of_none_means_no_ceiling_for_that_tool():
    """Absent and ``None`` are different answers: one inherits, one opts out."""
    from asgiref.sync import sync_to_async

    user = await sync_to_async(User.objects.create)(username="u")
    await sync_to_async(Widget.objects.create)(name="a", price=1, owner=user)

    toolset = SpecToolset(
        {"list_widgets": list_spec()},
        max_result_bytes=1,
        tool_max_result_bytes={"list_widgets": None},
    )
    tools = await toolset.get_tools(None)
    result = await toolset.call_tool("list_widgets", {}, ctx_for(user), tools["list_widgets"])

    assert [w["name"] for w in result] == ["a"]


def test_a_per_tool_ceiling_for_an_unknown_tool_is_a_typo():
    with pytest.raises(ValueError, match="unknown tool 'nope'"):
        SpecToolset({"list_widgets": list_spec()}, tool_max_result_bytes={"nope": 10})


async def test_the_page_ceiling_is_advertised_on_limit():
    toolset = SpecToolset({"list_widgets": list_spec()}, max_page_size=50)
    tools = await toolset.get_tools(None)
    props = tools["list_widgets"].tool_def.parameters_json_schema["properties"]
    assert props["limit"]["maximum"] == 50


@pytest.mark.django_db
def test_the_page_ceiling_is_clamped_as_well_as_advertised():
    """Advertising alone is a hint that nothing obliges a model to honour."""
    user = User.objects.create(username="u")
    for n in range(5):
        Widget.objects.create(name=f"w{n}", price=n, owner=user)

    result = _dispatch(list_spec(), user, {"limit": 1000}, max_page_size=2)
    assert len(result) == 2


@pytest.mark.django_db
def test_an_omitted_limit_becomes_the_ceiling():
    """The unbounded read is the one that hurts, and it is the one a model
    produces by simply not thinking about pagination."""
    user = User.objects.create(username="u")
    for n in range(5):
        Widget.objects.create(name=f"w{n}", price=n, owner=user)

    assert len(_dispatch(list_spec(), user, {}, max_page_size=3)) == 3
    # With no ceiling configured, an omitted limit still means "everything".
    assert len(_dispatch(list_spec(), user, {})) == 5


async def test_a_call_past_its_deadline_answers_instead_of_hanging():
    async def _slow():
        await asyncio.sleep(1)

    result = await _with_deadline(_slow(), 0.01, label="list_widgets")
    assert "longer than the 0.01s limit" in result["error"]


async def test_no_deadline_awaits_normally():
    async def _quick():
        return "done"

    assert await _with_deadline(_quick(), None, label="x") == "done"


# --- the overridable middle (A2 / A3 / A6) -----------------------------------
#
# Everything between argument intake and result rendering used to be sealed in a
# module-level private: a subclass could replace ``call_tool`` wholesale or
# nothing at all. These cover the two seams and the two knobs that ride on them.


class _Boom(Exception):
    """Stands in for an exception this package knows nothing about."""


def test_an_unmapped_exception_still_aborts_the_run():
    """The default is unchanged — a genuine bug must not become a tool result."""

    def boom(user):
        """Boom."""
        raise _Boom("kaboom")

    with pytest.raises(_Boom):
        _dispatch(ServiceSpec(service=boom, atomic=False), object())


def test_exception_map_turns_an_unknown_exception_into_a_result():
    """The motivating case: Django's ``ValidationError``, not DRF's.

    Raised by ``full_clean`` and any custom model validator, absent from the
    translated set, and so fatal to the run where the DRF twin would have been a
    retry. Broadening the built-in set fixes that one; a map fixes the class.
    """

    def boom(user):
        """Boom."""
        raise DjangoValidationError("that name is taken")

    result = _dispatch(
        ServiceSpec(service=boom, atomic=False),
        object(),
        toolset_kwargs={"exception_map": {DjangoValidationError: lambda exc: {"error": str(exc)}}},
    )
    assert "that name is taken" in result["error"]


def test_a_handler_may_retry_instead_of_returning():
    """Both outcomes of the model loop are reachable from one map."""

    def boom(user):
        """Boom."""
        raise _Boom("try again")

    with pytest.raises(ModelRetry, match="try again"):
        _dispatch(
            ServiceSpec(service=boom, atomic=False),
            object(),
            toolset_kwargs={
                "exception_map": {_Boom: lambda exc: (_ for _ in ()).throw(ModelRetry(str(exc)))}
            },
        )


def test_a_base_class_registration_catches_a_subclass():
    """Registering ``Exception`` should not require enumerating every subclass."""

    def boom(user):
        """Boom."""
        raise _Boom("anything")

    result = _dispatch(
        ServiceSpec(service=boom, atomic=False),
        object(),
        toolset_kwargs={"exception_map": {Exception: lambda exc: {"error": "caught"}}},
    )
    assert result == {"error": "caught"}


def test_the_most_specific_registration_wins():
    """MRO order, so a narrow handler is not shadowed by a broad one."""

    def boom(user):
        """Boom."""
        raise _Boom("x")

    result = _dispatch(
        ServiceSpec(service=boom, atomic=False),
        object(),
        toolset_kwargs={
            "exception_map": {
                Exception: lambda exc: {"error": "broad"},
                _Boom: lambda exc: {"error": "narrow"},
            }
        },
    )
    assert result == {"error": "narrow"}


def test_the_map_overrides_a_built_in_arm():
    """Deliberate: the built-ins are this package's guess, not a law.

    A ``ServiceError`` normally becomes ``{"error": …}``. A project that would
    rather the model retry must be able to say so, or the map is only half a
    seam.
    """

    def boom(user):
        """Boom."""
        raise ServiceError("nope")

    with pytest.raises(ModelRetry):
        _dispatch(
            ServiceSpec(service=boom, atomic=False),
            object(),
            toolset_kwargs={
                "exception_map": {
                    ServiceError: lambda exc: (_ for _ in ()).throw(ModelRetry(str(exc)))
                }
            },
        )


@pytest.mark.django_db
def test_http_request_is_forwarded_to_the_offline_context():
    """A3 — the parameter existed upstream all along and was never passed."""
    request = RequestFactory().get("/", HTTP_X_TENANT="acme")
    seen = {}

    def peek(user, request=None, **_):
        """Peek."""
        seen["tenant"] = request.META.get("HTTP_X_TENANT")
        return {"ok": True}

    _dispatch(
        ServiceSpec(service=peek, atomic=False),
        object(),
        toolset_kwargs={"http_request": request},
    )
    assert seen["tenant"] == "acme"


@pytest.mark.django_db
def test_get_http_request_can_vary_the_request_per_run():
    """The per-run form, mirroring ``get_user`` — the point of routing A3 through A2."""
    seen = {}

    def peek(user, request=None, **_):
        """Peek."""
        seen["tenant"] = request.META.get("HTTP_X_TENANT")
        return {"ok": True}

    def from_deps(ctx):
        return RequestFactory().get("/", HTTP_X_TENANT=ctx.deps.user)

    _dispatch(
        ServiceSpec(service=peek, atomic=False),
        "beta-corp",
        toolset_kwargs={"get_http_request": from_deps},
    )
    assert seen["tenant"] == "beta-corp"


def test_no_request_arrives_unless_one_was_configured():
    """The honest default — a request nothing declared is one nobody can audit."""
    assert _default_get_http_request(SimpleNamespace(deps=AgentDeps(user=object()))) is None


@pytest.mark.django_db
def test_build_context_is_overridable_and_sees_the_run():
    """A2 proper: a subclass reaches dispatch with the run's typed deps in hand."""
    seen = {}

    def peek(user, request=None, **_):
        """Peek."""
        seen["tenant"] = request.META.get("HTTP_X_TENANT")
        return {"ok": True}

    class TenantToolset(SpecToolset):
        def build_context(self, user, params, *, ctx, **kwargs):
            # The whole ask: per-run typed deps informing dispatch, with no
            # generic "set attributes on the request" knob in sight.
            # ``ctx.deps`` is the whole point: the tenant is per-run, so it
            # cannot be baked into the toolset at construction. Overriding
            # ``build_context`` rather than the extractor — ``_get_http_request``
            # is an instance attribute assigned in ``__init__``, so a method of
            # that name would be shadowed by it. This is the documented seam.
            return build_offline_context(
                user,
                params,
                http_request=RequestFactory().get("/", HTTP_X_TENANT=ctx.deps.user),
                **kwargs,
            )

    spec = ServiceSpec(service=peek, atomic=False, permission_classes=[AllowAny])
    toolset = TenantToolset({"t": spec})
    ctx = SimpleNamespace(deps=AgentDeps(user="gamma-ltd"))
    toolset._call_spec(spec, "gamma-ltd", {}, ctx=ctx)

    assert seen["tenant"] == "gamma-ltd"


def test_translate_exception_is_overridable_without_a_map():
    """The method is the seam; ``exception_map`` is the declarative shortcut."""

    def boom(user):
        """Boom."""
        raise _Boom("subclassed")

    class Translating(SpecToolset):
        def translate_exception(self, exc, *, ctx):
            if isinstance(exc, _Boom):
                return lambda e: {"error": f"handled {e}"}
            return super().translate_exception(exc, ctx=ctx)

    spec = ServiceSpec(service=boom, atomic=False, permission_classes=[AllowAny])
    toolset = Translating({"t": spec})
    ctx = SimpleNamespace(deps=AgentDeps(user=object()))

    assert toolset._call_spec(spec, object(), {}, ctx=ctx) == {"error": "handled subclassed"}


# --- the composing caller's view of a built toolset --------------------------


def test_specs_is_readable_without_a_run():
    """``get_tools`` is async and needs a ``RunContext``, so it cannot answer here.

    A caller composing this toolset into a name-dedup pass or a tool catalog does
    so at configuration time, with no run in sight. Without a public answer they
    reach for ``_specs``, and a private becomes load-bearing across a package
    boundary.
    """
    a, b = list_spec(), retrieve_spec()
    toolset = SpecToolset({"list": a, "get": b})

    assert dict(toolset.specs) == {"list": a, "get": b}


def test_specs_cannot_be_written_through():
    """A tool added here would have skipped every check the constructor ran."""
    toolset = SpecToolset({"list": list_spec()})

    with pytest.raises(TypeError):
        toolset.specs["sneaky"] = list_spec()  # ty: ignore[unsupported-operation]


def test_specs_reflects_a_registry_source_resolved():
    """A ``SpecRegistry`` in, a plain mapping out — the same normalisation."""
    registry = SpecRegistry()
    registry.register("list", list_spec())

    assert set(SpecToolset(registry).specs) == {"list"}


# --- agent audience -----------------------------------------------------------


def agent_list_spec(**kwargs):
    kwargs.setdefault("permission_classes", [AllowAny])
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=AgentWidgetSerializer,
        **kwargs,
    )


@pytest.mark.django_db
def test_hidden_fields_leave_the_payload_and_choices_are_spoken():
    """A toolset advertises no output schema, so the payload is all the model sees."""
    user = User.objects.create(username="u")
    Widget.objects.create(name="Sprocket", price=100, owner=user)

    rows = _dispatch(agent_list_spec(), user)

    assert rows == [{"id": 1, "name": "Sprocket", "status": "In stock"}]


@pytest.mark.django_db
async def test_call_tool_uses_the_projection_built_at_construction():
    user = await User.objects.acreate(username="u2")
    await Widget.objects.acreate(name="Sprocket", price=100, owner=user)
    toolset = SpecToolset({"widgets": agent_list_spec()})

    rows = await toolset.call_tool("widgets", {}, ctx_for(user), None)

    assert rows == [{"id": 1, "name": "Sprocket", "status": "In stock"}]


async def test_handle_instruction_present_only_when_a_tool_has_a_handle():
    with_handle = await SpecToolset({"widgets": agent_list_spec()}).get_instructions(None)
    assert _HANDLE_INSTRUCTION in with_handle

    # An unmarked serializer teaches the model nothing about handles.
    without = await SpecToolset({"list": list_spec()}).get_instructions(None)
    assert _HANDLE_INSTRUCTION not in without


@pytest.mark.parametrize("no_default", [None, UNSET])
def test_a_kwarg_declaring_no_default_contributes_nothing(no_default: Any) -> None:
    """Both sentinels the sister package has used for "no default" read as absent.

    drf-services used a plain ``None`` through 0.43 and switched to ``UNSET`` in
    0.44, so that ``default=None`` could mean an explicit null. A check of only
    ``is not None`` reads ``UNSET`` as a real value and hands the sentinel object
    to the spec as an argument. Parametrized over both so this keeps holding
    whichever side of that boundary the resolved version falls on.
    """
    args: dict[str, Any] = {}
    values = _pop_url_kwargs(
        [UrlKwarg(name="project_pk", type="integer", default=no_default)], args
    )
    assert values == {}


@pytest.mark.parametrize("no_default", [None, UNSET])
def test_a_required_kwarg_is_not_satisfied_by_the_no_default_sentinel(no_default: Any) -> None:
    """The requiredness check must still run, and must still name the kwarg.

    Reading the sentinel as a value skips the ``elif required`` arm entirely, so
    the model got no ``ModelRetry`` naming what it omitted -- the spec was handed
    the sentinel and failed further in, where the reason is far less legible.
    """
    with pytest.raises(ModelRetry, match="project_pk"):
        _pop_url_kwargs(
            [UrlKwarg(name="project_pk", type="integer", required=True, default=no_default)], {}
        )


@pytest.mark.parametrize("no_default", [None, UNSET])
def test_a_query_param_declaring_no_default_contributes_nothing(no_default: Any) -> None:
    values = _pop_query_params([QueryParam(name="fields", default=no_default)], {})
    assert values == {}


def test_a_real_default_still_reaches_the_spec() -> None:
    """The other half of the predicate: a declared default is still applied."""
    assert _pop_url_kwargs([UrlKwarg(name="pk", type="integer", default=7)], {}) == {"pk": 7}
    assert _pop_query_params([QueryParam(name="fields", default="id")], {}) == {"fields": "id"}


# --- what a permission class sees off HTTP -----------------------------------
#
# drf-services documents three attributes of the synthetic request / view a
# permission class may read off HTTP. These pin what this package puts in each,
# because a permission class is the only thing standing between a model and a
# spec, and every one of them is authored against the HTTP transport.


class RecordingPermission(BasePermission):
    """Allows everything, and records what it was shown."""

    seen: dict[str, Any] = {}

    def has_permission(self, request, view):
        RecordingPermission.seen = {
            "action": view.action,
            "kwargs": dict(view.kwargs),
            "query_params": dict(request.query_params.lists()),
            "user": request.user,
        }
        return True


async def _call(toolset: SpecToolset, name: str, user: Any, args: dict[str, Any] | None = None):
    tools = await toolset.get_tools(ctx_for(user))
    return await toolset.call_tool(name, dict(args or {}), ctx_for(user), tools[name])


@pytest.mark.django_db
async def test_the_tool_name_is_the_view_action_a_permission_class_reads():
    """``view.action`` used to be ``None`` for every spec this toolset exposed.

    A permission class branching on the action -- strict for the actions it
    knows, permissive otherwise -- then took the permissive arm on every call,
    so the same spec behind the same class was gated one way over HTTP and
    another way through an agent. The tool name is the identity the model
    called, and it is what the MCP transport reports for the same spec.
    """

    def noop(user):
        """Do it."""
        return {"ok": True}

    toolset = SpecToolset(
        {
            "suspend_account": ServiceSpec(
                service=noop, atomic=False, permission_classes=[RecordingPermission]
            )
        },
        descriptions={"suspend_account": "Suspend an account."},
    )
    RecordingPermission.seen = {}

    await _call(toolset, "suspend_account", object())

    assert RecordingPermission.seen["action"] == "suspend_account"


@pytest.mark.django_db
async def test_an_override_can_rewrite_the_action_it_is_handed():
    """The seam stays a seam: ``action`` arrives in the forwarding ``**kwargs``.

    A project whose permission class branches on viewset action names needs to
    supply one of those, and must not have to reimplement the whole context
    build to do it.
    """

    def noop(user):
        """Do it."""
        return {"ok": True}

    class RetrieveShaped(SpecToolset):
        def build_context(self, user, params, *, ctx, **kwargs):
            return super().build_context(user, params, ctx=ctx, **{**kwargs, "action": "retrieve"})

    toolset = RetrieveShaped(
        {
            "get_thing": ServiceSpec(
                service=noop, atomic=False, permission_classes=[RecordingPermission]
            )
        },
        descriptions={"get_thing": "Get a thing."},
    )
    RecordingPermission.seen = {}

    await _call(toolset, "get_thing", object())

    assert RecordingPermission.seen["action"] == "retrieve"


@pytest.mark.django_db
async def test_a_configured_requests_query_string_never_reaches_the_spec():
    """A tool declaring no ``QueryParam`` dispatches with an empty query string.

    ``build_offline_context`` replaces the wrapped request's ``GET`` only when
    ``query_params`` is not ``None``, so passing ``None`` for an empty
    declaration left the ambient endpoint's own query string live inside the
    spec -- a field-selection or ``filter_set`` channel neither the tool schema
    nor the model chose. The MCP transport passes a mapping at every call site
    for exactly this reason.
    """

    def noop(user):
        """Do it."""
        return {"ok": True}

    toolset = SpecToolset(
        {
            "do_it": ServiceSpec(
                service=noop, atomic=False, permission_classes=[RecordingPermission]
            )
        },
        descriptions={"do_it": "Do it."},
        http_request=RequestFactory().post("/agent/?query={id,secret}&fields=ssn"),
    )
    RecordingPermission.seen = {}

    await _call(toolset, "do_it", object())

    assert RecordingPermission.seen["query_params"] == {}


@pytest.mark.django_db
async def test_a_declared_query_param_still_reaches_the_spec_alongside_it():
    """The other half: replacing the query string is not dropping it."""

    def noop(user):
        """Do it."""
        return {"ok": True}

    toolset = SpecToolset(
        {
            "do_it": ServiceSpec(
                service=noop, atomic=False, permission_classes=[RecordingPermission]
            )
        },
        descriptions={"do_it": "Do it."},
        query_params=[QueryParam(name="fields")],
        http_request=RequestFactory().post("/agent/?fields=ssn&query={id}"),
    )
    RecordingPermission.seen = {}

    await _call(toolset, "do_it", object(), {"fields": "id,name"})

    assert RecordingPermission.seen["query_params"] == {"fields": ["id,name"]}


# --- the shared-request guarantee the declared floor provides ----------------


@pytest.mark.django_db
def test_one_configured_request_is_not_shared_between_two_dispatches():
    """Pins ``djangorestframework-services>=0.44``; lowering the floor fails here.

    ``http_request=`` captures one object for the toolset's lifetime, and
    ``call_tool`` dispatches through ``sync_to_async``, so two concurrent runs
    on one module-level toolset build their contexts from the same request.
    Below the declared floor ``build_offline_context`` wrapped that object
    itself, so the two contexts aliased it and the second call's query params
    silently became the first's -- a serializer shaped by another run's
    arguments, with no error anywhere. The floor wraps a shallow copy instead.
    """
    shared = RequestFactory().get("/agent/?fields=original")
    user = object()

    first = build_offline_context(user, {}, http_request=shared, query_params={"fields": "id,name"})
    second = build_offline_context(user, {}, http_request=shared, query_params={"fields": "id,ssn"})

    assert first.request._request is not second.request._request
    assert first.request.query_params["fields"] == "id,name"
    assert second.request.query_params["fields"] == "id,ssn"


@pytest.mark.django_db
async def test_a_dispatch_leaves_the_configured_request_untouched():
    """The second half: the caller's own request survives the call unchanged.

    A project handing the toolset the request it is serving must be able to keep
    reading it afterwards -- a later permission class, an audit hook, a
    ``request.method`` branch. Below the declared floor a dispatch forced
    ``method`` to ``POST``, replaced ``GET`` and reassigned ``user`` on that very
    object, and the effects outlived the call.
    """

    def noop(user):
        """Do it."""
        return {"ok": True}

    ambient = RequestFactory().get("/agent/?fields=original")
    ambient.user = "the-operator"
    toolset = SpecToolset(
        {"do_it": ServiceSpec(service=noop, atomic=False, permission_classes=[AllowAny])},
        descriptions={"do_it": "Do it."},
        query_params=[QueryParam(name="fields")],
        http_request=ambient,
    )

    await _call(toolset, "do_it", object(), {"fields": "id,name"})

    assert ambient.method == "GET"
    assert ambient.GET.dict() == {"fields": "original"}
    assert ambient.user == "the-operator"


# --- the tool catalog --------------------------------------------------------


async def test_every_tool_is_listed_by_default():
    """The documented posture: listing discloses, ``permission_classes`` gate.

    A permission whose answer depends on the arguments has none to read at
    listing time, so filtering by default would hide tools the caller can
    actually invoke.
    """
    toolset = SpecToolset({"list_widgets": list_spec(), "make": create_spec()})

    tools = await toolset.get_tools(ctx_for(object()))

    assert sorted(tools) == ["list_widgets", "make"]


async def test_is_tool_listed_can_narrow_the_catalog_for_one_run():
    """The seam for a deployment that does want a per-run catalog."""

    class StaffScoped(SpecToolset):
        async def is_tool_listed(self, name, ctx):
            return name != "make" or ctx.deps.user == "staff"

    toolset = StaffScoped({"list_widgets": list_spec(), "make": create_spec()})

    assert sorted(await toolset.get_tools(ctx_for("staff"))) == ["list_widgets", "make"]
    assert sorted(await toolset.get_tools(ctx_for("visitor"))) == ["list_widgets"]


async def test_a_hidden_tool_is_still_callable_and_still_gated():
    """Hiding is a disclosure decision, never an authorization one.

    Nothing routes tool calls through the catalog's absence, so an override that
    hides the wrong tool must not become the only thing standing between a model
    and a spec.
    """

    class HideEverything(SpecToolset):
        async def is_tool_listed(self, name, ctx):
            return False

    toolset = HideEverything({"denied": create_spec(permission_classes=[DenyAll])})

    assert await toolset.get_tools(ctx_for(object())) == {}
    with pytest.raises(PermissionDenied):
        await toolset.call_tool("denied", {}, ctx_for(object()), None)
