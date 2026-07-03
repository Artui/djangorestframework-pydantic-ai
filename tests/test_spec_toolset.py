from __future__ import annotations

from types import SimpleNamespace

import pytest
from django.contrib.auth.models import User
from pydantic_ai import ModelRetry
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

from rest_framework_pydantic_ai import AgentDeps, SpecToolset
from rest_framework_pydantic_ai.spec_toolset import (
    _call_spec,
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


# --- get_tools ---------------------------------------------------------------


async def test_get_tools_builds_function_tools():
    toolset = SpecToolset({"list_widgets": list_spec(), "create_widget": create_spec()})
    tools = await toolset.get_tools(None)
    assert set(tools) == {"list_widgets", "create_widget"}
    # ExternalToolset stamps "external" (deferred); we re-stamp "function" so the
    # run loop actually invokes call_tool.
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
