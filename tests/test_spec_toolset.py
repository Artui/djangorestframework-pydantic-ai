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
)

from drf_pydantic_ai import AgentDeps, SpecToolset
from drf_pydantic_ai.spec_toolset import (
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
