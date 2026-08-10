"""A ``SpecRegistry`` is accepted wherever a ``name -> spec`` mapping is."""

from __future__ import annotations

import pytest
from rest_framework.permissions import AllowAny
from rest_framework_services import (
    SelectorKind,
    SelectorSpec,
    ServiceSpec,
    SpecRegistry,
)

from rest_framework_pydantic_ai import QueryParam, SpecCapability, SpecToolset
from tests.testapp.models import Widget
from tests.testapp.serializers import WidgetSerializer


def list_widgets(user):
    """List widgets owned by the acting user."""
    return Widget.objects.filter(owner=user)


def create_widget(user, data):
    """Create a widget."""
    return Widget.objects.create(owner=user, **data)


def _list_spec() -> SelectorSpec:
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=WidgetSerializer,
        permission_classes=[AllowAny],
    )


def _create_spec() -> ServiceSpec:
    return ServiceSpec(service=create_widget, permission_classes=[AllowAny], atomic=False)


def _registry() -> SpecRegistry:
    registry = SpecRegistry()
    registry.register("list_widgets", _list_spec(), tags=("read", "public"))
    registry.register("create_widget", _create_spec(), tags=("write", "admin"))
    return registry


class TestToolsetFromRegistry:
    def test_exposes_every_entry_as_a_tool(self) -> None:
        toolset = SpecToolset(_registry())
        assert sorted(toolset._tool_defs) == ["create_widget", "list_widgets"]

    def test_matches_the_equivalent_dict(self) -> None:
        """``registry.specs()`` is the dict the toolset already accepted, so the
        two construction paths must be indistinguishable."""
        registry = _registry()
        from_registry = SpecToolset(registry)
        from_dict = SpecToolset(registry.specs())

        assert from_registry._specs == from_dict._specs
        assert from_registry._tool_defs.keys() == from_dict._tool_defs.keys()

    def test_read_only_hint_still_derives_from_spec_kind(self) -> None:
        toolset = SpecToolset(_registry())
        assert toolset._tool_defs["list_widgets"].metadata == {
            "annotations": {"readOnlyHint": True}
        }
        assert toolset._tool_defs["create_widget"].metadata == {
            "annotations": {"readOnlyHint": False}
        }

    def test_a_filtered_view_projects_a_subset(self) -> None:
        registry = _registry()
        reads = SpecToolset(registry.by_tag("read"), id="reads")

        assert list(reads._tool_defs) == ["list_widgets"]

    def test_two_toolsets_from_one_registry_are_independent(self) -> None:
        registry = _registry()
        reads = SpecToolset(registry.by_tag("read"), id="reads")
        writes = SpecToolset(registry.by_tag("write"), id="writes")

        assert list(reads._tool_defs) == ["list_widgets"]
        assert list(writes._tool_defs) == ["create_widget"]
        assert reads.id != writes.id

    def test_an_empty_registry_yields_no_tools(self) -> None:
        assert SpecToolset(SpecRegistry())._tool_defs == {}

    def test_registration_order_is_preserved(self) -> None:
        registry = SpecRegistry()
        registry.register("b_spec", _list_spec())
        registry.register("a_spec", _create_spec())

        assert list(SpecToolset(registry)._tool_defs) == ["b_spec", "a_spec"]

    def test_transport_knobs_still_apply(self) -> None:
        """The registry carries no transport config, so per-toolset knobs are
        unaffected by where the specs came from."""
        toolset = SpecToolset(
            _registry(),
            query_params=[QueryParam(name="tenant")],
        )
        props = toolset._tool_defs["list_widgets"].parameters_json_schema["properties"]
        assert "tenant" in props

    def test_a_registry_name_invalid_as_a_tool_name_still_fails_fast(self) -> None:
        """Registry names are free-form; provider tool names are not."""
        registry = SpecRegistry()
        registry.register("not a valid tool name", _list_spec())

        with pytest.raises(ValueError, match="not a valid tool name"):
            SpecToolset(registry)

    def test_per_tool_knobs_key_off_registry_names(self) -> None:
        toolset = SpecToolset(
            _registry(),
            tool_query_params={"list_widgets": [QueryParam(name="since")]},
        )
        assert "since" in toolset._tool_defs["list_widgets"].parameters_json_schema["properties"]
        other = toolset._tool_defs["create_widget"].parameters_json_schema
        assert "since" not in other.get("properties", {})

    def test_an_unknown_per_tool_key_still_raises(self) -> None:
        with pytest.raises(ValueError, match="typo"):
            SpecToolset(
                _registry(),
                tool_query_params={"typo": [QueryParam(name="x")]},
            )


class TestCapabilityFromRegistry:
    def test_wraps_a_registry(self) -> None:
        capability = SpecCapability(_registry())
        assert sorted(capability.get_toolset()._tool_defs) == ["create_widget", "list_widgets"]

    def test_matches_the_equivalent_dict(self) -> None:
        registry = _registry()
        assert (
            SpecCapability(registry).get_toolset()._specs
            == SpecCapability(registry.specs()).get_toolset()._specs
        )

    def test_filtered_views_give_independent_capabilities(self) -> None:
        registry = _registry()
        reads = SpecCapability(registry.by_tag("read"), id="reads")
        admin = SpecCapability(registry.by_tag("admin"), id="admin", defer_loading=True)

        assert list(reads.get_toolset()._tool_defs) == ["list_widgets"]
        assert list(admin.get_toolset()._tool_defs) == ["create_widget"]
        assert reads.defer_loading is False
        assert admin.defer_loading is True
        assert reads.id != admin.id

    def test_capability_id_mirrors_its_toolset(self) -> None:
        capability = SpecCapability(_registry(), id="reads")
        assert capability.id == capability.get_toolset().id == "reads"
