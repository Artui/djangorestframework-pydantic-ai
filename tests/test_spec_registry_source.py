"""A ``SpecRegistry`` is accepted wherever a ``name -> spec`` mapping is."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import django_service_specs
import pytest
from django.contrib.auth.models import User
from django.core.exceptions import ImproperlyConfigured
from django_service_specs import FieldAudience, FieldMarking, SelectorKind, SpecRegistry
from rest_framework.permissions import AllowAny, BasePermission
from rest_framework_services import SelectorSpec, ServiceSpec

from rest_framework_pydantic_ai import QueryParam, SpecCapability, SpecToolset, UrlKwarg
from tests.test_spec_toolset import ctx_for
from tests.testapp.models import Widget
from tests.testapp.serializers import AgentWidgetSerializer, WidgetSerializer


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


def _agent_list_spec() -> SelectorSpec:
    """The same read, through the serializer that carries agent markings."""
    return SelectorSpec(
        kind=SelectorKind.LIST,
        selector=list_widgets,
        output_serializer=AgentWidgetSerializer,
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


def _native_registry() -> SpecRegistry:
    # A django-service-specs spec, which the registry accepts because it
    # compiles to a kernel spec, and which drf-services cannot dispatch.
    registry = SpecRegistry()
    registry.register(
        "native_create",
        django_service_specs.ServiceSpec(
            service=create_widget, permissions=[django_service_specs.Unrestricted()]
        ),
    )
    return registry


_NATIVE_REFUSAL = (
    r"^SpecToolset tool 'native_create': django_service_specs\.ServiceSpec is not a "
    r"drf-services spec\. "
)


class TestANativeSpecIsRefused:
    """A kernel-native spec reached the toolset's internals and died on the first
    drf-services attribute it lacked, an ``AttributeError`` naming neither the
    tool nor what to do. The refusal is a ``TypeError`` because the value is the
    wrong kind, and it names the tool and both ways out."""

    def test_from_a_registry(self) -> None:
        with pytest.raises(TypeError, match=_NATIVE_REFUSAL) as raised:
            SpecToolset(_native_registry())

        message = str(raised.value)
        assert "rest_framework_services.ServiceSpec" in message
        assert "rest_framework_services.SelectorSpec" in message
        assert "django_service_specs.dispatch" in message

    def test_from_a_mapping(self) -> None:
        native = _native_registry().specs()

        with pytest.raises(TypeError, match=_NATIVE_REFUSAL):
            SpecToolset(native)

    def test_through_the_capability(self) -> None:
        with pytest.raises(TypeError, match=_NATIVE_REFUSAL):
            SpecCapability(_native_registry())


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


class TestTheEntrysOfflineContract:
    """What a caller with no HTTP request has to be told, declared on the spec.

    Over HTTP the URLconf supplies the route captures and the query string the
    read-shaping params, so a spec mounted on a view is already complete. Off
    HTTP nobody supplies them, and every agent transport needs the *identical*
    answer -- which is why the declaration belongs to the spec (``reads=`` and
    ``field_audiences=``), whose registry entry answers it as
    ``entry.agent_contract``, rather than to this toolset's constructor or an
    MCP server's registrar.
    """

    def _contract_registry(
        self,
        *,
        reads: tuple[UrlKwarg | QueryParam, ...] = (),
        field_audiences: dict[str, FieldMarking] | None = None,
    ) -> SpecRegistry:
        registry = SpecRegistry()
        registry.register(
            "list_widgets",
            replace(_agent_list_spec(), reads=reads, field_audiences=field_audiences),
        )
        return registry

    def test_url_kwargs_reach_the_tool_with_nothing_in_the_constructor(self) -> None:
        toolset = SpecToolset(self._contract_registry(reads=(UrlKwarg("tenant_pk"),)))

        props = toolset._tool_defs["list_widgets"].parameters_json_schema["properties"]
        assert "tenant_pk" in props
        assert toolset._tool_url_kwargs["list_widgets"] == (UrlKwarg("tenant_pk"),)

    def test_query_params_reach_the_tool_with_nothing_in_the_constructor(self) -> None:
        toolset = SpecToolset(self._contract_registry(reads=(QueryParam(name="since"),)))

        props = toolset._tool_defs["list_widgets"].parameters_json_schema["properties"]
        assert "since" in props
        assert toolset._tool_query_params["list_widgets"] == (QueryParam(name="since"),)

    def test_field_audiences_reach_the_projection(self) -> None:
        # The asymmetry this once closed -- one spec projecting a different field
        # set depending on which agent transport served it -- is now closed at
        # the source: the audiences are part of the spec's own output
        # declaration, so the registry and a bare mapping project alike.
        registry = self._contract_registry(field_audiences={"price": FieldMarking()})

        from_registry = SpecToolset(registry)._projections["list_widgets"]
        from_mapping = SpecToolset(registry.specs())._projections["list_widgets"]

        assert from_registry.audience("price") is FieldAudience.CONTENT
        assert from_mapping.audience("price") is FieldAudience.CONTENT

    def test_the_serializer_still_decides_when_the_contract_is_silent(self) -> None:
        toolset = SpecToolset(self._contract_registry())

        assert toolset._projections["list_widgets"].audience("price") is FieldAudience.HIDDEN

    def test_a_field_audiences_clash_is_refused_when_the_toolset_is_built(self) -> None:
        # ``name`` is already the serializer's label; claiming it for ``id`` as
        # well leaves two, and a record has one name. The refusal is
        # drf-services', from the spec's own output declaration, and the toolset
        # prefixes it with the tool: drf-services names the output, which a
        # registry of many specs shares across tools.
        registry = self._contract_registry(field_audiences={"id": FieldMarking.label()})

        with pytest.raises(
            ImproperlyConfigured,
            match=r"^SpecToolset tool 'list_widgets': Output: both 'id' and 'name' are marked",
        ) as raised:
            SpecToolset(registry)

        # Re-raised, not replaced: drf-services' own refusal is the cause.
        assert isinstance(raised.value.__cause__, ImproperlyConfigured)
        assert str(raised.value.__cause__).startswith("Output: both 'id' and 'name'")

    def test_a_field_audiences_key_naming_no_output_field_is_refused(self) -> None:
        # A marking for a field the output does not carry would silently project
        # nothing; drf-services refuses it, and the toolset names the tool.
        registry = self._contract_registry(field_audiences={"nope": FieldMarking.label()})

        with pytest.raises(ImproperlyConfigured) as raised:
            SpecToolset(registry)

        assert str(raised.value).startswith("SpecToolset tool 'list_widgets': ")
        assert "'nope'" in str(raised.value)

    def test_this_mounts_declaration_overrides_the_entrys_by_name(self) -> None:
        # The contract is a default. A constructor declaration is this mount's
        # word about this deployment, and wins.
        toolset = SpecToolset(
            self._contract_registry(reads=(QueryParam(name="since", description="entry"),)),
            query_params=[QueryParam(name="since", description="mount")],
        )

        (param,) = toolset._tool_query_params["list_widgets"]
        assert param.description == "mount"

    def test_a_per_tool_declaration_overrides_both(self) -> None:
        toolset = SpecToolset(
            self._contract_registry(reads=(UrlKwarg("tenant_pk"),)),
            url_kwargs=[UrlKwarg("tenant_pk", description="mount")],
            tool_url_kwargs={"list_widgets": [UrlKwarg("tenant_pk", description="tool")]},
        )

        (kwarg,) = toolset._tool_url_kwargs["list_widgets"]
        assert kwarg.description == "tool"

    def test_the_contract_still_answers_to_the_shared_channel_checks(self) -> None:
        # Nothing about arriving from an entry exempts a declaration from the
        # checks a constructor one answers to: both are merged first, and the
        # merged tuple is what is validated and what reaches the schema. The
        # entry's ``UrlKwarg`` and the mount's ``QueryParam`` meet only there --
        # one spec cannot declare the name twice, which drf-services refuses
        # when the spec is built.
        registry = self._contract_registry(reads=(UrlKwarg("tenant_pk"),))

        with pytest.raises(ValueError, match="cannot route to two channels"):
            SpecToolset(registry, query_params=[QueryParam(name="tenant_pk")])

    @pytest.mark.django_db(transaction=True)
    async def test_a_bare_mapping_advertises_no_read_and_still_routes_one(self) -> None:
        """``registry.specs()`` flattens the entry to its spec, and the toolset
        reads channels off the entry alone, so a bare mapping's tool advertises
        none of what the spec reads. The spec still carries its ``reads``, and
        drf-services routes a value sent under one of those names onto the
        offline view and its query string as it does for any caller, so the call
        is served: what the mapping drops is the advertisement, not the read.
        """
        spec = replace(
            _agent_list_spec(),
            permission_classes=[_RecordingPermission],
            reads=(UrlKwarg("tenant_pk"), QueryParam(name="since")),
        )
        registry = SpecRegistry()
        registry.register("list_widgets", spec)

        user = await User.objects.acreate(username="u")

        toolset = SpecToolset(registry.specs())
        tools = await toolset.get_tools(ctx_for(user))
        await toolset.call_tool(
            "list_widgets",
            {"tenant_pk": "5", "since": "2020"},
            ctx_for(user),
            tools["list_widgets"],
        )

        assert toolset._tool_url_kwargs["list_widgets"] == ()
        assert toolset._tool_query_params["list_widgets"] == ()
        props = toolset._tool_defs["list_widgets"].parameters_json_schema["properties"]
        assert "tenant_pk" not in props
        assert "since" not in props
        assert _RecordingPermission.seen == {
            "kwargs": {"tenant_pk": "5"},
            "query_params": {"since": ["2020"]},
        }


class _RecordingPermission(BasePermission):
    """Allows everything, and records the route captures and query it was shown."""

    seen: dict[str, Any] = {}

    def has_permission(self, request: Any, view: Any) -> bool:
        _RecordingPermission.seen = {
            "kwargs": dict(view.kwargs),
            "query_params": dict(request.query_params.lists()),
        }
        return True
