"""``tool_server_arguments``: values the server supplies for a tool's arguments, and names it withholds.

Written against the public signatures only -- ``SpecToolset``, ``SpecCapability``,
``ServerArguments`` and a real ``Agent`` run -- so each test compiles against a
toolset without the keyword and fails there on what it asserts, not on an import.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from django.contrib.auth.models import User
from django.core.exceptions import ImproperlyConfigured
from django_service_specs import (
    DEFAULT_POOL_SEEDS,
    SelectorKind,
    ServerArguments,
    SpecRegistry,
)
from pydantic_ai import Agent
from rest_framework.exceptions import ValidationError as DRFValidationError
from rest_framework.permissions import BasePermission
from rest_framework_services import SelectorSpec

from rest_framework_pydantic_ai import (
    AgentDeps,
    QueryParam,
    SpecCapability,
    SpecToolset,
    UrlKwarg,
)
from rest_framework_pydantic_ai.testing import tool_calling_model
from tests.test_spec_toolset import (
    _call,
    _input_schema_of,
    _name_list_spec,
    _NameSerializer,
    _paged_spec,
    _rows,
    _StrictRestqlWidget,
    _tool_return,
    ctx_for,
)
from tests.testapp.models import Widget


class _RecordingPermission(BasePermission):
    """Allows everything, and records the route captures and query string it was shown."""

    seen: dict[str, Any] = {}

    def has_permission(self, request: Any, view: Any) -> bool:
        _RecordingPermission.seen = {
            "kwargs": dict(view.kwargs),
            "query_params": dict(request.query_params.lists()),
        }
        return True


def vendors_in_scope(*, project_pk: str | None = None, draw_request_pk: str | None = None):
    """The vendors a document may be matched to: its project's, or its draw request's."""
    return [{"name": f"project={project_pk} draw={draw_request_pk}"}]


def vendors_of_project(*, project_pk: str):
    """The vendors of one project."""
    return [{"name": f"project={project_pk}"}]


def any_vendors():
    """Every vendor."""
    return [{"name": "any"}]


def _scoped_spec(selector: Any = vendors_in_scope, **kwargs: Any) -> SelectorSpec:
    kwargs.setdefault("permission_classes", [_RecordingPermission])
    kwargs.setdefault("reads", (UrlKwarg("project_pk"), UrlKwarg("draw_request_pk")))
    return _name_list_spec(selector, **kwargs)


def _required_spec() -> SelectorSpec:
    return _scoped_spec(vendors_of_project, reads=(UrlKwarg("project_pk", required=True),))


def _registry(**specs: SelectorSpec) -> SpecRegistry:
    registry = SpecRegistry()
    for name, spec in specs.items():
        registry.register(name, spec)
    return registry


_SCOPED = ServerArguments(values={"project_pk": "7"}, withheld=frozenset({"draw_request_pk"}))


# --- construction -------------------------------------------------------------


def test_a_tool_name_the_toolset_does_not_expose_is_refused_naming_it():
    """As every per-tool mapping is: a typo would otherwise bind nothing, silently."""
    with pytest.raises(ValueError, match=r"tool_server_arguments references unknown tool 'nope'"):
        SpecToolset(
            _registry(vendors=_scoped_spec()), tool_server_arguments={"nope": ServerArguments()}
        )


def test_a_name_the_spec_does_not_declare_is_refused_naming_the_tool():
    with pytest.raises(ImproperlyConfigured) as raised:
        SpecToolset(
            _registry(vendors=_scoped_spec()),
            tool_server_arguments={"vendors": ServerArguments(values={"tenant_pk": "1"})},
        )

    assert str(raised.value).startswith("SpecToolset tool 'vendors': ")
    assert "tenant_pk" in str(raised.value)


def test_a_bound_value_its_parameter_refuses_fails_at_construction():
    """A server value failing its own declaration is the server's bug, so it fails
    the mount rather than every call that reaches it."""
    spec = _scoped_spec(reads=(UrlKwarg("project_pk", type="integer"),))

    with pytest.raises(ImproperlyConfigured) as raised:
        SpecToolset(
            _registry(vendors=spec),
            tool_server_arguments={"vendors": ServerArguments(values={"project_pk": "seven"})},
        )

    assert str(raised.value).startswith("SpecToolset tool 'vendors': ")


def test_a_withheld_name_no_call_could_do_without_is_refused():
    """Withheld, the read is absent, and the selector declares it with no default."""
    with pytest.raises(ImproperlyConfigured) as raised:
        SpecToolset(
            _registry(vendors=_required_spec()),
            tool_server_arguments={"vendors": ServerArguments(withheld=frozenset({"project_pk"}))},
        )

    assert str(raised.value).startswith("SpecToolset tool 'vendors': ")
    assert "withholds project_pk" in str(raised.value)


def scoped_to_tenant(*, tenant: str):
    """The rows of one tenant."""
    return [{"name": tenant}]


def test_a_name_a_registered_seed_occupies_is_refused():
    """Checked against the toolset's own seeds: the kernel's default set does not
    hold ``tenant``, so a check that did not hear about them would pass this."""
    seeds = DEFAULT_POOL_SEEDS.extend(tenant=lambda: "acme")

    with pytest.raises(ImproperlyConfigured, match=r"tool 'rows': .*pool seed"):
        SpecToolset(
            {"rows": _name_list_spec(scoped_to_tenant)},
            pool_seeds=seeds,
            tool_server_arguments={"rows": ServerArguments(values={"tenant": "other"})},
        )


def test_a_withheld_name_this_mount_declares_required_is_refused():
    """The kernel reads the spec alone, so a ``UrlKwarg(required=True)`` declared
    at the mount for a withheld name passes its check, and the toolset would then
    refuse every call as missing an argument nobody may send."""
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ImproperlyConfigured) as raised,
    ):
        SpecToolset(
            {"vendors": _name_list_spec(vendors_in_scope)},
            tool_url_kwargs={"vendors": [UrlKwarg("project_pk", required=True)]},
            tool_server_arguments={"vendors": ServerArguments(withheld=frozenset({"project_pk"}))},
        )

    assert str(raised.value).startswith("SpecToolset tool 'vendors': ")
    assert "project_pk" in str(raised.value)


# --- the advertised schema ----------------------------------------------------


def test_bound_and_withheld_names_leave_the_schema_and_its_required():
    """Through the registry, where the reads reach the schema as this toolset's
    own channels, merged over the selector's parameters: the subtraction has to
    come after that merge, or a required ``UrlKwarg`` is written back."""
    registry = _registry(
        vendors=_scoped_spec(
            reads=(UrlKwarg("project_pk", required=True), UrlKwarg("draw_request_pk"))
        )
    )

    unbound = _input_schema_of(SpecToolset(registry), "vendors")
    bound = _input_schema_of(
        SpecToolset(registry, tool_server_arguments={"vendors": _SCOPED}), "vendors"
    )

    assert {"project_pk", "draw_request_pk"} <= set(unbound["properties"])
    assert unbound["required"] == ["project_pk"]
    assert not {"project_pk", "draw_request_pk"} & set(bound["properties"])
    assert "required" not in bound
    # Everything else is as it was.
    assert {k: v for k, v in unbound["properties"].items() if not k.endswith("_pk")} == bound[
        "properties"
    ]


def entry_on_page(*, page_pk: str, book: str, note: str = ""):
    """One entry, by its book and page."""
    return {"name": f"{book}{page_pk}{note}"}


def test_a_tool_with_no_channels_loses_the_bound_name_too():
    """A retrieve tool with no channel of its own takes the schema builder's early
    return, which has to subtract as the merged branch does."""
    spec = SelectorSpec(
        kind=SelectorKind.RETRIEVE,
        selector=entry_on_page,
        output_serializer=_NameSerializer,
        permission_classes=[_RecordingPermission],
    )
    toolset = SpecToolset(
        {"entry": spec},
        tool_server_arguments={"entry": ServerArguments(values={"page_pk": "3"})},
    )

    schema = _input_schema_of(toolset, "entry")
    assert set(schema["properties"]) == {"book", "note"}
    # What the binding does not answer for is still required.
    assert schema["required"] == ["book"]


def fields_free(*, user: Any):
    """Every vendor, shaped by the query string."""
    return [{"name": "v"}]


async def test_a_bound_query_param_is_not_offered_in_the_instructions():
    """The read-shaping line names every ``QueryParam`` a tool offers; a bound one
    is not offered, so it is not named."""
    registry = _registry(vendors=_scoped_spec(fields_free, reads=(QueryParam("fields"),)))
    unbound = SpecToolset(registry)
    bound = SpecToolset(
        registry, tool_server_arguments={"vendors": ServerArguments(values={"fields": "name"})}
    )

    assert "`fields`" in (await unbound.get_instructions(ctx_for()) or "")
    assert "fields" not in (await bound.get_instructions(ctx_for()) or "")


# --- each call ----------------------------------------------------------------


@pytest.mark.django_db(transaction=True)
async def test_a_binding_fills_a_required_read_the_model_never_sends():
    """The binding is applied before the call is checked for what it requires, so
    a required read the server supplies is never asked of the model. Applied any
    later, the model's empty call is answered with a missing-argument retry."""
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        _registry(vendors=_required_spec()),
        tool_server_arguments={"vendors": ServerArguments(values={"project_pk": "7"})},
    )
    agent = Agent(tool_calling_model("vendors", {}), deps_type=AgentDeps, toolsets=[toolset])

    result = await agent.run("go", deps=AgentDeps(user=user))

    assert _rows(_tool_return(result).content) == [{"name": "project=7"}]
    assert _RecordingPermission.seen["kwargs"] == {"project_pk": "7"}


@pytest.mark.django_db(transaction=True)
async def test_the_bound_value_outranks_the_one_the_model_sends(caplog):
    """Applied before the channels are split, so the bound value is the one
    ``view.kwargs`` carries. Applied after, the model's value is already on the
    route, which a selector reads over its arguments."""
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        _registry(vendors=_required_spec()),
        tool_server_arguments={"vendors": ServerArguments(values={"project_pk": "7"})},
    )
    agent = Agent(
        tool_calling_model("vendors", {"project_pk": "99"}), deps_type=AgentDeps, toolsets=[toolset]
    )

    with caplog.at_level(logging.WARNING, logger="rest_framework_pydantic_ai"):
        result = await agent.run("go", deps=AgentDeps(user=user))

    assert _rows(_tool_return(result).content) == [{"name": "project=7"}]
    assert _RecordingPermission.seen["kwargs"] == {"project_pk": "7"}
    (record,) = [r for r in caplog.records if "server supplies" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert "'vendors'" in record.getMessage()
    assert "project_pk" in record.getMessage()
    # The name, never the value: a binding is often an identifier the run is
    # scoped by, and a model's attempt at it is no business of the log's.
    assert "99" not in record.getMessage()
    assert "99" not in caplog.text


@pytest.mark.django_db(transaction=True)
async def test_a_withheld_name_takes_nothing_the_model_sends(caplog):
    """``"null"`` for an identifier the model was never shown is dropped before
    anything reads it, so it neither reaches the selector nor refuses the call."""
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        _registry(vendors=_scoped_spec()), tool_server_arguments={"vendors": _SCOPED}
    )

    with caplog.at_level(logging.WARNING, logger="rest_framework_pydantic_ai"):
        result = await _call(
            toolset, "vendors", user, {"project_pk": "99", "draw_request_pk": "null"}
        )

    assert _rows(result) == [{"name": "project=7 draw=None"}]
    assert _RecordingPermission.seen["kwargs"] == {"project_pk": "7"}
    assert "draw_request_pk, project_pk" in caplog.text


@pytest.mark.django_db(transaction=True)
async def test_nothing_is_logged_when_the_model_sends_no_server_name(caplog):
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        _registry(vendors=_scoped_spec()), tool_server_arguments={"vendors": _SCOPED}
    )

    with caplog.at_level(logging.WARNING, logger="rest_framework_pydantic_ai"):
        await _call(toolset, "vendors", user, {})

    assert "server supplies" not in caplog.text


@pytest.mark.django_db(transaction=True)
async def test_a_bound_query_param_reaches_the_query_string():
    """A bound ``QueryParam`` read takes the existing channel to
    ``request.query_params``, with the server's value over the model's."""
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        _registry(vendors=_scoped_spec(fields_free, reads=(QueryParam("fields"),))),
        tool_server_arguments={"vendors": ServerArguments(values={"fields": "name"})},
    )

    await _call(toolset, "vendors", user, {"fields": "secret"})

    assert _RecordingPermission.seen["query_params"] == {"fields": ["name"]}


@pytest.mark.django_db(transaction=True)
async def test_a_withheld_read_falls_back_to_its_declared_default():
    """Withholding means the caller does not choose, not null: the read's declared
    default answers, whatever the model sent."""
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        _registry(vendors=_scoped_spec(fields_free, reads=(QueryParam("fields", default="name"),))),
        tool_server_arguments={"vendors": ServerArguments(withheld=frozenset({"fields"}))},
    )

    await _call(toolset, "vendors", user, {"fields": "secret"})

    assert _RecordingPermission.seen["query_params"] == {"fields": ["name"]}


@pytest.mark.django_db(transaction=True)
async def test_a_tool_with_no_binding_is_untouched():
    """A binding is per tool: a sibling's call keeps what the model sent."""
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        _registry(vendors=_scoped_spec(), other=_scoped_spec()),
        tool_server_arguments={"vendors": _SCOPED},
    )

    result = await _call(toolset, "other", user, {"project_pk": "99"})

    assert _rows(result) == [{"name": "project=99 draw=None"}]
    assert "project_pk" in _input_schema_of(toolset, "other")["properties"]


@pytest.mark.django_db(transaction=True)
async def test_a_bound_selection_the_render_rejects_is_the_server_s_error_not_a_retry():
    """A bound read-shaping value is the server's, so a render that rejects it is a
    configuration bug that stays loud, as a rejected declared default does. A
    retry would name an argument the model was never offered and cannot change,
    and on a small budget end the run."""
    user = await User.objects.acreate(username="u")
    await Widget.objects.acreate(owner=user, name="a")
    toolset = SpecToolset(
        _registry(widgets=replace(_paged_spec(_StrictRestqlWidget), reads=(QueryParam("query"),))),
        tool_server_arguments={"widgets": ServerArguments(values={"query": "{name, bogus}"})},
    )

    with pytest.raises(DRFValidationError, match="`bogus` field is not found"):
        await _call(toolset, "widgets", user, {"query": "{id}"})


# --- a consumer scoping each run ----------------------------------------------
#
# Modelled on a production consumer that scopes a document-enrichment run to the
# document's project, or else its draw request. It subclassed the toolset to
# take every scope capture it had bound out of each tool's schema and out of
# each call's arguments, beside a ``UrlKwarg(default=<bound value>)`` that the
# model could still override. With the keyword, the project's part is one small
# function from its registry and its scope to the bindings; the toolset is not
# subclassed for it at all.

_SCOPE_NAMES = frozenset({"project_pk", "draw_request_pk"})


def scope_bindings(registry: SpecRegistry, scope: Mapping[str, str]) -> dict[str, ServerArguments]:
    """Each tool reading a scope name, bound to the run's scope, its other scope names withheld."""
    bindings: dict[str, ServerArguments] = {}
    for entry in registry.all():
        declared = {read.name for read in entry.spec.reads} & _SCOPE_NAMES
        if declared:
            values = {name: scope[name] for name in sorted(declared) if name in scope}
            bindings[entry.name] = ServerArguments(
                values=values, withheld=frozenset(declared - values.keys())
            )
    return bindings


@pytest.mark.django_db(transaction=True)
async def test_a_run_scoped_to_a_project_binds_it_and_withholds_the_sibling():
    user = await User.objects.acreate(username="u")
    registry = _registry(
        list_document_vendors=_scoped_spec(),
        list_vendors=_name_list_spec(any_vendors, permission_classes=[_RecordingPermission]),
    )
    toolset = SpecToolset(
        registry, tool_server_arguments=scope_bindings(registry, {"project_pk": "7"})
    )
    # The model reaches for the scope anyway -- a guessed project and a
    # placeholder for the draw request it was never shown.
    agent = Agent(
        tool_calling_model(
            "list_document_vendors", {"project_pk": "99", "draw_request_pk": "null"}
        ),
        deps_type=AgentDeps,
        toolsets=[toolset],
    )

    result = await agent.run("enrich", deps=AgentDeps(user=user))

    assert _rows(_tool_return(result).content) == [{"name": "project=7 draw=None"}]
    # Every reader of the call sees the one value: a permission class reading
    # the route included.
    assert _RecordingPermission.seen["kwargs"] == {"project_pk": "7"}
    schema = _input_schema_of(toolset, "list_document_vendors")
    assert not _SCOPE_NAMES & set(schema["properties"])
    # A tool reading no scope name has no binding.
    assert "list_vendors" not in scope_bindings(registry, {"project_pk": "7"})


# --- the capability -----------------------------------------------------------


@pytest.mark.django_db(transaction=True)
async def test_the_capability_forwards_the_bindings():
    user = await User.objects.acreate(username="u")
    capability = SpecCapability(
        _registry(vendors=_scoped_spec()), tool_server_arguments={"vendors": _SCOPED}
    )
    toolset = capability.get_toolset()

    result = await _call(toolset, "vendors", user, {"project_pk": "99"})

    assert _rows(result) == [{"name": "project=7 draw=None"}]
    assert "project_pk" not in _input_schema_of(toolset, "vendors")["properties"]


# --- the mount-level channel declarations are deprecated ----------------------


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("query_params", [QueryParam("fields")]),
        ("tool_query_params", {"vendors": [QueryParam("fields")]}),
        ("url_kwargs", [UrlKwarg("tenant_pk")]),
        ("tool_url_kwargs", {"vendors": [UrlKwarg("tenant_pk")]}),
    ],
)
@pytest.mark.parametrize("build", [SpecToolset, SpecCapability], ids=["toolset", "capability"])
def test_a_mount_level_channel_declaration_is_deprecated(build, knob, value):
    """Each names its replacement, and points at the line that passed it, through
    the capability as well as directly."""
    with pytest.warns(DeprecationWarning) as record:
        build({"vendors": _name_list_spec(any_vendors)}, **{knob: value})

    (warning,) = [w for w in record if issubclass(w.category, DeprecationWarning)]
    message = str(warning.message)
    assert f"{knob}=" in message
    assert "reads=" in message
    assert "tool_server_arguments" in message
    assert warning.filename == __file__


def test_no_deprecation_is_raised_for_a_mount_declaring_no_channel():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        SpecToolset(_registry(vendors=_scoped_spec()), tool_server_arguments={"vendors": _SCOPED})


def test_the_bindings_are_read_once_when_the_toolset_is_built():
    """A mapping changed after the toolset is built binds nothing new: the check
    ran on what was passed, and an unchecked binding would reach every call."""
    bindings = {"vendors": _SCOPED}
    toolset = SpecToolset(_registry(vendors=_scoped_spec()), tool_server_arguments=bindings)

    bindings["vendors"] = ServerArguments()

    assert "project_pk" not in _input_schema_of(toolset, "vendors")["properties"]


@pytest.mark.django_db(transaction=True)
async def test_a_malformed_value_for_a_bound_name_never_refuses_the_call():
    """The model's value is dropped before the shape check reads it, so an integer
    for a string it was never offered costs no retry, under ``REJECT`` too."""
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        {"vendors": _name_list_spec(vendors_of_project, permission_classes=[_RecordingPermission])},
        tool_server_arguments={"vendors": ServerArguments(values={"project_pk": "7"})},
    )

    result = await _call(toolset, "vendors", user, {"project_pk": 12345})

    assert _rows(result) == [{"name": "project=7"}]
