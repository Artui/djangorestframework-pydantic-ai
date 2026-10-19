"""``tool_server_arguments``: values the server supplies for a tool's arguments, and names it withholds.

Written against the public signatures only -- ``SpecToolset``, ``SpecCapability``,
``ServerArguments`` and a real ``Agent`` run -- so each test compiles against a
toolset without the keyword and fails there on what it asserts, not on an import.
"""

from __future__ import annotations

import _thread
import logging
import re
import warnings
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
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
    _dual_declared_spec,
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
    (record,) = [r for r in caplog.records if "dropped caller-supplied" in r.getMessage()]
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
    # Worded for both kinds of name: a withheld one is not a value the server
    # supplies, only one the model may not choose.
    assert (
        "dropped caller-supplied argument(s) draw_request_pk, project_pk, which "
        "tool_server_arguments keeps from the model"
    ) in caplog.text


@pytest.mark.django_db(transaction=True)
async def test_nothing_is_logged_when_the_model_sends_no_server_name(caplog):
    user = await User.objects.acreate(username="u")
    toolset = SpecToolset(
        _registry(vendors=_scoped_spec()), tool_server_arguments={"vendors": _SCOPED}
    )

    with caplog.at_level(logging.WARNING, logger="rest_framework_pydantic_ai"):
        await _call(toolset, "vendors", user, {})

    assert "dropped caller-supplied" not in caplog.text


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
# document's project, or else its draw request. It subclassed the toolset to keep
# its bound scope out of each tool's schema and each call's arguments, under one
# rule: a tool counts as scoped by the run only if the run binds one of its scope
# names; its optional siblings are then the server's as well, and a required one
# the run does not bind stays the model's to send. With the keyword, that rule is
# one small function from the registry and the run's scope to the bindings, and
# the toolset is not subclassed for it at all.
#
# The function is the registry page's own example, run as written, so the page
# cannot drift from what these tests hold.

_DOCS_PAGE = Path(__file__).resolve().parent.parent / "docs" / "spec-registry.md"


def _docs_recipe(registry: SpecRegistry, scope: Mapping[str, str]) -> dict[str, Any]:
    """Run the registry page's per-run scope example over ``registry`` and ``scope``.

    Returns the names the block defined: ``scope_bindings`` and the ``toolset``
    it builds.
    """
    blocks = re.findall(r"```python\n(.*?)```", _DOCS_PAGE.read_text(), flags=re.DOTALL)
    (block,) = [block for block in blocks if "def scope_bindings" in block]
    namespace: dict[str, Any] = {
        "ServerArguments": ServerArguments,
        "SpecToolset": SpecToolset,
        "registry": registry,
        "scope": scope,
    }
    exec(block, namespace)
    return namespace


@pytest.mark.django_db(transaction=True)
async def test_a_run_scoped_to_a_project_binds_it_and_withholds_the_sibling():
    user = await User.objects.acreate(username="u")
    registry = _registry(
        list_document_vendors=_scoped_spec(),
        list_vendors=_name_list_spec(any_vendors, permission_classes=[_RecordingPermission]),
    )
    recipe = _docs_recipe(registry, {"project_pk": "7"})
    toolset = recipe["toolset"]
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
    assert not {"project_pk", "draw_request_pk"} & set(schema["properties"])
    # A tool reading no scope name has no binding.
    assert "list_vendors" not in recipe["scope_bindings"](registry, {"project_pk": "7"})


@pytest.mark.django_db(transaction=True)
async def test_a_tool_reading_only_scope_names_the_run_does_not_bind_keeps_them():
    """A tool reading only ``draw_request_pk``, in a run scoped to a project, is
    not scoped by the run: it has no binding, so the model still sends its draw
    request and the selector receives it. Holds the recipe's skip of a tool with
    nothing bound; without it the name would be withheld, gone from the schema,
    and every call would reach the selector with no draw request."""
    user = await User.objects.acreate(username="u")
    registry = _registry(list_draw_vendors=_scoped_spec(reads=(UrlKwarg("draw_request_pk"),)))
    recipe = _docs_recipe(registry, {"project_pk": "7"})
    toolset = recipe["toolset"]

    result = await _call(toolset, "list_draw_vendors", user, {"draw_request_pk": "3"})

    assert _rows(result) == [{"name": "project=None draw=3"}]
    assert "draw_request_pk" in _input_schema_of(toolset, "list_draw_vendors")["properties"]
    assert recipe["scope_bindings"](registry, {"project_pk": "7"}) == {}


def vendors_of_document(*, document_pk: str, project_pk: str | None = None):
    """The vendors a document may be matched to, within its project if one is given."""
    return [{"name": f"project={project_pk} document={document_pk}"}]


@pytest.mark.django_db(transaction=True)
async def test_a_required_scope_name_the_run_does_not_bind_stays_the_model_s():
    """A scoped tool's sibling it cannot run without is not withheld: withheld, no
    call could supply it, which the toolset refuses when it is built. Holds the
    recipe's ``required`` check; the model is still asked for the document, and
    only the project is the server's."""
    user = await User.objects.acreate(username="u")
    registry = _registry(
        list_document_vendors=_scoped_spec(
            vendors_of_document,
            reads=(UrlKwarg("project_pk"), UrlKwarg("document_pk", required=True)),
        )
    )
    toolset = _docs_recipe(registry, {"project_pk": "7"})["toolset"]

    result = await _call(
        toolset, "list_document_vendors", user, {"project_pk": "99", "document_pk": "5"}
    )

    assert _rows(result) == [{"name": "project=7 document=5"}]
    schema = _input_schema_of(toolset, "list_document_vendors")
    assert "project_pk" not in schema["properties"]
    assert schema["required"] == ["document_pk"]


@pytest.mark.django_db(transaction=True)
async def test_a_bound_capture_reaches_the_extras_and_the_route_alike():
    """The quickstart's scope over a reflected ``**extras`` key: declared as the
    spec's ``UrlKwarg`` read and bound, the selector receives the server's value
    in its extras and a scoping provider finds it on ``view.kwargs``, while the
    model is offered neither. Unbound, the read is advertised once, and required."""
    seen: dict[str, Any] = {}

    def scope_provider(view: Any) -> dict[str, Any]:
        seen["view_kwargs"] = dict(view.kwargs)
        return {}

    user = await User.objects.acreate(username="u")
    await Widget.objects.acreate(name="cheap", price=5, owner=user)
    await Widget.objects.acreate(name="dear", price=15, owner=user)
    spec = replace(
        _dual_declared_spec(),
        kwargs=scope_provider,
        reads=(UrlKwarg("project_pk", type="integer", required=True),),
    )
    toolset = SpecToolset(
        _registry(list_widgets=spec),
        tool_server_arguments={"list_widgets": ServerArguments(values={"project_pk": 7})},
    )

    result = await _call(toolset, "list_widgets", user, {"project_pk": 99})

    assert [row["name"] for row in _rows(result)] == ["cheap"]
    assert seen["view_kwargs"] == {"project_pk": 7}
    assert "project_pk" not in _input_schema_of(toolset, "list_widgets").get("properties", {})
    # Declared and not bound, the name is advertised once, the read's
    # declaration over the reflected key, and required as the read says.
    unbound = _input_schema_of(SpecToolset(_registry(list_widgets=spec)), "list_widgets")
    assert unbound["properties"]["project_pk"]["type"] == "integer"
    assert unbound["required"] == ["project_pk"]


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
    assert message.startswith(f"{build.__name__}'s {knob}= is deprecated")
    assert "reads=" in message
    assert "tool_server_arguments" in message
    assert warning.filename == __file__


# Compiled into a namespace of the test's choosing, so the frames a warning is
# attributed through carry whatever module name the test needs them to.
_DEPRECATED_BUILD = """
def build():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        SpecToolset({"vendors": spec}, url_kwargs=[UrlKwarg("tenant_pk")])
    return [w for w in record if issubclass(w.category, DeprecationWarning)]


def run_as_thread(outcome, done):
    try:
        outcome["warnings"] = build()
    except BaseException as error:
        outcome["error"] = error
    finally:
        done.release()
"""


def _probe(module: str) -> dict[str, Any]:
    """``_DEPRECATED_BUILD`` compiled as code of ``module``, in a file named
    ``<module>``, so a warning attributed to one of its frames says so."""
    namespace: dict[str, Any] = {
        "__name__": module,
        "warnings": warnings,
        "SpecToolset": SpecToolset,
        "UrlKwarg": UrlKwarg,
        "spec": _name_list_spec(any_vendors),
    }
    exec(compile(_DEPRECATED_BUILD, f"<{module}>", "exec"), namespace)
    return namespace


def test_a_package_whose_name_only_begins_like_this_one_is_the_caller():
    """The frame walk skips this package's modules by dotted name, so a package
    named ``rest_framework_pydantic_ai_extras`` is outside it and is the line the
    warning names. Holds the trailing dot in ``_in_this_package``: a bare prefix
    match would skip this caller too and name the test instead."""
    (warning,) = _probe("rest_framework_pydantic_ai_extras")["build"]()

    assert warning.filename == "<rest_framework_pydantic_ai_extras>"


def test_a_walk_reaching_the_bottom_of_the_stack_inside_this_package_still_warns():
    """A function a thread is started on from C has no frame beneath it. When
    that frame is this package's, the walk stops on it and warns there rather
    than stepping off the stack. Holds the ``f_back`` guard in
    ``_warn_mount_channels``, without which the walk reads the globals of
    ``None`` and the toolset is never built."""
    outcome: dict[str, Any] = {}
    done = _thread.allocate_lock()
    done.acquire()

    _thread.start_new_thread(
        _probe("rest_framework_pydantic_ai.probe")["run_as_thread"], (outcome, done)
    )

    assert done.acquire(timeout=30)
    assert "error" not in outcome, outcome.get("error")
    (warning,) = outcome["warnings"]
    assert warning.filename == "<rest_framework_pydantic_ai.probe>"


def tasks_in_status(*, status: str = "open"):
    """The tasks in one status, the open ones when none is named."""
    return [{"name": status}]


def _status_registry() -> SpecRegistry:
    """A tool whose selector takes ``status``, which its spec also reads as a ``QueryParam``."""
    return _registry(
        tasks=_name_list_spec(
            tasks_in_status,
            permission_classes=[_RecordingPermission],
            reads=(QueryParam("status"),),
        )
    )


@pytest.mark.django_db(transaction=True)
async def test_a_mount_url_kwarg_over_a_spec_s_read_of_its_input_serves_as_both_routes_do():
    """A mount's ``UrlKwarg("status")`` over a spec's ``QueryParam("status")``
    read of a selector parameter declares where the tool's input comes from, so
    the registry builds and serves it as ``registry.specs()`` does: the value on
    ``view.kwargs``, and on the query string under the spec's read. Up to 0.35.x
    the registry refused it as one name on two channels."""
    user = await User.objects.acreate(username="u")
    registry = _status_registry()
    served = []
    for source in (registry, registry.specs()):
        with pytest.warns(DeprecationWarning):
            toolset = SpecToolset(source, url_kwargs=[UrlKwarg("status")])
        result = await _call(toolset, "tasks", user, {"status": "done"})
        served.append(
            (_rows(result), _RecordingPermission.seen, _input_schema_of(toolset, "tasks"))
        )

    assert served[0] == served[1]
    assert served[0][:2] == (
        [{"name": "done"}],
        {"kwargs": {"status": "done"}, "query_params": {"status": ["done"]}},
    )


@pytest.mark.parametrize(
    ("selector", "url_kwarg"),
    [
        (tasks_in_status, UrlKwarg("status", default="open")),
        (any_vendors, UrlKwarg("status")),
    ],
    ids=["default-fills-the-parameter", "not-an-input"],
)
def test_a_mount_url_kwarg_over_a_read_the_tool_does_not_take_is_refused_from_the_registry(
    selector, url_kwarg
):
    """Where the tool does not take the name as an input -- a ``UrlKwarg`` with a
    ``default`` fills the selector's ``status`` itself, and ``any_vendors`` has
    none -- the spec's read is a channel of its own, and the registry refuses it
    beside the ``UrlKwarg`` as one name on two channels. ``registry.specs()``
    reads no channel off the spec, so it builds. The first case holds
    ``_parameter_reads`` reading the tool's ``url_kwargs``: read without them,
    the parameter would count as the model's and the registry would build."""
    registry = _registry(
        tasks=_name_list_spec(
            selector, permission_classes=[_RecordingPermission], reads=(QueryParam("status"),)
        )
    )
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="both a QueryParam and a UrlKwarg on tool 'tasks'"),
    ):
        SpecToolset(registry, url_kwargs=[url_kwarg])
    with pytest.warns(DeprecationWarning):
        SpecToolset(registry.specs(), url_kwargs=[url_kwarg])


def test_no_deprecation_is_raised_for_a_mount_declaring_no_channel():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        SpecToolset(_registry(vendors=_scoped_spec()), tool_server_arguments={"vendors": _SCOPED})


@pytest.mark.django_db(transaction=True)
async def test_the_bindings_are_read_once_when_the_toolset_is_built():
    """A mapping changed after the toolset is built binds nothing new: the check
    ran on what was passed, and an unchecked binding would reach every call. So
    the call still serves the bound project, and the schema still omits it."""
    user = await User.objects.acreate(username="u")
    bindings = {"vendors": _SCOPED}
    toolset = SpecToolset(_registry(vendors=_scoped_spec()), tool_server_arguments=bindings)

    bindings["vendors"] = ServerArguments()
    result = await _call(toolset, "vendors", user, {"project_pk": "99"})

    assert _rows(result) == [{"name": "project=7 draw=None"}]
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
