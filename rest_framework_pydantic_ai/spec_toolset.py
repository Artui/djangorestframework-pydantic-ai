"""``SpecToolset`` — expose drf-services specs as a Pydantic-AI toolset.

A thin adapter that turns a ``name -> spec`` mapping into agent tools, executing
each call through drf-services' transport-neutral surface — ``dispatch_spec``
plus its off-HTTP helpers (``build_offline_context`` / ``enforce_permissions`` /
``spec_to_json_schema`` / ``render_spec_output``). No MCP server and no AG-UI
bridge is in the path: a plain ``pydantic_ai.Agent`` calls the specs in-process.

One call is ``_call_spec``, which mirrors a DRF view in order: pop the args
the transport owns (pagination, registered query params and URL kwargs), build
the off-HTTP context, enforce ``spec.permission_classes`` — ``dispatch_spec``
deliberately does not, so a naive adapter would skip authorization — then
dispatch and render. The failure-kind mapping onto the model loop lives in the
same function's arms, and is tabulated for callers in ``docs/quickstart.md``.

**Every failure this toolset decides is terminal is *raised*, never returned.**
Pydantic-AI marks an ordinary return ``outcome="success"``, so a refusal handed
back as a value is a failure only the model's reader can see: everything
downstream -- a transport streaming the result, a log, a client rendering the
call -- gets a successful call carrying prose. ``ToolFailed`` says the same
sentence to the model and marks the return ``outcome="failed"``, which is the
one field that makes the failure legible to anything that is not an LLM.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from types import MappingProxyType
from typing import Any, TypeGuard, cast

from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.db import connections
from django.http import HttpRequest
from pydantic_ai import ModelRetry, RunContext, ToolFailed
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_core import SchemaValidator, core_schema
from rest_framework.exceptions import PermissionDenied
from rest_framework.exceptions import ValidationError as DRFValidationError
from rest_framework.settings import api_settings
from rest_framework_services import (
    DEFAULT_JSON_SCHEMA_REGISTRY,
    DEFAULT_PAGE_SIZE,
    DEFAULT_POOL_SEEDS,
    UNSET,
    ActionUnavailable,
    AdditionalInputRequired,
    Affordance,
    ArgumentBinding,
    AudienceProjection,
    DispatchResult,
    FieldAudience,
    JsonSchemaRegistry,
    OfflineContract,
    OutputPage,
    PoolSeeds,
    SelectorKind,
    SelectorSpec,
    ServiceError,
    ServiceSpec,
    ServiceValidationError,
    SpecRegistry,
    UnknownArguments,
    audience_projection_for_spec,
    base_pool,
    build_offline_context,
    can_present_nothing,
    dispatch_spec,
    enforce_permissions,
    operation_affordances,
    output_to_json_schema,
    paginate_output,
    provider_keys,
    render_for_audience,
    server_owned_keys,
    spec_to_json_schema,
    unmet_operation_affordance,
)
from rest_framework_services.dispatch.unguarded_specs import unguarded_specs
from rest_framework_services.types.progress_reporter import ProgressReporter
from rest_framework_services.types.validate_channel_names import validate_channel_names

from rest_framework_pydantic_ai.types.agent_conventions import AgentConventions
from rest_framework_pydantic_ai.types.query_param import QueryParam
from rest_framework_pydantic_ai.types.url_kwarg import UrlKwarg

Spec = ServiceSpec[Any, Any, Any] | SelectorSpec[Any, Any]
# Widening the parameter beats a ``from_registry`` constructor, which would have
# to restate every keyword of the signature it forwards to (and then drift).
SpecSource = Mapping[str, Spec] | SpecRegistry
UserExtractor = Callable[[RunContext[Any]], Any]
ProgressExtractor = Callable[[RunContext[Any]], ProgressReporter | None]
HttpRequestExtractor = Callable[[RunContext[Any]], HttpRequest | None]

_ContextBuilder = Callable[..., Any]
_ExceptionTranslator = Callable[[BaseException], "ExceptionHandler | None"]
# The four result-shaping seams. ``Callable[..., X]`` rather than a written-out
# signature because each is bound through a forwarding lambda that injects
# ``ctx``, and a precise type would have to describe the *unbound* shape while
# the call site uses the bound one.
_PageShaper = Callable[..., "OutputPage"]
_OutputRenderer = Callable[..., Any]
_ExtrasBuilder = Callable[..., "dict[str, Any]"]
_ResultBounder = Callable[..., Any]

ExceptionHandler = Callable[[BaseException], Any]
"""Turns one exception into a tool result.

Three answers, and the choice is about what the model should do next. Raise
``ToolFailed`` for something it should report and stop on — that is what the
built-in arms do, and it marks the tool return ``outcome="failed"`` so a
transport can say so too. Raise ``ModelRetry`` to hand the call back for another
attempt. Or **return** a value, which becomes the tool's result verbatim and is
marked ``outcome="success"``: the escape hatch for an exception that is not
really a failure — a "nothing matched" a caller would rather express as an empty
payload than as a refusal.

Raising anything else aborts the run, which is the right answer for a genuine
bug.
"""

logger = logging.getLogger("rest_framework_pydantic_ai")
"""The package's one logger.

A dispatch behind a DRF view leaves an access-log line; the same spec called by
a model leaves nothing, and that includes a permission denial, which over HTTP
is a 403 in the log and here is an exception the run loop absorbs into a
message. Timings go to ``DEBUG``, denials to ``WARNING``; named for the package
so ``LOGGING`` can set the two independently.
"""


def _run_extra(ctx: RunContext[Any]) -> dict[str, Any]:
    """Correlation fields for one tool call's log lines.

    A dispatch behind a DRF view lands in an access log beside a request id; the
    same spec called by a model produced lines naming only the tool and the
    toolset. One chat turn does not need more than that -- there is one run --
    but the shape this package is otherwise undocumented for does: a worker
    fanning several runs out concurrently interleaved their lines with nothing
    to separate them by.

    Passed through ``extra=`` rather than formatted into the message so a
    structured handler can index the fields and a plain one stays readable. The
    four names are pydantic-ai's own, and none of them collides with a
    ``LogRecord`` attribute -- ``extra`` overwriting one of those raises.
    """
    return {
        "run_id": ctx.run_id,
        "conversation_id": ctx.conversation_id,
        "run_step": ctx.run_step,
        "tool_call_id": ctx.tool_call_id,
    }


def _usage_extra(ctx: RunContext[Any]) -> dict[str, Any]:
    """The run's usage so far, as of this tool call.

    **Cumulative for the run, not attributable to this call** -- a tool call
    spends no tokens itself. What it is good for is the shape a chatbox does not
    have: a long autonomous run where the interesting question is which tool call
    the budget was standing at when the run went wrong, and that is answerable
    only if the number is stamped on each line as it goes.

    Enforcing a budget is deliberately not done here. ``UsageLimits`` belongs to
    ``Agent.run``, which can stop the run; a toolset can only refuse the next
    tool call, which is the wrong instrument and a second place for the limit to
    live. ``ctx.usage_limits`` is readable from an
    [`enforce_result_bytes`][rest_framework_pydantic_ai.SpecToolset.enforce_result_bytes]
    override for a project that wants to taper its results as a run gets long.
    """
    usage = ctx.usage
    return {
        "run_input_tokens": usage.input_tokens,
        "run_output_tokens": usage.output_tokens,
        "run_requests": usage.requests,
        "run_tool_calls": usage.tool_calls,
    }


def _resolve_specs(specs: SpecSource) -> Mapping[str, Spec]:
    """Normalise a ``SpecSource`` to the plain mapping the internals expect."""
    return specs.specs() if isinstance(specs, SpecRegistry) else specs


def _resolve_contracts(specs: SpecSource) -> Mapping[str, OfflineContract]:
    """Each registry entry's ``OfflineContract``, by tool name.

    ``SpecRegistry.specs()`` flattens an entry to its spec, which is everything
    the dispatch internals need and drops the one thing this toolset cannot
    derive: what a caller with **no HTTP request** has to be told. Over HTTP the
    URLconf supplies the route captures and the query string supplies the
    read-shaping params; here nobody does, and the entry is where a project that
    also runs an MCP server has already said so.

    A bare mapping carries no entries and so no contracts. That toolset declares
    its channels in the constructor, as it always has.
    """
    if not isinstance(specs, SpecRegistry):
        return {}
    return {
        entry.name: entry.agent_contract
        for entry in specs.all()
        if entry.agent_contract is not None
    }


# The absent contract, so a lookup miss reads like an entry that declared
# nothing rather than needing a branch at every use.
_NO_CONTRACT = OfflineContract()


# List-selector pagination args own these names; a registered ``QueryParam`` or
# ``UrlKwarg`` may not shadow them. ``ordering`` stays reserved even for a spec
# whose ``filter_set`` owns it: a channel registered under that name pops the
# value at call time, so the FilterSet would never see it.
_RESERVED_PARAM_NAMES = frozenset({"page", "limit", "ordering"})

# Tool names are surfaced verbatim to the model provider, which constrains them
# to this shape (OpenAI / Anthropic function-name rules).
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# A no-op: the advertised parameter schemas are advisory (``spec_to_json_schema``
# output, not a Pydantic model) and the real validation is the spec's own input
# serializer at dispatch time.
_TOOL_ARGS_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())

# The ``page`` tool arg a list selector accepts on top of its filter fields;
# ``limit`` is ``_limit_param_schema``, because its wording depends on the
# toolset's ceiling. Ordering is not here: it belongs to the spec whenever the
# spec's own schema advertises it (see ``_spec_ordering_argument``).
_PAGE_PARAM_SCHEMA: dict[str, Any] = {
    "type": "integer",
    "minimum": 1,
    "description": "1-based page number.",
}
# ``page`` no longer says "requires `limit`", because it no longer does: every
# list result is a page, so an omitted ``limit`` is the default page size rather
# than "everything". The pair used to be advertised and then not honoured — the
# schema claimed pagination while the payload was a bare list — and the wording
# was the last place that claim was still qualified.


def _served_page_size(max_page_size: int | None) -> int:
    """The rows a list tool serves to a call that names no ``limit``.

    ``paginate_output``'s own rule, stated here because the schema and the
    instructions are written at construction, long before any call: an omitted
    ``limit`` is ``DEFAULT_PAGE_SIZE``, then clamped *down* to the ceiling and
    never raised to it. So ``max_page_size=3`` serves 3 and ``max_page_size=500``
    still serves 100. The number used to be formatted into a module-level string
    from ``DEFAULT_PAGE_SIZE`` alone, and a toolset with a lower ceiling told the
    model "Defaults to 100" beside ``maximum: 3``.

    It describes the default ``shape_page``. An override that serves a different
    page size to an omitted ``limit`` has to say so in its own descriptions.
    ``test_the_limit_description_states_the_default_a_call_is_served`` holds the
    lower bound against what a call is served, and
    ``test_a_ceiling_above_the_default_page_size_leaves_the_stated_default_alone``
    holds the ``min``: replacing it with the ceiling fails that one.
    """
    if max_page_size is None:
        return DEFAULT_PAGE_SIZE
    return min(DEFAULT_PAGE_SIZE, max_page_size)


def _limit_param_schema(max_page_size: int | None) -> dict[str, Any]:
    """A list tool's ``limit`` arg: the default it is served, and its ceiling.

    The ceiling is advertised as well as clamped: a schema with no ``maximum``
    invites a request for 100 000 rows, and telling the model is cheaper than
    correcting it.
    """
    schema: dict[str, Any] = {
        "type": "integer",
        "minimum": 1,
        "description": (
            f"Maximum number of items per page. Defaults to {_served_page_size(max_page_size)}; "
            "the result reports `totalPages` and `hasNext`."
        ),
    }
    if max_page_size is not None:
        schema["maximum"] = max_page_size
    return schema


@dataclass(frozen=True)
class _PageArgs:
    """A list selector's stripped pagination tool args.

    Pagination only. Sorting is never carried here: the spec's own schema is what
    advertises a sort argument, and whatever declared it — a ``filter_set``'s
    ``OrderingFilter``, or the selector callable itself — is what applies it.
    """

    page: int | None
    limit: int | None


# The binding every call is dispatched under, and so the one every input schema
# is read under: ``AUTO``, which binds a selector's arguments spread and a
# service's as one ``data`` bundle. One name for both ends, because
# ``spec_to_json_schema`` lists a serializer-less service's own parameters under
# a ``SPREAD_*`` binding and ``REJECT`` admits them only there, so a schema read
# under another binding than the call's advertises what the call refuses.
# ``test_a_serializer_less_service_advertises_what_dispatch_admits`` holds both
# ends: either one changed alone, it fails.
_ARGUMENT_BINDING = ArgumentBinding.AUTO


class SpecToolset(AbstractToolset[Any]):
    """Exposes drf-services specs as a Pydantic-AI toolset.

    Build it from a ``name -> spec`` mapping and hand it to an ``Agent``:

        toolset = SpecToolset({
            "list_orders": orders_selector_spec,   # SelectorSpec -> read-only tool
            "create_order": create_order_spec,     # ServiceSpec  -> mutation tool
        })
        agent = Agent(model, deps_type=AgentDeps, toolsets=[toolset])

    Each key becomes one tool: the description is the spec's selector/service
    docstring, the parameter schema comes from ``spec_to_json_schema`` (with a
    list selector's ``page`` / ``limit`` args merged in), the ``return_schema``
    comes from the same spec's projected output path, and the ``readOnlyHint``
    annotation is derived from the spec kind (selectors read, services mutate).

    **The parameter schema asks for what a call needs and nothing the toolset
    fills.** A selector parameter named after a pool seed, a defaulted
    ``UrlKwarg`` or a key a typed ``kwargs=`` provider returns is not
    advertised; every other selector parameter without a default is required,
    except one the provider *may* fill, which stays advertised but optional:
    any parameter, beside a provider whose return annotation does not name a
    ``TypedDict``, and a key the provider may decline with ``UNSET``, both as
    drf-services' ``provider_keys`` reads the annotation. A name a
    ``build_context`` override fills is invisible here, so it has to be
    declared (see ``build_context``). A single-item service also advertises its
    target lookup -- the parameters of its ``collection_selector_spec``, or
    failing that its ``instance_selector_spec``, such as a ``pk`` -- beside its
    input serializer's fields, the serializer's property winning a shared name,
    and less every key a callable in the call marks ``NotClientInput``, which
    dispatch keeps from the lookup. A
    call leaving out a required selector parameter is handed back as
    ``ModelRetry`` naming each one left out, before anything runs. A required
    serializer field is the serializer's to report, once the call runs, so a
    call missing both hears about the field on its next turn.

    **A list selector's result is a page, always.** It comes back as
    ``{"items": [...], "page": 1, "totalPages": N, "hasNext": bool}`` — never a
    bare list — with at most
    [`DEFAULT_PAGE_SIZE`][rest_framework_services.dispatch.paginate_output.DEFAULT_PAGE_SIZE]
    rows unless ``max_page_size`` lowers it. That is the input contract this
    toolset has always published being honoured: ``page`` and ``limit`` were
    advertised on every list tool and the payload was a bare slice, so a model
    asking for a collection got 50 of 51 rows with nothing saying more existed.
    ``hasNext`` is what it was missing.

    **Filtering needs no declaration here, and ordering belongs to the
    ``filter_set``.** A ``SelectorSpec.filter_set``'s fields are already
    generated into the tool's input schema and flow through as ordinary
    ``params``, which ``dispatch_spec`` hands the FilterSet as ``filter_data``.
    That includes ordering: a FilterSet carrying an ``OrderingFilter`` named
    ``ordering`` advertises the argument itself — drf-services reflects the
    filter's public choices into the schema as an enum — and the toolset keeps
    its hands off the value, which the FilterSet validates and applies through
    its own ``param_map``.

    **A ``many=True`` service takes its list as one named argument.** Its input
    validates as a JSON array and a model's arguments are always an object, so the
    list travels under the argument ``ServiceSpec.many_argument`` names --
    ``items`` unless the spec names another -- and nothing may be sent beside it
    but this toolset's own ``query_params`` / ``url_kwargs``. An invalid item comes
    back as a ``ModelRetry`` keyed under that argument and then by the item's
    index. The result is the rendered list, advertised as an array.

    For anything the keywords below do not cover,
    [`build_context`][rest_framework_pydantic_ai.SpecToolset.build_context] and
    [`translate_exception`][rest_framework_pydantic_ai.SpecToolset.translate_exception]
    are overridable and both receive the live ``RunContext``, which is how
    per-run typed deps reach dispatch.

    Args:
        specs: The ``name -> spec`` mapping to expose, one tool per key. A
            ``SpecRegistry`` is accepted anywhere the mapping is (drf-services
            0.27+) — the shared
            declaration site for a project exposing the same specs over more than
            one transport, so the agent reads the source MCP and the HTTP views
            read. **Prefer the registry over ``registry.specs()``**: an entry
            carries an
            [`OfflineContract`][rest_framework_services.types.offline_contract.OfflineContract]
            and the flattened mapping does not, so a contract's ``url_kwargs``,
            ``query_params`` and ``field_audiences`` are silently absent from a
            toolset built off the mapping. A filtered view is itself a registry, so several toolsets can
            be projected from one declaration with no shared state
            (``SpecToolset(registry.by_tag("read"), id="reads")``). Only the
            names come from it; everything else here is transport-specific, which
            the registry deliberately carries none of.
        id: Identifies this toolset, and keys a wrapping
            [`SpecCapability`][rest_framework_pydantic_ai.SpecCapability]'s
            ``defer_loading`` catalog entry — so give each projection of one
            registry its own.
        instructions: Replaces the conventions block
            [`get_instructions`][rest_framework_pydantic_ai.SpecToolset.get_instructions]
            derives from the specs. ``None`` derives it. What it does **not**
            replace is the per-step list of operations that are unavailable right
            now, which is appended after it whenever an operation condition leaves
            one out of the catalog, unless ``conventions`` sets
            ``unavailable_heading`` to ``None``, which drops that list with or
            without an override: the override replaces conventions, and which
            operations are offered on a given step is state no override written in
            advance could have described.

            Refused beside a ``conventions=`` that changes a line only the
            derived block says, because the override would ignore it without a
            word. The fields that land elsewhere -- ``unavailable_heading``,
            ``handle_field_description``, ``query_param_on_pages`` and
            ``missing_arguments`` -- apply with or without it.
        conventions: The wording the model is told, one line per field -- an
            [`AgentConventions`][rest_framework_pydantic_ai.AgentConventions].
            ``None`` is ``AgentConventions()``, today's wording. A field changes
            what its line says and never whether it appears, which the toolset
            still decides per step; ``None`` on a field drops its line. Unlike
            ``instructions=``, an override of one line leaves every other line
            derived, conditional and current.
        get_user: Reads the acting identity off the run context. Defaults to
            ``ctx.deps.user`` (the
            [`AgentDeps`][rest_framework_pydantic_ai.AgentDeps] shape).
        get_progress: Reads the run's ``ProgressReporter`` sink off the run
            context, for a spec that reports progress. Defaults to
            ``ctx.deps.progress``, tolerating a deps type without the field.
        unknown_arguments: What to do with a tool arg outside the spec's declared
            input set — a key the model invented. ``UnknownArguments.REJECT``
            surfaces it as a ``ModelRetry`` so the model self-corrects,
            ``IGNORE`` drops it, ``PASSTHROUGH`` forwards it to the callable.
            Specs whose declared set is open (a ``filter_set``, a ``**kwargs``
            selector) are unaffected.
        query_params: Read-shaping
            [`QueryParam`][rest_framework_services.types.query_param.QueryParam]
            args that seed
            ``request.query_params`` over the off-HTTP path — the extensible
            generalization of ``page`` / ``limit`` / ``ordering``. Each is
            advertised as a tool arg, then popped at call time and handed to
            ``build_offline_context(query_params=…)``, never to the spec as an
            input, so ``unknown_arguments`` never sees it. For whatever reads
            ``request.query_params`` **directly** — django-restql field
            selection, a serializer branching on the query string — with no
            toolset awareness of the library.
        tool_query_params: ``query_params`` for one tool only, keyed by tool
            name. A per-tool param overrides a toolset-wide one of the same name.

            Both are **this mount's** declarations, and both override the entry's
            own ``OfflineContract`` by name. Where a project runs more than one
            agent transport, the contract is the better home: the operation needs
            the identical params whichever transport calls it, and declaring them
            per mount is how two mounts come to disagree.
        url_kwargs:
            [`UrlKwarg`][rest_framework_services.types.url_kwarg.UrlKwarg] args
            — URL route captures (``parent_pk``) seeded into
            ``build_offline_context(kwargs=…)`` and spread by drf-services into
            the selector / target pools, authoritative over ``params``.
            Advertised then popped like ``query_params``. Use them for a
            URL-derived value **not** already in the tool schema: a scoping
            ``spec.kwargs`` provider reading ``view.kwargs`` (which ``params``
            alone cannot cover), or a closed-surface route capture. A selector
            reading the value from its ``**extras: Unpack[TypedDict]`` needs none
            — drf-services reflects the key and delivers it through ``params`` —
            though a key may be both reflected and registered, in which case the
            ``UrlKwarg`` schema wins and the authoritative ``kwargs=`` spread
            still reaches the selector. A name cannot be both a ``QueryParam``
            and a ``UrlKwarg`` on one tool: a value cannot route to two channels.
        tool_url_kwargs: ``url_kwargs`` for one tool only, same override rule,
            and the same preference for the entry's ``OfflineContract`` where one
            exists.
        host: The origin the synthesized request reports, so
            ``build_absolute_uri`` builds real absolute URLs — DRF's
            ``FileField`` and the ``Hyperlinked*`` fields call it for every value
            once a ``request`` is in the serializer context, which off the HTTP
            path it always is. Accepts ``"example.com"``, ``"example.com:8000"``
            or a full origin like ``"https://example.com"``, whose scheme decides
            whether links are https. Nothing is inferred: only the project knows
            its public origin, and a guess emits confidently wrong links that
            look valid. Unset, those fields produce relative URLs, which is what
            they fall back to on their own. Toolset-wide only — an origin is a
            property of the deployment, not of a tool.
        max_retries: Each tool's retry budget: how many times a
            ``ModelRetry`` is fed back to the model before the
            run aborts with ``UnexpectedModelBehavior``. The default matches
            pydantic-ai's own function-tool default.
        max_result_bytes: Ceiling on a rendered result, measured on the encoded
            payload because what is being protected is the model's context
            window. Over it the call **fails** with a model-readable
            ``ToolFailed`` — never truncates, because a partial payload looks
            complete.
        tool_max_result_bytes: ``max_result_bytes`` per tool. An explicit
            ``None`` opts that tool out; an absent key inherits the default.
        max_page_size: Clamps a list tool's ``limit`` *and* advertises the
            ceiling as JSON-Schema ``maximum``. It lowers the default page size
            only when it is below ``DEFAULT_PAGE_SIZE``: an omitted ``limit`` is
            served ``min(DEFAULT_PAGE_SIZE, max_page_size)``, and that is the
            default the ``limit`` description and the instructions state. A
            ceiling above it raises what a call may ask for, not what it gets by
            asking for nothing. Unset, a list tool still returns at most
            ``DEFAULT_PAGE_SIZE`` rows per page — the unbounded read is the one
            that hurts, and it is what a model produces by not thinking about
            pagination.
        thread_sensitive: Whether every dispatch shares one thread. ``True``
            (the default, and asgiref's) is what keeps Django's thread-local
            database connections coherent, and is why it is not flipped for you.

            **The cost is that concurrent tool calls serialise, process-wide.**
            asgiref's ``single_thread_executor`` is a *class* attribute, so it is
            one thread shared by every toolset instance and every concurrent run
            in the process -- not one per toolset. Pydantic-AI genuinely runs
            function tools in parallel within a segment, so four 0.30s calls
            under one model step take ~1.2s rather than ~0.3s. That is invisible
            in a chat turn calling one tool and severe in a fan-out.

            Set ``False`` only when you know the dispatched work is safe off the
            main thread -- typically because each call opens and closes its own
            connection, or you pass an ``executor`` you control.
        executor: A ``ThreadPoolExecutor`` to run dispatch on, instead of
            asgiref's shared single thread. Only consulted when
            ``thread_sensitive`` is ``False``, which is asgiref's own rule.
        dispatch_timeout: Seconds bounding one call, so the model gets an answer
            instead of a hang. It does not *stop* the work: the dispatch runs in
            a ``sync_to_async`` thread and asyncio cannot interrupt a thread
            parked in a database driver's socket read, so the query runs to
            completion regardless. Pair it with a database statement timeout.
        require_permissions: Refuse to construct a toolset containing a spec with
            no ``permission_classes``. Over HTTP that means *inherit*; here there
            is nothing to inherit from, so it means *ungated*. ``False``
            downgrades the refusal to an ``UnguardedSpecWarning`` while
            migrating.
        descriptions: Overrides ``spec.description`` per tool — the docstring an
            API developer reads is rarely the sentence a model needs. A tool left
            with no description anywhere gets an ``UndescribedToolWarning``.
        http_request: The ``HttpRequest`` the off-HTTP context is built from.
            **Incidental request data, never an auth channel:** it exists so a
            serializer or scoping provider reading ``request.META`` finds
            something plausible. The acting identity is the user, and passing an
            authenticated request authorizes nothing. Its **query string never
            reaches the spec**: every call replaces it with the declared
            ``query_params`` for that tool, empty declaration included, so the
            ambient endpoint's own query string cannot shape a result. Its
            headers and ``META`` are what it contributes; drf-services wraps a
            copy, so nothing a dispatch does is visible on it afterwards.
        get_http_request: ``http_request`` resolved per run from ``RunContext``,
            the way ``get_user`` is. Wins over a static ``http_request``.
        exception_map: Maps an exception type to a handler returning the tool's
            result (or raising ``ModelRetry``). Matched along
            the MRO, most specific first, and consulted **before** the built-in
            arms, so a project can override those too.
        json_schema_registry: Consumer rules for turning a custom serializer
            field, django-filter filter or Python type into a JSON Schema
            fragment — a
            [`JsonSchemaRegistry`][rest_framework_services.types.json_schema_registry.JsonSchemaRegistry],
            threaded into every schema this toolset generates (input *and*
            return). Without it a project's own field type reaches the model as
            ``{}`` — "any value" — which is the schema saying nothing at exactly
            the field the model is most likely to get wrong. Build one by
            extending the shared default:
            ``DEFAULT_JSON_SCHEMA_REGISTRY.extend(fields=[(MoneyField, {"type": "string"})])``.
        pool_seeds: The project's own always-available pool seeds -- a
            [`PoolSeeds`][rest_framework_services.types.pool_seeds.PoolSeeds]
            registry of names such as a tenant, a locale or a clock, which over
            HTTP hang off ``request`` and off it have no channel. Handed to every
            dispatch this toolset makes, so a service, a selector or an
            affordance's condition declaring one receives it as it would from
            ``dispatch_spec(pool_seeds=)`` called directly; and to the per-step
            check that leaves out an operation whose condition is unmet, so the
            catalog and the instructions are decided against the same seeds the
            call is refused with. A registered name is also reserved: the model
            cannot supply it, an argument of that name is neither spread nor
            refused as unknown, no tool's input schema advertises it, and a
            ``QueryParam`` / ``UrlKwarg`` declaring it is refused here with
            ``ImproperlyConfigured``. Toolset-wide, with no
            per-tool or per-call form: a seed is ambient to the deployment, and
            what varies per call belongs in its resolver, which declares
            ``user`` / ``request`` to receive them.

    Raises:
        ImproperlyConfigured: A spec has no ``permission_classes`` and
            ``require_permissions`` is set, a ``QueryParam`` / ``UrlKwarg`` on
            a ``many=True`` spec's tool -- declared here or on its entry's
            ``OfflineContract`` -- shares the name its list travels under, one
            is named after a registered pool seed, a selector takes a parameter
            that a list tool's ``page`` / ``limit`` or one of the tool's
            ``QueryParam`` declarations takes out of the call before it runs, or
            ``instructions=`` is given
            beside a ``conventions=`` that changes a line of the block it
            replaces.
        ValueError: A tool name is outside ``^[a-zA-Z0-9_-]{1,64}$``, a per-tool
            mapping names a tool this toolset does not expose, or one name is
            registered on both parameter channels.
    """

    def __init__(
        self,
        specs: SpecSource,
        *,
        id: str = "drf-specs",
        instructions: str | None = None,
        conventions: AgentConventions | None = None,
        get_user: UserExtractor | None = None,
        get_progress: ProgressExtractor | None = None,
        unknown_arguments: UnknownArguments = UnknownArguments.REJECT,
        query_params: Sequence[QueryParam] = (),
        tool_query_params: Mapping[str, Sequence[QueryParam]] | None = None,
        url_kwargs: Sequence[UrlKwarg] = (),
        tool_url_kwargs: Mapping[str, Sequence[UrlKwarg]] | None = None,
        host: str | None = None,
        max_retries: int = 1,
        max_result_bytes: int | None = None,
        tool_max_result_bytes: Mapping[str, int | None] | None = None,
        max_page_size: int | None = None,
        dispatch_timeout: float | None = None,
        require_permissions: bool = True,
        descriptions: Mapping[str, str] | None = None,
        http_request: HttpRequest | None = None,
        get_http_request: HttpRequestExtractor | None = None,
        exception_map: Mapping[type[BaseException], ExceptionHandler] | None = None,
        json_schema_registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
        pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
        thread_sensitive: bool = True,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        resolved = _resolve_specs(specs)
        contracts = _resolve_contracts(specs)
        _validate_tool_names(resolved)
        _validate_permissions(resolved, require=require_permissions)
        _validate_query_params(query_params, tool_query_params, resolved)
        _validate_url_kwargs(url_kwargs, tool_url_kwargs, resolved)
        self._descriptions: dict[str, str] = _validate_descriptions(resolved, descriptions)
        self._id = id
        self._instructions_override = instructions
        self._conventions = conventions if conventions is not None else _DEFAULT_CONVENTIONS
        _validate_conventions_beside_instructions(self._conventions, instructions)
        self._specs: dict[str, Spec] = dict(resolved)
        self._get_user: UserExtractor = get_user or _default_get_user
        self._get_progress: ProgressExtractor = get_progress or _default_get_progress
        # One knob at two lifetimes, so the static form resolves to the per-run
        # one rather than living beside it.
        self._get_http_request: HttpRequestExtractor = get_http_request or (
            (lambda ctx: http_request) if http_request is not None else _default_get_http_request
        )
        self._exception_map: dict[type[BaseException], ExceptionHandler] = dict(exception_map or {})
        self._unknown_arguments: UnknownArguments = unknown_arguments
        self._json_schema_registry = json_schema_registry
        self._pool_seeds = pool_seeds
        self._host = host
        self._max_retries = max_retries
        self._max_page_size = max_page_size
        self._dispatch_timeout = dispatch_timeout
        self._thread_sensitive = thread_sensitive
        self._executor = executor
        overrides: Mapping[str, int | None] = tool_max_result_bytes or {}
        for tool_name in overrides:
            if tool_name not in self._specs:
                raise ValueError(
                    f"tool_max_result_bytes references unknown tool {tool_name!r}; "
                    f"known tools: {sorted(self._specs)}."
                )
        self._tool_max_result_bytes: dict[str, int | None] = {
            # ``.get`` with the toolset default, so a stored ``None`` wins over
            # it: an explicit "no ceiling here" is not an absent key.
            name: overrides.get(name, max_result_bytes)
            for name in self._specs
        }
        # Effective declarations per tool, built once — they are static.
        #
        # The entry's contract is the base and this mount's constructor
        # declarations override it by name: the contract says what the operation
        # needs off HTTP, which every agent transport needs identically, while
        # the constructor is one mount's word about one deployment.
        self._tool_query_params: dict[str, tuple[QueryParam, ...]] = {
            name: _merge_query_params(
                _merge_query_params(contracts.get(name, _NO_CONTRACT).query_params, query_params),
                (tool_query_params or {}).get(name, ()),
            )
            for name in self._specs
        }
        self._tool_url_kwargs: dict[str, tuple[UrlKwarg, ...]] = {
            name: _merge_url_kwargs(
                _merge_url_kwargs(contracts.get(name, _NO_CONTRACT).url_kwargs, url_kwargs),
                (tool_url_kwargs or {}).get(name, ()),
            )
            for name in self._specs
        }
        _validate_no_param_channel_overlap(self._tool_query_params, self._tool_url_kwargs)
        # Checked on the **merged** tuples, not the raw declarations: a
        # toolset-wide and a per-tool entry of the same name are an intentional
        # override, which the shared check would read as a duplicate in the
        # pre-merge concatenation. Post-merge is also what reaches the schema.
        for name in self._specs:
            _validate_channel_declarations(
                name, self._tool_query_params[name], "query_params", seeds=pool_seeds
            )
            _validate_channel_declarations(
                name, self._tool_url_kwargs[name], "url_kwargs", seeds=pool_seeds
            )
        # Also post-merge, for the same reason and one more: a contract's
        # declarations exist only in the merged tuples.
        _validate_many_argument_channels(
            self._specs, self._tool_query_params, self._tool_url_kwargs
        )
        # The spec's side of the same collisions, so post-merge as well: a
        # contract's ``QueryParam`` takes an input's value as a mount's does.
        _validate_inputs_a_channel_takes(
            self._specs,
            self._tool_query_params,
            self._tool_url_kwargs,
            pool_seeds=pool_seeds,
            registry=json_schema_registry,
        )
        # Agent markings are pure in the serializer, like the schemas below, so
        # they are resolved once rather than paying a serializer instantiation
        # on every tool call.
        #
        # **Built before the tool definitions, which now read them.** A tool's
        # ``return_schema`` describes the payload the model actually receives,
        # and that payload is projected — so the schema has to be projected by
        # the same declaration, or it would advertise a field the render drops.
        self._projections: dict[str, AudienceProjection] = {
            name: audience_projection_for_spec(
                spec,
                overrides=contracts.get(name, _NO_CONTRACT).field_audiences,
                name=f"Tool {name!r}",
            )
            for name, spec in self._specs.items()
        }
        # Schemas derive purely from the specs (no DB), so the tool defs are built
        # once up front. ``ToolDefinition`` defaults to ``kind="function"`` — the
        # in-process kind the run loop routes into ``call_tool``.
        self._tool_defs: dict[str, ToolDefinition] = {
            name: _build_tool_def(
                name,
                spec,
                self._tool_query_params[name],
                self._tool_url_kwargs[name],
                self._descriptions.get(name),
                self._max_page_size,
                projection=self._projections[name],
                registry=json_schema_registry,
                pool_seeds=pool_seeds,
                conventions=self._conventions,
            )
            for name, spec in self._specs.items()
        }
        # The tools whose availability has to be *asked* each step: those with an
        # affordance answered without a row. Static, like everything above, so
        # the question "does anything here need asking?" costs a step nothing,
        # and a toolset where the answer is no never builds a pool or hops.
        #
        # drf-services' own selection rather than a restatement of it: which
        # conditions are callables and which are row expressions is one line it
        # draws in one place, and a second copy here could only drift from it.
        self._conditioned: tuple[str, ...] = tuple(
            name for name, spec in self._specs.items() if operation_affordances(spec)
        )
        # Per instance, so the cache dies with the toolset rather than a
        # module-level cache keeping every toolset it has seen alive. Keyed by
        # which tools are left out and nothing else: the block is a pure
        # function of that, so an entry holds no user's answer and serves any
        # run that arrives at the same combination. Bounded because the key
        # space is every subset of ``_conditioned``, however few of them a real
        # deployment visits.
        self._instructions_without: Callable[[frozenset[str]], str | None] = lru_cache(
            maxsize=_INSTRUCTIONS_MEMO_SIZE
        )(self._derive_instructions_without)

    @property
    def id(self) -> str | None:
        return self._id

    @property
    def specs(self) -> Mapping[str, Spec]:
        """The resolved ``name -> spec`` mapping this toolset exposes.

        The synchronous answer to "what tools are these?", for a caller composing
        this toolset at configuration time — a name-dedup pass, a tool catalog —
        with no run in sight, since ``get_tools`` is ``async`` and needs a
        ``RunContext``. Read-only (a ``MappingProxyType``), so enumerating it
        cannot add a tool that skipped the constructor's permission and
        description checks.
        """
        return MappingProxyType(self._specs)

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """The tool catalog this run is offered, one entry per listed spec.

        **The catalog is not permission-filtered, by design.** Every tool is
        advertised to every run and ``permission_classes`` gate the *call*: a
        denied tool is one the model can see and cannot use. Filtering by
        default would be worse in three ways — a permission whose answer depends
        on the arguments has none to read at listing time and would deny a tool
        the caller can in fact invoke; ``get_tools`` runs once per model step, so
        a DB-backed check becomes a query per spec per step; and a tool the model
        never sees is one it cannot ask about, which is how a run ends in a
        guess instead of a denial. Nothing row-level is exposed either way — a
        listing carries a name, a description and an input schema.

        **An operation whose condition is unmet right now is left out, and that is
        not the same decision.** A spec's ``Affordance`` answered without a row --
        a callable ``when``, "the books are open" -- is asked each step, and a
        tool it fails is omitted until it holds again. Each of the three
        arguments above has an answer here. Such a condition reads only the
        seeds -- the user, the request, a registered seed -- and never an
        argument, so it cannot hide a tool the caller could in fact invoke:
        whatever the model would send, the call would be refused. Only the specs
        declaring one are asked, so a toolset that declares none pays nothing,
        not even a thread hop. And the model can still ask about what it cannot
        see, because
        [`get_instructions`][rest_framework_pydantic_ai.SpecToolset.get_instructions]
        names every tool left out this way together with its reason, which is
        what it needs to tell the user why rather than guess.

        A condition on the row (an ORM expression) is never asked here: there is
        no row at listing time, and it goes on being answered per object at the
        call. A condition that raises aborts the step, as it would the call --
        one that cannot be answered is not a ``no``.

        **The omission is applied beside ``is_tool_listed``, not inside it**, so
        an override of that seam -- which is free to return ``True`` -- cannot
        switch it off by not calling ``super()``. Neither is an authorization
        decision: the call enforces every affordance whatever this listed, and a
        model calling a name left out gets pydantic-ai's unknown-tool retry.

        This and ``get_instructions`` each ask the conditions for themselves,
        once per step apiece, rather than sharing one answer. The run context
        carries no key that could safely scope a shared answer to one run and one
        step: ``run_id`` may be supplied by the caller and is reused when a
        failed run is retried under it, and a context built by hand has none.
        A memo keyed on it could hand one run's -- one user's -- answer to
        another, where asking twice costs a second evaluation of conditions that
        read only seeds. The price is that a condition flipping between the two
        reads can leave one step's catalog and its instructions disagreeing
        about one tool; the call's own enforcement is authoritative either way.

        Override [`is_tool_listed`][rest_framework_pydantic_ai.SpecToolset.is_tool_listed]
        when a deployment does want a narrower catalog.
        """
        unavailable = await self._unavailable_operations(ctx)
        return {
            name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self._max_retries,
                args_validator=_TOOL_ARGS_VALIDATOR,
            )
            for name, tool_def in self._tool_defs.items()
            if name not in unavailable and await self.is_tool_listed(name, ctx)
        }

    async def is_tool_listed(self, name: str, ctx: RunContext[Any]) -> bool:
        """Whether ``name`` belongs in this run's catalog. ``True`` by default.

        The seam for a deployment that wants a per-run catalog — hiding a
        staff-only tool from a non-staff run, or scoping the catalog to a
        tenant read off ``ctx.deps``. Hiding a tool is a *disclosure* decision,
        never an authorization one: the call is gated by
        ``spec.permission_classes`` whatever this returns, so an override that
        wrongly returns ``True`` grants nothing.

        Not consulted for a tool an unmet operation condition already left out
        of this step's catalog, and returning ``True`` does not put one back:
        that omission is applied by ``get_tools`` itself, so an override need
        not call ``super()`` to keep it.

        ``async`` because ``get_tools`` is, and it is called once per tool per
        model step. An override that queries the database must wrap that work in
        ``asgiref.sync.sync_to_async`` — Django refuses ORM access on the event
        loop, exactly as ``call_tool`` has to for dispatch.
        """
        del name, ctx  # unused by the default; present so an override has both
        return True

    async def get_instructions(self, ctx: RunContext[Any]) -> str | None:
        """Teach the model this toolset's conventions.

        The per-tool descriptions and parameter schemas say what each tool *is*,
        but not how the family behaves: that list tools accept ``page`` /
        ``limit`` / ``ordering``, that a business failure comes back as a readable
        failed result (a final answer, not a reason to retry) while a
        bad argument comes back as a retry request, and that a permission error
        is final. Pydantic-AI appends the block to the system prompt each turn,
        for a toolset attached directly *or* wrapped by a capability.

        **Per step, when an operation condition leaves a tool out.** The block is
        then derived from the tools this step offers, so no line advises about
        one it does not, and it ends by naming each tool left out with its
        ``reason``: the model sees neither the tool nor, otherwise, any sign it
        exists, and a user asking for it would get a guess where a sentence was
        available. The conditions are asked here and in ``get_tools``
        separately -- see that method for why, and for what it costs. With an
        ``instructions`` override, the override stands in for the derived block
        and the unavailable tools are still appended after it, because an
        override replaces the conventions and cannot have described which
        operations a given step would lack. ``unavailable_heading=None`` in the
        ``conventions`` drops that list either way, override or not.

        A toolset declaring no such condition is untouched by any of this: it
        asks nothing, and returns the same string every step.

        Each line is worded by the toolset's ``conventions``, which change what a
        line says and never whether it appears.

        Returns:
            The ``instructions`` override when one was given, else a block
            derived from the specs — each line conditional on something in this
            toolset being able to act on it, so the prompt carries no advice that
            cannot fire — followed in either case by the operations unavailable
            this step, when there are any and ``unavailable_heading`` is not
            ``None``. ``None`` when ``conventions`` dropped
            every line that would have been said and nothing is unavailable.
        """
        unavailable = await self._unavailable_operations(ctx)
        block: str | None
        if self._instructions_override is not None:
            block = self._instructions_override
        elif unavailable:
            block = self._instructions_without(frozenset(unavailable))
        else:
            block = self._derived_instructions
        heading = self._conventions.unavailable_heading
        # One arc for two reasons to say nothing more. ``not unavailable`` is held
        # by every toolset declaring no condition; ``heading is None`` by
        # ``test_none_drops_a_block_line_and_nothing_else[unavailable_heading]``,
        # where a tool *is* left out.
        if not unavailable or heading is None:
            return block
        listed = _unavailable_instruction(unavailable, heading)
        # A block with every line dropped is ``None``, and the list then stands
        # alone (``test_with_every_line_dropped_the_unavailable_list_stands_alone``).
        return listed if block is None else f"{block}\n{listed}"

    @cached_property
    def _derived_instructions(self) -> str | None:
        """The conventions block, built once and kept.

        A pure function of six attributes ``__init__`` assigns and nothing
        reassigns -- the specs, the merged query-param declarations, the
        projections, the schema registry, the page ceiling and the conventions,
        which are frozen -- so recomputing
        it is recomputing the same string. Pydantic-AI asks for instructions on **every model
        step**, and the derivation walks each list spec through
        ``spec_to_json_schema`` to find what that spec calls its sort.

        This is tidiness, not a fix: the measured cost is tens of microseconds
        per spec, against model round trips measured in seconds. What earns the
        memo is that a step should not pay for an answer that was settled at
        construction.

        ``cached_property`` rather than computing it in ``__init__`` so a toolset
        given an ``instructions=`` override never derives the block it replaces.
        The consequence is the ordinary one for a memo: a subclass that mutates
        ``_specs`` after the first ``get_instructions`` is describing tools the
        block will not mention. Nothing in this package mutates it, and the
        public ``specs`` property is read-only precisely so nothing outside can.
        """
        return _derive_instructions(
            self._specs,
            self._tool_query_params,
            self._projections,
            registry=self._json_schema_registry,
            page_size=_served_page_size(self._max_page_size),
            pool_seeds=self._pool_seeds,
            conventions=self._conventions,
        )

    def _derive_instructions_without(self, omitted: frozenset[str]) -> str | None:
        """The conventions block for the tools a step offers, ``omitted`` aside.

        Reached through ``_instructions_without``, the per-instance memo
        ``__init__`` wraps it in, and only when some tool *is* omitted -- the
        full set is ``_derived_instructions``, cached for the toolset's life.
        The lines naming what was omitted are not part of it: they carry each
        tool's ``reason``, and rendering them per step keeps this memo a
        function of which tools are offered and nothing else.
        """
        offered = {name: spec for name, spec in self._specs.items() if name not in omitted}
        return _derive_instructions(
            offered,
            {name: self._tool_query_params[name] for name in offered},
            {name: self._projections[name] for name in offered},
            registry=self._json_schema_registry,
            page_size=_served_page_size(self._max_page_size),
            pool_seeds=self._pool_seeds,
            conventions=self._conventions,
        )

    async def _unavailable_operations(self, ctx: RunContext[Any]) -> dict[str, Affordance]:
        """Each tool an operation condition refuses right now, with that condition.

        Returns before building anything when no spec declares such a
        condition, which is the ordinary toolset: no pool, no request, no thread
        hop. Otherwise one hop answers every conditioned spec, rather than one
        hop apiece -- ``aunmet_operation_affordance`` would take one per spec,
        and see ``_answer_operation_conditions`` for the reason this cannot use
        it anyway.

        The user is read here, on the loop, exactly where ``call_tool`` reads
        it, so an extractor that is only safe off the database is treated alike.
        """
        if not self._conditioned:
            return {}
        user = self._get_user(ctx)
        return await sync_to_async(
            self._answer_operation_conditions,
            thread_sensitive=self._thread_sensitive,
            executor=self._executor,
        )(user, ctx)

    def _answer_operation_conditions(
        self, user: Any, ctx: RunContext[Any]
    ) -> dict[str, Affordance]:
        """Ask every conditioned spec, once, against the pool its call would see.

        **Run in the dispatch thread and released like a dispatch.** A condition
        is user code and may query, so it cannot run on the event loop; and a
        query here opens a connection on the thread it lands on, which off HTTP
        nothing closes -- the leak ``_call_spec_releasing_connections`` exists
        for. drf-services' ``aunmet_operation_affordance`` hops through
        ``sync_to_async`` on its own and leaves that connection open, and the
        leak it leaves would not stay its own: a connection already open on the
        shared thread is in every later dispatch's ``held_before``, so each of
        them would then leave it alone too. It also ignores this toolset's
        ``thread_sensitive`` and ``executor``, which a condition should honour
        as the call it stands in front of does. So the hop is this toolset's and
        the evaluation inside it is drf-services' synchronous
        ``unmet_operation_affordance``.

        **The pool is built the way the call's is.** The request comes from
        [`build_context`][rest_framework_pydantic_ai.SpecToolset.build_context],
        the seam the call builds its request through, so a condition reading
        ``request`` sees the same kind of object -- a DRF ``Request`` around the
        configured ``http_request`` or a synthetic one -- and whatever an
        override puts on it. It is built with no arguments and no action,
        because a listing has neither: the query string is empty, which is also
        what a tool declaring no ``QueryParam`` dispatches with. The toolset's
        ``pool_seeds`` are handed over twice, as drf-services asks: resolved into
        the pool by ``base_pool(seeds=)``, and admitted into what the condition
        is shown by ``unmet_operation_affordance(reserved=)``, which hands a
        condition only the reserved names. Drop either and a condition reading a
        registered seed fails to bind here while the call, given the same seeds
        by ``dispatch_spec(pool_seeds=)``, answers it -- both halves are held by
        ``test_a_condition_reading_a_registered_seed_is_asked_with_it``. So the
        condition sees the names ``dispatch_spec`` would give it.

        One request and one pool serve every spec asked, because a condition
        reads only seeds and the seeds do not depend on which operation is
        being asked about.
        """
        with _releasing_connections_opened_here():
            # ``query_params={}`` rather than ``None`` for the reason the call
            # path gives: ``None`` would leave a configured ``http_request``'s
            # own query string live on the request the condition reads.
            context = self.build_context(user, {}, ctx=ctx, query_params={}, host=self._host)
            pool = base_pool(user=user, request=context.request, seeds=self._pool_seeds)
            unavailable: dict[str, Affordance] = {}
            for name in self._conditioned:
                unmet = unmet_operation_affordance(
                    self._specs[name], pool, reserved=self._pool_seeds.reserved
                )
                if unmet is not None:
                    unavailable[name] = unmet
            return unavailable

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        spec = self._specs[name]
        user = self._get_user(ctx)
        started: float = time.perf_counter()
        try:
            # The whole pipeline touches the ORM, which Django forbids on the
            # async event loop — run it in a thread. ``dict(tool_args)`` is a
            # private copy, so popping the transport's args never mutates the
            # caller's dict.
            result = await _with_deadline(
                sync_to_async(
                    self._call_spec_releasing_connections,
                    thread_sensitive=self._thread_sensitive,
                    executor=self._executor,
                )(
                    spec,
                    user,
                    dict(tool_args),
                    ctx=ctx,
                    # The tool name is this call's view action, the same
                    # identity the MCP transport reports for the same spec.
                    action=name,
                    unknown_arguments=self._unknown_arguments,
                    query_params=self._tool_query_params[name],
                    url_kwargs=self._tool_url_kwargs[name],
                    max_page_size=self._max_page_size,
                    max_result_bytes=self._tool_max_result_bytes[name],
                    projection=self._projections[name],
                    label=name,
                    progress=self._get_progress(ctx),
                    host=self._host,
                    # The dispatch path asks the same question the schema
                    # builder did — *what does this spec call its sort?* — so
                    # it has to ask it against the same registry, or the two
                    # could answer differently for one spec.
                    json_schema_registry=self._json_schema_registry,
                    pool_seeds=self._pool_seeds,
                ),
                self._dispatch_timeout,
                label=name,
                extra=_run_extra(ctx),
            )
        except PermissionDenied:
            # The one failure with no other trace: a ``ModelRetry`` reaches the
            # model and a ``ToolFailed`` reaches the answer as a failed result,
            # but a denial aborts the run and is absorbed by whatever drives it.
            # Logged at the boundary, then re-raised untouched.
            logger.warning(
                "Permission denied calling tool %r on toolset %r",
                name,
                self._id,
                extra=_run_extra(ctx),
            )
            raise
        logger.debug(
            "Tool %r on toolset %r took %.1f ms",
            name,
            self._id,
            (time.perf_counter() - started) * 1000,
            extra=_run_extra(ctx) | _usage_extra(ctx),
        )
        return result

    # The front half of a call: argument intake through dispatch. Deliberately
    # not offered beside these: a generic "set arbitrary attributes on the
    # synthetic request" parameter, which would make ambient-state-on-the-request
    # the default posture for everyone. An override keeps the honest path the
    # easy one -- which is why the back half below is four more overrides rather
    # than four more constructor knobs.

    def build_context(
        self,
        user: Any,
        params: Mapping[str, Any],
        *,
        ctx: RunContext[Any],
        action: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
        query_params: Mapping[str, Any] | None = None,
        host: str | None = None,
    ) -> Any:
        """Build the off-HTTP context one call dispatches under.

        Override to vary the synthetic request per run. The default forwards to
        drf-services' ``build_offline_context``, resolving ``http_request``
        through the configured extractor.

        **An ``http_request`` here is incidental request data — never an auth
        channel.** The acting identity is ``user``, resolved from ``ctx.deps``,
        and nothing downstream re-derives it from the request; supplying an
        authenticated one authorizes nothing and would put a second, invisible
        identity in the call.

        ``action`` is the tool name, landing on the synthetic view as
        ``view.action`` — one of the three attributes (``request`` /
        ``action`` / ``kwargs``) drf-services documents a permission class as
        being able to read off HTTP, and left unset it reads as ``None`` for
        every spec alike. Rewrite it in an override (it arrives in ``**kwargs``
        in the forwarding form) when a permission class branches on the viewset
        action names it knows.

        **A name an override fills has to be declared.** The tool schema is
        built from what the toolset can read, and an override is code: a
        selector parameter it fills through ``kwargs`` (which become
        ``view.kwargs``) is, to the schema, a parameter without a default like
        any other, so it is advertised as required, and a call leaving it out is
        handed back as ``ModelRetry`` before the override runs. Mark the
        selector parameter with drf-services' ``NotClientInput``
        (``project_pk: Annotated[int, NotClientInput]``): the name is left out
        of the schema, so the model is never asked for it, and the override
        fills it. Under the default ``unknown_arguments`` a call that sends it
        anyway is handed back as an unexpected argument where the selector's
        input set is closed (no ``filter_set``, no ``**kwargs``); otherwise, and
        under the other policies, drf-services drops the model's value before
        the selector reads its arguments, so the override's value is the one it
        receives
        (``test_a_value_the_model_sends_for_a_marked_name_never_reaches_the_selector``).
        Where the value can be
        resolved from what a seed resolver receives, register it in
        ``pool_seeds=`` and resolve it there instead, since dispatch fills a
        seed from its resolver and drops a route capture of the same name.

        A ``UrlKwarg`` with a ``default`` also keeps a call that leaves the name
        out from being refused, but it is not a way to keep the name from the
        model: it stays advertised, as an optional argument the model can see
        and send, and the override's value replaces whatever the model sent,
        with nothing said to the model. A model asking for project 5 is served
        project 42's rows.
        ``test_a_name_a_build_context_override_fills_is_kept_from_the_model``
        holds the marker,
        ``test_a_name_a_build_context_override_fills_is_declared_to_the_toolset``
        the seed and the defaulted ``UrlKwarg``, and
        ``test_a_name_only_a_build_context_override_fills_is_asked_of_the_model``
        the undeclared case.

        **Also called when the catalog is listed**, once per ``get_tools`` and
        once per ``get_instructions``, whenever some spec declares an
        ``Affordance`` answered without a row: that condition is asked against
        this context's ``request``, so it reads the same object at listing time
        as at the call. That call carries no arguments, no ``action`` and an
        empty ``query_params``, since a listing has none of them, and runs in
        the dispatch thread as the call's does. An override that reads
        ``action`` to decide something should expect ``None`` there.
        """
        return build_offline_context(
            user,
            params,
            http_request=self._get_http_request(ctx),
            action=action,
            kwargs=kwargs,
            query_params=query_params,
            host=host,
        )

    def translate_exception(
        self, exc: BaseException, *, ctx: RunContext[Any]
    ) -> ExceptionHandler | None:
        """Return a handler for ``exc``, or ``None`` to leave it to the defaults.

        The default consults ``exception_map`` by walking the exception's MRO,
        so a handler registered for a base class catches its subclasses and the
        **most specific** registration wins. Override for a decision the map
        cannot express — one that has to read the run's deps.

        This runs ahead of every built-in arm, so it is also where a program reads
        a refusal's code: an ``ActionUnavailable`` arrives here with ``.code``
        intact, before the default renders it into the sentence
        ``<reason> (code: <code>)``. A handler returned for it replaces that
        sentence along with everything else the default would have done.
        """
        del ctx  # unused by the default; present so an override has it
        for klass in type(exc).__mro__:
            handler = self._exception_map.get(cast(type[BaseException], klass))
            if handler is not None:
                return handler
        return None

    # The back half of a call: everything between the dispatch returning and the
    # tool result going back to the model. These were module-level privates
    # reached only from a module-level function, so a project wanting to change
    # any one of them had to replace ``call_tool`` wholesale or nothing at all --
    # the same gap 0.14.0 closed on the front half.

    def shape_page(
        self,
        rows: Any,
        *,
        ctx: RunContext[Any],
        page: int | None,
        limit: int | None,
        max_page_size: int | None,
    ) -> OutputPage:
        """Slice a list selector's rows into the page this call serves.

        Called only for list selectors -- the specs that advertise pagination
        arguments and return something sliceable. The default is drf-services'
        shared ``paginate_output`` (which counts the rows and clamps both bounds)
        followed by forcing evaluation, so nothing downstream holds a lazy
        queryset.

        **No sorting happens here, and an override should not add any.** Ordering
        belongs to whatever the spec declared it on, which has already applied it
        to the queryset by the time this runs; an ``order_by`` here would replace
        that sort rather than compose with it.
        """
        del ctx  # unused by the default; present so an override has it
        return _shape_list(rows, page=page, limit=limit, max_page_size=max_page_size)

    def render_output(
        self,
        spec: Spec,
        value: Any,
        *,
        ctx: RunContext[Any],
        projection: AudienceProjection | None,
        many: bool,
        request: Any,
        view: Any,
        extras: dict[str, Any],
    ) -> Any:
        """Turn a dispatch result into the payload the model reads.

        The default is drf-services' ``render_for_audience``: ``render_spec_output``
        plus the serializer's audience markings, using the projection this toolset
        resolved once at registration.

        **This is the opt-out of projecting**, and the only one. Passing
        ``projection=None`` is not it -- ``render_for_audience`` reads ``None`` as
        "derive one from the spec", so the payload is projected anyway and a
        serializer is instantiated per call to decide how. An override that wants
        the unprojected payload calls ``render_spec_output`` directly.

        Two cases want that. A **chaining pipeline** feeding one spec's output into
        the next needs the handles the next step reads by, and drf-services says so
        in ``render_for_audience``'s own docstring: project them away and the next
        step has nothing to key on. And a serializer with a single ``ChoiceField``
        whose display differs from its value is projected **without any marking
        being declared** -- ``choice_labels`` is derived from the field itself --
        so "we marked nothing, so nothing is projected" is not true, and an
        override is how a project that means it says so.
        """
        del ctx  # unused by the default; present so an override has it
        return _render_output(
            spec,
            value,
            projection=projection,
            many=many,
            request=request,
            view=view,
            extras=extras,
        )

    def output_extras(
        self,
        spec: Spec,
        value: Any,
        *,
        ctx: RunContext[Any],
        many: bool,
        dispatch_result: DispatchResult | None = None,
    ) -> dict[str, Any]:
        """The resolved-data pool a spec's output-context provider may read.

        Keyed the same way the HTTP path keys it, deliberately: ``result`` for a
        service, ``instance`` for a selector, ``page`` for a list. A provider
        written against one transport therefore reads the same names under the
        other.

        Note what the **default pool** leaves out, because it is the question this
        seam attracts: drf-services'
        [`DispatchResult.service_result`][rest_framework_services.types.dispatch_result.DispatchResult]
        -- the flags carrier an upsert's ``created`` rides on. The HTTP path does
        not put it in this pool either; it feeds a callable ``success_status`` and
        a ``response_finalizer``, both of which are status-code machinery a
        toolset has no wire for, and applying either here would be inventing one.
        A spec whose *model-visible* outcome depends on such a flag should put the
        flag in its output serializer, where both transports can see it.

        **``dispatch_result`` is the escape hatch for a project that disagrees**,
        and it carries the whole result rather than the one field: ``instance``
        (the pre-mutation target, resolved once, so an override reading it cannot
        get a different answer by resolving again) and ``data`` (the validated
        input) were dropped on the same floor. The loss it repairs is conditional
        -- with no ``output_selector_spec`` the service's return *is* ``value``,
        and only a re-fetch replacing ``value`` puts the flags out of reach -- but
        the upsert that wants ``created`` is exactly the spec that re-fetches.

        Args:
            dispatch_result: The dispatch's full
                [`DispatchResult`][rest_framework_services.types.dispatch_result.DispatchResult].
                Keyword-only, and **always supplied by this toolset** -- an
                override that does not accept it raises ``TypeError`` on the next
                tool call. That break is deliberate: the parameter is the only
                honest signature for a seam that is handed the carrier, and a
                failure the author sees on the first call is worth more than a
                gate that quietly stops receiving what it asked for. Declare
                ``dispatch_result``, or a ``**kwargs``, to take it. The default is
                ``None`` only so the method stays callable by something other than
                this toolset -- a test asserting the default pool, say.
        """
        del ctx, dispatch_result  # unused by the default; present so an override has them
        return _output_extras(spec, value, many=many)

    def enforce_result_bytes(
        self, payload: Any, *, ctx: RunContext[Any], max_bytes: int | None, label: str
    ) -> Any:
        """Return ``payload``, or raise a model-readable refusal when it is over budget.

        Measured on the envelope, because the envelope is what is sent. Override to
        bound on something other than serialized length -- a row count, a per-run
        running total read off ``ctx.usage``, a taper against ``ctx.usage_limits``
        as a run gets long -- or to shape the refusal differently.

        **The default raises ``ToolFailed``**; an override is free to return a
        value instead, and that value becomes the tool's result marked
        ``outcome="success"``. Delegating to ``super()`` and inspecting what comes
        back is the one shape that changed -- the refusal used to be a returned
        ``{"error": …}`` and is now an exception.

        ``ctx`` is not merely passed through to an override here: the default
        reads this run's correlation fields off it and stamps them on the
        ``WARNING`` a fired bound emits, which is the one line saying why a
        result the model expected came back as a refusal.
        """
        return _enforce_result_bytes(
            payload, max_bytes=max_bytes, label=label, extra=_run_extra(ctx)
        )

    def _call_spec(
        self, spec: Spec, user: Any, args: dict[str, Any], *, ctx: RunContext[Any], **kw: Any
    ) -> Any:
        """Bind the six seams to this run, then run the shared pipeline.

        Separate from the module-level function of the same name because that one
        has to stay usable without a toolset. The toolset's ``conventions`` are
        bound here too, beside the seams, so every route into a call words its
        retries the same way.
        """
        return _call_spec(
            spec,
            user,
            args,
            build_context=lambda *a, **kwargs: self.build_context(*a, ctx=ctx, **kwargs),
            translate_exception=lambda exc: self.translate_exception(exc, ctx=ctx),
            shape_page=lambda *a, **kwargs: self.shape_page(*a, ctx=ctx, **kwargs),
            render_output=lambda *a, **kwargs: self.render_output(*a, ctx=ctx, **kwargs),
            output_extras=lambda *a, **kwargs: self.output_extras(*a, ctx=ctx, **kwargs),
            enforce_result_bytes=lambda *a, **kwargs: self.enforce_result_bytes(
                *a, ctx=ctx, **kwargs
            ),
            conventions=self._conventions,
            **kw,
        )

    def _call_spec_releasing_connections(
        self, spec: Spec, user: Any, args: dict[str, Any], *, ctx: RunContext[Any], **kw: Any
    ) -> Any:
        """``_call_spec``, plus the connection cleanup the thread hop owes.

        Django opens a database connection **per thread** and closes one per
        *request* — the ``request_finished`` receiver, ``close_old_connections``.
        Off HTTP there is no request, and on the default
        ``thread_sensitive=True`` the dispatch runs on asgiref's process-wide
        ``single_thread_executor``, so the connection it opens there outlives
        the call, the toolset and the agent, with nothing that will ever close
        it. Two separate projects hit this and could not fix it from outside,
        because ``django.db.connections`` is thread-local: only asgiref's thread
        can close asgiref's connection. The symptom lands somewhere else
        entirely — a test session that will not drop its database because it "is
        being accessed by other users", naming neither this package nor the
        thread holding the handle.

        **Only what this call opened is released**, which is what makes the
        cleanup safe on every thread it can land on rather than only the one it
        was written for. ``sync_to_async`` runs on the *caller's* thread whenever
        there is a synchronous frame above the loop (asgiref's
        ``current_thread_executor``) — a WSGI request driving an agent through
        ``async_to_sync`` — and that thread's connection is the request's, quite
        possibly mid-``atomic``. It was open before this call, so it is not ours
        and is not touched. An unconditional close here is the obvious fix and it
        is wrong: it severs the caller's own transaction.

        The bookkeeping stays true across calls rather than decaying: a dispatch
        that opened a connection closes it, so the next one on the same shared
        thread finds the thread as it was and owns what it opens in turn. A
        connection someone *else* opened on that thread — a consumer's own
        ``sync_to_async`` ORM work — stays theirs forever.

        ``close()`` rather than Django's ``close_if_unusable_or_obsolete()``,
        which is what ``close_old_connections`` calls: that one honours
        ``CONN_MAX_AGE``, a budget for reusing a connection *across requests*.
        There are no requests on this thread, so a connection held back for the
        next one is held forever — and the alias would then read as "not ours"
        on the following dispatch, quietly restoring the leak at any
        ``CONN_MAX_AGE`` above zero.

        Every clause above is held by a named test, because a 100% branch gate
        cannot see the difference between them: mutating each one out in turn
        says that ``test_a_dispatch_closes_the_connection_it_opened_on_the_shared_thread``
        holds the cleanup itself, its ``persistent-connections`` parameter alone
        holds ``close()`` over ``close_if_unusable_or_obsolete()``, and
        ``test_a_dispatch_leaves_a_connection_it_did_not_open_alone`` is the only
        thing standing between this and the unconditional version.

        The bookkeeping itself is ``_releasing_connections_opened_here``, shared
        with the listing-time evaluation of operation conditions, which makes the
        same hop and owes the same cleanup.
        """
        with _releasing_connections_opened_here():
            return self._call_spec(spec, user, args, ctx=ctx, **kw)


@contextmanager
def _releasing_connections_opened_here() -> Iterator[None]:
    """Close, on the way out, each connection this thread opened inside the block.

    The cleanup ``SpecToolset._call_spec_releasing_connections`` documents,
    clause by clause. It is a helper rather than that method's body
    because two hops owe it -- a dispatch, and the listing-time evaluation of
    operation conditions -- and the second one leaking would disarm the first.
    """
    # ``initialized_only`` so asking the question does not itself build a
    # wrapper for every configured alias on this thread.
    held_before = {
        conn.alias for conn in connections.all(initialized_only=True) if conn.connection is not None
    }
    try:
        yield
    finally:
        for conn in connections.all(initialized_only=True):
            # No ``conn.connection is not None`` here: Django's ``close()``
            # returns immediately on a wrapper that never connected, so the
            # extra conjunct would change nothing and no test could hold it.
            if conn.alias not in held_before:
                conn.close()


def _validate_permissions(specs: Mapping[str, Spec], *, require: bool) -> None:
    """Refuse — or warn about — specs with nothing gating them off HTTP.

    **``permission_classes=None`` means *inherit*, and off HTTP there is nothing
    to inherit from.** Over HTTP it is a correct, working configuration: the
    view's own ``permission_classes`` and DRF's ``DEFAULT_PERMISSION_CLASSES``
    apply. A toolset has neither, so a spec properly guarded behind a viewset,
    with passing HTTP tests, becomes callable by whatever the agent decides to
    call the moment it is handed to a model, with no signal anywhere.

    ``ImproperlyConfigured`` rather than the
    ``ValueError`` the checks below raise: this is a deployment misconfiguration
    rather than a coding error, and it is what the MCP transport raises for the
    same check, so a consumer running both catches one thing.
    """
    unguarded: list[str] = unguarded_specs(specs)
    if not unguarded:
        return
    names: str = ", ".join(repr(name) for name in sorted(unguarded))
    problem = (
        f"SpecToolset was given spec(s) with no permission_classes: {names}. "
        "A toolset dispatches off HTTP, where neither a viewset's "
        "permission_classes nor REST_FRAMEWORK's DEFAULT_PERMISSION_CLASSES "
        "apply — so nothing gates these calls and the model can make any of "
        "them. Set spec.permission_classes on each."
    )
    if require:
        raise ImproperlyConfigured(
            f"{problem} To downgrade this to a warning while you migrate, pass "
            "require_permissions=False."
        )
    warnings.warn(
        f"{problem} This is a warning because require_permissions=False.",
        UnguardedSpecWarning,
        stacklevel=3,
    )


class UnguardedSpecWarning(UserWarning):
    """A spec was exposed as a tool with no ``permission_classes``.

    Its own category so a consumer migrating a large registry can silence it
    deliberately — ``warnings.filterwarnings("ignore", category=…)`` — rather
    than by muting ``UserWarning`` across the process.
    """


class UndescribedToolWarning(UserWarning):
    """A spec was exposed as a tool with nothing to tell the model it exists for.

    Its own category, separate from ``UnguardedSpecWarning``, because the
    two are silenced by different people: one is a security posture, the other
    is prompt quality.
    """


def _validate_descriptions(
    specs: Mapping[str, Spec],
    descriptions: Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve each tool's description, warning about the ones that say nothing.

    A key naming a tool this toolset does not expose is a typo and raises, on the
    same reasoning as ``tool_query_params``: dropping it silently would leave the
    tool carrying the description its author thought they had replaced.

    Warning rather than raising for a blank one, unlike the permission check: an
    undescribed tool degrades an answer, an unguarded one exposes data.
    """
    for tool_name in descriptions or {}:
        if tool_name not in specs:
            raise ValueError(
                f"descriptions references unknown tool {tool_name!r}; known tools: {sorted(specs)}."
            )
    resolved: dict[str, str] = {}
    blank: list[str] = []
    for name, spec in specs.items():
        text: str = ((descriptions or {}).get(name) or _spec_description(spec) or "").strip()
        if not text:
            blank.append(name)
            continue
        resolved[name] = text
    if blank:
        names: str = ", ".join(repr(name) for name in sorted(blank))
        warnings.warn(
            f"SpecToolset tool(s) {names} have no description: neither a "
            "descriptions={...} entry nor a docstring on the spec's callable. A "
            "model picks tools almost entirely by description, so an undescribed "
            "tool is one it will call at the wrong time or not at all.",
            UndescribedToolWarning,
            stacklevel=3,
        )
    return resolved


def _validate_tool_names(specs: Mapping[str, Spec]) -> None:
    """Fail fast when a tool name violates the model provider's name constraint."""
    invalid = sorted(name for name in specs if not _TOOL_NAME_RE.match(name))
    if invalid:
        raise ValueError(
            "SpecToolset tool names must match ^[a-zA-Z0-9_-]{1,64}$ (model provider "
            f"function-name constraint); invalid name(s): {invalid}."
        )


def _validate_query_params(
    query_params: Sequence[QueryParam],
    tool_query_params: Mapping[str, Sequence[QueryParam]] | None,
    specs: Mapping[str, Spec],
) -> None:
    """Fail fast on a per-tool key naming a tool this toolset does not expose.

    Runs before the merge, unlike the name-level checks in
    ``_validate_channel_declarations``: the merge indexes by tool name, so a
    typo'd key would otherwise be dropped silently.
    """
    for tool_name in tool_query_params or {}:
        if tool_name not in specs:
            raise ValueError(
                f"tool_query_params references unknown tool {tool_name!r}; "
                f"known tools: {sorted(specs)}."
            )


def _merge_query_params(
    toolset_wide: Sequence[QueryParam], per_tool: Sequence[QueryParam]
) -> tuple[QueryParam, ...]:
    """Toolset-wide params, then per-tool overriding by name (per-tool wins)."""
    merged: dict[str, QueryParam] = {qp.name: qp for qp in toolset_wide}
    for qp in per_tool:
        merged[qp.name] = qp
    return tuple(merged.values())


def _validate_url_kwargs(
    url_kwargs: Sequence[UrlKwarg],
    tool_url_kwargs: Mapping[str, Sequence[UrlKwarg]] | None,
    specs: Mapping[str, Spec],
) -> None:
    """Fail fast on a per-tool key naming a tool this toolset does not expose.

    See ``_validate_query_params`` — name-level checks run post-merge.
    """
    for tool_name in tool_url_kwargs or {}:
        if tool_name not in specs:
            raise ValueError(
                f"tool_url_kwargs references unknown tool {tool_name!r}; "
                f"known tools: {sorted(specs)}."
            )


def _validate_many_argument_channels(
    specs: Mapping[str, Spec],
    tool_query_params: Mapping[str, Sequence[QueryParam]],
    tool_url_kwargs: Mapping[str, Sequence[UrlKwarg]],
) -> None:
    """Refuse a declared channel named after the argument a tool's list travels under.

    A ``QueryParam`` or ``UrlKwarg`` is popped out of the model's arguments before
    dispatch, so one sharing ``spec.many_argument`` would take the list with it:
    every call would reach drf-services without its list and come back as a retry
    saying the argument is required -- the argument the model had just sent.

    Its own check rather than ``many_argument`` added to the ``reserved`` names
    handed to drf-services' shared validator, which would refuse the same
    declaration with a list of the dispatcher's pool seeds and no word about the
    list. ``ImproperlyConfigured``, as that validator raises for a channel name the
    transport owns, which this is for one tool.
    """
    for tool_name, spec in specs.items():
        if not _takes_a_list(spec):
            continue
        argument: str = spec.many_argument
        channels: tuple[tuple[str, Sequence[QueryParam] | Sequence[UrlKwarg]], ...] = (
            ("QueryParam", tool_query_params[tool_name]),
            ("UrlKwarg", tool_url_kwargs[tool_name]),
        )
        for kind, declarations in channels:
            if any(declaration.name == argument for declaration in declarations):
                raise ImproperlyConfigured(
                    f"SpecToolset tool {tool_name!r} takes its list under the argument "
                    f"{argument!r} (its spec declares many=True), and a {kind} named "
                    f"{argument!r} is declared for it too. A {kind} is taken out of the "
                    "arguments before dispatch, so the list would never reach the service. "
                    "Rename the declaration, or name the list's argument with "
                    "ServiceSpec(many_argument=...)."
                )


# The names ``_pop_pagination`` takes out of every list tool's call. Not
# ``_RESERVED_PARAM_NAMES``: ``ordering`` is left in the call for a spec that
# advertises it, so a selector taking ``ordering`` receives it.
_PAGINATION_ARGUMENTS = frozenset({"page", "limit"})


def _validate_inputs_a_channel_takes(
    specs: Mapping[str, Spec],
    tool_query_params: Mapping[str, Sequence[QueryParam]],
    tool_url_kwargs: Mapping[str, Sequence[UrlKwarg]],
    *,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
) -> None:
    """Refuse a spec input that one of this toolset's own arguments takes away.

    The spec's side of the collision ``_validate_channel_declarations`` refuses
    from the channel's. That check keeps a ``QueryParam`` or ``UrlKwarg`` off
    ``page`` / ``limit``; this one keeps the spec's inputs off them, and off a
    ``QueryParam``'s name, because an input of such a name registers, is
    advertised, and then never receives what the model sent. A required one
    failed every call carrying it; a defaulted one ran on its default whatever
    the call asked for. Two cases, each refused with a default or without:

    - on a list tool, a selector parameter named ``page`` or ``limit``, which
      ``_pop_pagination`` takes out of every call to serve the page. Only a list
      tool paginates, so a retrieve selector's ``page`` reaches it, and the
      condition is held by ``test_a_retrieve_selector_keeps_a_parameter_named_page``.
      Read off the spec's own reflected input, ``own`` below, which no
      provider's keys narrow;
    - on any tool, a name one of the tool's ``QueryParam`` declarations shares,
      whose value ``_pop_query_params`` routes to ``request.query_params``
      instead. Read below.

    **A ``QueryParam`` is checked against what the tool advertises as the
    call's own input**, ``_caller_schema``'s properties, the reader the tool's
    schema is built from, so the refusal and the schema cannot disagree about
    which names are the model's. That is a selector's parameters and
    ``filter_set`` fields
    (``test_a_filter_set_field_a_query_param_shadows_is_refused``), or a
    service's input serializer fields and its target lookup's parameters, read
    under the binding the call is dispatched with. Each part of that set is held by a test
    that fails without it, because the refusal is one branch whichever part
    answers:

    - **the target lookup's parameters**, which a service tool advertises
      beside its serializer's fields, so the model sends them: a required one
      was answered as a missing argument of a call that carried it, which an
      agent resent until it ran out of retries, and a defaulted one resolved
      the row on its default
      (``test_a_lookup_parameter_a_query_param_takes_is_refused``);
    - **less what a ``kwargs=`` provider declares it fills**, its
      ``provider_keys`` ``filled`` set, which owns the parameter whatever the
      model sends. A provider reading ``request.query_params`` into the
      parameter is the ordinary way to route a query parameter to a selector,
      and nothing is lost there
      (``test_a_query_param_a_typed_provider_hands_the_selector_is_served``);
    - **keeping a key the provider may decline**, and every key of a provider
      whose keys cannot be read: on a call where it does not fill the key, only
      the model can, and the ``QueryParam`` has popped the model's value
      (``test_a_query_param_on_a_key_a_provider_may_leave_to_the_model_is_refused``);
    - **less the keys the server keeps from the call**, ``server_owned_keys``,
      which the model is never offered and dispatch never hands the caller's
      value for, so a ``QueryParam`` of that name takes nothing
      (``test_a_query_param_named_after_a_key_the_server_keeps_is_not_refused``);
    - **and a service's input serializer fields**, which validate the arguments
      left once the ``QueryParam`` has taken its own, whatever a callable hides
      under the name (``test_a_serializer_field_a_query_param_shadows_is_refused``).

    The tool's ``UrlKwarg`` declarations are passed, so the read is the
    schema's own, and no test holds them: a ``UrlKwarg`` fills only its own
    name, and one sharing a ``QueryParam``'s name is refused by
    ``_validate_no_param_channel_overlap`` before this runs. A registered pool
    seed is in the same position, refused as a ``QueryParam`` name by
    ``_validate_channel_declarations``.

    **No name is exempted for an input serializer, as drf-mcp exempts one.**
    drf-mcp lays a selector tool's validated input back over the stripped
    arguments, so a field it declares under the name does reach the selector.
    A ``SelectorSpec`` declares no ``input_serializer`` and this toolset takes
    none for a selector tool, so nothing here lays a stripped value back.

    A ``UrlKwarg`` sharing a parameter's name stays allowed: its value reaches
    the selector through ``view.kwargs``, the documented way to route a capture
    the selector also reads.
    """
    for tool_name, spec in specs.items():
        own = frozenset(
            cast(
                "dict[str, Any]",
                spec_to_json_schema(
                    spec, phase="input", registry=registry, argument_binding=_ARGUMENT_BINDING
                ),
            ).get("properties", {})
        )
        label = f"SpecToolset tool {tool_name!r}"
        pagination = sorted(own & _PAGINATION_ARGUMENTS) if _is_list_selector(spec) else []
        if pagination:
            raise ImproperlyConfigured(
                f"{label}: the selector takes parameter(s) {pagination!r}, but `page` and "
                "`limit` are a list tool's pagination arguments, which the toolset takes "
                "out of the call before the selector runs, so the parameter would never "
                "receive the model's value. Rename the parameter."
            )
        advertised = _caller_schema(
            spec, tool_url_kwargs[tool_name], pool_seeds=pool_seeds, registry=registry
        ).get("properties", {})
        shadowed = sorted(
            frozenset(advertised)
            & {query_param.name for query_param in tool_query_params[tool_name]}
        )
        if shadowed:
            raise ImproperlyConfigured(
                f"{label}: {_inputs_taken_by(spec, shadowed, own)} that the tool also "
                "declares as a QueryParam. A QueryParam's value is taken out of the call "
                "and routed to request.query_params before the spec runs, so the input "
                "would never receive the model's value. Fill the parameter from "
                "request.query_params with a kwargs= provider whose TypedDict declares "
                "it, or read the value there in the callable and drop the input, or drop "
                "the QueryParam so the argument reaches the spec."
            )


def _inputs_taken_by(spec: Spec, names: list[str], own: frozenset[str]) -> str:
    """Which callable takes each name a ``QueryParam`` shadows, for the refusal.

    A selector tool's names are the selector's. A service tool's are its own
    where its input schema lists them, its serializer's fields, and otherwise
    its target lookup's, which ``_caller_schema`` merged in beside them. The
    lookup's wording is held by ``test_a_lookup_parameter_a_query_param_takes_is_refused``.
    """
    if isinstance(spec, SelectorSpec):
        return f"the selector takes input(s) {names!r}"
    served = [name for name in names if name in own]
    looked_up = [name for name in names if name not in own]
    parts: list[str] = []
    if served:
        parts.append(f"the service takes input(s) {served!r}")
    if looked_up:
        parts.append(f"its target lookup takes input(s) {looked_up!r}")
    return " and ".join(parts)


def _validate_channel_declarations(
    tool_name: str, declarations: Sequence[Any], kind: str, *, seeds: PoolSeeds
) -> None:
    """Apply drf-services' shared channel checks to one tool's merged tuple.

    Those cover the dispatcher's pool seeds (``request`` / ``user`` / ``data`` /
    …, which a caller must not be able to route a value onto) and the
    contradiction of ``required=True`` with a ``default``. The transport-side
    names stay ours to contribute, belonging to the adapter rather than the
    dispatcher: the ``page`` / ``limit`` / ``ordering`` the MCP transport
    reserves too.

    **A registered pool seed is reserved like a built-in one**, so its name is
    passed in beside those. Dispatch strips a reserved name from the URL kwargs
    it hands a selector, so a ``UrlKwarg`` named after a seed would be accepted
    here, advertised to the model, and then silently dropped on every call.
    ``test_a_channel_named_after_a_registered_seed_is_refused`` holds it.
    """
    validate_channel_names(
        label=f"SpecToolset tool {tool_name!r}",
        kind=kind,
        declarations=declarations,
        reserved=_RESERVED_PARAM_NAMES | seeds.names,
    )


def _merge_url_kwargs(
    toolset_wide: Sequence[UrlKwarg], per_tool: Sequence[UrlKwarg]
) -> tuple[UrlKwarg, ...]:
    """Toolset-wide kwargs, then per-tool overriding by name (per-tool wins)."""
    merged: dict[str, UrlKwarg] = {uk.name: uk for uk in toolset_wide}
    for uk in per_tool:
        merged[uk.name] = uk
    return tuple(merged.values())


def _validate_no_param_channel_overlap(
    tool_query_params: Mapping[str, Sequence[QueryParam]],
    tool_url_kwargs: Mapping[str, Sequence[UrlKwarg]],
) -> None:
    """Fail fast when a name is both a QueryParam and a UrlKwarg on one tool.

    Both channels pop the arg at call time, so a shared name would route to only
    one of ``query_params=`` / ``kwargs=`` (whichever pops first) — an ambiguity
    the caller must resolve, not the toolset.
    """
    for tool_name, query_params in tool_query_params.items():
        clash = sorted(
            {qp.name for qp in query_params} & {uk.name for uk in tool_url_kwargs[tool_name]}
        )
        if clash:
            raise ValueError(
                f"name(s) {clash} are registered as both a QueryParam and a UrlKwarg on "
                f"tool {tool_name!r}; a value cannot route to two channels."
            )


async def _with_deadline(
    awaitable: Any,
    seconds: float | None,
    *,
    label: str,
    extra: Mapping[str, Any] | None = None,
) -> Any:
    """Await ``awaitable``, answering the model instead of hanging past ``seconds``.

    ``None`` awaits without a deadline, so the resolved bound goes straight in.

    **This does not stop the work.** The dispatch runs in a ``sync_to_async``
    thread and asyncio cannot interrupt a thread parked in a database driver's
    socket read, so the query runs to completion regardless; what the deadline
    buys is a terminal answer rather than a run that never returns.

    That answer is a ``ToolFailed`` rather than an ordinary exception, for the
    same reason the byte ceiling's is: the model can respond to it by asking for
    less, and killing the run denies it the chance. It is not a ``ModelRetry``
    either -- a retry budget is finite, and a run should not die because a model
    spent it on progressively narrower queries against a slow table.

    **A returned ``{"error": …}`` had those two properties as well, and lost a
    third.** Pydantic-AI marks a returned value ``outcome="success"``, so an
    abandoned call was indistinguishable on the wire from one that answered --
    the deadline fired, the operator's log line was written, and the client drew
    a completed call. The exception says the same sentence to the model and says
    "failed" to everything downstream of it.

    ``extra`` is the caller's log correlation, threaded in rather than derived:
    this is a module-level helper with no ``RunContext``, and its one call site
    inside ``call_tool`` has one. A timeout is a run misbehaving, which is
    precisely when a line that cannot be tied back to its run is worth least --
    concurrent runs interleave, and every one of them times out the same way.
    """
    if seconds is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        # The operator hears about it too: a tool that intermittently answers
        # "took too long" and logs nothing is indistinguishable from a bug.
        logger.warning("Tool %r exceeded its %.1fs dispatch timeout", label, seconds, extra=extra)
        raise ToolFailed(
            f"This call took longer than the {seconds:g}s limit and was "
            "abandoned. Narrow the request — add or tighten a filter, or "
            "lower `limit` — and call again."
        ) from exc


def _default_get_progress(ctx: RunContext[Any]) -> ProgressReporter | None:
    """Read ``ctx.deps.progress``, tolerating a deps type that has no such field.

    ``getattr`` rather than attribute access: a project with its own deps class
    need not declare the field, and a missing sink is the ordinary case.
    """
    return getattr(getattr(ctx, "deps", None), "progress", None)


def _default_get_http_request(ctx: RunContext[Any]) -> HttpRequest | None:
    """No request unless one was configured.

    **Not read off ``ctx.deps``, unlike the user and the progress sink.** A
    request that appeared by default would be one nothing declared, failing
    silently: a serializer would build absolute URLs against whatever host
    happened to be in scope.
    """
    del ctx
    return None


def _default_get_user(ctx: RunContext[Any]) -> Any:
    """Read the acting user off ``ctx.deps.user`` (the ``AgentDeps`` default)."""
    return ctx.deps.user


_DEFAULT_CONVENTIONS = AgentConventions()
"""The wording a toolset given no ``conventions=`` says, and the module-level default.

Built once because it is frozen and validated at construction; private because
``AgentConventions()`` is the public way to say the same thing.
"""

_BLOCK_CONVENTIONS = (
    "base",
    "pagination",
    "ordering",
    "handles",
    "read_shaping",
    "read_shaping_on_pages",
)
"""The fields only the derived block says, so ``instructions=`` would ignore them.

The other four land somewhere an override does not reach: the heading of the
per-step unavailable list (appended after the override), a handle field's output
schema, a ``QueryParam``'s schema and the render retry, and the missing-argument
retry. ``test_the_fixture_reaches_every_field`` and the ``_LANDS`` table beside
it say where each field lands.
"""

_INSTRUCTIONS_MEMO_SIZE = 32
"""How many combinations of omitted tools one toolset keeps a derived block for.

Every subset of the conditioned tools is a possible key, which is why there is a
bound at all; in practice a deployment moves between a handful -- the books open
or closed -- and an evicted entry costs one derivation, tens of microseconds.
"""


def _validate_conventions_beside_instructions(
    conventions: AgentConventions, instructions: str | None
) -> None:
    """Refuse a block line changed beside the override that replaces the block.

    Ignored configuration fails loudly here, and the block's lines are exactly
    what ``instructions=`` stands in for. Compared against the defaults field by
    field, so ``AgentConventions()`` passed explicitly is not a change, and
    neither is changing only the fields that land outside the block. Every
    changed field is named at once, so one restart fixes them all.
    """
    if instructions is None:
        return
    changed = [
        name
        for name in _BLOCK_CONVENTIONS
        if getattr(conventions, name) != getattr(_DEFAULT_CONVENTIONS, name)
    ]
    if changed:
        raise ImproperlyConfigured(
            f"conventions= changes {', '.join(changed)}, which only the derived instructions "
            "block says, and instructions= replaces that block, so the change would be "
            "ignored. Drop instructions= to keep the derived block with your wording, or "
            "change only unavailable_heading, handle_field_description, query_param_on_pages "
            "or missing_arguments beside it."
        )


def _render(template: str | None, **values: Any) -> str | None:
    """One field of ``AgentConventions`` as the model reads it, or ``None`` if dropped.

    Every field is rendered, placeholders or not, so a doubled brace means one
    brace in all of them alike (``test_a_doubled_brace_reaches_the_model_as_one``).
    The fields were validated against these exact values' names and types when
    the conventions were built, so this cannot raise.
    """
    return None if template is None else template.format(**values)


def _listed(names: Sequence[str]) -> str:
    """Names as a ``{names}`` placeholder receives them: each in backticks, comma-joined."""
    return ", ".join(f"`{name}`" for name in names)


def _unavailable_instruction(unavailable: Mapping[str, Affordance], heading: str) -> str:
    """The heading, then one line per tool left out: its name and its ``reason``.

    In the toolset's declaration order, which is the order ``unavailable`` was
    built in, so the same step always reads the same way. The ``code`` is left
    out on purpose: it is for programs and for tying a refusal to a row's
    ``affordances``, and a model relaying this to a person has no use for it.

    Said once, above the names, so each tool left out costs one short line. What
    the heading has to carry is the thing the catalog's silence cannot: the tool
    exists, and there is a reason it is absent that can be passed on to a person
    as written -- a condition's ``reason`` is a sentence written for people and
    models alike.
    """
    # Rendered like every other field, so a doubled brace reads as one here too
    # (``test_a_doubled_brace_reaches_the_model_as_one``).
    lines = [heading.format()]
    lines.extend(f"  - `{name}`: {affordance.reason}" for name, affordance in unavailable.items())
    return "\n".join(lines)


def _has_handle(projection: AudienceProjection) -> bool:
    """Whether any field on this tool's output is an opaque identifier."""
    return any(marking.audience is FieldAudience.HANDLE for marking in projection.fields.values())


def _derive_instructions(
    specs: Mapping[str, Spec],
    tool_query_params: Mapping[str, Sequence[QueryParam]],
    projections: Mapping[str, AudienceProjection] | None = None,
    *,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
    page_size: int = DEFAULT_PAGE_SIZE,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    conventions: AgentConventions = _DEFAULT_CONVENTIONS,
) -> str | None:
    """Build the conventions block from the specs / query params / ordering.

    Every line is conditional on something being able to act on it: pagination
    only with a list selector present, ordering only where some tool actually has
    a sort argument, read-shaping only with a ``QueryParam``. Advice a model
    cannot use is not neutral — it is budget spent teaching it about an argument
    that will be rejected.

    **The conditions are decided here and the words come from ``conventions``.**
    A consumer changes what a line says, never whether it appears, so an
    overridden pagination line is still absent from a toolset with no list tool
    and from a step that offers none. A field set to ``None`` drops its line;
    ``None`` comes back when every line was dropped, so a toolset that says
    nothing adds nothing to the prompt.

    **The specs are the only source of an ordering argument**, so this asks them
    and nothing else: whatever the schema advertises is what the model may send.
    ``pool_seeds`` is the toolset's, so a sort a registered seed fills is asked
    about as the schema was built.
    """
    lines: list[str | None] = [_render(conventions.base)]
    if any(_is_list_selector(spec) for spec in specs.values()):
        # The same number each ``limit`` description states, for the reason
        # ``_served_page_size`` gives.
        lines.append(_render(conventions.pagination, page_size=page_size))
    # Deduplicated, in first-seen order: one toolset can carry several tools
    # whose sorts are declared under different names, and a model reading one
    # block for the whole toolset has to be told each. Named rather than said as
    # ``ordering`` outright, because a toolset whose ``OrderingFilter`` is called
    # ``sorting`` would otherwise be told about an argument that does not exist.
    ordering_names: list[str] = []
    for spec in specs.values():
        advertised = _spec_ordering_argument(spec, pool_seeds=pool_seeds, registry=registry)
        if advertised is not None and advertised not in ordering_names:
            ordering_names.append(advertised)
    if ordering_names:
        lines.append(_render(conventions.ordering, names=_listed(ordering_names)))
    # A toolset with no handle anywhere gains nothing from being told how to
    # treat one, and this block is prepended to every run.
    if any(_has_handle(projection) for projection in (projections or {}).values()):
        lines.append(_render(conventions.handles))
    query_param_names = sorted({qp.name for params in tool_query_params.values() for qp in params})
    if query_param_names:
        line = _render(conventions.read_shaping, names=_listed(query_param_names))
        # Keyed on a list tool *declaring* one, not on a list tool and a
        # ``QueryParam`` both being present: a toolset whose only read-shaping
        # param is on a retrieve tool returns no page for it to be misread against.
        scope = (
            _render(conventions.read_shaping_on_pages)
            if any(
                _is_list_selector(spec) and tool_query_params.get(name)
                for name, spec in specs.items()
            )
            else None
        )
        # The continuation goes with the line it continues. One arc: ``line is
        # not None`` is held by ``test_none_drops_a_block_line_and_nothing_else
        # [read_shaping]``; ``scope is not None`` by its ``[read_shaping_on_pages]``
        # case and by ``test_an_overridden_line_follows_a_tool_a_condition_leaves_out``,
        # whose read-shaping param is on a tool that returns no page.
        if line is not None and scope is not None:
            line = f"{line} {scope}"
        lines.append(line)
    said = [line for line in lines if line is not None]
    return "\n".join(said) if said else None


def _build_tool_def(
    name: str,
    spec: Spec,
    query_params: Sequence[QueryParam] = (),
    url_kwargs: Sequence[UrlKwarg] = (),
    description: str | None = None,
    max_page_size: int | None = None,
    *,
    projection: AudienceProjection | None = None,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    conventions: AgentConventions = _DEFAULT_CONVENTIONS,
) -> ToolDefinition:
    """One tool definition: what the model may send, and what it gets back.

    ``include_return_schema`` is deliberately **left at its default**, which
    resolves to ``False``. Populating ``return_schema`` costs nothing; putting it
    on the wire costs context on every turn of every run, and whether that trade
    is worth it depends on the model and the size of the serializer — neither of
    which this package knows. Pydantic-AI already owns the opt-in, at both
    scopes: ``SpecToolset(...).include_return_schemas()`` for one toolset, or the
    ``IncludeToolReturnSchemas`` capability for a run. A knob here would be a
    third way to say the same thing, and the one a consumer composing through
    ``SpecCapability`` still could not reach.
    """
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema=_input_schema(
            spec,
            query_params,
            url_kwargs,
            max_page_size,
            registry=registry,
            pool_seeds=pool_seeds,
            conventions=conventions,
        ),
        return_schema=_return_schema(
            spec, projection=projection, registry=registry, conventions=conventions
        ),
        metadata={"annotations": {"readOnlyHint": isinstance(spec, SelectorSpec)}},
    )


def _return_schema(
    spec: Spec,
    *,
    projection: AudienceProjection | None = None,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
    conventions: AgentConventions = _DEFAULT_CONVENTIONS,
) -> dict[str, Any] | None:
    """The shape of what this tool returns, or ``None`` when the spec declares none.

    Generated from the **projected** output path rather than from the raw
    serializer, so it describes the payload the model is actually handed:
    ``render_for_audience`` drops hidden fields and substitutes a choice field's
    display value, and a schema still advertising either would be worse than no
    schema at all — a model asking for a field the render removes gets nothing
    back and no reason why.

    ``paginate`` is ``_is_list_selector`` and not ``kind is LIST``, because it
    has to answer *does this toolset wrap the result*, not *is the result a
    collection*. A ``ServiceSpec`` whose
    ``output_selector_spec`` is a LIST returns a bare array here — the toolset
    paginates list *selectors* only — so claiming the envelope for it would
    re-introduce, one spec kind over, exactly the schema-versus-payload
    disagreement this release exists to close.

    ``affordances`` declares the object the render adds to every item when the
    selector spec it renders through declares them: per name, whether the
    operation is available and, when it is not, the ``code`` -- enumerated from
    the declaration -- and ``reason``. Leaving it out advertised items without the
    key while every payload carried it -- the same disagreement, over the one key
    that tells a model what it can do next. A spec declaring none passes ``None``
    and its schema is unchanged.

    ``None`` for a spec with no ``output_serializer`` is the correct answer and
    not a gap: drf-services refuses to fabricate a shape it cannot derive, and a
    guessed one would be a claim the payload never has to honour.

    ``kind`` is ``LIST`` for a ``many=True`` service whatever its
    ``output_selector_spec`` declares. That selector is ``RETRIEVE`` by
    convention, because its kind describes one row, while drf-services renders
    the list it returns as a list; the kind alone advertised an object for a
    payload that is always an array. Otherwise the kind is the rendered spec's,
    which is the kind dispatch presents: a single-row service declaring a
    ``LIST`` output is an array whether or not that output has a ``selector`` to
    re-read through, since dispatch presents the service's own return as the
    list when it has none
    (``test_a_service_whose_output_is_a_list_is_an_array_not_an_envelope``), and
    raises ``ImproperlyConfigured`` for a return that is not a set of rows rather
    than present one row against this array
    (``test_a_list_output_with_nothing_to_reread_refuses_a_single_row``).

    The root admits ``null`` exactly where drf-services' ``can_present_nothing``
    says dispatch may present ``None``, so this schema and every other route's
    output schema state the same ``null``.
    """
    rendered = _rendered_selector_spec(spec)
    if rendered is None:
        return None
    return output_to_json_schema(
        rendered.output_serializer,
        kind=SelectorKind.LIST if _takes_a_list(spec) else rendered.kind,
        paginate=_is_list_selector(spec),
        projection=projection,
        # The fallback for a handle declaring no wording of its own. drf-services
        # supplies none on purpose, and ``None`` here leaves such a field
        # undescribed, which is what dropping the line means.
        handle_description=_render(conventions.handle_field_description),
        registry=registry,
        affordances=rendered.affordances,
        # Whether dispatch may hand the model ``None``, as drf-services answers
        # it for every route that states an output schema: an ``allow_none``
        # retrieve that finds nothing, a service whose re-read can find no row,
        # and a service declaring ``ServiceSpec(allow_none=True)``. Never a
        # list, and never off the nested spec's ``allow_none``, which dispatch
        # does not read. Read off ``spec``, not ``rendered``: the re-read and
        # the service's declaration are on the service. Held by
        # ``test_a_service_whose_reread_can_find_no_row_admits_null`` and
        # ``test_a_service_declaring_allow_none_admits_null``, and its limits by
        # ``test_a_service_admits_null_only_where_dispatch_may_present_it``.
        allow_none=can_present_nothing(spec),
    )


def _rendered_selector_spec(spec: Spec) -> SelectorSpec[Any, Any] | None:
    """The selector spec a tool's output renders through, or ``None`` for none.

    A selector renders through itself; a service through its
    ``output_selector_spec``, and a service without one returns its value
    unrendered. The output serializer, the ``kind`` that decides between an item
    and a collection, and the ``affordances`` each item carries are all read off
    this one object, so the schema cannot take one from the spec and another from
    an assumption.

    Dispatching on the class rather than reading attributes off ``spec`` is what
    keeps ``affordances`` right, and it is how drf-services' render path decides
    the same question: a ``ServiceSpec``'s *own* ``affordances`` are the conditions
    that service is checked against before it runs, a different declaration that
    is never rendered. Read off the service, the schema would advertise answers
    no payload carries.
    """
    if isinstance(spec, SelectorSpec):
        return spec
    return spec.output_selector_spec


def _spec_description(spec: Spec) -> str | None:
    """The tool description: the docstring of the spec's selector / service."""
    callable_ = spec.selector if isinstance(spec, SelectorSpec) else spec.service
    return inspect.getdoc(callable_) if callable_ is not None else None


def _input_schema(
    spec: Spec,
    query_params: Sequence[QueryParam] = (),
    url_kwargs: Sequence[UrlKwarg] = (),
    max_page_size: int | None = None,
    *,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    conventions: AgentConventions = _DEFAULT_CONVENTIONS,
) -> dict[str, Any]:
    """The tool's parameter schema, with list-selector pagination + registered
    query params + URL kwargs merged into ``properties``.

    The base is what the spec itself asks of the model, from
    ``_caller_schema``: a selector's reflected parameters, less the names this
    toolset fills, or a service's input with its target lookup merged in.

    **No sort argument is contributed here.** A tool that can sort says so in its
    own reflected schema — a ``filter_set``'s ``OrderingFilter``, or an
    ``ordering`` parameter on the selector callable — and that reflected property
    passes straight through. Writing one into ``extra`` would land it *over* the
    reflected properties in the merge below, replacing a FilterSet's public
    choices with a second vocabulary the FilterSet would then reject.

    The registered declarations are merged **over** the reflected properties, so
    an explicit ``UrlKwarg`` for a key drf-services already reflected (from a
    selector's ``Unpack[TypedDict]``, or a serializer field) wins — it is the
    intentional one, and the channel the value arrives by.

    The reflected ``required`` list is preserved and *extended* by any
    ``UrlKwarg(required=True)``. A key that is both reflected-required and
    registered-required appears once: that is one statement made twice, not two
    requirements.

    A ``many=True`` spec's reflected schema is already the object a model can
    send -- one required array property named by ``spec.many_argument``, and
    ``additionalProperties: false`` -- so the merge below applies to it unchanged.
    The ``false`` is carried over and stays true: a declared channel is a
    property, not an additional one, and ``_call_spec`` pops it before drf-services
    refuses anything sent beside the list.
    """
    schema = _caller_schema(spec, url_kwargs, pool_seeds=pool_seeds, registry=registry)
    extra: dict[str, Any] = {}
    paged = _is_list_selector(spec)
    if paged:
        extra["page"] = _PAGE_PARAM_SCHEMA
        extra["limit"] = _limit_param_schema(max_page_size)
    scope = _render(conventions.query_param_on_pages) if paged else None
    extra.update({qp.name: _query_param_schema(qp, scope=scope) for qp in query_params})
    extra.update({uk.name: uk.json_schema() for uk in url_kwargs})
    required: list[str] = list(schema.get("required", []))
    required.extend(uk.name for uk in url_kwargs if uk.required and uk.name not in required)
    if not extra:
        return schema
    merged: dict[str, Any] = {
        **schema,
        "type": "object",
        "properties": {**schema.get("properties", {}), **extra},
    }
    if required:
        merged["required"] = required
    return merged


def _caller_schema(
    spec: Spec,
    url_kwargs: Sequence[UrlKwarg] = (),
    *,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
) -> dict[str, Any]:
    """What ``spec`` itself asks of the model, before this toolset's own channels.

    A selector tool asks for its selector's parameters, as ``_selector_inputs``
    reflects them. A service tool asks for its input serializer's fields and for
    its **target lookup**, the selector ``_called_selector`` names: drf-services
    hands the arguments it validates against the serializer to the selector that
    resolves the row or the set as well, and its unknown-argument check admits
    what that selector declares. Described by the serializer alone, a tool
    renaming a row never told the model about the ``pk`` naming which row, and a
    call without one reached the lookup as a ``TypeError``.

    **A serializer field sharing its name with a lookup parameter keeps the
    serializer's property**: the serializer validates the value, so its
    declaration is the more precise one, and the lookup's is written first and
    overlaid. **Requiredness is the union**, because the lookup cannot run
    without a parameter it requires whatever the serializer says about the name,
    and a ``partial`` serializer says nothing is required.
    ``test_a_serializer_field_keeps_its_schema_when_a_lookup_shares_its_name``
    holds both, and a name both sides require is listed once
    (``test_a_name_the_lookup_and_the_serializer_both_require_is_listed_once``).

    A service whose lookup advertises nothing keeps its schema exactly, down to
    a bare ``{"type": "object"}`` gaining no empty ``properties``
    (``test_a_service_without_a_lookup_keeps_its_schema_byte_for_byte``), and a
    merge requiring nothing writes no empty ``required``
    (``test_a_lookup_that_can_run_without_arguments_requires_none``).

    **The lookup's keys are merged less every key the call keeps from it**,
    ``server_owned_keys(spec)``: one the service, a precondition or the lookup
    marks ``NotClientInput``. The lookup's own reflection drops only the keys
    the lookup marks, and asked of the service's spec rather than the nested
    lookup's, the set has the others too. Dispatch drops the caller's value for
    each before the lookup reads it and ``REJECT`` refuses it, so advertising
    one offered an argument every call carrying it was refused for
    (``test_a_key_a_precondition_hides_is_not_advertised_beside_the_lookup``).
    **A field the input serializer declares under such a name stays**, because
    the serializer's properties are overlaid after the subtraction: the field is
    client input, validated into ``data``, whatever a callable hides under its
    name (``test_a_serializer_field_under_a_hidden_name_stays_advertised``).

    The service's own schema is read under ``_ARGUMENT_BINDING``, the binding it
    is dispatched under, so a service with no input serializer lists what
    ``REJECT`` admits of it.

    ``spec_to_json_schema(phase="input")`` always returns a dict (only the
    output phase is nullable), so the result is narrowed for the type-checker.
    """
    called = _called_selector(spec)
    reflected: dict[str, Any] = (
        {}
        if called is None
        else _selector_inputs(called, url_kwargs, pool_seeds=pool_seeds, registry=registry)[0]
    )
    if isinstance(spec, SelectorSpec):
        return reflected
    schema = cast(
        "dict[str, Any]",
        spec_to_json_schema(
            spec, phase="input", registry=registry, argument_binding=_ARGUMENT_BINDING
        ),
    )
    owned = server_owned_keys(spec)
    properties: dict[str, Any] = {
        name: value for name, value in reflected.get("properties", {}).items() if name not in owned
    }
    if not properties:
        return schema
    properties.update(schema.get("properties", {}))
    required = [
        *(name for name in reflected.get("required", []) if name not in owned),
        *schema.get("required", []),
    ]
    merged: dict[str, Any] = {**schema, "type": "object", "properties": properties}
    if required:
        merged["required"] = list(dict.fromkeys(required))
    return merged


def _required_arguments(
    spec: Spec,
    url_kwargs: Sequence[UrlKwarg] = (),
    *,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
) -> tuple[str, ...]:
    """The arguments a call to ``spec`` cannot go without, as its schema requires them.

    Read off the same ``_selector_inputs`` the schema is built from, so what a
    call is refused for and what the model was told cannot drift apart. Only
    the selector's half: a service's serializer fields are checked by the
    serializer, which answers a missing one itself.

    Less ``server_owned_keys(spec)``, as ``_caller_schema`` merges the lookup's
    keys less them: a key the call keeps from the lookup is not advertised, and
    a call is not refused for leaving it out. One nothing fills is a
    configuration error that fails as the lookup's own ``TypeError``, as
    drf-services fails it, rather than a retry asking the model for an argument
    ``REJECT`` then refuses
    (``test_a_call_is_not_refused_for_a_key_no_caller_can_send``). A selector
    tool's own reflection has already left the set out, so the subtraction
    changes nothing there.
    """
    called = _called_selector(spec)
    if called is None:
        return ()
    owned = server_owned_keys(spec)
    checked = _selector_inputs(called, url_kwargs, pool_seeds=pool_seeds, registry=registry)[1]
    return tuple(name for name in checked if name not in owned)


def _called_selector(spec: Spec) -> SelectorSpec[Any, Any] | None:
    """The selector spec a call to ``spec`` hands the model's arguments to, if any.

    A selector tool's own spec. For a service, the **target lookup** drf-services
    resolves the row or the set through: its ``collection_selector_spec`` or its
    ``instance_selector_spec``, whichever it declares, and ``None`` for neither.
    The collection arm is held by
    ``test_a_collection_lookup_is_advertised_as_an_instance_lookup_is``.

    There is no precedence between the two lookups and no ``many=True`` arm,
    because drf-services refuses both shapes when the spec is built: a lookup
    dispatch would never call, beside the other lookup or beside a list payload
    that resolves no target
    (``test_a_lookup_dispatch_never_calls_cannot_reach_the_toolset``). So a
    ``many=True`` service declares no lookup to read, and its schema stays the
    list alone, closed with ``additionalProperties: false``.
    """
    if isinstance(spec, SelectorSpec):
        return spec
    if spec.collection_selector_spec is not None:
        return spec.collection_selector_spec
    return spec.instance_selector_spec


def _selector_inputs(
    spec: SelectorSpec[Any, Any],
    url_kwargs: Sequence[UrlKwarg] = (),
    *,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """A selector's input schema as this toolset calls it, and what a call needs.

    A signature alone cannot say which parameters the caller sends and which the
    transport fills, so drf-services' reflection is told: ``supplied`` names
    what this toolset fills in the pool ``dispatch_spec`` calls the selector
    with, read from the sources that pool is built from, less the model's own
    arguments. A supplied name is not advertised, and every other parameter
    without a default is required, except a name a provider *may* fill (below).
    The union has three parts, each held by a test, because deleting one leaves
    every branch covered:

    - ``pool_seeds.reserved``, the set dispatch reserves against the model's
      arguments: drf-services' own seeds and the registered ones. ``base_pool``
      fills ``request`` / ``user`` / ``progress`` and every registered seed, and
      dispatch strips every reserved name from what the model sends, so a
      ``data`` or ``instance`` the selector path never fills is not the model's
      to send either. drf-services adds its own seeds to any ``supplied`` set
      itself, so the registered half is the part only this toolset can state:
      held by ``test_a_registered_seed_outranks_a_model_argument_of_the_same_name``
      and ``test_a_service_tool_advertises_its_target_lookup``.
      ``test_a_dispatcher_seed_is_neither_advertised_nor_required`` holds the
      built-in half's outcome, whichever of the two states it.
    - the name of every ``UrlKwarg`` declaring a ``default``, which fills the
      pool through ``view.kwargs`` whenever the model leaves it out. Held by
      ``test_a_url_kwarg_answers_for_the_parameter_it_fills``. One with no
      default reaches the pool only when the model sends it, so it is the
      model's to send, and the selector's signature still says whether it must;
      ``test_a_url_kwarg_with_no_default_leaves_the_selector_to_require_it``
      holds that filter. A ``build_context`` override filling a name through
      ``view.kwargs`` is code nothing here can read, which is why its docstring
      asks for the name to be declared as one of these or as a seed.
    - the keys the spec's ``kwargs=`` provider always fills, as drf-services'
      ``provider_keys`` reads them: its ``filled`` set. Held by
      ``test_a_name_a_typed_provider_returns_is_not_asked_for``.

    **The provider's annotation is read by drf-services, not here.** This
    toolset and drf-mcp each kept a copy of that reader, and the copies drifted
    from each other and from dispatch. What the reading decides is held on the
    tool definition the model sees, one test per shape:
    ``test_a_key_holding_unset_inside_a_container_is_not_offered_to_the_model``,
    ``test_an_annotation_imported_only_for_type_checking_leaves_the_keys_readable``,
    ``test_a_generic_typed_dict_declines_what_its_binding_declines`` and
    ``test_a_providers_return_annotation_decides_what_the_model_is_asked_for``.

    **A name the provider may fill, without saying it will, is advertised and
    not required for lacking a default**: every name, beside a provider whose
    keys cannot be read (``provider_keys`` answers ``None``), and a key it may
    decline with ``UNSET``, leaving the caller's value through (its
    ``declinable`` set). ``required`` keeps such a name only where the
    reflection requires it without ``supplied`` (an ``InputRequired`` marker, a
    required ``TypedDict`` key), and keeps no name the toolset fills even then.
    Nor is the call checked for it here, since only the assembled pool can say
    whether it arrived: drf-services checks that pool itself, and refuses a
    parameter nothing filled as a missing argument the model can send on its
    next turn. Each half of that is held by a test:

    - every other name staying required:
      ``test_a_selector_parameter_without_a_default_is_required``;
    - the untyped provider:
      ``test_an_untyped_provider_leaves_every_parameter_optional``;
    - a declinable key: ``test_a_key_the_provider_may_decline_stays_the_models_to_send``;
    - a marker standing: ``test_a_marked_parameter_stays_required_beside_an_untyped_provider``;
    - a marker on a name the toolset fills:
      ``test_a_marked_parameter_a_url_kwarg_fills_is_not_required_beside_an_untyped_provider``;
    - no check for such a name:
      ``test_an_untyped_provider_filling_a_marked_parameter_is_not_refused`` and
      the ``filled`` case of the declinable-key test.

    The second value is what ``_call_spec`` refuses a call without: the
    schema's own ``required`` less every name a provider may fill, so the call
    and the schema cannot disagree.
    """
    provided = provider_keys(spec.kwargs)
    defaulted = {uk.name for uk in url_kwargs if _declares_default(uk.default)}
    filled = frozenset() if provided is None else provided.filled
    supplied = pool_seeds.reserved | defaulted | filled
    schema = cast(
        "dict[str, Any]",
        spec_to_json_schema(spec, phase="input", registry=registry, supplied=supplied),
    )
    uninformed = cast("dict[str, Any]", spec_to_json_schema(spec, phase="input", registry=registry))
    declared: list[str] = uninformed.get("required", [])
    # ``None`` is every name: a provider whose keys cannot be read may fill any.
    may_fill = None if provided is None else provided.declinable
    required: list[str] = schema.get("required", [])
    checked = tuple(name for name in required if may_fill is not None and name not in may_fill)
    kept = [name for name in required if name in checked or name in declared]
    cut: dict[str, Any] = {key: value for key, value in schema.items() if key != "required"}
    if kept:
        cut["required"] = kept
    return cut, checked


def _query_param_schema(query_param: QueryParam, *, scope: str | None) -> dict[str, Any]:
    """One ``QueryParam``'s property, told what it applies to when the tool pages.

    Every list selector here returns a page, and the declared description is
    written by someone thinking of a row -- "fields to return" -- while
    the model reads it beside a documented ``{"items": [...], ...}`` result. The
    scope sentence closes that gap at the parameter, where the model is choosing
    a value. Appended after the declared text, which stays first because it is
    the part that says what the param *is*; with no declared text it is the
    whole description.

    ``scope`` is ``None`` on a tool that does not page, and on one that does when
    the conventions dropped the sentence; either way the declaration stands as
    written.
    """
    schema = query_param.json_schema()
    if scope is not None:
        declared = schema.get("description")
        schema["description"] = f"{declared} {scope}" if declared else scope
    return schema


def _is_list_selector(spec: Spec) -> bool:
    return isinstance(spec, SelectorSpec) and spec.kind == SelectorKind.LIST


def _takes_a_list(spec: Spec) -> TypeGuard[ServiceSpec[Any, Any, Any]]:
    """Whether ``spec`` is a ``many=True`` service: a list in, and a list out.

    A ``TypeGuard`` so a caller reading ``many_argument`` after it type-checks
    without restating the ``isinstance``.

    The ``isinstance`` is not a formality -- a ``SelectorSpec`` has no ``many`` to
    read -- and is held by every toolset exposing a selector. ``spec.many`` is held
    by ``test_a_service_tool_reads_its_output_serializer_one_level_down`` on the
    return schema and ``test_a_channel_may_share_a_name_no_list_travels_under`` on
    the channel check.
    """
    return isinstance(spec, ServiceSpec) and spec.many


def _spec_ordering_argument(
    spec: Spec,
    *,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
) -> str | None:
    """The argument name a list spec advertises its sort under, or ``None``.

    **The one signal, read by both the schema builder and the dispatch path, so
    they cannot disagree** — the invariant being *whatever the schema advertises,
    the dispatch must deliver*.

    **Returns the name rather than a yes/no, and that is the fix.** This used to
    ask whether the reflected schema contained the literal ``"ordering"``, which
    silently assumed the one name the documentation happens to suggest.
    A project following that to the letter is fine; a project whose
    ``OrderingFilter`` is called ``sorting`` was not. The sort still applied —
    the FilterSet reads ``params`` either way — but the usage instruction was
    dropped, and the value was never popped out of the callable's kwarg pool, so
    a selector declaring ``**kwargs`` received a read-shaping argument it never
    asked for. Which is precisely the hazard
    :func:`_pop_filter_ordering` exists to prevent, arriving through the door it
    was watching.

    **Duck-typed, not an ``isinstance`` test against ``django_filters``.**
    django-filter is an optional extra this package never imports, and a spec's
    ``filter_set`` is duck-typed all the way down. ``get_ordering_value`` is
    defined by ``OrderingFilter`` and by nothing else in the filter hierarchy, so
    asking for it identifies the sort filter under whatever name it was
    declared, including a project's own subclass.

    Falls back to the literal ``"ordering"``, which covers the case a
    ``filter_set`` cannot answer for: a list selector with no ``filter_set``
    whose *callable* declares an ``ordering`` parameter, reflected into the
    schema like any other selector argument and consumed by the callable itself.

    Restricted to list selectors because that is the only kind the toolset ever
    contributes a sort argument to: elsewhere there is no ownership to contest,
    and a service whose input serializer happens to have a field named
    ``ordering`` must not be read as a clash.

    **Read off the reflection the tool's schema is built from**, told the same
    names this toolset fills. A selector whose ``ordering`` parameter a typed
    provider or a registered seed fills is offered no sort, so no instruction
    may teach one and a sort the model sends anyway is refused. Read without
    them, this answered ``"ordering"`` for a property the schema had dropped:
    held by ``test_an_ordering_the_toolset_fills_is_refused_from_the_model`` and
    ``test_an_ordering_the_toolset_fills_gets_no_usage_line``. Without
    ``url_kwargs``, because they cannot change the answer: no ``UrlKwarg`` may
    be named ``ordering`` (``_RESERVED_PARAM_NAMES``), and a ``filter_set``'s
    sort is reflected whatever ``supplied`` says.
    """
    if not _is_list_selector(spec):
        return None
    # ``_is_list_selector`` has just said it is one; it answers ``bool``.
    selector = cast("SelectorSpec[Any, Any]", spec)
    reflected = _selector_inputs(selector, pool_seeds=pool_seeds, registry=registry)[0]
    properties = reflected.get("properties", {})
    filter_set = getattr(spec, "filter_set", None)
    for name, declared in getattr(filter_set, "base_filters", {}).items():
        # Advertised as well as declared: a filter the schema does not carry is
        # not something the model can send, so claiming it would break the
        # advertise/deliver invariant in the other direction.
        if hasattr(declared, "get_ordering_value") and name in properties:
            return cast("str", name)
    return "ordering" if "ordering" in properties else None


def _call_spec(
    spec: Spec,
    user: Any,
    args: dict[str, Any],
    *,
    action: str | None = None,
    unknown_arguments: UnknownArguments = UnknownArguments.REJECT,
    query_params: Sequence[QueryParam] = (),
    url_kwargs: Sequence[UrlKwarg] = (),
    max_page_size: int | None = None,
    max_result_bytes: int | None = None,
    projection: AudienceProjection | None = None,
    label: str = "",
    progress: ProgressReporter | None = None,
    host: str | None = None,
    json_schema_registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    build_context: _ContextBuilder = build_offline_context,
    translate_exception: _ExceptionTranslator | None = None,
    shape_page: _PageShaper | None = None,
    render_output: _OutputRenderer | None = None,
    output_extras: _ExtrasBuilder | None = None,
    enforce_result_bytes: _ResultBounder | None = None,
    conventions: AgentConventions = _DEFAULT_CONVENTIONS,
) -> Any:
    """Run ``spec`` under an off-HTTP context and render the result.

    Synchronous on purpose — ``SpecToolset.call_tool`` runs it in a thread so
    the ORM stays off the event loop.

    The six keyword seams are what
    [`SpecToolset`][rest_framework_pydantic_ai.SpecToolset] threads its
    overridable methods through; the defaults keep this function usable on its
    own. ``build_context`` and ``translate_exception`` cover the front half of a
    call, and ``shape_page`` / ``render_output`` / ``output_extras`` /
    ``enforce_result_bytes`` the back half.

    They are ``None``-defaulted rather than bound to their module-level
    implementations in the signature because those are defined below this
    function, and a default expression is evaluated at ``def`` time.

    ``output_extras`` is called with the whole ``DispatchResult`` as
    ``dispatch_result=``, unconditionally and by every route --
    [`SpecToolset`][rest_framework_pydantic_ai.SpecToolset] binds its public
    method here untouched. A seam supplied by a caller must accept that keyword.

    ``action`` becomes ``view.action`` on the synthetic view, so a permission
    class reading it sees the tool name rather than ``None``.

    ``conventions`` words the two retries this function writes itself: a missing
    argument, and a read-shaping value the render rejected on a paged tool.
    """
    shape: _PageShaper = shape_page or _shape_list
    render: _OutputRenderer = render_output or _render_output
    extras_for: _ExtrasBuilder = output_extras or _output_extras
    bound: _ResultBounder = enforce_result_bytes or _enforce_result_bytes
    # **First, against the arguments as the model sent them.** A selector -- the
    # tool's own, or the lookup a service resolves its target through -- that is
    # called without a parameter it has no default for raises ``TypeError``,
    # which used to escape the run. The names are the ones the tool's schema
    # requires of the selector, from the same reflection, and a name a channel
    # fills is never among them: the channel's own declaration answers for it.
    # Nor is one a provider only may fill, which only the assembled pool can
    # settle. The serializer's fields are not checked here; it reports those.
    # It runs before the permission check, which drf-mcp's does not: drf-mcp's
    # listing hides a tool its caller is denied, so answering there first would
    # confirm the tool exists. This catalog is not permission-filtered (see
    # ``get_tools``), so the names asked for here are ones every run was shown.
    missing = [
        name
        for name in _required_arguments(
            spec, url_kwargs, pool_seeds=pool_seeds, registry=json_schema_registry
        )
        if name not in args
    ]
    if missing:
        raise _missing_arguments(missing, conventions)
    page_args = _pop_pagination(spec, args, pool_seeds=pool_seeds, registry=json_schema_registry)
    # **Read before the pop, because the pop erases the answer.** It seeds a
    # declared ``default`` for every name the caller left out, so afterwards a
    # value nobody sent is indistinguishable from one the model chose. The render
    # arm below needs exactly that distinction: a default that breaks the render
    # is a configuration bug and has to stay loud, while a value the model sent
    # is something it can correct. An explicit ``None`` counts as omitted, which
    # is how ``QueryParam``'s own contract says a transport reads a null.
    supplied_query_params = _supplied_query_params(query_params, args)
    # Both channels pop before dispatch so their values never reach the spec as
    # inputs, where ``unknown_arguments`` (REJECT by default) would flag them.
    # That is what makes the provider-only case work: a ``project_pk`` a scoping
    # provider reads off ``view.kwargs`` is never a spec input.
    query_param_values = _pop_query_params(query_params, args)
    url_kwarg_values = _pop_url_kwargs(url_kwargs, args, conventions=conventions)
    # Last of the pops, because the filter data it returns is built from whatever
    # ``args`` is left holding once every other channel has taken its own.
    filter_data = _pop_filter_ordering(
        spec, args, pool_seeds=pool_seeds, registry=json_schema_registry
    )
    context = build_context(
        user,
        args,
        action=action,
        kwargs=url_kwarg_values or None,
        # **Always a mapping, never ``None``.** ``build_offline_context``
        # replaces the wrapped request's ``GET`` only when this is not ``None``,
        # so a tool declaring no ``QueryParam`` would otherwise leave a
        # configured ``http_request``'s own query string live inside the spec —
        # a ``?query=`` / ``?fields=`` serializer or a ``filter_set`` reshaping
        # the result through a channel neither the tool schema nor the model
        # chose. An empty declaration means an empty query string.
        query_params=query_param_values,
        host=host,
    )
    # Two-layer authorization, mirroring a DRF view: this call runs the
    # class-level ``has_permission`` (create / list-payload targets) and the
    # ``on_target_resolved`` hook below runs ``has_object_permission`` on the
    # resolved row (update / retrieve). ``dispatch_spec`` never consults
    # ``permission_classes`` itself, so without both an object-owned row would be
    # reachable by any acting user. A denial raises ``PermissionDenied``
    # uncaught below, aborting the run exactly as it would over HTTP.
    enforce_permissions(spec, context)
    try:
        result = dispatch_spec(
            spec,
            user=user,
            params=args,
            request=context.request,
            view=context.view,
            unknown_arguments=unknown_arguments,
            # The binding the input schema was read under (see the constant).
            argument_binding=_ARGUMENT_BINDING,
            on_target_resolved=enforce_permissions,
            # Accepted and forwarded, never constructed — see ``AgentDeps.progress``.
            # ``None`` becomes drf-services' no-op seed.
            progress=progress,
            # ``None`` for every call but a filter-owned ordering, where it is the
            # channel that separates the FilterSet's data from the callable's
            # arguments — the two are one flat mapping off HTTP otherwise.
            filter_data=filter_data,
            # A model's arguments are always an object, so a ``many=True`` spec's
            # list arrives under the one argument ``spec.many_argument`` names --
            # the argument its input schema advertises. A no-op on every other
            # spec, which is why it is passed unconditionally rather than behind a
            # ``spec.many`` branch this function would then have to keep in step
            # with drf-services' idea of which specs take a list.
            many_as_argument=True,
            # The project's registered seeds: resolved into the pool, reserved
            # against the model's arguments, exempt from unknown-argument
            # accounting, and shown to the affordances this call enforces --
            # all of it drf-services', keyed off the one registry handed here.
            pool_seeds=pool_seeds,
        )
    except BaseException as exc:
        # **One arm, not a chain, because the consumer's map has to be consulted
        # first *and* fall through when it declines.** Written as a leading
        # ``except`` clause instead, every arm below it becomes unreachable — the
        # ``raise`` for an unclaimed exception leaves the ``try`` entirely rather
        # than trying the next clause.
        handler = translate_exception(exc) if translate_exception is not None else None
        if handler is not None:
            return handler(exc)
        if isinstance(exc, DRFValidationError | ServiceValidationError):
            # ``ServiceValidationError`` is a ``ServiceError`` subclass, so it
            # must be matched here, before the business-error case below. Both
            # mean "the arguments were wrong", so the model retries with the
            # detail -- flattened to text, because ``str`` of a DRF detail is the
            # Python repr of its ``ErrorDetail`` objects.
            raise ModelRetry(_format_validation_detail(exc.detail)) from exc
        if isinstance(exc, AdditionalInputRequired):
            # **Must precede the ``ServiceError`` case below** — this is a
            # subclass of it, and the generic handler would report a request for
            # input as a terminal failure. ``ModelRetry`` is already the "here is
            # what to fix, call me again" channel, so the answer comes back as an
            # ordinary argument on the next call.
            raise ModelRetry(_missing_input_prompt(exc)) from exc
        if isinstance(exc, ServiceError):
            # **Raised, not returned, with the rule's own message.** A business
            # rule that refused — a conflict, a state the operation cannot run
            # against — is ``ToolFailed``'s own description: the call is done, it
            # failed definitively, and the model should adapt rather than repeat
            # it. Returning a value instead made the failure *unreadable one hop
            # out*: pydantic-ai marks an ordinary return ``outcome="success"``,
            # so a transport streaming the result had nothing but the payload's
            # wording to tell a refusal from a row, and every one of them
            # rendered as a completed call. ``ToolFailed`` carries the same
            # sentence to the model, spends no retry budget, and prepends no
            # correction instructions — the three properties the returned dict
            # was chosen for — while marking the return ``outcome="failed"``.
            if isinstance(exc, ActionUnavailable):
                # **Inside this branch, not ahead of it, and still behind the
                # consumer's map.** A subclass of ``ServiceConflict`` and so of
                # ``ServiceError``: it fails the call exactly as its parent does,
                # and the only difference is what the sentence carries. The
                # message is the only channel -- ``ToolFailed`` takes nothing
                # else, and it is what a transport forwards as the result -- so
                # a refusal that dropped its ``code`` here left the model reading
                # a reason it could not tie to the ``affordances`` answer naming
                # the same rule, with the name surviving only on ``__cause__``,
                # where no model looks.
                raise ToolFailed(_refusal_message(exc)) from exc
            raise ToolFailed(str(exc)) from exc
        raise

    if result.kind == "not_found":
        # "a missing resource" is the first case ``ToolFailed`` names, and the
        # wording the model reads is the one it read before. Raised here rather
        # than in the ``except`` arm because drf-services reports an unresolved
        # target as a *result kind*, not an exception.
        raise ToolFailed("not found")

    value = result.value
    # ``page_args`` is non-None exactly for list selectors — the only specs that
    # advertise pagination args and return a (lazy) queryset to slice.
    page: OutputPage | None = None
    if page_args is not None:
        # Deliberately unguarded against ``FieldError``. Sorting is applied by
        # whatever declared it, upstream of here, and Django validates a
        # plain-string ``order_by`` eagerly — so a ``filter_set`` whose
        # ``param_map`` names something that is not a column raises inside
        # ``dispatch_spec``, not at this line. An arm here would never fire.
        page = shape(
            value,
            page=page_args.page,
            limit=page_args.limit,
            max_page_size=max_page_size,
        )
        value = page.items
    many = result.kind == "list"
    # **The carrier, not the three fields read above.** ``result.kind`` and
    # ``result.value`` were all this function ever took off the dispatch, so
    # ``service_result`` (an upsert's ``created``), ``instance`` (the
    # pre-mutation target) and ``data`` (the validated input) reached no seam at
    # all -- and ``output_extras`` documents itself as the escape hatch for
    # exactly the first of those. Handing over the whole result closes the three
    # together and costs nothing: it is already in scope.
    #
    # Built ahead of the render rather than as its argument so the ``try`` below
    # holds the render and nothing else. Nothing here reads the query string,
    # so an error out of it is never the caller's to correct.
    extras = extras_for(spec, value, many=many, dispatch_result=result)
    try:
        rendered = render(
            spec,
            value,
            projection=projection,
            many=many,
            request=context.request,
            view=context.view,
            extras=extras,
        )
    except (DRFValidationError, ServiceValidationError) as exc:
        # **A read-shaping param is the one caller input used while rendering**,
        # not while dispatching -- a ``fields`` or ``query`` selection is read
        # by the output serializer's ``to_representation`` -- so a bad one fails
        # here, after the ``try`` above has closed, and used to escape the run
        # as a raw ``ValidationError``. Wrapped around the call rather than
        # inside ``_render_output`` so a consumer's ``render_output`` override
        # is covered too.
        #
        # The consumer's map first, and its handler's value returned as-is,
        # exactly as on the dispatch path: one ``translate_exception`` should not
        # have to know which half of the call an error came from.
        handler = translate_exception(exc) if translate_exception is not None else None
        if handler is not None:
            return handler(exc)
        # **Only when the caller shaped the render.** With no read-shaping value
        # sent, nothing the model could change would change the outcome -- a
        # serializer that raises unprompted, or a declared default it rejects --
        # so a retry would spend the budget on a server bug and hide it.
        # Validation errors only, for the same reason: an ``AttributeError`` in
        # a serializer is a server bug whatever the caller sent.
        #
        # One arc to coverage, and its parts live in how the names were
        # gathered, so what holds each is named here. Deleting this condition
        # fails ``test_render_error_with_no_query_param_supplied_still_raises``,
        # ``test_render_error_from_a_seeded_default_still_raises`` and
        # ``test_render_error_with_an_explicit_null_still_raises``; gathering the
        # names after the pop fails the second alone, and counting a null as
        # supplied the third alone.
        if not supplied_query_params:
            raise
        raise ModelRetry(
            _render_rejection_message(
                supplied_query_params,
                exc.detail,
                scope=_render(conventions.query_param_on_pages) if page is not None else None,
            )
        ) from exc
    if page is not None:
        # **After the render, never before.** The projection lands on the rows;
        # ``items`` / ``page`` / ``totalPages`` / ``hasNext`` are the envelope's
        # own keys and belong to no serializer, so a projection walking them
        # would look for markings that cannot exist.
        rendered = page.envelope(rendered)
    # Measured on the envelope, because the envelope is what is sent. It adds
    # three small scalars and the "there is more" signal the model needs in
    # order to act on a refusal that tells it to lower `limit`.
    return bound(rendered, max_bytes=max_result_bytes, label=label)


def _enforce_result_bytes(
    payload: Any,
    *,
    max_bytes: int | None,
    label: str,
    extra: Mapping[str, Any] | None = None,
) -> Any:
    """Return ``payload``, or raise a model-readable refusal when it is over budget.

    **Fails; never truncates.** A list cut at the byte ceiling is
    indistinguishable from a list that had that many rows, so a model would
    answer confidently from data it does not know is missing.

    Raised as ``ToolFailed`` rather than as ``ModelRetry``: the model should
    narrow the request and it *can*, but a retry budget is finite and a run
    should not die because a model spent it on progressively smaller queries.
    Also logged at ``WARNING``, since a bound that fires invisibly reads to an
    operator as "the tool is broken".

    **The refusal used to be a returned ``{"error": …}``, which understated it
    by exactly one field.** The dispatch did succeed; the result was refused,
    and no data reached the model. Pydantic-AI marks a returned value
    ``outcome="success"``, so a caller one hop out saw a completed call carrying
    a payload it had no way to read as a refusal. ``failed`` is the truthful
    marking for a call that produced nothing usable, and it costs the model
    nothing: the sentence it reads is unchanged.

    ``extra`` carries the caller's log correlation. Defaulted rather than
    required because this function is also the standalone bound in ``_call_spec``,
    which has no ``RunContext`` to derive one from;
    [`SpecToolset.enforce_result_bytes`][rest_framework_pydantic_ai.SpecToolset.enforce_result_bytes]
    always supplies it.
    """
    if max_bytes is None:
        return payload
    size: int = len(json.dumps(payload, default=str).encode("utf-8"))
    if size <= max_bytes:
        return payload
    logger.warning(
        "Result bound exceeded: tool %r produced %d bytes over a %d byte ceiling",
        label,
        size,
        max_bytes,
        extra=extra,
    )
    raise ToolFailed(
        f"This result was {size} bytes, over the {max_bytes} byte ceiling. "
        "Narrow the request — add or tighten a filter, lower `limit`, or "
        "select fewer fields — and call again. The result was not "
        "truncated: a partial payload would look complete."
    )


def _missing_input_prompt(exc: AdditionalInputRequired) -> str:
    """The service's message, plus the names it wants the answer back under.

    ``schema`` is a JSON-Schema *properties* mapping keyed by input name, so the
    keys alone are what the model needs: it is about to call the same tool again,
    and those are the arguments to add. The full schema is deliberately not
    rendered — the tool's own parameter schema already describes them, and a
    second, differently shaped description in prose is how a model ends up
    inventing a nested object.
    """
    if not exc.schema:
        return str(exc)
    names: str = ", ".join(f"`{name}`" for name in exc.schema)
    return f"{exc} Call this tool again, additionally supplying: {names}."


def _refusal_message(exc: ActionUnavailable) -> str:
    """The sentence a refused call fails with: the reason, then the rule's code.

    ``The books are closed. (code: books_closed)``. drf-services raises
    ``ActionUnavailable`` when one of a spec's ``Affordance`` conditions is not
    met, carrying that affordance's ``reason`` as the message and its ``code`` as
    an attribute, and asks a transport serving an agent to pass on both. Here the
    ``ToolFailed`` message is the only thing a model receives and the only thing a
    transport forwards, so both go into it.

    **The code is what connects the refusal to what the model already read.** A
    selector tool's rows carry ``affordances: {<name>: {"available": false,
    "code": ..., "reason": ...}}``, and the reason is a sentence a project may
    reword between releases; the code is the part that stays put. Labelled rather
    than bare so a reader does not take ``books_closed`` for more of the sentence,
    and last so the reason still reads first.

    **Written for the model and for a person reading a tool card, not for a
    program.** The format is a convention across the family's agent transports
    rather than this toolset's own -- django-pydantic-agent's drf-mcp bridge is
    meant to render the same refusal as the same text, so changing it here means
    changing it there. A program that branches on the code should read ``.code``
    off the exception instead, either in
    [`translate_exception`][rest_framework_pydantic_ai.SpecToolset.translate_exception]
    or on the ``ToolFailed``'s ``__cause__``. Parsing it back out of the sentence
    couples that program to wording this function is free to change.
    """
    return f"{exc.message} (code: {exc.code})"


def _format_validation_detail(detail: Any) -> str:
    """A DRF or service validation detail as text a model can act on.

    ``str(exc.detail)`` is what the retry carried before, and for a DRF error that
    is the Python repr of its ``ErrorDetail`` objects --
    ``{'name': [ErrorDetail(string='This field is required.', code='required')]}``
    -- which spends the model's attention on a class name and a ``code`` it has no
    use for, around the one sentence it needs.

    One line per message, each prefixed with the path to the argument it is about:
    ``name: This field is required.``, a nested serializer's field as
    ``address.city: ...``, a position in a list as ``tags[0]: ...``. A message not
    about any one argument -- a bare string, a list of strings, or DRF's
    ``non_field_errors`` -- is the line on its own, because printing that key
    would read as the name of an argument to fix. The ``code`` is dropped: it is
    for programs, and a program reads it off the exception in
    [`translate_exception`][rest_framework_pydantic_ai.SpecToolset.translate_exception].
    """
    return "\n".join(_validation_detail_lines(detail, path=""))


def _validation_detail_lines(detail: Any, *, path: str) -> Iterator[str]:
    """Walk a detail depth-first, yielding ``path: message`` for every leaf."""
    if isinstance(detail, Mapping):
        for key, value in detail.items():
            yield from _validation_detail_lines(value, path=_validation_detail_path(path, key))
    elif isinstance(detail, list | tuple):
        for index, value in enumerate(detail):
            # A message in a list belongs to the list's own path -- ``name: [a, b]``
            # is two messages about ``name`` -- while a nested structure is one
            # entry of several (a ``many=True`` serializer's rows, which DRF
            # aligns by position with ``{}`` for a valid one) and needs its
            # position to be found again.
            nested = isinstance(value, Mapping | list | tuple)
            yield from _validation_detail_lines(
                value, path=_validation_detail_path(path, index) if nested else path
            )
    else:
        yield f"{path}: {detail}" if path else str(detail)


def _validation_detail_path(path: str, key: Any) -> str:
    """Extend ``path`` by one step: ``[i]`` for a position, ``.name`` for a field.

    An ``int`` key is a position whether it came from a list or from the dict a
    DRF ``ListField`` / ``DictField`` keys its child errors by. The non-field key
    adds nothing, so its messages print against the object that holds them.
    """
    if isinstance(key, int):
        return f"{path}[{key}]"
    if key == api_settings.NON_FIELD_ERRORS_KEY:
        return path
    return f"{path}.{key}" if path else str(key)


def _render_rejection_message(names: Sequence[str], detail: Any, *, scope: str | None) -> str:
    """The retry for a render the caller's read-shaping values broke.

    For example ``"`fields` was rejected while rendering the result: Unknown
    field `items`."`` Names the argument first, because the detail alone -- the
    serializer's own wording -- says nothing about *which* argument the model has to
    change, and "while rendering the result" tells it the rest of the call was
    accepted.

    **Several supplied names are all named, joined with "or".** The serializer
    does not say which one it refused, so singling one out would be a guess
    presented as a fact, and "and ... were" would claim all of them were. The
    MCP transport words it the same way.

    On a paged tool the scope sentence follows, because the likeliest way to
    write a bad selection there is against the page envelope, which is exactly
    the shape the tool's result documents -- saying so turns that into one retry.
    ``scope`` is that sentence as the conventions word it, ``None`` on a tool that
    does not page or where the conventions dropped it.
    """
    quoted = [f"`{name}`" for name in names]
    subject = quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} or {quoted[-1]}"
    message = (
        f"{subject} was rejected while rendering the result: {_format_validation_detail(detail)}"
    )
    # A serializer's message may or may not carry a full stop (DRF's own do,
    # django-restql's do not); the sentence should end in exactly one either way.
    if not message.endswith((".", "!", "?")):
        message += "."
    if scope is not None:
        message = f"{message} {scope}"
    return message


def _pop_pagination(
    spec: Spec,
    args: dict[str, Any],
    *,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
) -> _PageArgs | None:
    """Strip + validate the list-selector args this toolset owns.

    The tool schema advertises ``page`` / ``limit`` as integers, but the
    toolset's argument validator is a no-op (the schema is advisory), so a model
    that sends ``limit="2"`` reaches here untyped. Coerce and validate rather
    than letting a ``TypeError`` abort the run, mapping a bad value to
    ``ModelRetry``.

    **The coercion stays here rather than moving to drf-services' shaper, which
    takes both values already parsed.** That is the one place the two agent
    transports legitimately differ: an MCP server answering a public endpoint has
    to clamp a malformed value and serve *something*, while an in-process toolset
    can hand the model its own mistake back and get a corrected call. Clamping is
    about a page's bounds; this is about bad input, and only the second is a
    policy.

    Clamping ``limit`` to ``max_page_size`` is therefore **not** done here any
    more: ``paginate_output(max_page_size=…)`` applies it at the slice, where the
    ``totalPages`` the caller is told about is computed from the same number. Two
    clamps meant two places for that number to be, and only one of them was ever
    reported back.

    **``ordering`` is left entirely alone for a spec that advertises it**, so the
    value reaches the ``filter_set`` — or the selector callable — that declared
    it. Nothing sorts from here. For a spec that advertises none, the argument is
    popped and refused: it was never offered, so a model that sent one is told
    that rather than having the value fall through to the spec and come back as a
    generic unknown-argument rejection.
    """
    if not _is_list_selector(spec):
        return None
    # Popped in a fixed order (limit, ordering, page) so that a call carrying two
    # bad arguments is corrected on the same one every time.
    limit: int | None = _coerce_positive_int(args.pop("limit", None), "limit")
    if _spec_ordering_argument(spec, pool_seeds=pool_seeds, registry=registry) is None:
        _refuse_unadvertised_ordering(args.pop("ordering", None))
    return _PageArgs(page=_coerce_positive_int(args.pop("page", None), "page"), limit=limit)


def _pop_filter_ordering(
    spec: Spec,
    args: dict[str, Any],
    *,
    pool_seeds: PoolSeeds = DEFAULT_POOL_SEEDS,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
) -> dict[str, Any] | None:
    """Route a filter-owned ``ordering`` out of the callable's args into filter data.

    Returns the mapping ``dispatch_spec(filter_data=…)`` should hand the
    ``filter_set``, or ``None`` to leave the default alone (``params`` is the
    filter source, which is what every other call wants).

    **``filter_data`` replaces ``params`` as the filter source rather than
    adding to it**, so the returned mapping carries the remaining args too;
    returning ``{"ordering": …}`` on its own would silently drop every other
    filter the model supplied.

    **Popped, not left in ``params``.** ``ordering`` is the FilterSet's
    argument, not the selector callable's: leaving it in ``params`` would reach
    the FilterSet by the same route, but it would also land in the callable's
    kwarg pool, where a selector declaring ``**kwargs`` receives it as a surprise
    argument it never asked for.

    **The two checks here are not redundant with each other.**
    :func:`_spec_ordering_argument` answers *what the spec calls* its sort; the
    ``filter_set`` check answers *what to hand it to*. A ``filter_set``
    advertised it, so the FilterSet is the consumer, and it reads ``filter_data``
    rather than the callable's arguments. When only the selector's own signature
    advertised it there is no FilterSet to reach, so the value stays in
    ``params`` — popping it would starve the one thing that asked for it.

    **The name is asked for rather than assumed.** A FilterSet whose
    ``OrderingFilter`` is called anything other than ``ordering`` used to fall
    through here, leaving its value in the callable's kwarg pool — the exact
    surprise the paragraph above says this function exists to prevent.
    """
    if not isinstance(spec, SelectorSpec) or spec.filter_set is None:
        return None
    name = _spec_ordering_argument(spec, pool_seeds=pool_seeds, registry=registry)
    if name is None or name not in args:
        return None
    ordering: Any = args.pop(name)
    return {**args, name: ordering}


def _declares_default(default: Any) -> bool:
    """Whether a ``QueryParam`` / ``UrlKwarg`` actually declares a default.

    Tolerates both sentinels the sister package has used for "no default": plain
    ``None`` up to drf-services 0.43, and ``UNSET`` from 0.44, where the change was
    made so that ``default=None`` could mean an explicit null. Checking only
    ``is not None`` reads ``UNSET`` as a real value and hands the sentinel object
    to the spec as an argument -- and, for a ``required=True`` kwarg, satisfies the
    requiredness check with it. Written to accept either so this package keeps
    working across that boundary without a floor raise.
    """
    return default is not None and default is not UNSET


def _supplied_query_params(
    query_params: Sequence[QueryParam], args: Mapping[str, Any]
) -> list[str]:
    """The declared read-shaping names the caller actually sent a value for.

    In declaration order, so a message naming several reads the same way on
    every call. A ``None`` is not a value: ``{"fields": null}`` is the shape a
    model emits for a param it chose not to fill, and ``QueryParam`` documents
    that a transport treats it as omitted.
    """
    return [qp.name for qp in query_params if args.get(qp.name) is not None]


def _pop_query_params(query_params: Sequence[QueryParam], args: dict[str, Any]) -> dict[str, Any]:
    """Strip the registered query params from ``args`` into a plain ``dict``.

    A declared param the model supplied is popped; one it omitted contributes its
    ``default`` if set, else nothing. The result is handed to
    ``build_offline_context(query_params=…)`` (which stringifies as on HTTP).

    **An explicit ``None`` is omitted, not forwarded.** ``QueryParam`` documents
    it that way -- ``{"fields": null}`` is how a model says it chose not to fill
    the param, and the ``default`` still applies -- and the stringifying above
    is why it matters: forwarded, the null reached the serializer as the four
    characters ``None``, which a strict selection parser refuses as malformed,
    ending a run over an argument the model had declined to send. Popped either
    way, so ``unknown_arguments`` never sees the key.
    """
    values: dict[str, Any] = {}
    for query_param in query_params:
        supplied: Any = args.pop(query_param.name, None)
        if supplied is not None:
            values[query_param.name] = supplied
        elif _declares_default(query_param.default):
            values[query_param.name] = query_param.default
    return values


def _pop_url_kwargs(
    url_kwargs: Sequence[UrlKwarg],
    args: dict[str, Any],
    *,
    conventions: AgentConventions = _DEFAULT_CONVENTIONS,
) -> dict[str, Any]:
    """Strip the registered URL kwargs from ``args`` into a plain ``dict``.

    A declared kwarg the model supplied is popped; one it omitted contributes its
    ``default`` if set, else nothing. The result is handed to
    ``build_offline_context(kwargs=…)``.

    A kwarg registered ``required=True`` that the model omitted raises
    ``ModelRetry`` naming it, so the model gets a chance to supply it on the next
    turn: schema ``required`` is only a hint, and without this the run would fail
    deeper in, where the reason is far less legible. (Registration forbids
    ``required`` alongside a ``default``, so such a kwarg is never satisfiable
    from the declaration.)
    """
    values: dict[str, Any] = {}
    missing: list[str] = []
    for url_kwarg in url_kwargs:
        if url_kwarg.name in args:
            values[url_kwarg.name] = args.pop(url_kwarg.name)
        elif _declares_default(url_kwarg.default):
            values[url_kwarg.name] = url_kwarg.default
        elif url_kwarg.required:
            missing.append(url_kwarg.name)
    if missing:
        raise _missing_arguments(missing, conventions)
    return values


def _missing_arguments(names: Sequence[str], conventions: AgentConventions) -> ModelRetry:
    """The retry for a call that left out arguments it cannot run without.

    One sentence for both checks that ask -- a required ``UrlKwarg`` and a
    parameter the tool's schema requires -- so the model reads the same wording
    whichever it omitted, worded by ``conventions.missing_arguments``. Sorted, so
    a message naming several reads the same way on every call (the ``several``
    case of ``test_a_selector_call_missing_a_required_argument_is_handed_back``).
    The field cannot be ``None`` or blank: a retry has to say something.
    """
    return ModelRetry(conventions.missing_arguments.format(names=_listed(sorted(names))))


def _coerce_positive_int(value: Any, name: str) -> int | None:
    """Coerce a pagination arg to a positive int; ``ModelRetry`` on anything else.

    Accepts an ``int`` or an all-digit ``str`` (``"2"``); rejects booleans,
    floats, negatives, zero, and non-numeric strings.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass — never a valid count
        raise ModelRetry(f"`{name}` must be a positive integer.")
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, str) and value.strip().isdigit():
        coerced = int(value)
    else:
        raise ModelRetry(f"`{name}` must be a positive integer.")
    if coerced < 1:
        raise ModelRetry(f"`{name}` must be a positive integer.")
    return coerced


def _refuse_unadvertised_ordering(value: Any) -> None:
    """``ModelRetry`` when a model sends a sort to a tool that advertises none.

    Reached only for a list selector whose reflected schema carries no sort
    argument — a spec that advertises one keeps its value, which travels on to
    whatever declared it. So the argument was genuinely never offered and the
    model invented it, and saying exactly that is more use than the generic
    unknown-argument rejection the value would otherwise collect at dispatch.

    **A deliberate divergence from the MCP transport**, which silently ignores an
    ordering value it does not recognise. For a model that is the worst outcome
    available: it asked for newest-first, received insertion order, and has no
    way to find out.
    """
    if value is not None:
        raise ModelRetry("This tool does not accept an `ordering` argument; omit it.")


def _shape_list(
    value: Any, *, page: int | None, limit: int | None, max_page_size: int | None
) -> OutputPage:
    """Paginate a list selector's queryset and force it to evaluate.

    The slicing itself is ``paginate_output``, drf-services' shared shaper, which
    also counts the rows and clamps both bounds. This function is what is left
    once that moved out: force evaluation.

    **No sorting happens here.** Ordering belongs to whatever the spec declared
    it on, which has already applied it to the (lazy) queryset by the time this
    runs; a second ``order_by`` from the transport would replace that sort rather
    than compose with it.

    Forces evaluation (``list(...)``) so that *nothing downstream* holds a lazy
    queryset once this returns: a serializer re-evaluating one would run the
    query a second time, and the envelope's counts and its rows could then come
    from two different reads. ``replace`` rather than mutation because
    ``OutputPage`` is frozen, and here rather than at the call site because the
    guarantee belongs to whatever produces the page.
    """
    shaped: OutputPage = paginate_output(
        value,
        page=page,
        limit=limit,
        max_page_size=max_page_size,
    )
    return replace(shaped, items=list(shaped.items))


def _render_output(
    spec: Spec,
    value: Any,
    *,
    projection: AudienceProjection | None,
    many: bool,
    request: Any,
    view: Any,
    extras: dict[str, Any],
) -> Any:
    """Render a dispatch result for the model to read.

    The default is drf-services' ``render_for_audience`` -- ``render_spec_output``
    plus the serializer's audience markings -- with the projection this toolset
    built once at registration rather than derived per call.

    **Passing ``projection=None`` is not an opt-out**: ``render_for_audience``
    reads it as "derive one from the spec", which projects anyway and costs a
    serializer instantiation. The way out is to render with ``render_spec_output``
    instead, which is the whole reason this step is a seam -- see
    [`SpecToolset.render_output`][rest_framework_pydantic_ai.SpecToolset.render_output].
    """
    return render_for_audience(
        spec,
        value,
        projection=projection,
        many=many,
        request=request,
        view=view,
        extras=extras,
    )


def _output_extras(
    spec: Spec, value: Any, *, many: bool, dispatch_result: DispatchResult | None = None
) -> dict[str, Any]:
    """The resolved-data keyword a spec's output-context provider may declare.

    ``dispatch_result`` is accepted and ignored: the pool is keyed the way the
    HTTP path keys it, and a key the HTTP path does not supply would be handed to
    a provider that never declared it. It is in the signature so this function
    keeps satisfying the seam's contract, and so an override calling ``super()``
    can forward what it was given rather than filtering the call.
    """
    del dispatch_result  # see above: deliberately not in the default pool
    if many:
        return {"page": value}
    if isinstance(spec, ServiceSpec):
        return {"result": value}
    return {"instance": value}


__all__ = ["SpecToolset"]
