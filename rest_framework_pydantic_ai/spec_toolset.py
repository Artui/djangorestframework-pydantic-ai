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
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, cast

from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.http import HttpRequest
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_core import SchemaValidator, core_schema
from rest_framework.exceptions import PermissionDenied
from rest_framework.exceptions import ValidationError as DRFValidationError
from rest_framework_services import (
    DEFAULT_JSON_SCHEMA_REGISTRY,
    DEFAULT_PAGE_SIZE,
    UNSET,
    AdditionalInputRequired,
    AudienceProjection,
    FieldAudience,
    JsonSchemaRegistry,
    OfflineContract,
    OutputPage,
    SelectorKind,
    SelectorSpec,
    ServiceError,
    ServiceSpec,
    ServiceValidationError,
    SpecRegistry,
    UnknownArguments,
    audience_projection_for_spec,
    build_offline_context,
    dispatch_spec,
    enforce_permissions,
    output_to_json_schema,
    paginate_output,
    render_for_audience,
    spec_to_json_schema,
)
from rest_framework_services.dispatch.unguarded_specs import unguarded_specs
from rest_framework_services.types.progress_reporter import ProgressReporter
from rest_framework_services.types.validate_channel_names import validate_channel_names

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

Return the value the tool should return — ``{"error": …}`` for something the
model should report and stop on — or raise ``ModelRetry`` to
hand it back for another attempt. Raising anything else aborts the run, which is
the right answer for a genuine bug.
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

# Tool args a list selector accepts on top of its filter fields. Ordering is not
# here: it belongs to the spec whenever the spec's own schema advertises it (see
# ``_spec_ordering_argument``).
_LIST_PARAM_SCHEMA: dict[str, Any] = {
    "page": {
        "type": "integer",
        "minimum": 1,
        "description": "1-based page number.",
    },
    "limit": {
        "type": "integer",
        "minimum": 1,
        "description": (
            f"Maximum number of items per page. Defaults to {DEFAULT_PAGE_SIZE}; "
            "the result reports `totalPages` and `hasNext`."
        ),
    },
}
# ``page`` no longer says "requires `limit`", because it no longer does: every
# list result is a page, so an omitted ``limit`` is the default page size rather
# than "everything". The pair used to be advertised and then not honoured — the
# schema claimed pagination while the payload was a bare list — and the wording
# was the last place that claim was still qualified.


@dataclass(frozen=True)
class _PageArgs:
    """A list selector's stripped pagination tool args.

    Pagination only. Sorting is never carried here: the spec's own schema is what
    advertises a sort argument, and whatever declared it — a ``filter_set``'s
    ``OrderingFilter``, or the selector callable itself — is what applies it.
    """

    page: int | None
    limit: int | None


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
            derives from the specs. ``None`` derives it.
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
            ``{"error": …}`` — never truncates, because a partial payload looks
            complete.
        tool_max_result_bytes: ``max_result_bytes`` per tool. An explicit
            ``None`` opts that tool out; an absent key inherits the default.
        max_page_size: Clamps a list tool's ``limit`` *and* advertises the
            ceiling as JSON-Schema ``maximum``. Lowers the default page size
            with it, so an omitted ``limit`` becomes the ceiling rather than
            ``DEFAULT_PAGE_SIZE``. Unset, a list tool still returns at most
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

    Raises:
        ImproperlyConfigured: A spec has no ``permission_classes`` and
            ``require_permissions`` is set.
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
            _validate_channel_declarations(name, self._tool_query_params[name], "query_params")
            _validate_channel_declarations(name, self._tool_url_kwargs[name], "url_kwargs")
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
            )
            for name, spec in self._specs.items()
        }

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

        Override [`is_tool_listed`][rest_framework_pydantic_ai.SpecToolset.is_tool_listed]
        when a deployment does want a narrower catalog.
        """
        return {
            name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self._max_retries,
                args_validator=_TOOL_ARGS_VALIDATOR,
            )
            for name, tool_def in self._tool_defs.items()
            if await self.is_tool_listed(name, ctx)
        }

    async def is_tool_listed(self, name: str, ctx: RunContext[Any]) -> bool:
        """Whether ``name`` belongs in this run's catalog. ``True`` by default.

        The seam for a deployment that wants a per-run catalog — hiding a
        staff-only tool from a non-staff run, or scoping the catalog to a
        tenant read off ``ctx.deps``. Hiding a tool is a *disclosure* decision,
        never an authorization one: the call is gated by
        ``spec.permission_classes`` whatever this returns, so an override that
        wrongly returns ``True`` grants nothing.

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
        ``{"error": …}`` result (a final answer, not a reason to retry) while a
        bad argument comes back as a retry request, and that a permission error
        is final. Pydantic-AI appends the block to the system prompt each turn,
        for a toolset attached directly *or* wrapped by a capability.

        Returns:
            The ``instructions`` override when one was given, else a block
            derived from the specs — each line conditional on something in this
            toolset being able to act on it, so the prompt carries no advice that
            cannot fire.
        """
        if self._instructions_override is not None:
            return self._instructions_override
        return _derive_instructions(
            self._specs,
            self._tool_query_params,
            self._projections,
            registry=self._json_schema_registry,
        )

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
                    self._call_spec,
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
                ),
                self._dispatch_timeout,
                label=name,
            )
        except PermissionDenied:
            # The one failure with no other trace: a ``ModelRetry`` reaches the
            # model and a ``{"error": …}`` reaches the answer, but a denial
            # aborts the run and is absorbed by whatever drives it. Logged at
            # the boundary, then re-raised untouched.
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
        self, spec: Spec, value: Any, *, ctx: RunContext[Any], many: bool
    ) -> dict[str, Any]:
        """The resolved-data pool a spec's output-context provider may read.

        Keyed the same way the HTTP path keys it, deliberately: ``result`` for a
        service, ``instance`` for a selector, ``page`` for a list. A provider
        written against one transport therefore reads the same names under the
        other.

        Note what is **not** here, because it is the question this seam attracts:
        drf-services' ``DispatchResult.service_result`` -- the flags carrier an
        upsert's ``created`` rides on. The HTTP path does not put it in this pool
        either; it feeds a callable ``success_status`` and a ``response_finalizer``,
        both of which are status-code machinery a toolset has no wire for. A spec
        whose *model-visible* outcome depends on such a flag should put the flag in
        its output serializer, where both transports can see it. An override here
        is the escape hatch for a project that disagrees.
        """
        del ctx  # unused by the default; present so an override has it
        return _output_extras(spec, value, many=many)

    def enforce_result_bytes(
        self, payload: Any, *, ctx: RunContext[Any], max_bytes: int | None, label: str
    ) -> Any:
        """Return ``payload``, or a model-readable refusal when it is over budget.

        Measured on the envelope, because the envelope is what is sent. Override to
        bound on something other than serialized length -- a row count, a per-run
        running total read off ``ctx.usage`` -- or to shape the refusal differently.
        """
        del ctx  # unused by the default; present so an override has it
        return _enforce_result_bytes(payload, max_bytes=max_bytes, label=label)

    def _call_spec(
        self, spec: Spec, user: Any, args: dict[str, Any], *, ctx: RunContext[Any], **kw: Any
    ) -> Any:
        """Bind the six seams to this run, then run the shared pipeline.

        Separate from the module-level function of the same name because that one
        has to stay usable without a toolset.
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
            **kw,
        )


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


def _validate_channel_declarations(tool_name: str, declarations: Sequence[Any], kind: str) -> None:
    """Apply drf-services' shared channel checks to one tool's merged tuple.

    Those cover the dispatcher's pool seeds (``request`` / ``user`` / ``data`` /
    …, which a caller must not be able to route a value onto) and the
    contradiction of ``required=True`` with a ``default``. The transport-side
    names stay ours to contribute, belonging to the adapter rather than the
    dispatcher: the ``page`` / ``limit`` / ``ordering`` the MCP transport
    reserves too.
    """
    validate_channel_names(
        label=f"SpecToolset tool {tool_name!r}",
        kind=kind,
        declarations=declarations,
        reserved=_RESERVED_PARAM_NAMES,
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


async def _with_deadline(awaitable: Any, seconds: float | None, *, label: str) -> Any:
    """Await ``awaitable``, answering the model instead of hanging past ``seconds``.

    ``None`` awaits without a deadline, so the resolved bound goes straight in.

    **This does not stop the work.** The dispatch runs in a ``sync_to_async``
    thread and asyncio cannot interrupt a thread parked in a database driver's
    socket read, so the query runs to completion regardless; what the deadline
    buys is a terminal answer rather than a run that never returns.

    That answer is an ``{"error": …}`` rather than an exception, for the same
    reason the byte ceiling's is: the model can respond to it by asking for less,
    and killing the run denies it the chance.
    """
    if seconds is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except (TimeoutError, asyncio.TimeoutError):
        # The operator hears about it too: a tool that intermittently answers
        # "took too long" and logs nothing is indistinguishable from a bug.
        logger.warning("Tool %r exceeded its %.1fs dispatch timeout", label, seconds)
        return {
            "error": (
                f"This call took longer than the {seconds:g}s limit and was "
                "abandoned. Narrow the request — add or tighten a filter, or "
                "lower `limit` — and call again."
            )
        }


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


# The conventions block ``SpecToolset.get_instructions`` teaches the model.
_BASE_INSTRUCTIONS = (
    "The following tools call Django REST Framework services and selectors.\n"
    "- A successful call returns the tool's data. A business-rule failure returns a JSON "
    'object like {"error": "..."} — that is a final answer explaining why the operation '
    "could not complete; read it and report it, do not retry the same call.\n"
    "- An invalid or missing argument comes back as a retry request naming the problem; "
    "correct the argument and call again.\n"
    "- A permission error is final: the current user may not perform that call — do not "
    "retry it.\n"
    "- Only pass documented parameters; unknown arguments are rejected."
)

_LIST_INSTRUCTION = (
    "- Read-only tools that return a collection always return one page, shaped "
    '{"items": [...], "page": 1, "totalPages": N, "hasNext": true|false}. They accept optional '
    f"`limit` (items per page, default {DEFAULT_PAGE_SIZE}) and `page` (1-based). When "
    "`hasNext` is true there are more items than you were shown: ask for the next `page`, or "
    "narrow the request with a filter — never answer as if the page were the whole collection."
)

_HANDLE_INSTRUCTION = (
    "- Some tools return opaque identifier fields, described as such in the tool's "
    "output. Pass them to other tools that ask for one; refer to records by their "
    "name in anything you say, never by the identifier."
)

_HANDLE_DESCRIPTION = (
    "An opaque identifier. Pass it to other tools that ask for one; refer to the record "
    "by its name in anything you say, never by this value."
)
"""Fallback wording for a ``HANDLE`` field that declares no description of its own.

drf-services deliberately supplies none: what a reader should *do* with an
identifier depends on the reader, and only the transport knows its audience.
Ours is a model reading a tool's output schema, so the sentence is the per-field
half of ``_HANDLE_INSTRUCTION`` — same advice, at the field that needs it.
"""


def _ordering_instruction(names: Sequence[str]) -> str:
    """The sort-usage line, naming the arguments the toolset actually advertises.

    **Parameterised because the name is not fixed.** The prose used to say
    ``ordering`` outright, which was a second hard-coding of the same assumption
    the dispatch made — so a toolset whose ``OrderingFilter`` is called
    ``sorting`` would have been handed an instruction naming an argument that
    does not exist, which is worse than no instruction at all.

    Several names can be live at once: one tool's filter may be ``ordering`` and
    another's ``sorting``, and a model reading one block for the whole toolset
    has to be told both.
    """
    listed = ", ".join(f"`{name}`" for name in names)
    return (
        f"- Some collection tools also accept {listed}. It takes exactly one of the values "
        "listed in that tool's schema (a sortable name, or the same name prefixed with `-` for "
        "descending) — not a comma-separated list, and not an arbitrary column."
    )


def _has_handle(projection: AudienceProjection) -> bool:
    """Whether any field on this tool's output is an opaque identifier."""
    return any(marking.audience is FieldAudience.HANDLE for marking in projection.fields.values())


def _derive_instructions(
    specs: Mapping[str, Spec],
    tool_query_params: Mapping[str, Sequence[QueryParam]],
    projections: Mapping[str, AudienceProjection] | None = None,
    *,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
) -> str:
    """Build the conventions block from the specs / query params / ordering.

    Every line is conditional on something being able to act on it: pagination
    only with a list selector present, ordering only where some tool actually has
    a sort argument, read-shaping only with a ``QueryParam``. Advice a model
    cannot use is not neutral — it is budget spent teaching it about an argument
    that will be rejected.

    **The specs are the only source of an ordering argument**, so this asks them
    and nothing else: whatever the schema advertises is what the model may send.
    """
    lines = [_BASE_INSTRUCTIONS]
    if any(_is_list_selector(spec) for spec in specs.values()):
        lines.append(_LIST_INSTRUCTION)
    # Deduplicated, in first-seen order: one toolset can carry several tools
    # whose sorts are declared under different names.
    ordering_names: list[str] = []
    for spec in specs.values():
        advertised = _spec_ordering_argument(spec, registry=registry)
        if advertised is not None and advertised not in ordering_names:
            ordering_names.append(advertised)
    if ordering_names:
        lines.append(_ordering_instruction(ordering_names))
    # A toolset with no handle anywhere gains nothing from being told how to
    # treat one, and this block is prepended to every run.
    if any(_has_handle(projection) for projection in (projections or {}).values()):
        lines.append(_HANDLE_INSTRUCTION)
    query_param_names = sorted({qp.name for params in tool_query_params.values() for qp in params})
    if query_param_names:
        joined = ", ".join(f"`{name}`" for name in query_param_names)
        lines.append(
            f"- Some tools accept read-shaping parameters ({joined}) that adjust the shape "
            "of the returned data without filtering it."
        )
    return "\n".join(lines)


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
            spec, query_params, url_kwargs, max_page_size, registry=registry
        ),
        return_schema=_return_schema(spec, projection=projection, registry=registry),
        metadata={"annotations": {"readOnlyHint": isinstance(spec, SelectorSpec)}},
    )


def _return_schema(
    spec: Spec,
    *,
    projection: AudienceProjection | None = None,
    registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY,
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

    ``None`` for a spec with no ``output_serializer`` is the correct answer and
    not a gap: drf-services refuses to fabricate a shape it cannot derive, and a
    guessed one would be a claim the payload never has to honour.
    """
    serializer, kind = _output_serializer_and_kind(spec)
    return output_to_json_schema(
        serializer,
        kind=kind,
        paginate=_is_list_selector(spec),
        projection=projection,
        handle_description=_HANDLE_DESCRIPTION,
        registry=registry,
    )


def _output_serializer_and_kind(spec: Spec) -> tuple[type | None, SelectorKind | None]:
    """Where a spec keeps its output serializer, and under which kind.

    A selector holds both itself; a service holds them one level down on its
    ``output_selector_spec``. drf-services exposes ``output_serializer_for`` for
    the first half but nothing for the pair, and the ``kind`` is what decides
    between an item and a collection — so the two are read together here rather
    than deriving one from the spec and the other from an assumption.
    """
    if isinstance(spec, SelectorSpec):
        return spec.output_serializer, spec.kind
    nested = spec.output_selector_spec
    return (None, None) if nested is None else (nested.output_serializer, nested.kind)


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
) -> dict[str, Any]:
    """The tool's parameter schema, with list-selector pagination + registered
    query params + URL kwargs merged into ``properties``.

    **No sort argument is contributed here.** A tool that can sort says so in its
    own reflected schema — a ``filter_set``'s ``OrderingFilter``, or an
    ``ordering`` parameter on the selector callable — and that reflected property
    passes straight through. Writing one into ``extra`` would land it *over* the
    reflected properties in the merge below, replacing a FilterSet's public
    choices with a second vocabulary the FilterSet would then reject.

    ``spec_to_json_schema(phase="input")`` always returns a dict (only the
    output phase is nullable), so the result is narrowed for the type-checker.
    The registered declarations are merged **over** the reflected properties, so
    an explicit ``UrlKwarg`` for a key drf-services already reflected (from a
    selector's ``Unpack[TypedDict]``) wins — it is the intentional one.

    The reflected ``required`` list is preserved and *extended* by any
    ``UrlKwarg(required=True)``. A key that is both reflected-required and
    registered-required appears once: that is one statement made twice, not two
    requirements.
    """
    schema = cast("dict[str, Any]", spec_to_json_schema(spec, phase="input", registry=registry))
    extra: dict[str, Any] = {}
    if _is_list_selector(spec):
        extra.update(_LIST_PARAM_SCHEMA)
        if max_page_size is not None:
            # Advertised as well as clamped: a schema with no ``maximum`` invites
            # a request for 100 000 rows, and telling the model is cheaper than
            # correcting it.
            extra["limit"] = {**_LIST_PARAM_SCHEMA["limit"], "maximum": max_page_size}
    extra.update({qp.name: qp.json_schema() for qp in query_params})
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


def _is_list_selector(spec: Spec) -> bool:
    return isinstance(spec, SelectorSpec) and spec.kind == SelectorKind.LIST


def _spec_ordering_argument(
    spec: Spec, *, registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY
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
    """
    if not _is_list_selector(spec):
        return None
    reflected = cast("dict[str, Any]", spec_to_json_schema(spec, phase="input", registry=registry))
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
    build_context: _ContextBuilder = build_offline_context,
    translate_exception: _ExceptionTranslator | None = None,
    shape_page: _PageShaper | None = None,
    render_output: _OutputRenderer | None = None,
    output_extras: _ExtrasBuilder | None = None,
    enforce_result_bytes: _ResultBounder | None = None,
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

    ``action`` becomes ``view.action`` on the synthetic view, so a permission
    class reading it sees the tool name rather than ``None``.
    """
    shape: _PageShaper = shape_page or _shape_list
    render: _OutputRenderer = render_output or _render_output
    extras_for: _ExtrasBuilder = output_extras or _output_extras
    bound: _ResultBounder = enforce_result_bytes or _enforce_result_bytes
    page_args = _pop_pagination(spec, args, registry=json_schema_registry)
    # Both channels pop before dispatch so their values never reach the spec as
    # inputs, where ``unknown_arguments`` (REJECT by default) would flag them.
    # That is what makes the provider-only case work: a ``project_pk`` a scoping
    # provider reads off ``view.kwargs`` is never a spec input.
    query_param_values = _pop_query_params(query_params, args)
    url_kwarg_values = _pop_url_kwargs(url_kwargs, args)
    # Last of the pops, because the filter data it returns is built from whatever
    # ``args`` is left holding once every other channel has taken its own.
    filter_data = _pop_filter_ordering(spec, args, registry=json_schema_registry)
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
            on_target_resolved=enforce_permissions,
            # Accepted and forwarded, never constructed — see ``AgentDeps.progress``.
            # ``None`` becomes drf-services' no-op seed.
            progress=progress,
            # ``None`` for every call but a filter-owned ordering, where it is the
            # channel that separates the FilterSet's data from the callable's
            # arguments — the two are one flat mapping off HTTP otherwise.
            filter_data=filter_data,
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
            # detail.
            raise ModelRetry(str(exc.detail)) from exc
        if isinstance(exc, AdditionalInputRequired):
            # **Must precede the ``ServiceError`` case below** — this is a
            # subclass of it, and the generic handler would report a request for
            # input as a terminal failure. ``ModelRetry`` is already the "here is
            # what to fix, call me again" channel, so the answer comes back as an
            # ordinary argument on the next call.
            raise ModelRetry(_missing_input_prompt(exc)) from exc
        if isinstance(exc, ServiceError):
            return {"error": str(exc)}
        raise

    if result.kind == "not_found":
        return {"error": "not found"}

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
    rendered = render(
        spec,
        value,
        projection=projection,
        many=many,
        request=context.request,
        view=context.view,
        extras=extras_for(spec, value, many=many),
    )
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


def _enforce_result_bytes(payload: Any, *, max_bytes: int | None, label: str) -> Any:
    """Return ``payload``, or a model-readable refusal when it is over budget.

    **Fails; never truncates.** A list cut at the byte ceiling is
    indistinguishable from a list that had that many rows, so a model would
    answer confidently from data it does not know is missing.

    Returned as an ``{"error": …}`` result rather than raised as
    ``ModelRetry``: the model should narrow the request and it *can*, but a
    retry budget is finite and a run should not die because a model spent it on
    progressively smaller queries. Also logged at ``WARNING``, since a bound that
    fires invisibly reads to an operator as "the tool is broken".
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
    )
    return {
        "error": (
            f"This result was {size} bytes, over the {max_bytes} byte ceiling. "
            "Narrow the request — add or tighten a filter, lower `limit`, or "
            "select fewer fields — and call again. The result was not "
            "truncated: a partial payload would look complete."
        )
    }


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


def _pop_pagination(
    spec: Spec, args: dict[str, Any], *, registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY
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
    if _spec_ordering_argument(spec, registry=registry) is None:
        _refuse_unadvertised_ordering(args.pop("ordering", None))
    return _PageArgs(page=_coerce_positive_int(args.pop("page", None), "page"), limit=limit)


def _pop_filter_ordering(
    spec: Spec, args: dict[str, Any], *, registry: JsonSchemaRegistry = DEFAULT_JSON_SCHEMA_REGISTRY
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
    name = _spec_ordering_argument(spec, registry=registry)
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


def _pop_query_params(query_params: Sequence[QueryParam], args: dict[str, Any]) -> dict[str, Any]:
    """Strip the registered query params from ``args`` into a plain ``dict``.

    A declared param the model supplied is popped; one it omitted contributes its
    ``default`` if set, else nothing. The result is handed to
    ``build_offline_context(query_params=…)`` (which stringifies as on HTTP).
    """
    values: dict[str, Any] = {}
    for query_param in query_params:
        if query_param.name in args:
            values[query_param.name] = args.pop(query_param.name)
        elif _declares_default(query_param.default):
            values[query_param.name] = query_param.default
    return values


def _pop_url_kwargs(url_kwargs: Sequence[UrlKwarg], args: dict[str, Any]) -> dict[str, Any]:
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
        names = ", ".join(f"`{name}`" for name in sorted(missing))
        raise ModelRetry(f"Missing required argument(s): {names}.")
    return values


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


def _output_extras(spec: Spec, value: Any, *, many: bool) -> dict[str, Any]:
    """The resolved-data keyword a spec's output-context provider may declare."""
    if many:
        return {"page": value}
    if isinstance(spec, ServiceSpec):
        return {"result": value}
    return {"instance": value}


__all__ = ["SpecToolset"]
