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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from asgiref.sync import sync_to_async
from django.core.exceptions import FieldError, ImproperlyConfigured
from django.http import HttpRequest
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_core import SchemaValidator, core_schema
from rest_framework.exceptions import PermissionDenied
from rest_framework.exceptions import ValidationError as DRFValidationError
from rest_framework_services import (
    AdditionalInputRequired,
    SelectorKind,
    SelectorSpec,
    ServiceError,
    ServiceSpec,
    ServiceValidationError,
    SpecRegistry,
    UnknownArguments,
    build_offline_context,
    dispatch_spec,
    enforce_permissions,
    render_spec_output,
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


def _resolve_specs(specs: SpecSource) -> Mapping[str, Spec]:
    """Normalise a ``SpecSource`` to the plain mapping the internals expect."""
    return specs.specs() if isinstance(specs, SpecRegistry) else specs


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
# ``_spec_owns_ordering``).
_LIST_PARAM_SCHEMA: dict[str, Any] = {
    "page": {
        "type": "integer",
        "minimum": 1,
        "description": "1-based page number (requires `limit`).",
    },
    "limit": {
        "type": "integer",
        "minimum": 1,
        "description": "Maximum number of items to return.",
    },
}


@dataclass(frozen=True)
class _PageArgs:
    """A list selector's stripped pagination tool args.

    ``ordering`` is ``None`` whenever the spec advertises ordering itself;
    ``_shape_list`` reads it, so a filter-owned value never reaches
    ``queryset.order_by`` from here.
    """

    page: int | None
    limit: int | None
    ordering: str | None


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
    list selector's ``page`` / ``limit`` args merged in), and the
    ``readOnlyHint`` annotation is derived from the spec kind (selectors read,
    services mutate).

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
            read. A filtered view is itself a registry, so several toolsets can
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
        tool_url_kwargs: ``url_kwargs`` for one tool only, same override rule.
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
            ceiling as JSON-Schema ``maximum``. With it set, an omitted ``limit``
            becomes the ceiling rather than "everything" — the unbounded read is
            the one that hurts, and it is what a model produces by not thinking
            about pagination.
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
        ordering_fields: **Deprecated** second ordering vocabulary, kept for a
            list selector with no ``filter_set`` and therefore no other route. It
            declares what such a tool may sort by, advertised as an enum on an
            ``ordering`` argument and validated against it; the values are raw
            ORM paths, because the toolset applies them with
            ``queryset.order_by``. Nothing declared means no ``ordering``
            argument at all. Declaring it for a tool whose spec already
            advertises ``ordering`` raises: public filter choices and ORM paths
            are two vocabularies for one argument name, and quietly preferring
            either is how a schema and its dispatch come to disagree.
        tool_ordering_fields: ``ordering_fields`` per tool. Per-tool *replaces*
            the toolset-wide set rather than merging with it.
        http_request: The ``HttpRequest`` the off-HTTP context is built from.
            **Incidental request data, never an auth channel:** it exists so a
            serializer or scoping provider reading ``request.META`` finds
            something plausible. The acting identity is the user, and passing an
            authenticated request authorizes nothing.
        get_http_request: ``http_request`` resolved per run from ``RunContext``,
            the way ``get_user`` is. Wins over a static ``http_request``.
        exception_map: Maps an exception type to a handler returning the tool's
            result (or raising ``ModelRetry``). Matched along
            the MRO, most specific first, and consulted **before** the built-in
            arms, so a project can override those too.

    Raises:
        ImproperlyConfigured: A spec has no ``permission_classes`` and
            ``require_permissions`` is set.
        ValueError: A tool name is outside ``^[a-zA-Z0-9_-]{1,64}$``, a per-tool
            mapping names a tool this toolset does not expose, one name is
            registered on both parameter channels, or a tool declares ordering
            through both ``ordering_fields`` and its ``filter_set``.
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
        ordering_fields: Sequence[str] = (),
        tool_ordering_fields: Mapping[str, Sequence[str]] | None = None,
        http_request: HttpRequest | None = None,
        get_http_request: HttpRequestExtractor | None = None,
        exception_map: Mapping[type[BaseException], ExceptionHandler] | None = None,
    ) -> None:
        resolved = _resolve_specs(specs)
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
        self._host = host
        self._max_retries = max_retries
        self._max_page_size = max_page_size
        self._dispatch_timeout = dispatch_timeout
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
        self._tool_query_params: dict[str, tuple[QueryParam, ...]] = {
            name: _merge_query_params(query_params, (tool_query_params or {}).get(name, ()))
            for name in self._specs
        }
        self._tool_url_kwargs: dict[str, tuple[UrlKwarg, ...]] = {
            name: _merge_url_kwargs(url_kwargs, (tool_url_kwargs or {}).get(name, ()))
            for name in self._specs
        }
        self._tool_ordering_fields: dict[str, tuple[str, ...]] = {
            name: tuple((tool_ordering_fields or {}).get(name, ordering_fields))
            for name in self._specs
        }
        _validate_ordering_fields(self._specs, self._tool_ordering_fields)
        _validate_no_param_channel_overlap(self._tool_query_params, self._tool_url_kwargs)
        # Checked on the **merged** tuples, not the raw declarations: a
        # toolset-wide and a per-tool entry of the same name are an intentional
        # override, which the shared check would read as a duplicate in the
        # pre-merge concatenation. Post-merge is also what reaches the schema.
        for name in self._specs:
            _validate_channel_declarations(name, self._tool_query_params[name], "query_params")
            _validate_channel_declarations(name, self._tool_url_kwargs[name], "url_kwargs")
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
                self._tool_ordering_fields[name],
                self._max_page_size,
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
        return {
            name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self._max_retries,
                args_validator=_TOOL_ARGS_VALIDATOR,
            )
            for name, tool_def in self._tool_defs.items()
        }

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
            self._specs, self._tool_query_params, self._tool_ordering_fields
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
                sync_to_async(self._call_spec)(
                    spec,
                    user,
                    dict(tool_args),
                    ctx=ctx,
                    unknown_arguments=self._unknown_arguments,
                    query_params=self._tool_query_params[name],
                    url_kwargs=self._tool_url_kwargs[name],
                    ordering_fields=self._tool_ordering_fields[name],
                    max_page_size=self._max_page_size,
                    max_result_bytes=self._tool_max_result_bytes[name],
                    label=name,
                    progress=self._get_progress(ctx),
                    host=self._host,
                ),
                self._dispatch_timeout,
                label=name,
            )
        except PermissionDenied:
            # The one failure with no other trace: a ``ModelRetry`` reaches the
            # model and a ``{"error": …}`` reaches the answer, but a denial
            # aborts the run and is absorbed by whatever drives it. Logged at
            # the boundary, then re-raised untouched.
            logger.warning("Permission denied calling tool %r on toolset %r", name, self._id)
            raise
        logger.debug(
            "Tool %r on toolset %r took %.1f ms",
            name,
            self._id,
            (time.perf_counter() - started) * 1000,
        )
        return result

    # The two seams into the middle of a call. Deliberately not offered beside
    # them: a generic "set arbitrary attributes on the synthetic request"
    # parameter, which would make ambient-state-on-the-request the default
    # posture for everyone. An override keeps the honest path the easy one.

    def build_context(
        self,
        user: Any,
        params: Mapping[str, Any],
        *,
        ctx: RunContext[Any],
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
        """
        return build_offline_context(
            user,
            params,
            http_request=self._get_http_request(ctx),
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

    def _call_spec(
        self, spec: Spec, user: Any, args: dict[str, Any], *, ctx: RunContext[Any], **kw: Any
    ) -> Any:
        """Bind the two seams to this run, then run the shared pipeline.

        Separate from the module-level function of the same name because that one
        has to stay usable without a toolset.
        """
        return _call_spec(
            spec,
            user,
            args,
            build_context=lambda *a, **kwargs: self.build_context(*a, ctx=ctx, **kwargs),
            translate_exception=lambda exc: self.translate_exception(exc, ctx=ctx),
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


def _validate_ordering_fields(
    specs: Mapping[str, Spec],
    tool_ordering_fields: Mapping[str, Sequence[str]],
) -> None:
    """Refuse a tool that declares ordering twice, then deprecate the second way.

    **Two vocabularies cannot share one argument name.** A FilterSet's
    ``OrderingFilter`` speaks *public* names it maps through its own
    ``param_map`` — several of which resolve to annotation aliases — while
    ``ordering_fields`` values are raw ORM paths, because the toolset applies
    them with ``queryset.order_by``. One enum would overwrite the other in the
    schema, and whichever lost would be advertised to the model and then refused
    at dispatch, so the combination fails at configuration time instead.
    """
    clashing: list[str] = sorted(
        name
        for name, fields in tool_ordering_fields.items()
        if fields and _spec_owns_ordering(specs[name])
    )
    if clashing:
        names: str = ", ".join(repr(name) for name in clashing)
        raise ValueError(
            f"SpecToolset tool(s) {names} declare ordering twice: through "
            "ordering_fields / tool_ordering_fields, and through a filter_set that "
            "already advertises an `ordering` argument in the tool schema. Those are "
            "two vocabularies for one argument — a FilterSet's OrderingFilter exposes "
            "public choices it maps itself, ordering_fields values are raw ORM paths — "
            "and only one can be advertised. Drop ordering_fields for these tool(s); "
            "the FilterSet's OrderingFilter already advertises ordering."
        )
    declared: list[str] = sorted(name for name, fields in tool_ordering_fields.items() if fields)
    if declared:
        names = ", ".join(repr(name) for name in declared)
        warnings.warn(
            "SpecToolset's ordering_fields / tool_ordering_fields are deprecated and "
            "will be removed in a future release. Declare a django-filter "
            "OrderingFilter named `ordering` on the selector's filter_set instead: it "
            "is the one place ordering is declared, its choices are already reflected "
            "into the tool schema, and it validates and applies the value itself. "
            f"Tool(s) still declaring ordering_fields: {names}.",
            DeprecationWarning,
            stacklevel=3,
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
    "- Read-only tools that return a collection accept optional `page` and `limit`: `limit` "
    "caps the number of items and `page` (1-based, requires `limit`) selects the page."
)

_ORDERING_INSTRUCTION = (
    "- Some collection tools also accept `ordering`. It takes exactly one of the values "
    "listed in that tool's schema (a sortable name, or the same name prefixed with `-` for "
    "descending) — not a comma-separated list, and not an arbitrary column."
)


def _derive_instructions(
    specs: Mapping[str, Spec],
    tool_query_params: Mapping[str, Sequence[QueryParam]],
    tool_ordering_fields: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Build the conventions block from the specs / query params / ordering.

    Every line is conditional on something being able to act on it: pagination
    only with a list selector present, ordering only where some tool actually has
    an ``ordering`` argument, read-shaping only with a ``QueryParam``. Advice a
    model cannot use is not neutral — it is budget spent teaching it about an
    argument that will be rejected.

    **Both sources of an ordering argument count**, so a tool whose
    ``filter_set`` advertises ``ordering`` is covered as well as one declaring
    ``tool_ordering_fields``.
    """
    lines = [_BASE_INSTRUCTIONS]
    if any(_is_list_selector(spec) for spec in specs.values()):
        lines.append(_LIST_INSTRUCTION)
    if any((tool_ordering_fields or {}).values()) or any(
        _spec_owns_ordering(spec) for spec in specs.values()
    ):
        lines.append(_ORDERING_INSTRUCTION)
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
    ordering_fields: Sequence[str] = (),
    max_page_size: int | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema=_input_schema(
            spec, query_params, url_kwargs, ordering_fields, max_page_size
        ),
        metadata={"annotations": {"readOnlyHint": isinstance(spec, SelectorSpec)}},
    )


def _spec_description(spec: Spec) -> str | None:
    """The tool description: the docstring of the spec's selector / service."""
    callable_ = spec.selector if isinstance(spec, SelectorSpec) else spec.service
    return inspect.getdoc(callable_) if callable_ is not None else None


def _input_schema(
    spec: Spec,
    query_params: Sequence[QueryParam] = (),
    url_kwargs: Sequence[UrlKwarg] = (),
    ordering_fields: Sequence[str] = (),
    max_page_size: int | None = None,
) -> dict[str, Any]:
    """The tool's parameter schema, with list-selector pagination + registered
    query params + URL kwargs merged into ``properties``.

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
    schema = cast("dict[str, Any]", spec_to_json_schema(spec, phase="input"))
    extra: dict[str, Any] = {}
    if _is_list_selector(spec):
        extra.update(_LIST_PARAM_SCHEMA)
        if max_page_size is not None:
            # Advertised as well as clamped: a schema with no ``maximum`` invites
            # a request for 100 000 rows, and telling the model is cheaper than
            # correcting it.
            extra["limit"] = {**_LIST_PARAM_SCHEMA["limit"], "maximum": max_page_size}
        if ordering_fields and not _spec_owns_ordering(spec):
            # **Never written over a reflected ``ordering``.** ``extra`` is
            # merged over ``properties`` below, so writing here for a spec that
            # advertises its own would replace the filter's public choices with
            # ORM paths under the same property name — an enum the FilterSet
            # would then reject. The constructor refuses that combination; this
            # is the same rule at the one place the overwrite could happen.
            extra["ordering"] = {
                "enum": _ordering_values(ordering_fields),
                "description": "Sort order. Prefix a field with `-` for descending.",
            }
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


def _spec_owns_ordering(spec: Spec) -> bool:
    """True when a list spec's own reflected schema already advertises ``ordering``.

    **The one signal, read by both the schema builder and the dispatch path, so
    they cannot disagree** — the invariant being *whatever the schema advertises,
    the dispatch must deliver*.

    **Deliberately not an ``isinstance`` test against ``django_filters``.**
    django-filter is an optional extra the package never imports — a spec's
    ``filter_set`` is duck-typed all the way down — and the property that matters
    is what the tool *advertised*, not which library produced it. A list selector
    whose callable happens to declare an ``ordering`` parameter is owned for the
    same reason.

    Restricted to list selectors because that is the only kind the toolset ever
    contributes an ``ordering`` argument to: elsewhere there is no ownership to
    contest, and a service whose input serializer happens to have a field named
    ``ordering`` must not be read as a clash.
    """
    if not _is_list_selector(spec):
        return False
    reflected = cast("dict[str, Any]", spec_to_json_schema(spec, phase="input"))
    return "ordering" in reflected.get("properties", {})


def _call_spec(
    spec: Spec,
    user: Any,
    args: dict[str, Any],
    *,
    unknown_arguments: UnknownArguments = UnknownArguments.REJECT,
    query_params: Sequence[QueryParam] = (),
    url_kwargs: Sequence[UrlKwarg] = (),
    ordering_fields: Sequence[str] = (),
    max_page_size: int | None = None,
    max_result_bytes: int | None = None,
    label: str = "",
    progress: ProgressReporter | None = None,
    host: str | None = None,
    build_context: _ContextBuilder = build_offline_context,
    translate_exception: _ExceptionTranslator | None = None,
) -> Any:
    """Run ``spec`` under an off-HTTP context and render the result.

    Synchronous on purpose — ``SpecToolset.call_tool`` runs it in a thread so
    the ORM stays off the event loop.

    ``build_context`` and ``translate_exception`` are the two seams
    [`SpecToolset`][rest_framework_pydantic_ai.SpecToolset] threads its
    overridable methods through; the defaults
    keep this function usable on its own.
    """
    page_args = _pop_pagination(spec, args, ordering_fields, max_page_size)
    # Both channels pop before dispatch so their values never reach the spec as
    # inputs, where ``unknown_arguments`` (REJECT by default) would flag them.
    # That is what makes the provider-only case work: a ``project_pk`` a scoping
    # provider reads off ``view.kwargs`` is never a spec input.
    query_param_values = _pop_query_params(query_params, args)
    url_kwarg_values = _pop_url_kwargs(url_kwargs, args)
    # Last of the pops, because the filter data it returns is built from whatever
    # ``args`` is left holding once every other channel has taken its own.
    filter_data = _pop_filter_ordering(spec, args)
    context = build_context(
        user,
        args,
        kwargs=url_kwarg_values or None,
        query_params=query_param_values or None,
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
    if page_args is not None:
        try:
            value = _shape_list(value, page_args)
        except FieldError as exc:
            raise ModelRetry(f"invalid ordering: {exc}") from exc
    many = result.kind == "list"
    rendered = render_spec_output(
        spec,
        value,
        many=many,
        request=context.request,
        view=context.view,
        extras=_output_extras(spec, value, many=many),
    )
    return _enforce_result_bytes(rendered, max_result_bytes, label=label)


def _enforce_result_bytes(payload: Any, max_bytes: int | None, *, label: str) -> Any:
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
    spec: Spec,
    args: dict[str, Any],
    ordering_fields: Sequence[str] = (),
    max_page_size: int | None = None,
) -> _PageArgs | None:
    """Strip + validate the list-selector args this toolset owns.

    The tool schema advertises ``page`` / ``limit`` as integers and ``ordering``
    as an enum, but the toolset's argument validator is a no-op (the schema is
    advisory), so a model that sends ``limit="2"`` or ``ordering=["a"]`` reaches
    here untyped. Coerce and validate rather than letting a ``TypeError`` abort
    the run, mapping a bad value to ``ModelRetry``.

    **``ordering`` is left entirely alone for a spec that advertises it**, so the
    value reaches the ``filter_set`` that declared it. For a spec that does not,
    the pop is unconditional: the argument was never offered, so a model that
    sent one is corrected here rather than having the value fall through to the
    spec as an unknown argument.
    """
    if not _is_list_selector(spec):
        return None
    limit: int | None = _coerce_positive_int(args.pop("limit", None), "limit")
    if max_page_size is not None:
        # Clamped rather than rejected: a model that did not think to paginate is
        # exactly the caller that needs the ceiling applied for it.
        limit = max_page_size if limit is None else min(limit, max_page_size)
    ordering: str | None = None
    if not _spec_owns_ordering(spec):
        ordering = _coerce_ordering(args.pop("ordering", None), ordering_fields)
    return _PageArgs(
        page=_coerce_positive_int(args.pop("page", None), "page"),
        limit=limit,
        ordering=ordering,
    )


def _pop_filter_ordering(spec: Spec, args: dict[str, Any]) -> dict[str, Any] | None:
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
    ``_spec_owns_ordering`` answers *whether* the spec owns ``ordering``; the
    ``filter_set`` check answers *what to hand it to*. A ``filter_set``
    advertised it, so the FilterSet is the consumer, and it reads ``filter_data``
    rather than the callable's arguments. When only the selector's own signature
    advertised it there is no FilterSet to reach, so the value stays in
    ``params`` — popping it would starve the one thing that asked for it.
    """
    if not isinstance(spec, SelectorSpec) or spec.filter_set is None:
        return None
    if not _spec_owns_ordering(spec) or "ordering" not in args:
        return None
    ordering: Any = args.pop("ordering")
    return {**args, "ordering": ordering}


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
        elif query_param.default is not None:
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
        elif url_kwarg.default is not None:
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


def _coerce_ordering(value: Any, allowed: Sequence[str]) -> str | None:
    """Validate ``ordering`` against the declared enum; ``ModelRetry`` otherwise.

    **A deliberate divergence from the MCP transport**, which silently ignores an
    ordering value it does not recognise. For a model that is the worst outcome
    available: it asked for newest-first, received oldest-first, and has no way
    to find out. A retry naming the accepted values is self-correcting.
    """
    if value is None:
        return None
    values: list[str] = _ordering_values(allowed)
    if not values:
        # Never advertised, so the model invented it; saying so beats a "must be
        # one of: " with an empty list after it. Reached only for a spec that
        # advertises no ``ordering`` of its own — a filter-owned one never
        # arrives here, which is what makes this answer true when it is given.
        raise ModelRetry("This tool does not accept an `ordering` argument; omit it.")
    options: str = ", ".join(f"`{v}`" for v in values)
    if value not in values:
        raise ModelRetry(f"`ordering` must be one of: {options}; got {value!r}.")
    return value


def _ordering_values(fields: Sequence[str]) -> list[str]:
    """Each declared field, ascending then descending — the advertised enum.

    Same construction as the MCP transport's ``ordering_fields``, so a project
    exposing one registry over both surfaces advertises one vocabulary.
    """
    values: list[str] = []
    for field in fields:
        values.append(field)
        values.append(f"-{field}")
    return values


def _shape_list(value: Any, page_args: _PageArgs) -> list[Any]:
    """Paginate a list selector's queryset, ordering it only when ordering is ours.

    Forces evaluation (``list(...)``) so a ``FieldError`` surfaces here — where
    ``_call_spec`` turns it into a ``ModelRetry`` — rather than later inside the
    serializer. That covers both ordering routes, since a FilterSet's own
    ``order_by`` is applied to the lazy queryset upstream and only raises once
    this materializes it.

    ``page_args.ordering`` is ``None`` whenever the spec advertises ``ordering``
    itself, so the sort a FilterSet already applied is never applied twice, in a
    second vocabulary, over the top of it.
    """
    queryset = value
    if page_args.ordering:
        queryset = queryset.order_by(page_args.ordering)
    return list(_paginate(queryset, page_args.page, page_args.limit))


def _paginate(queryset: Any, page: int | None, limit: int | None) -> Any:
    if limit is None:
        return queryset
    offset = ((page or 1) - 1) * limit
    return queryset[offset : offset + limit]


def _output_extras(spec: Spec, value: Any, *, many: bool) -> dict[str, Any]:
    """The resolved-data keyword a spec's output-context provider may declare."""
    if many:
        return {"page": value}
    if isinstance(spec, ServiceSpec):
        return {"result": value}
    return {"instance": value}


__all__ = ["SpecToolset"]
