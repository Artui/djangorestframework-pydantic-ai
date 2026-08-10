"""``SpecToolset`` — expose drf-services specs as a Pydantic-AI toolset.

A thin adapter that turns a ``name -> spec`` mapping into agent tools, executing
each call through drf-services' transport-neutral surface — ``dispatch_spec``
plus its off-HTTP helpers (``build_offline_context`` / ``enforce_permissions`` /
``spec_to_json_schema`` / ``render_spec_output``). There is **no MCP server and
no AG-UI bridge** in the path: a plain ``pydantic_ai.Agent`` calls the specs
in-process.

The acting identity flows through ``RunContext.deps``: by default the toolset
reads ``ctx.deps.user`` (the :class:`~rest_framework_pydantic_ai.types.agent_deps.AgentDeps`
shape); projects that thread identity differently pass a ``get_user`` extractor.

Each call mirrors what a DRF view does, in order:

1. strip a list selector's ``page`` / ``limit`` / ``ordering`` tool args (ordering
   and pagination are transport concerns, kept off the spec) plus any registered
   :class:`~rest_framework_pydantic_ai.QueryParam` args (read-shaping query
   params that seed ``request.query_params``, not spec inputs);
2. build the off-HTTP context (synthetic request + view + principal, with the
   registered query params seeded into ``request.query_params``);
3. **enforce ``spec.permission_classes``** — ``dispatch_spec`` deliberately does
   not, so a naive adapter would skip authorization;
4. dispatch the spec, then render the result through the spec's serializer.

Error semantics map drf-services' failure kinds onto Pydantic-AI's model-loop:

- input validation errors (DRF ``ValidationError`` from the input serializer, or
  drf-services' ``ServiceValidationError`` from a service) →
  :class:`pydantic_ai.ModelRetry`, so the model self-corrects with the field
  errors instead of the run dying;
- business ``ServiceError`` and an unresolved instance (``not_found``) → a
  model-readable ``{"error": ...}`` payload;
- an ``ordering`` value outside the declared enum → ``ModelRetry`` (naming the
  values that are accepted);
- a denied ``permission_classes`` check raises ``PermissionDenied`` and aborts
  the run, exactly as it would over HTTP.
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
# Anywhere a ``name -> spec`` mapping is accepted, a ``SpecRegistry`` is too —
# it *is* that mapping plus tags and ordering, and ``registry.specs()`` hands
# back the plain dict. Widening the parameter beats a ``from_registry``
# constructor, which would have to restate every keyword of the signature it
# forwards to (and then drift from it).
SpecSource = Mapping[str, Spec] | SpecRegistry
UserExtractor = Callable[[RunContext[Any]], Any]
ProgressExtractor = Callable[[RunContext[Any]], ProgressReporter | None]
HttpRequestExtractor = Callable[[RunContext[Any]], HttpRequest | None]

_ContextBuilder = Callable[..., Any]
_ExceptionTranslator = Callable[[BaseException], "ExceptionHandler | None"]

ExceptionHandler = Callable[[BaseException], Any]
"""Turns one exception into a tool result.

Return the value the tool should return — ``{"error": …}`` for something the
model should report and stop on — or raise :class:`pydantic_ai.ModelRetry` to
hand it back for another attempt. Raising anything else aborts the run, which is
the right answer for a genuine bug.
"""

logger = logging.getLogger("rest_framework_pydantic_ai")
"""The package's one logger.

⚠ **A toolset call is invisible without this, and it is the call that most
needs to be visible.** A dispatch behind a DRF view leaves an access-log line;
the same spec called by a model leaves nothing — no method, no path, no status
— so "the agent did something odd" has no record to check it against. That is
also where a permission denial lands: over HTTP it is a ``403`` in the log,
here it is an exception the run loop absorbs into a message.

Timings go to ``DEBUG`` (one line per call, which is noise in production and
exactly what you want while tuning), denials to ``WARNING``. Named for the
package so ``LOGGING`` can set the two independently of everything else.
"""


def _resolve_specs(specs: SpecSource) -> Mapping[str, Spec]:
    """Normalise a ``SpecSource`` to the plain mapping the internals expect."""
    return specs.specs() if isinstance(specs, SpecRegistry) else specs


# List-selector pagination args own these names; a registered ``QueryParam`` or
# ``UrlKwarg`` may not shadow them.
_RESERVED_PARAM_NAMES = frozenset({"page", "limit", "ordering"})

# Tool names are surfaced verbatim to the model provider, which constrains them
# to this shape (OpenAI / Anthropic function-name rules). Validated at
# construction so a bad key fails fast instead of at the provider boundary.
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Tool args pass through unvalidated: the parameter schemas advertised to the
# model come from ``spec_to_json_schema`` (advisory, not a Pydantic model), and
# the real validation is the spec's own input serializer at dispatch time — so
# the per-tool validator is a no-op, exactly the double-validation split a DRF
# view has (JSON parsing at the transport, field validation in the serializer).
_TOOL_ARGS_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())

# Tool args a list selector accepts on top of its filter fields. Ordering and
# pagination stay transport-side, so the adapter — not the spec — exposes them
# and slices the result.
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
    """A list selector's stripped ordering / pagination tool args."""

    page: int | None
    limit: int | None
    ordering: str | None


class SpecToolset(AbstractToolset[Any]):
    """Exposes drf-services specs as a Pydantic-AI toolset.

    Build it from a ``name -> spec`` mapping and hand it to an ``Agent``::

        toolset = SpecToolset({
            "list_orders": orders_selector_spec,   # SelectorSpec -> read-only tool
            "create_order": create_order_spec,     # ServiceSpec  -> mutation tool
        })
        agent = Agent(model, deps_type=AgentDeps, toolsets=[toolset])

    Each key becomes one tool: the description is the spec's selector/service
    docstring, the parameter schema comes from ``spec_to_json_schema`` (with a
    list selector's ``page`` / ``limit`` / ``ordering`` args merged in), and the
    ``readOnlyHint`` annotation is derived from the spec kind (selectors read,
    services mutate).

    ``specs`` also accepts a
    :class:`~rest_framework_services.registry.spec_registry.SpecRegistry`
    (drf-services 0.27+) — the shared declaration site for a project that
    exposes the same specs over more than one transport, so the agent reads
    the same source MCP and the HTTP views do instead of repeating the list::

        toolset = SpecToolset(registry)                     # every entry
        read_only = SpecToolset(registry.by_tag("read"), id="reads")

    A filtered view is itself a registry, so several toolsets can be projected
    from one declaration with no shared state. Names come from the registry;
    everything else on this signature stays per-toolset, because it is
    transport-specific and the registry deliberately carries none of it.

    ``get_user`` overrides how the acting identity is read off the run context;
    it defaults to ``ctx.deps.user``.

    ``unknown_arguments`` controls what happens to tool args outside a spec's
    declared input set — a hallucinated key the model invented. It defaults to
    :attr:`~rest_framework_services.UnknownArguments.REJECT`, which surfaces the
    unexpected key as a :class:`pydantic_ai.ModelRetry` so the model
    self-corrects; specs whose declared set is open (a ``filter_set`` or
    ``**kwargs`` selector) are unaffected. Pass ``IGNORE`` to silently drop them
    or ``PASSTHROUGH`` to forward them to the callable.

    ``query_params`` / ``tool_query_params`` register read-shaping
    :class:`~rest_framework_pydantic_ai.QueryParam` args that seed
    ``request.query_params`` over the off-HTTP path — the extensible generalization
    of ``page`` / ``limit`` / ``ordering``. ``query_params`` applies to **every** tool;
    ``tool_query_params`` maps a tool name to params for that tool only (a per-tool
    param overrides a toolset-wide one of the same name). Each is advertised as a
    tool arg, then popped at call time and handed to
    ``build_offline_context(query_params=…)`` — never to the spec as an input, so
    ``unknown_arguments`` never sees it. This is for whatever reads
    ``request.query_params`` **directly** — django-restql field selection, a custom
    serializer branching on the query string — with zero toolset awareness of the
    specific library.

    A ``SelectorSpec.filter_set`` does **not** need this: its fields are already
    generated into the tool's input schema by ``spec_to_json_schema`` and flow
    through as ordinary ``params`` (which ``dispatch_spec`` hands the FilterSet as
    ``filter_data``), so the model can filter with no extra declaration.

    ``url_kwargs`` / ``tool_url_kwargs`` register
    :class:`~rest_framework_pydantic_ai.UrlKwarg` args — URL route captures
    (``parent_pk``) seeded into ``build_offline_context(kwargs=…)`` and spread by
    drf-services into the selector / target pools, authoritative over ``params``.
    Same shape as ``query_params`` / ``tool_query_params`` (toolset-wide + per-tool,
    per-tool overriding by name), advertised as a tool arg then popped at call
    time. Use them for a URL-derived value **not** already in the tool schema — a
    scoping ``spec.kwargs`` provider that reads ``view.kwargs`` (the case
    ``params`` alone cannot cover), or a closed-surface route capture. A selector
    that reads the value from its ``**extras: Unpack[TypedDict]`` needs none —
    drf-services reflects the key and delivers it through ``params`` — though a
    key may be *both* reflected and ``UrlKwarg``-registered (the ``UrlKwarg``
    schema wins, and the authoritative ``kwargs=`` spread still reaches the
    selector). A name cannot be both a ``QueryParam`` and a ``UrlKwarg`` on the
    same tool (a value cannot route to two channels).

    ``host`` gives the synthesized request an origin, so ``build_absolute_uri``
    builds real absolute URLs — DRF's ``FileField`` and the ``Hyperlinked*``
    fields call it for every value once a ``request`` is in the serializer
    context, which off the HTTP path it always is. Accepts ``"example.com"``,
    ``"example.com:8000"``, or a full origin like ``"https://example.com"``
    whose scheme decides whether links are https. There is no default and none is
    inferred: only the project knows its public origin, and guessing one would
    emit confidently-wrong links that look valid. Left unset, those fields
    produce **relative** URLs — usable, and what they fall back to on their own
    when no request is in the context. Toolset-wide only, with no per-tool
    variant: an origin is a property of the deployment, not of a tool.

    ``max_retries`` is each tool's retry budget: how many times a
    :class:`pydantic_ai.ModelRetry` (a validation failure, a bad ``ordering``
    field) is fed back to the model before the run aborts with
    ``UnexpectedModelBehavior``. Defaults to ``1``, matching pydantic-ai's own
    function-tool default.

    ``instructions`` overrides the auto-derived conventions block that
    :meth:`get_instructions` teaches the model (pagination + error contract);
    pass a string to replace it, or leave it ``None`` to derive from the specs.

    **Safety and bounds.**

    ``require_permissions`` (default ``True``) refuses to construct a toolset
    containing a spec with no ``permission_classes``. Over HTTP that means
    *inherit*; here there is nothing to inherit from, so it means *ungated*.
    Pass ``False`` to downgrade to an :class:`UnguardedSpecWarning` while
    migrating.

    ``descriptions`` overrides ``spec.description`` per tool — the docstring an
    API developer reads is rarely the sentence a model needs — and a tool left
    with no description anywhere gets an :class:`UndescribedToolWarning`.

    ``ordering_fields`` / ``tool_ordering_fields`` declare what a list tool may
    sort by, advertised as an enum on an ``ordering`` argument and validated
    against it. Nothing declared means no ``ordering`` argument at all; per-tool
    *replaces* the toolset-wide set rather than merging with it.

    ``max_result_bytes`` / ``tool_max_result_bytes`` cap a rendered result,
    measured on the encoded payload because the thing being protected is the
    model's context window. Over the ceiling the call **fails** with a
    model-readable ``{"error": …}`` — never truncates, because a partial payload
    looks complete. A per-tool ``None`` opts that tool out; a per-tool key
    simply absent inherits the toolset default.

    ``max_page_size`` clamps a list tool's ``limit`` *and* advertises the
    ceiling as JSON-Schema ``maximum``. With it set, an omitted ``limit``
    becomes the ceiling rather than "everything" — the unbounded read is the one
    that hurts, and it is what a model produces by not thinking about
    pagination.

    **Extending dispatch.**

    ``http_request`` / ``get_http_request`` supply the ``HttpRequest`` the
    off-HTTP context is built from — static, or resolved per run from
    ``RunContext`` the way ``get_user`` is. ⛔ **Incidental request data, never
    an auth channel:** it exists so a serializer or scoping provider that reads
    ``request.META`` finds something plausible. The acting identity is ``user``,
    and passing an authenticated request authorizes nothing.

    ``exception_map`` maps an exception type to a handler returning the tool's
    result (or raising :class:`pydantic_ai.ModelRetry`). Registrations are
    matched along the MRO — most specific wins — and are consulted **before**
    the built-in arms, so a project can override those too.

    For anything these do not cover, :meth:`build_context` and
    :meth:`translate_exception` are overridable and both receive the live
    ``RunContext``, which is how per-run typed deps reach dispatch.

        ``dispatch_timeout`` (seconds) bounds one call, answering the model instead
    of hanging. ⚠ It does not *stop* the work: the dispatch runs in a
    ``sync_to_async`` thread and asyncio cannot interrupt a thread parked in a
    database driver's socket read, so the query runs to completion regardless.
    Pair it with a database statement timeout for the other half.
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
        # A static request and a per-run extractor are the same knob at two
        # lifetimes, so one resolves to the other rather than living beside it:
        # ``http_request=`` becomes an extractor returning it, and a supplied
        # ``get_http_request=`` wins outright. Same shape as ``get_user``.
        self._get_http_request: HttpRequestExtractor = get_http_request or (
            (lambda ctx: http_request) if http_request is not None else _default_get_http_request
        )
        self._exception_map: dict[type[BaseException], ExceptionHandler] = dict(exception_map or {})
        self._unknown_arguments: UnknownArguments = unknown_arguments
        self._host = host
        self._max_retries = max_retries
        self._max_page_size = max_page_size
        self._dispatch_timeout = dispatch_timeout
        # ``UNSET`` is spelled here as "key absent": a per-tool ``None`` is a
        # deliberate "no ceiling for this one", which is not the same as saying
        # nothing and inheriting the toolset default.
        overrides: Mapping[str, int | None] = tool_max_result_bytes or {}
        for tool_name in overrides:
            if tool_name not in self._specs:
                raise ValueError(
                    f"tool_max_result_bytes references unknown tool {tool_name!r}; "
                    f"known tools: {sorted(self._specs)}."
                )
        self._tool_max_result_bytes: dict[str, int | None] = {
            # ``.get`` with the toolset default: a stored ``None`` wins over it,
            # which is exactly the absent-vs-explicit distinction above.
            name: overrides.get(name, max_result_bytes)
            for name in self._specs
        }
        # The effective (deduped) query params for each tool: toolset-wide first,
        # then per-tool overriding by name. Built once — declarations are static.
        self._tool_query_params: dict[str, tuple[QueryParam, ...]] = {
            name: _merge_query_params(query_params, (tool_query_params or {}).get(name, ()))
            for name in self._specs
        }
        # Same shape for URL kwargs (the ``kwargs=`` channel).
        self._tool_url_kwargs: dict[str, tuple[UrlKwarg, ...]] = {
            name: _merge_url_kwargs(url_kwargs, (tool_url_kwargs or {}).get(name, ()))
            for name in self._specs
        }
        # Ordering is opt-in and per-tool, matching the MCP transport: a field
        # nobody declared is not orderable, so a model cannot ask a list tool to
        # sort by an arbitrary column (an unindexed one is a table scan, and a
        # non-existent one is a wasted turn). Per-tool *replaces* rather than
        # merges — the sensible sort keys for `list_invoices` and `list_users`
        # have nothing to do with each other, unlike a query param, which is
        # usually a cross-cutting concern.
        self._tool_ordering_fields: dict[str, tuple[str, ...]] = {
            name: tuple((tool_ordering_fields or {}).get(name, ordering_fields))
            for name in self._specs
        }
        _validate_no_param_channel_overlap(self._tool_query_params, self._tool_url_kwargs)
        # Run drf-services' shared checks on the **merged** tuples, not the raw
        # declarations: toolset-wide and per-tool entries of the same name are an
        # intentional override, and the shared check would read the pre-merge
        # concatenation as a duplicate. Post-merge is also what actually reaches
        # the schema, so it is the honest thing to validate.
        for name in self._specs:
            _validate_channel_declarations(name, self._tool_query_params[name], "query_params")
            _validate_channel_declarations(name, self._tool_url_kwargs[name], "url_kwargs")
        # Schemas derive purely from the specs (no DB), so the tool defs are
        # built once up front. ``ToolDefinition`` defaults to ``kind="function"``
        # — the in-process kind the run loop routes into ``call_tool``.
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
        is final. Pydantic-AI appends this block to the system prompt each turn —
        for a toolset attached directly *or* wrapped by a capability — so the
        model doesn't rediscover the conventions by failing a call.

        Returns the ``instructions`` override when one was given, else a block
        derived from the specs: the pagination line appears only when a list
        selector is present, and the read-shaping line only when some
        ``QueryParam`` is declared, keeping the prompt free of advice that can't
        fire.
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
            # The whole pipeline touches the ORM (validation, dispatch,
            # serializer rendering), which Django forbids on the async event
            # loop — run it in a thread. ``dict(tool_args)`` is a private copy
            # so popping pagination / query-param args never mutates the
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
            # ⚠ The one failure with no other trace. A ``ModelRetry`` reaches
            # the model and a ``{"error": …}`` reaches the answer, but a denial
            # aborts the run and is absorbed by whatever is driving it — so
            # without this line, "the agent said it couldn't do that" has
            # nothing behind it. Logged at the boundary, then re-raised
            # untouched.
            logger.warning("Permission denied calling tool %r on toolset %r", name, self._id)
            raise
        logger.debug(
            "Tool %r on toolset %r took %.1f ms",
            name,
            self._id,
            (time.perf_counter() - started) * 1000,
        )
        return result

    # ----- the overridable middle -----
    #
    # ⚠ Everything between argument intake and result rendering used to be
    # sealed inside a module-level private, so a subclass could replace
    # ``call_tool`` wholesale or nothing at all. These two methods are the seam.
    # Both receive the live ``RunContext``, which is the point: a project's
    # per-run typed deps — a tenant, a feature-flag snapshot — are what usually
    # need to reach dispatch, and they have no other route in.
    #
    # ⛔ **Deliberately not offered: a generic "set arbitrary attributes on the
    # synthetic request" parameter.** It was the obvious shortcut and it makes
    # ambient-state-on-the-request the default posture for everyone. An
    # override keeps the honest path the easy one.

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
        drf-services' :func:`build_offline_context`, resolving
        ``http_request`` through :attr:`get_http_request`.

        ⚠ **An ``http_request`` here is incidental request data — never an auth
        channel.** It is there so a serializer or a scoping provider that reads
        ``request.META`` / ``request.headers`` finds something plausible. The
        acting identity is ``user``, resolved from ``ctx.deps``, and nothing
        downstream re-derives it from the request; passing an authenticated
        request does **not** authorize anything, and relying on it would put a
        second, invisible identity in the call.
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
        **most specific** registration wins.

        ⭐ **The motivating case is one line long.** ``django.core.exceptions
        .ValidationError`` — Django's, not DRF's — is raised by ``full_clean``
        and by any custom model validator, is not in the translated set, and so
        kills the whole run where the DRF twin would have become a
        ``ModelRetry``. Broadening the built-in set would have fixed that one
        case; a map fixes the class of them, and lets a project override the
        built-in arms too when it knows better than this package does.
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

        Separate from :meth:`call_tool` because the pipeline is synchronous and
        runs in a thread; separate from the module-level function because that
        one has to stay usable without a toolset.
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

    ⚠ **The defect is not "``None`` means unguarded". It is "``None`` means
    *inherit*, and off HTTP there is nothing to inherit from."** Over HTTP,
    ``permission_classes=None`` is a correct, working configuration: the view's
    own ``permission_classes`` and DRF's ``DEFAULT_PERMISSION_CLASSES`` apply.
    A toolset has neither. So a spec that is properly guarded behind a viewset,
    with passing HTTP tests, becomes callable by whatever the agent decides to
    call the moment it is handed to a model — with no signal anywhere.

    The predicate itself lives in drf-services, which owns the off-HTTP
    dispatch semantics that create the asymmetry and is therefore the package
    that knows why ``None`` is ambiguous. It raises nothing and defaults
    nothing; the policy — this flag, this default, this wording — stays here.

    ⛔ **Defaults to refusing, matching the MCP transport.** Warning first and
    tightening a minor later would leave the hole open for the length of that
    minor and cost a second release to close, for consumers who would have to
    act either way.

    :exc:`~django.core.exceptions.ImproperlyConfigured` rather than the
    ``ValueError`` the checks below raise: this is a deployment misconfiguration
    rather than a coding error, it is what Django raises for the same shape of
    problem, and it is what the MCP transport raises for this exact check —
    a consumer running both catches one thing.
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
        # The way *out*, not the way in: the check has already raised, so
        # pointing at the flag that would have prevented it reads as a bug
        # report against the library.
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
    than by muting ``UserWarning`` across the process and losing everything
    else with it.
    """


class UndescribedToolWarning(UserWarning):
    """A spec was exposed as a tool with nothing to tell the model it exists for.

    Separate category from :class:`UnguardedSpecWarning` for the same reason,
    and because the two are silenced by different people: one is a security
    posture, the other is prompt quality.
    """


def _validate_descriptions(
    specs: Mapping[str, Spec],
    descriptions: Mapping[str, str] | None,
) -> dict[str, str]:
    """Resolve each tool's description, warning about the ones that say nothing.

    ⚠ **A missing description is not a cosmetic problem — it is the tool being
    invisible.** A model chooses tools almost entirely from their descriptions;
    one that has only a name is either never called or called for the wrong
    reason, and both failures look like "the agent is bad at this" rather than
    like a registration defect. It is also fixable exactly once, here, where
    every name is known.

    ``descriptions`` overrides ``spec.description`` per tool — the same spec
    serves an HTTP API whose docstring is written for a developer and an agent
    that needs to be told when to reach for it, and those are not the same
    sentence. A key naming a tool this toolset does not expose is a typo and
    raises, on the same reasoning as ``tool_query_params``: silently dropping it
    would leave the tool with the description the author thought they had
    replaced.

    Warning rather than raising, unlike the permission check: an undescribed
    tool degrades an answer, an unguarded one exposes data.
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

    Name-level checks (reserved names, duplicates) run **after** the merge, in
    :func:`_validate_channel_declarations` — see there for why post-merge is the
    honest place. This one has to run first, because the merge indexes by tool
    name and a typo'd key would otherwise be silently dropped.
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

    See :func:`_validate_query_params` — name-level checks run post-merge.
    """
    for tool_name in tool_url_kwargs or {}:
        if tool_name not in specs:
            raise ValueError(
                f"tool_url_kwargs references unknown tool {tool_name!r}; "
                f"known tools: {sorted(specs)}."
            )


def _validate_channel_declarations(tool_name: str, declarations: Sequence[Any], kind: str) -> None:
    """Apply drf-services' shared channel checks to one tool's merged tuple.

    Adds what the local checks above never covered: the dispatcher's pool seeds
    (``request`` / ``user`` / ``data`` / …, which a caller must not be able to
    route a value onto) and the contradiction of ``required=True`` with a
    ``default``. The pagination names stay ours to contribute — this toolset
    reserves ``order`` where the MCP transport reserves ``ordering``, which is
    exactly the kind of divergence that let the two adapters' copies of these
    types drift apart in the first place.
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

    ⚠ **This does not stop the work, and saying so matters.** The dispatch runs
    in a ``sync_to_async`` thread — which is where every ORM-backed spec spends
    its time — and asyncio cannot interrupt a thread parked in a database
    driver's socket read. The query runs to completion (or until the database
    ends it) regardless. What the deadline buys is a *terminal answer* for the
    model instead of a run that never returns; pair it with a database statement
    timeout for the other half.

    The timeout becomes an ``{"error": …}`` rather than an exception, for the
    same reason the byte ceiling does: it is a condition the model can respond
    to by asking for less, and killing the run denies it the chance.
    """
    if seconds is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except (TimeoutError, asyncio.TimeoutError):
        # The operator hears about it too — a tool that intermittently answers
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

    ``getattr`` rather than attribute access because a project with its own deps
    class predates this field, and a missing sink is the ordinary case — not
    something to make a caller declare their way out of.
    """
    return getattr(getattr(ctx, "deps", None), "progress", None)


def _default_get_http_request(ctx: RunContext[Any]) -> HttpRequest | None:
    """No request unless one was configured — the honest default.

    ⛔ **Not read off ``ctx.deps``, unlike the user and the progress sink.** A
    request that appears by default is one nothing declared, and the failure
    mode is silent: a serializer starts building absolute URLs against whatever
    host happened to be in scope. It arrives because a project asked for it.
    """
    del ctx
    return None


def _default_get_user(ctx: RunContext[Any]) -> Any:
    """Read the acting user off ``ctx.deps.user`` (the ``AgentDeps`` default)."""
    return ctx.deps.user


# The conventions block :meth:`SpecToolset.get_instructions` teaches the model.
# Kept out of per-tool descriptions because they describe the *family*'s behaviour
# (the error contract, pagination) rather than any one tool.
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
    "listed in that tool's schema (a field name, or the same name prefixed with `-` for "
    "descending) — not a comma-separated list, and not an arbitrary column."
)


def _derive_instructions(
    specs: Mapping[str, Spec],
    tool_query_params: Mapping[str, Sequence[QueryParam]],
    tool_ordering_fields: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Build the conventions block from the specs / query params / ordering.

    Every line is conditional on something being able to act on it: pagination
    only with a list selector present, ordering only where some tool declares
    fields, read-shaping only with a ``QueryParam``. Advice a model cannot use
    is not neutral — it is budget spent teaching it about an argument that will
    be rejected.
    """
    lines = [_BASE_INSTRUCTIONS]
    if any(_is_list_selector(spec) for spec in specs.values()):
        lines.append(_LIST_INSTRUCTION)
    if any((tool_ordering_fields or {}).values()):
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

    The reflected ``required`` list is preserved and *extended*: drf-services
    contributes keys the selector's own ``TypedDict`` marks ``InputRequired``,
    and a ``UrlKwarg(required=True)`` adds the ones declared here. A key that is
    both reflected-required and registered-required appears once — the two are
    the same statement made in two places, not two requirements.
    """
    schema = cast("dict[str, Any]", spec_to_json_schema(spec, phase="input"))
    extra: dict[str, Any] = {}
    if _is_list_selector(spec):
        extra.update(_LIST_PARAM_SCHEMA)
        if max_page_size is not None:
            # Advertised *and* clamped, because the two fail differently: a
            # schema with no ``maximum`` invites a request for 100 000 rows, and
            # a schema on its own is a hint that nothing obliges a model to
            # honour. Telling it is cheaper than correcting it.
            extra["limit"] = {**_LIST_PARAM_SCHEMA["limit"], "maximum": max_page_size}
        if ordering_fields:
            # An enum, not a free string: it is the only shape that tells the
            # model what it may sort by, and the only one this can validate
            # without asking the database.
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
    :class:`SpecToolset` threads its overridable methods through; the defaults
    keep this function usable on its own.
    """
    page_args = _pop_pagination(spec, args, ordering_fields, max_page_size)
    # Pop the registered query params out of the spec args and seed them into the
    # off-HTTP request's ``query_params`` (for whatever reads them directly —
    # restql, a custom serializer; not a ``filter_set``, which reads the spec
    # args as ``filter_data``). Popped first so they never reach the spec as
    # inputs, so ``unknown_arguments`` (REJECT by default) can't flag them.
    query_param_values = _pop_query_params(query_params, args)
    # Pop the registered URL kwargs into the off-HTTP view's ``kwargs`` (from
    # where drf-services spreads them into the selector / target pools). Popping
    # them out of the spec args is what makes the provider-only case work: a
    # ``project_pk`` a scoping provider reads off ``view.kwargs`` (never a spec
    # input) would otherwise be flagged by ``unknown_arguments``.
    url_kwarg_values = _pop_url_kwargs(url_kwargs, args)
    context = build_context(
        user,
        args,
        kwargs=url_kwarg_values or None,
        query_params=query_param_values or None,
        host=host,
    )
    # Two-layer authorization, mirroring a DRF view: the upfront call runs the
    # class-level ``has_permission`` (covers create / list-payload targets), and
    # the ``on_target_resolved`` hook runs ``has_object_permission`` on the
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
        )
    except BaseException as exc:
        # ⚠ **One arm, not a chain, because the consumer's map has to be
        # consulted first *and* fall through when it declines.** Written as a
        # leading ``except`` clause instead, every arm below it becomes
        # unreachable — the ``raise`` for an unclaimed exception leaves the
        # ``try`` entirely rather than trying the next clause.
        #
        # The map wins over the built-ins deliberately: those encode this
        # package's guess at what a model should see, and a project that knows
        # better should be able to say so.
        handler = translate_exception(exc) if translate_exception is not None else None
        if handler is not None:
            return handler(exc)
        if isinstance(exc, DRFValidationError | ServiceValidationError):
            # Input-serializer validation raises DRF's ``ValidationError``; a
            # service may raise drf-services' ``ServiceValidationError`` (a
            # ``ServiceError`` subclass — matched here, before the business-error
            # case below). Both mean "the arguments were wrong", so the model
            # retries with the detail.
            raise ModelRetry(str(exc.detail)) from exc
        if isinstance(exc, AdditionalInputRequired):
            # ⚠ **Must precede the ``ServiceError`` case below** — this is a
            # subclass of it, and the generic handler would report a request for
            # input as a terminal failure. drf-services subclasses it
            # deliberately so an unaware transport still says something
            # sensible; matching it first is what a transport that can do better
            # owes in return.
            #
            # ⭐ A model *is* the thing that can answer, and ``ModelRetry`` is
            # already the "here is what to fix, call me again" channel — so the
            # service's message plus the shape of what it wants is exactly a
            # retry prompt. No dialog, no elicitation surface, no second result
            # type: the answer comes back as an ordinary argument on the next
            # call, which is the whole premise of the exception.
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

    ⛔ **Fails; never truncates.** A partial payload looks complete — a list cut
    at the byte ceiling is indistinguishable from a list that had that many rows
    — so a model would answer confidently from data it does not know is missing.
    Refusing costs a turn; truncating costs correctness silently, which is the
    trade this whole wave keeps making the same way.

    ⚠ **The ceiling protects the context window, so it is measured on the
    encoded payload** rather than on a row count. Ten rows of a wide serializer
    and ten thousand of a narrow one are the same number of rows and nothing
    like the same number of tokens.

    Returned as an ``{"error": …}`` result rather than raised as
    :class:`ModelRetry`: the model should narrow the request and it *can*, but
    a retry budget is finite and a run should not die because a model spent it
    on progressively smaller queries. The message names the size, the ceiling
    and the remedies, because its reader is deciding what to do next.

    ⚠ Also logged at ``WARNING``: the model was told, and until this line the
    operator was not — a bound that fires invisibly reads to everyone else as
    "the tool is broken".
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
    keys alone are what the model needs — it is about to call the same tool
    again, and those are the arguments to add. The full schema is deliberately
    not rendered: the tool's own parameter schema already describes them, and a
    second, differently-shaped description in prose is how a model ends up
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
    """Strip + validate ``page`` / ``limit`` / ``ordering`` from a list selector's args.

    The tool schema advertises ``page`` / ``limit`` as integers and ``ordering``
    as an enum, but the toolset's argument validator is a no-op (the schema is
    advisory), so a model that sends ``limit="2"`` or ``ordering=["a"]`` reaches
    here untyped. Rather than let a ``TypeError`` / ``AttributeError`` abort the
    run, coerce and validate, mapping a bad value to :class:`ModelRetry` so the
    model corrects it.

    ``ordering`` is popped whether or not any fields are declared: with none, it
    was never advertised, so a model that sent it anyway is corrected rather
    than having the value fall through to the spec as an unknown argument.
    """
    if not _is_list_selector(spec):
        return None
    limit: int | None = _coerce_positive_int(args.pop("limit", None), "limit")
    if max_page_size is not None:
        # Clamped rather than rejected, and an omitted ``limit`` becomes the
        # ceiling rather than "everything": the unbounded read is the one that
        # hurts, and a model that did not think to paginate is exactly the
        # caller that needs the default applied for it.
        limit = max_page_size if limit is None else min(limit, max_page_size)
    return _PageArgs(
        page=_coerce_positive_int(args.pop("page", None), "page"),
        limit=limit,
        ordering=_coerce_ordering(args.pop("ordering", None), ordering_fields),
    )


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
    ``ModelRetry`` naming it, so the model gets a chance to supply the argument
    on the next turn. Schema ``required`` is only a hint — models omit required
    arguments routinely — and without this the run would fail deeper in, where
    the reason is far less legible. (Registration forbids ``required`` alongside
    a ``default``, so such a kwarg is never satisfiable from the declaration.)
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

    ⚠ **A deliberate divergence from the MCP transport, in the direction this
    wave keeps arguing for.** drf-mcp *silently ignores* an ordering value it
    does not recognise — the rows come back in whatever order the queryset had,
    and nothing says so. For a model that is the worst outcome available: it
    asked for newest-first, received oldest-first, and has no way to find out.
    A retry naming the accepted values costs one turn and is self-correcting.
    """
    if value is None:
        return None
    values: list[str] = _ordering_values(allowed)
    if not values:
        # Never advertised, so the model invented it. Saying so beats a
        # "must be one of: " with an empty list after it.
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
    """Order + paginate a list selector's queryset.

    Forces evaluation (``list(...)``) so a ``FieldError`` surfaces here — where
    ``_call_spec`` turns it into a ``ModelRetry`` — rather than later inside the
    serializer. ``ordering`` is enum-validated before it reaches this point, so
    the remaining way to raise one is a declared field that is not a real
    column, which is an author's error rather than a model's.
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
