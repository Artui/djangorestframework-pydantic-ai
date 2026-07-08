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

1. strip a list selector's ``page`` / ``limit`` / ``order`` tool args (ordering
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
- a bad ``order`` field → ``ModelRetry`` (the model picked a column that does
  not exist);
- a denied ``permission_classes`` check raises ``PermissionDenied`` and aborts
  the run, exactly as it would over HTTP.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from asgiref.sync import sync_to_async
from django.core.exceptions import FieldError
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.external import ExternalToolset
from rest_framework.exceptions import ValidationError as DRFValidationError
from rest_framework_services import (
    SelectorKind,
    SelectorSpec,
    ServiceError,
    ServiceSpec,
    ServiceValidationError,
    UnknownArguments,
    build_offline_context,
    dispatch_spec,
    enforce_permissions,
    render_spec_output,
    spec_to_json_schema,
)

from rest_framework_pydantic_ai.types.query_param import QueryParam

Spec = ServiceSpec[Any, Any, Any] | SelectorSpec[Any, Any]
UserExtractor = Callable[[RunContext[Any]], Any]

# List-selector pagination args own these names; a registered ``QueryParam`` may
# not shadow them.
_RESERVED_PARAM_NAMES = frozenset({"page", "limit", "order"})

# Tool names are surfaced verbatim to the model provider, which constrains them
# to this shape (OpenAI / Anthropic function-name rules). Validated at
# construction so a bad key fails fast instead of at the provider boundary.
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

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
    "order": {
        "type": "string",
        "description": (
            "Comma-separated fields to order by; prefix a field with `-` for descending."
        ),
    },
}


@dataclass(frozen=True)
class _PageArgs:
    """A list selector's stripped ordering / pagination tool args."""

    page: int | None
    limit: int | None
    order: str | None


class SpecToolset(ExternalToolset[Any]):
    """Exposes drf-services specs as a Pydantic-AI toolset.

    Build it from a ``name -> spec`` mapping and hand it to an ``Agent``::

        toolset = SpecToolset({
            "list_orders": orders_selector_spec,   # SelectorSpec -> read-only tool
            "create_order": create_order_spec,     # ServiceSpec  -> mutation tool
        })
        agent = Agent(model, deps_type=AgentDeps, toolsets=[toolset])

    Each key becomes one tool: the description is the spec's selector/service
    docstring, the parameter schema comes from ``spec_to_json_schema`` (with a
    list selector's ``page`` / ``limit`` / ``order`` args merged in), and the
    ``readOnlyHint`` annotation is derived from the spec kind (selectors read,
    services mutate).

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
    of ``page`` / ``limit`` / ``order``. ``query_params`` applies to **every** tool;
    ``tool_query_params`` maps a tool name to params for that tool only (a per-tool
    param overrides a toolset-wide one of the same name). Each is advertised as a
    tool arg, then popped at call time and handed to
    ``build_offline_context(query_params=…)`` — never to the spec as an input, so
    ``unknown_arguments`` never sees it. Whatever reads query params (a
    ``filter_set`` driven off ``request.query_params``, django-restql, a custom
    serializer) then works with zero toolset awareness of the specific library.
    """

    def __init__(
        self,
        specs: Mapping[str, Spec],
        *,
        id: str = "drf-specs",
        get_user: UserExtractor | None = None,
        unknown_arguments: UnknownArguments = UnknownArguments.REJECT,
        query_params: Sequence[QueryParam] = (),
        tool_query_params: Mapping[str, Sequence[QueryParam]] | None = None,
    ) -> None:
        _validate_tool_names(specs)
        _validate_query_params(query_params, tool_query_params, specs)
        self._specs: dict[str, Spec] = dict(specs)
        self._get_user: UserExtractor = get_user or _default_get_user
        self._unknown_arguments: UnknownArguments = unknown_arguments
        # The effective (deduped) query params for each tool: toolset-wide first,
        # then per-tool overriding by name. Built once — declarations are static.
        self._tool_query_params: dict[str, tuple[QueryParam, ...]] = {
            name: _merge_query_params(query_params, (tool_query_params or {}).get(name, ()))
            for name in self._specs
        }
        # Schemas derive purely from the specs (no DB), so the tool defs are
        # built once up front and handed to ExternalToolset.
        super().__init__(
            [
                _build_tool_def(name, spec, self._tool_query_params[name])
                for name, spec in self._specs.items()
            ],
            id=id,
        )

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Re-stamp the base tools ``kind="function"`` so the run loop calls us.

        ``ExternalToolset`` marks every tool ``kind="external"``, which
        Pydantic-AI *defers* — it yields the call back to the caller and ends
        the run, never invoking ``call_tool``. This toolset executes specs
        in-process, so the tools must run like ordinary function tools.
        """
        tools = await super().get_tools(ctx)
        return {
            name: replace(tool, tool_def=replace(tool.tool_def, kind="function"))
            for name, tool in tools.items()
        }

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        spec = self._specs[name]
        user = self._get_user(ctx)
        # The whole pipeline touches the ORM (validation, dispatch, serializer
        # rendering), which Django forbids on the async event loop — run it in a
        # thread. ``dict(tool_args)`` is a private copy so popping pagination /
        # query-param args never mutates the caller's dict.
        return await sync_to_async(_call_spec)(
            spec,
            user,
            dict(tool_args),
            unknown_arguments=self._unknown_arguments,
            query_params=self._tool_query_params[name],
        )


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
    """Fail fast on an unknown per-tool key or a reserved query-param name."""
    declared = list(query_params)
    for tool_name, params in (tool_query_params or {}).items():
        if tool_name not in specs:
            raise ValueError(
                f"tool_query_params references unknown tool {tool_name!r}; "
                f"known tools: {sorted(specs)}."
            )
        declared.extend(params)
    reserved = sorted({qp.name for qp in declared} & _RESERVED_PARAM_NAMES)
    if reserved:
        raise ValueError(
            f"QueryParam name(s) {reserved} are reserved for list-selector "
            "pagination (page / limit / order)."
        )


def _merge_query_params(
    toolset_wide: Sequence[QueryParam], per_tool: Sequence[QueryParam]
) -> tuple[QueryParam, ...]:
    """Toolset-wide params, then per-tool overriding by name (per-tool wins)."""
    merged: dict[str, QueryParam] = {qp.name: qp for qp in toolset_wide}
    for qp in per_tool:
        merged[qp.name] = qp
    return tuple(merged.values())


def _default_get_user(ctx: RunContext[Any]) -> Any:
    """Read the acting user off ``ctx.deps.user`` (the ``AgentDeps`` default)."""
    return ctx.deps.user


def _build_tool_def(
    name: str, spec: Spec, query_params: Sequence[QueryParam] = ()
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=_spec_description(spec),
        parameters_json_schema=_input_schema(spec, query_params),
        metadata={"annotations": {"readOnlyHint": isinstance(spec, SelectorSpec)}},
    )


def _spec_description(spec: Spec) -> str | None:
    """The tool description: the docstring of the spec's selector / service."""
    callable_ = spec.selector if isinstance(spec, SelectorSpec) else spec.service
    return inspect.getdoc(callable_) if callable_ is not None else None


def _input_schema(spec: Spec, query_params: Sequence[QueryParam] = ()) -> dict[str, Any]:
    """The tool's parameter schema, with list-selector pagination + registered
    query params merged into ``properties``.

    ``spec_to_json_schema(phase="input")`` always returns a dict (only the
    output phase is nullable), so the result is narrowed for the type-checker.
    """
    schema = cast("dict[str, Any]", spec_to_json_schema(spec, phase="input"))
    extra: dict[str, Any] = {}
    if _is_list_selector(spec):
        extra.update(_LIST_PARAM_SCHEMA)
    extra.update({qp.name: qp.json_schema() for qp in query_params})
    if not extra:
        return schema
    return {
        **schema,
        "type": "object",
        "properties": {**schema.get("properties", {}), **extra},
    }


def _is_list_selector(spec: Spec) -> bool:
    return isinstance(spec, SelectorSpec) and spec.kind == SelectorKind.LIST


def _call_spec(
    spec: Spec,
    user: Any,
    args: dict[str, Any],
    *,
    unknown_arguments: UnknownArguments = UnknownArguments.REJECT,
    query_params: Sequence[QueryParam] = (),
) -> Any:
    """Run ``spec`` under an off-HTTP context and render the result.

    Synchronous on purpose — ``SpecToolset.call_tool`` runs it in a thread so
    the ORM stays off the event loop.
    """
    page_args = _pop_pagination(spec, args)
    # Pop the registered query params out of the spec args and seed them into the
    # off-HTTP request's ``query_params`` (whatever reads them — filter_set,
    # restql, a custom serializer). Popped first so they never reach the spec as
    # inputs, so ``unknown_arguments`` (REJECT by default) can't flag them.
    query_param_values = _pop_query_params(query_params, args)
    context = build_offline_context(user, args, query_params=query_param_values or None)
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
        )
    except (DRFValidationError, ServiceValidationError) as exc:
        # Input-serializer validation raises DRF's ``ValidationError``; a service
        # may raise drf-services' ``ServiceValidationError`` (a ``ServiceError``
        # subclass — caught here, before the business-error clause below). Both
        # mean "the arguments were wrong", so the model retries with the detail.
        raise ModelRetry(str(exc.detail)) from exc
    except ServiceError as exc:
        return {"error": str(exc)}
    if result.kind == "not_found":
        return {"error": "not found"}

    value = result.value
    # ``page_args`` is non-None exactly for list selectors — the only specs that
    # advertise pagination args and return a (lazy) queryset to slice.
    if page_args is not None:
        try:
            value = _shape_list(value, page_args)
        except FieldError as exc:
            raise ModelRetry(f"invalid order parameter: {exc}") from exc
    many = result.kind == "list"
    return render_spec_output(
        spec,
        value,
        many=many,
        request=context.request,
        view=context.view,
        extras=_output_extras(spec, value, many=many),
    )


def _pop_pagination(spec: Spec, args: dict[str, Any]) -> _PageArgs | None:
    """Strip + validate ``page`` / ``limit`` / ``order`` from a list selector's args.

    The tool schema advertises ``page`` / ``limit`` as integers and ``order`` as
    a string, but ``ExternalToolset`` installs a no-op argument validator, so a
    model that sends ``limit="2"`` or ``order=["a"]`` reaches here untyped. Rather
    than let a ``TypeError`` / ``AttributeError`` abort the run, coerce and
    validate, mapping a bad value to :class:`ModelRetry` so the model corrects it.
    """
    if not _is_list_selector(spec):
        return None
    return _PageArgs(
        page=_coerce_positive_int(args.pop("page", None), "page"),
        limit=_coerce_positive_int(args.pop("limit", None), "limit"),
        order=_coerce_order(args.pop("order", None)),
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


def _coerce_order(value: Any) -> str | None:
    """Require ``order`` to be a string; ``ModelRetry`` otherwise."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelRetry("`order` must be a comma-separated string of field names.")
    return value


def _shape_list(value: Any, page_args: _PageArgs) -> list[Any]:
    """Order + paginate a list selector's queryset.

    Forces evaluation (``list(...)``) so an invalid ``order`` field raises its
    ``FieldError`` here — where ``_call_spec`` turns it into a ``ModelRetry`` —
    rather than later inside the serializer.
    """
    queryset = value
    fields = _split_order(page_args.order)
    if fields:
        queryset = queryset.order_by(*fields)
    return list(_paginate(queryset, page_args.page, page_args.limit))


def _split_order(order: str | None) -> list[str]:
    if not order:
        return []
    return [field.strip() for field in order.split(",") if field.strip()]


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
