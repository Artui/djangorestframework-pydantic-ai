"""``SpecToolset`` — expose drf-services specs as a Pydantic-AI toolset.

A thin adapter that turns a ``name -> spec`` mapping into agent tools, executing
each call through drf-services' transport-neutral surface — ``dispatch_spec``
plus its off-HTTP helpers (``build_offline_context`` / ``enforce_permissions`` /
``spec_to_json_schema`` / ``render_spec_output``). There is **no MCP server and
no AG-UI bridge** in the path: a plain ``pydantic_ai.Agent`` calls the specs
in-process.

The acting identity flows through ``RunContext.deps``: by default the toolset
reads ``ctx.deps.user`` (the :class:`~drf_pydantic_ai.types.agent_deps.AgentDeps`
shape); projects that thread identity differently pass a ``get_user`` extractor.

Each call mirrors what a DRF view does, in order:

1. strip a list selector's ``page`` / ``limit`` / ``order`` tool args (ordering
   and pagination are transport concerns, kept off the spec);
2. build the off-HTTP context (synthetic request + view + principal);
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
from collections.abc import Callable, Mapping
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
    build_offline_context,
    dispatch_spec,
    enforce_permissions,
    render_spec_output,
    spec_to_json_schema,
)

Spec = ServiceSpec[Any, Any, Any] | SelectorSpec[Any, Any]
UserExtractor = Callable[[RunContext[Any]], Any]

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
    """

    def __init__(
        self,
        specs: Mapping[str, Spec],
        *,
        id: str = "drf-specs",
        get_user: UserExtractor | None = None,
    ) -> None:
        self._specs: dict[str, Spec] = dict(specs)
        self._get_user: UserExtractor = get_user or _default_get_user
        # Schemas derive purely from the specs (no DB), so the tool defs are
        # built once up front and handed to ExternalToolset.
        super().__init__(
            [_build_tool_def(name, spec) for name, spec in self._specs.items()],
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
        # thread. ``dict(tool_args)`` is a private copy so popping pagination
        # args never mutates the caller's dict.
        return await sync_to_async(_call_spec)(spec, user, dict(tool_args))


def _default_get_user(ctx: RunContext[Any]) -> Any:
    """Read the acting user off ``ctx.deps.user`` (the ``AgentDeps`` default)."""
    return ctx.deps.user


def _build_tool_def(name: str, spec: Spec) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=_spec_description(spec),
        parameters_json_schema=_input_schema(spec),
        metadata={"annotations": {"readOnlyHint": isinstance(spec, SelectorSpec)}},
    )


def _spec_description(spec: Spec) -> str | None:
    """The tool description: the docstring of the spec's selector / service."""
    callable_ = spec.selector if isinstance(spec, SelectorSpec) else spec.service
    return inspect.getdoc(callable_) if callable_ is not None else None


def _input_schema(spec: Spec) -> dict[str, Any]:
    """The tool's parameter schema, with list-selector pagination args merged in.

    ``spec_to_json_schema(phase="input")`` always returns a dict (only the
    output phase is nullable), so the result is narrowed for the type-checker.
    """
    schema = cast("dict[str, Any]", spec_to_json_schema(spec, phase="input"))
    if not _is_list_selector(spec):
        return schema
    return {
        **schema,
        "type": "object",
        "properties": {**schema.get("properties", {}), **_LIST_PARAM_SCHEMA},
    }


def _is_list_selector(spec: Spec) -> bool:
    return isinstance(spec, SelectorSpec) and spec.kind == SelectorKind.LIST


def _call_spec(spec: Spec, user: Any, args: dict[str, Any]) -> Any:
    """Run ``spec`` under an off-HTTP context and render the result.

    Synchronous on purpose — ``SpecToolset.call_tool`` runs it in a thread so
    the ORM stays off the event loop.
    """
    page_args = _pop_pagination(spec, args)
    context = build_offline_context(user, args)
    enforce_permissions(spec, context)
    try:
        result = dispatch_spec(
            spec,
            user=user,
            params=args,
            request=context.request,
            view=context.view,
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
    """Strip ``page`` / ``limit`` / ``order`` from a list selector's args."""
    if not _is_list_selector(spec):
        return None
    return _PageArgs(
        page=args.pop("page", None),
        limit=args.pop("limit", None),
        order=args.pop("order", None),
    )


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
