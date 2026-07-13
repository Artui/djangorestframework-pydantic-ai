"""``SpecCapability`` — a Pydantic-AI capability wrapping ``SpecToolset``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from rest_framework_services import SelectorKind, SelectorSpec, UnknownArguments

from rest_framework_pydantic_ai.spec_toolset import Spec, SpecToolset, UserExtractor
from rest_framework_pydantic_ai.types.query_param import QueryParam

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
    "- Read-only tools that return a collection accept optional `page`, `limit`, and "
    "`order`: `limit` caps the number of items, `page` (1-based, requires `limit`) selects "
    "the page, and `order` is a comma-separated list of fields (prefix a field with `-` for "
    "descending)."
)


class SpecCapability(AbstractCapability[Any]):
    """A Pydantic-AI capability exposing drf-services specs as tools *and* teaching
    the model the conventions ``SpecToolset`` alone leaves in human docs.

    :class:`~rest_framework_pydantic_ai.SpecToolset` ships the tools but says
    nothing to the model about how they behave: that list tools accept ``page`` /
    ``limit`` / ``order``, that a business failure comes back as a readable
    ``{"error": …}`` result (a final answer, not a reason to retry) while a bad
    argument comes back as a retry request, or that a permission error is final.
    ``SpecCapability`` wraps the toolset and carries those conventions to the model
    through ``get_instructions()`` — the system prompt Pydantic-AI appends each
    turn — so the model doesn't rediscover them by failing a call.

    Construct it the same way as ``SpecToolset`` (it forwards the toolset knobs)::

        agent = Agent(
            model,
            deps_type=AgentDeps,
            capabilities=[SpecCapability({
                "list_orders": orders_selector_spec,   # SelectorSpec -> read-only tool
                "create_order": create_order_spec,     # ServiceSpec  -> mutation tool
            })],
        )

    or wrap an already-built toolset with :meth:`from_toolset`. Either way the
    exposed tool set is identical to the toolset's; only the instructions are added.

    ``instructions`` overrides the auto-derived conventions text. ``defer_loading``
    (which needs the stable ``id``) hides the whole spec toolset and its
    instructions behind Pydantic-AI's native ``load_capability`` tool until the
    model loads it — progressive disclosure for a large spec map. The remaining
    keywords (``get_user`` / ``unknown_arguments`` / ``query_params`` /
    ``tool_query_params`` / ``max_retries``) are forwarded verbatim to the
    ``SpecToolset`` it builds; see there for their semantics.
    """

    def __init__(
        self,
        specs: Mapping[str, Spec],
        *,
        id: str = "drf-specs",
        defer_loading: bool = False,
        instructions: str | None = None,
        get_user: UserExtractor | None = None,
        unknown_arguments: UnknownArguments = UnknownArguments.REJECT,
        query_params: Sequence[QueryParam] = (),
        tool_query_params: Mapping[str, Sequence[QueryParam]] | None = None,
        max_retries: int = 1,
    ) -> None:
        toolset = SpecToolset(
            specs,
            id=id,
            get_user=get_user,
            unknown_arguments=unknown_arguments,
            query_params=query_params,
            tool_query_params=tool_query_params,
            max_retries=max_retries,
        )
        self._configure(toolset, defer_loading=defer_loading, instructions=instructions)

    @classmethod
    def from_toolset(
        cls,
        toolset: SpecToolset,
        *,
        defer_loading: bool = False,
        instructions: str | None = None,
    ) -> SpecCapability:
        """Wrap an already-built :class:`SpecToolset` (the compose path).

        The capability adopts the toolset's ``id`` and derives its instructions
        from the same specs, so ``SpecCapability.from_toolset(ts)`` and
        ``SpecCapability(specs, …)`` produce the same model-facing behaviour.
        """
        self = cls.__new__(cls)
        self._configure(toolset, defer_loading=defer_loading, instructions=instructions)
        return self

    def _configure(
        self,
        toolset: SpecToolset,
        *,
        defer_loading: bool,
        instructions: str | None,
    ) -> None:
        # ``AbstractCapability`` is a ``@dataclass(init=False)``; set its fields as
        # plain instance attributes — the blessed pattern (see django-ag-ui's
        # ``AuditCapability``). ``id`` mirrors the toolset's so ``defer_loading``'s
        # catalog entry and the toolset resolve under one identity.
        self.id = toolset.id
        self.description = None
        self.defer_loading = defer_loading
        self._toolset = toolset
        self._instructions = (
            instructions if instructions is not None else _derive_instructions(toolset)
        )

    def get_toolset(self) -> SpecToolset:
        return self._toolset

    def get_instructions(self) -> str | None:
        return self._instructions


def _derive_instructions(toolset: SpecToolset) -> str:
    """Build the conventions block from the toolset's own specs / query params.

    Same-package read of ``SpecToolset``'s ``_specs`` / ``_tool_query_params``:
    they are the source of truth for which conventions actually apply, so the
    pagination line appears only when a list selector exists and the read-shaping
    line only when some ``QueryParam`` is declared — keeping the system prompt
    free of advice that can't fire.
    """
    lines = [_BASE_INSTRUCTIONS]
    if any(_is_list_selector(spec) for spec in toolset._specs.values()):
        lines.append(_LIST_INSTRUCTION)
    query_param_names = sorted(
        {qp.name for params in toolset._tool_query_params.values() for qp in params}
    )
    if query_param_names:
        joined = ", ".join(f"`{name}`" for name in query_param_names)
        lines.append(
            f"- Some tools accept read-shaping parameters ({joined}) that adjust the shape "
            "of the returned data without filtering it."
        )
    return "\n".join(lines)


def _is_list_selector(spec: Spec) -> bool:
    return isinstance(spec, SelectorSpec) and spec.kind is SelectorKind.LIST
