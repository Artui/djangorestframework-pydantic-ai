"""``SpecCapability`` — a Pydantic-AI capability wrapping ``SpecToolset``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from rest_framework_services import UnknownArguments

from rest_framework_pydantic_ai.spec_toolset import Spec, SpecToolset, UserExtractor
from rest_framework_pydantic_ai.types.query_param import QueryParam
from rest_framework_pydantic_ai.types.url_kwarg import UrlKwarg


class SpecCapability(AbstractCapability[Any]):
    """A Pydantic-AI capability exposing drf-services specs as tools.

    :class:`~rest_framework_pydantic_ai.SpecToolset` is a first-class toolset you
    can attach directly (``Agent(toolsets=[SpecToolset(...)])``); it already
    exposes the tools *and* teaches the model its conventions through
    ``get_instructions`` (that list tools accept ``page`` / ``limit`` / ``order``,
    and how errors come back). ``SpecCapability`` wraps that toolset to add the
    capability-only feature: ``defer_loading``. It does **not** re-emit the
    conventions — Pydantic-AI collects them from the owned toolset's
    ``get_instructions`` automatically, so wrapping vs. attaching directly yields
    the same model-facing instructions, exactly once.

    Construct it the same way as ``SpecToolset`` (it forwards the toolset knobs)::

        agent = Agent(
            model,
            deps_type=AgentDeps,
            capabilities=[SpecCapability({
                "list_orders": orders_selector_spec,   # SelectorSpec -> read-only tool
                "create_order": create_order_spec,     # ServiceSpec  -> mutation tool
            })],
        )

    or wrap an already-built toolset with :meth:`from_toolset` (the compose path).
    Either way the exposed tool set and instructions are the toolset's.

    ``instructions`` overrides the toolset's auto-derived conventions text — it is
    forwarded to the ``SpecToolset`` this builds. ``defer_loading`` (which needs
    the stable ``id``) hides the whole spec toolset and its instructions behind
    Pydantic-AI's native ``load_capability`` tool until the model loads it —
    progressive disclosure for a large spec map. The remaining keywords
    (``get_user`` / ``unknown_arguments`` / ``query_params`` /
    ``tool_query_params`` / ``url_kwargs`` / ``tool_url_kwargs`` /
    ``max_retries``) are forwarded verbatim to the ``SpecToolset`` it builds; see
    there for their semantics.
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
        url_kwargs: Sequence[UrlKwarg] = (),
        tool_url_kwargs: Mapping[str, Sequence[UrlKwarg]] | None = None,
        max_retries: int = 1,
    ) -> None:
        toolset = SpecToolset(
            specs,
            id=id,
            instructions=instructions,
            get_user=get_user,
            unknown_arguments=unknown_arguments,
            query_params=query_params,
            tool_query_params=tool_query_params,
            url_kwargs=url_kwargs,
            tool_url_kwargs=tool_url_kwargs,
            max_retries=max_retries,
        )
        self._configure(toolset, defer_loading=defer_loading)

    @classmethod
    def from_toolset(
        cls,
        toolset: SpecToolset,
        *,
        defer_loading: bool = False,
    ) -> SpecCapability:
        """Wrap an already-built :class:`SpecToolset` (the compose path).

        The capability adopts the toolset's ``id``, and its tools and
        instructions are the toolset's own — set an ``instructions`` override on
        the ``SpecToolset`` itself if you need one, so ``from_toolset(ts)`` and
        ``SpecCapability(specs, …)`` behave identically.
        """
        self = cls.__new__(cls)
        self._configure(toolset, defer_loading=defer_loading)
        return self

    def _configure(self, toolset: SpecToolset, *, defer_loading: bool) -> None:
        # ``AbstractCapability`` is a ``@dataclass(init=False)``; set its fields as
        # plain instance attributes — the blessed pattern (see django-ag-ui's
        # ``AuditCapability``). ``id`` mirrors the toolset's so ``defer_loading``'s
        # catalog entry and the toolset resolve under one identity. No
        # ``get_instructions`` override: Pydantic-AI collects the owned toolset's
        # instructions, so re-emitting here would duplicate them in the prompt.
        self.id = toolset.id
        self.description = None
        self.defer_loading = defer_loading
        self._toolset = toolset

    def get_toolset(self) -> SpecToolset:
        return self._toolset
