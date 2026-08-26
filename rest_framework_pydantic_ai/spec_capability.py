"""``SpecCapability`` — a Pydantic-AI capability wrapping ``SpecToolset``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from django.http import HttpRequest
from pydantic_ai.capabilities import AbstractCapability
from rest_framework_services import UnknownArguments

from rest_framework_pydantic_ai.spec_toolset import (
    ExceptionHandler,
    HttpRequestExtractor,
    ProgressExtractor,
    SpecSource,
    SpecToolset,
    UserExtractor,
)
from rest_framework_pydantic_ai.types.query_param import QueryParam
from rest_framework_pydantic_ai.types.url_kwarg import UrlKwarg


class SpecCapability(AbstractCapability[Any]):
    """A Pydantic-AI capability exposing drf-services specs as tools.

    [`SpecToolset`][rest_framework_pydantic_ai.SpecToolset] is a first-class toolset you
    can attach directly (``Agent(toolsets=[SpecToolset(...)])``), and it already
    exposes the tools *and* teaches the model its conventions. This wraps one to
    add the two capability-only knobs, ``defer_loading`` and the
    ``description`` its catalog entry is chosen by. It does **not** re-emit
    those conventions — Pydantic-AI collects them from the owned toolset — so
    wrapping and attaching directly yield the same instructions, exactly once.

    Construct it the same way as ``SpecToolset`` (it forwards the toolset knobs):

        agent = Agent(
            model,
            deps_type=AgentDeps,
            capabilities=[SpecCapability({
                "list_orders": orders_selector_spec,   # SelectorSpec -> read-only tool
                "create_order": create_order_spec,     # ServiceSpec  -> mutation tool
            })],
        )

    or wrap an already-built toolset with
    [`from_toolset`][rest_framework_pydantic_ai.SpecCapability.from_toolset]
    (the compose path). Either way the exposed tool set and instructions are the
    toolset's.

    **Everything else ``SpecToolset`` accepts, this accepts, and means there.**
    That is a guarantee rather than a list, enforced name-by-name by the
    forwarding tests: a knob added to the toolset and forgotten here is not a
    missing feature but an *unreachable* one for every consumer composing through
    a capability. See [`SpecToolset`][rest_framework_pydantic_ai.SpecToolset] for what
    each does; the safety-relevant ones are ``require_permissions``,
    ``max_result_bytes``, ``max_page_size`` and ``dispatch_timeout``.

    Args:
        specs: As ``SpecToolset``, including a
            ``SpecRegistry``
            or a filtered view of one, so a project declaring its specs once can
            project several capabilities from them.
        defer_loading: Hide the whole spec toolset and its instructions behind
            Pydantic-AI's native ``load_capability`` tool until the model loads
            it — progressive disclosure for a large spec map.
        id: As ``SpecToolset``. It keys ``defer_loading``'s catalog entry, so
            give each capability projected from one registry its own.
        description: One line saying what this capability is for, rendered
            beside ``id`` in ``defer_loading``'s catalog. **Give one to every
            deferred capability:** Pydantic-AI's loader renders ``- {id}:
            {description}`` when there is one and a bare ``- {id}`` when there
            is not, so several undescribed capabilities leave the model
            choosing between names alone — the guess-or-load-everything outcome
            deferring exists to avoid. Not the same knob as ``descriptions``,
            which relabels individual *tools*.
    """

    def __init__(
        self,
        specs: SpecSource,
        *,
        id: str = "drf-specs",
        defer_loading: bool = False,
        description: str | None = None,
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
        toolset = SpecToolset(
            specs,
            id=id,
            instructions=instructions,
            get_user=get_user,
            get_progress=get_progress,
            unknown_arguments=unknown_arguments,
            query_params=query_params,
            tool_query_params=tool_query_params,
            url_kwargs=url_kwargs,
            tool_url_kwargs=tool_url_kwargs,
            host=host,
            max_retries=max_retries,
            max_result_bytes=max_result_bytes,
            tool_max_result_bytes=tool_max_result_bytes,
            max_page_size=max_page_size,
            dispatch_timeout=dispatch_timeout,
            require_permissions=require_permissions,
            descriptions=descriptions,
            ordering_fields=ordering_fields,
            tool_ordering_fields=tool_ordering_fields,
            http_request=http_request,
            get_http_request=get_http_request,
            exception_map=exception_map,
        )
        self._configure(toolset, defer_loading=defer_loading, description=description)

    @classmethod
    def from_toolset(
        cls,
        toolset: SpecToolset,
        *,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> SpecCapability:
        """Wrap an already-built
        [`SpecToolset`][rest_framework_pydantic_ai.SpecToolset] (the compose
        path).

        The capability adopts the toolset's ``id``, and its tools and
        instructions are the toolset's own — set an ``instructions`` override on
        the ``SpecToolset`` itself if you need one, so ``from_toolset(ts)`` and
        ``SpecCapability(specs, …)`` behave identically.

        ``description`` means what it does on the constructor: the catalog line
        a deferred capability is chosen by. It has no toolset counterpart to
        adopt, so pass it here.
        """
        self = cls.__new__(cls)
        self._configure(toolset, defer_loading=defer_loading, description=description)
        return self

    def _configure(
        self, toolset: SpecToolset, *, defer_loading: bool, description: str | None
    ) -> None:
        # ``AbstractCapability`` is a ``@dataclass(init=False)``, so its fields are
        # set as plain instance attributes. ``id`` mirrors the toolset's, so
        # ``defer_loading``'s catalog entry and the toolset resolve under one
        # identity. No ``get_instructions`` override: Pydantic-AI collects the
        # owned toolset's, and re-emitting here would duplicate them.
        self.id = toolset.id
        self.description = description
        self.defer_loading = defer_loading
        self._toolset = toolset

    def get_toolset(self) -> SpecToolset:
        return self._toolset
