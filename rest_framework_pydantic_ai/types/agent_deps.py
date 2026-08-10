"""``AgentDeps`` — the default dependency carrier a :class:`SpecToolset` reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rest_framework_services.types.progress_reporter import ProgressReporter


@dataclass
class AgentDeps:
    """Dependencies a Pydantic-AI agent passes to a :class:`SpecToolset`.

    Carries the acting ``user`` so the toolset can run each spec under the same
    off-HTTP context and permission checks a DRF view would apply. Pass an
    instance as ``deps`` when running the agent::

        agent = Agent(model, deps_type=AgentDeps, toolsets=[toolset])
        await agent.run("create an order for …", deps=AgentDeps(user=request.user))

    ``SpecToolset`` reads ``ctx.deps.user`` by default. A project that threads
    identity differently — a richer principal, a lookup keyed off a token — can
    keep its own deps type and hand ``SpecToolset`` a ``get_user`` extractor
    instead of using this class.
    """

    user: Any

    progress: ProgressReporter | None = None
    """Where a spec's ``progress(...)`` calls go for this run, or ``None``.

    A plain callable ``(progress, *, total, message, meta) -> None``. The
    toolset forwards it into the kwarg pool and does nothing else with it.

    Typed, unlike :attr:`user` beside it — the two look alike and are not.
    ``user`` is genuinely ``Any``: a Django user, a custom principal, whatever
    ``get_user`` returns. A reporter has a Protocol this package already
    depends on, so ``Any`` here would only mean a caller passing the wrong
    shape finds out inside their own service body.

    ⛔ **It arrives here rather than on the toolset because the toolset must
    never construct one.** This package is a pydantic-ai adapter; it is driven
    by AG-UI, by A2A, by a management command, by a worker. Each of those has a
    different idea of where a progress report should go — an SSE frame, a task
    record, a log line — and a toolset that picked one would have chosen a
    transport it does not own. The caller knows; it passes a callable.

    ``None`` costs nothing: drf-services substitutes its no-op, so a service
    declaring ``progress`` runs unchanged whether or not anyone is listening."""


__all__ = ["AgentDeps"]
