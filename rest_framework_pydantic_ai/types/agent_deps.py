"""``AgentDeps`` — the default dependency carrier a
[`SpecToolset`][rest_framework_pydantic_ai.SpecToolset] reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rest_framework_services.types.progress_reporter import ProgressReporter


@dataclass
class AgentDeps:
    """Dependencies a Pydantic-AI agent passes to a
    [`SpecToolset`][rest_framework_pydantic_ai.SpecToolset].

    Pass an instance as ``deps`` when running the agent::

        agent = Agent(model, deps_type=AgentDeps, toolsets=[toolset])
        await agent.run("create an order for …", deps=AgentDeps(user=request.user))

    ``user`` carries the acting identity — a Django user, a custom principal,
    whatever ``get_user`` returns, hence ``Any`` — so the toolset can run each
    spec under the same off-HTTP context and permission checks a DRF view would
    apply. ``SpecToolset`` reads ``ctx.deps.user`` by default; a project that
    threads identity differently — a richer principal, a lookup keyed off a token
    — can keep its own deps type and hand ``SpecToolset`` a ``get_user``
    extractor instead of using this class.
    """

    user: Any

    progress: ProgressReporter | None = None
    """Where a spec's ``progress(...)`` calls go for this run, or ``None``.

    A plain callable ``(progress, *, total, message, meta) -> None``. The toolset
    forwards it into the kwarg pool and does nothing else with it.

    **It arrives here rather than on the toolset because the toolset must never
    construct one.** This package is driven by AG-UI, by A2A, by a management
    command, by a worker, each with a different idea of where a progress report
    should go — an SSE frame, a task record, a log line — and a toolset that
    picked one would have chosen a transport it does not own.

    ``None`` costs nothing: drf-services substitutes its no-op, so a service
    declaring ``progress`` runs unchanged whether or not anyone is listening."""


__all__ = ["AgentDeps"]
