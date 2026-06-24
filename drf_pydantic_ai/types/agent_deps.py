"""``AgentDeps`` — the default dependency carrier a :class:`SpecToolset` reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


__all__ = ["AgentDeps"]
