from __future__ import annotations

from rest_framework_pydantic_ai import AgentDeps


def test_agent_deps_carries_user():
    sentinel = object()
    deps = AgentDeps(user=sentinel)
    assert deps.user is sentinel
