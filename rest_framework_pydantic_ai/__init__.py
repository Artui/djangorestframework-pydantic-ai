"""Expose djangorestframework-services specs as a Pydantic-AI toolset."""

from rest_framework_pydantic_ai.spec_toolset import SpecToolset
from rest_framework_pydantic_ai.types.agent_deps import AgentDeps
from rest_framework_pydantic_ai.version import __version__

__all__ = ["AgentDeps", "SpecToolset", "__version__"]
