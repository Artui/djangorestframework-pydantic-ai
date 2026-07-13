"""Expose djangorestframework-services specs as a Pydantic-AI toolset."""

from rest_framework_pydantic_ai.spec_capability import SpecCapability
from rest_framework_pydantic_ai.spec_toolset import SpecToolset
from rest_framework_pydantic_ai.types.agent_deps import AgentDeps
from rest_framework_pydantic_ai.types.query_param import QueryParam
from rest_framework_pydantic_ai.version import __version__

__all__ = ["AgentDeps", "QueryParam", "SpecCapability", "SpecToolset", "__version__"]
