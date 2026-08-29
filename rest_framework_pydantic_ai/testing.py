"""Test doubles for driving a toolset through a real ``Agent`` run loop.

Asserting on ``call_tool`` directly answers *does the tool work*. It does not
answer the questions that only the run loop can: that the tools execute
in-process rather than being deferred to the client, that a ``ModelRetry``
is fed back for another attempt instead of aborting, that the toolset's
``get_instructions`` actually reaches the model. Those need an ``Agent``, and an
``Agent`` needs a model.

``FunctionModel`` is the model with no provider and no network behind it, and
the two recipes here are the ones this package's own suite grew and then kept
re-deriving. They lived in a test file, which is to say outside the wheel, so
every consumer wanting the same coverage wrote them again from the pydantic-ai
docs.

**Test doubles, not fixtures.** Nothing here touches Django, the database or
``pytest``, so it composes with whatever a project already uses.

```python
from pydantic_ai import Agent
from rest_framework_pydantic_ai import AgentDeps, SpecToolset
from rest_framework_pydantic_ai.testing import tool_calling_model

agent = Agent(tool_calling_model("list_widgets"), deps_type=AgentDeps, toolsets=[toolset])
result = await agent.run("go", deps=AgentDeps(user=user))
```
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel


def tool_calling_model(
    tool_name: str,
    args: Mapping[str, Any] | None = None,
    *,
    retry_args: Mapping[str, Any] | None = None,
    final_text: str = "done",
) -> FunctionModel:
    """A model that calls one tool, optionally corrects itself once, then stops.

    Args:
        tool_name: The tool to call. It must be one the toolset advertises --
            an unknown name is a retry from the tool manager, not a failure the
            model sees, so the run would loop.
        args: The arguments for the first call.
        retry_args: The arguments for the second call, used only if the first
            raised ``ModelRetry``. Defaults to ``args``, which makes the model
            repeat itself -- fine for asserting the retry *reached* it, and
            wrong for asserting recovery, since a spec that rejected those
            arguments once will reject them again and the run will not settle.
        final_text: The output once a tool call has returned.

    Returns:
        A ``FunctionModel``: no provider, no network, no key.

    The branch order matters and is not arbitrary. A retry prompt is checked
    **before** a tool return, because a retried call's message history carries
    both and the tool-return arm would end the run on the failed attempt.
    """
    first: dict[str, Any] = dict(args or {})
    second: dict[str, Any] = dict(retry_args) if retry_args is not None else dict(first)

    def model_fn(messages: Any, info: Any) -> ModelResponse:
        last = messages[-1]
        if any(part.part_kind == "retry-prompt" for part in last.parts):
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=second)])
        if any(part.part_kind == "tool-return" for part in last.parts):
            return ModelResponse(parts=[TextPart(final_text)])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=first)])

    return FunctionModel(model_fn)


def instruction_capturing_model(
    captured: dict[str, Any], *, final_text: str = "done"
) -> FunctionModel:
    """A model that records the instructions it was given and calls nothing.

    Writes ``captured["instructions"]`` on every request, so the last value is
    what the final request carried.

    The caller owns the dict rather than getting one back with the model,
    because the assertion happens after ``agent.run`` returns and a tuple return
    reads worse at every call site than a name that is already in scope:

    ```python
    captured: dict[str, Any] = {}
    agent = Agent(instruction_capturing_model(captured), toolsets=[toolset])
    await agent.run("go", deps=deps)
    assert "Sort with" in captured["instructions"]
    ```

    A toolset's ``get_instructions`` is collected by pydantic-ai rather than by
    anything here, so this is the only way to assert the conventions a
    ``SpecToolset`` contributes actually arrive -- and they arrive on the
    request, not on the toolset.
    """

    def model_fn(messages: Any, info: Any) -> ModelResponse:
        captured["instructions"] = messages[-1].instructions
        return ModelResponse(parts=[TextPart(final_text)])

    return FunctionModel(model_fn)


__all__ = ["instruction_capturing_model", "tool_calling_model"]
