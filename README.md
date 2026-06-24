# djangorestframework-pydantic-ai

[![CI](https://github.com/Artui/djangorestframework-pydantic-ai/workflows/tests/badge.svg)](https://github.com/Artui/djangorestframework-pydantic-ai/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/djangorestframework-pydantic-ai.svg)](https://pypi.org/project/djangorestframework-pydantic-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/djangorestframework-pydantic-ai.svg)](https://pypi.org/project/djangorestframework-pydantic-ai/)
[![Django versions](https://img.shields.io/pypi/djversions/djangorestframework-pydantic-ai.svg)](https://pypi.org/project/djangorestframework-pydantic-ai/)
[![Docs](https://img.shields.io/badge/docs-artui.github.io-blue.svg)](https://artui.github.io/djangorestframework-pydantic-ai/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Artui/djangorestframework-pydantic-ai/gh-pages/coverage.json)](https://github.com/Artui/djangorestframework-pydantic-ai/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/pypi/l/djangorestframework-pydantic-ai.svg)](LICENSE)

Expose [`djangorestframework-services`](https://github.com/Artui/djangorestframework-services)
services and selectors as a [Pydantic-AI](https://ai.pydantic.dev) toolset, so a
plain `pydantic_ai.Agent` can call them as tools — **no MCP server and no AG-UI
bridge** in the path.

Every tool call routes through drf-services' transport-neutral surface
(`dispatch_spec` plus its off-HTTP helpers), so the same validation,
permissions, and serializer rendering your DRF views apply also apply here —
just without the HTTP hop.

## Install

```bash
pip install djangorestframework-pydantic-ai
```

It depends only on `djangorestframework-services` and `pydantic-ai-slim`. A model
provider is pulled in separately, the usual Pydantic-AI way (e.g.
`pip install "pydantic-ai-slim[anthropic]"`).

## Quickstart

```python
from pydantic_ai import Agent
from drf_pydantic_ai import AgentDeps, SpecToolset

toolset = SpecToolset({
    "list_orders":  orders_selector_spec,   # SelectorSpec -> read-only tool
    "create_order": create_order_spec,      # ServiceSpec  -> mutation tool
})

agent = Agent("anthropic:claude-opus-4-8", deps_type=AgentDeps, toolsets=[toolset])

result = await agent.run(
    "create an order of 3 widgets",
    deps=AgentDeps(user=request.user),
)
```

The agent acts as `deps.user`: each call builds an off-HTTP request/view context,
**enforces the spec's `permission_classes`**, dispatches the spec, and renders
the result through the spec's serializer. List selectors gain `page` / `limit` /
`order` tool args.

See the [documentation](https://artui.github.io/djangorestframework-pydantic-ai/)
for the full reference.

## License

MIT
