# djangorestframework-pydantic-ai

Expose [`djangorestframework-services`](https://github.com/Artui/djangorestframework-services)
services and selectors as a [Pydantic-AI](https://ai.pydantic.dev) toolset, so a
plain `pydantic_ai.Agent` can call them as tools — **no MCP server and no AG-UI
bridge** in the path.

## What this is

A single thin adapter, [`SpecToolset`](reference.md#drf_pydantic_ai.SpecToolset),
that turns a `name -> spec` mapping into agent tools. Every call routes through
drf-services' transport-neutral surface:

- **`dispatch_spec`** executes the service or selector (validation, instance
  resolution, output-selector re-fetch);
- **`build_offline_context`** supplies a synthetic request / view / principal so
  `permission_classes`, `extend_queryset`, and context providers keep working;
- **`enforce_permissions`** runs `spec.permission_classes` — `dispatch_spec`
  deliberately does not, so building a toolset naively on it would skip
  authorization;
- **`spec_to_json_schema`** derives each tool's parameter schema;
- **`render_spec_output`** renders the result through the spec's serializer.

The result is that an agent driving these tools sees exactly what an HTTP client
would: the same validation errors, the same `permission_classes` checks (both
class-level `has_permission` and object-level `has_object_permission` on the
resolved row), the same rendered payloads — without the network hop, and without
standing up an MCP server.

## How it compares

If you already run an MCP server, `django-ag-ui`'s `DrfMcpToolset` exposes its
tools to Pydantic-AI through the in-process MCP surface. `SpecToolset` is for the
case where you want the specs as tools **without** any MCP layer: it depends only
on `djangorestframework-services` and `pydantic-ai-slim`.

## Next steps

- [Quickstart](quickstart.md) — wire a toolset to an agent.
- [Reference](reference.md) — the public API.
