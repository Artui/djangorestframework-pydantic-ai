# Quickstart

## 1. Have some specs

`SpecToolset` works with the `ServiceSpec` and `SelectorSpec` objects you already
define for `djangorestframework-services`. A read selector and a write service:

```python
from rest_framework_services import SelectorKind, SelectorSpec, ServiceSpec

def list_orders(user):
    """List the current user's orders."""
    return Order.objects.filter(owner=user)

list_orders_spec = SelectorSpec(
    kind=SelectorKind.LIST,
    selector=list_orders,
    output_serializer=OrderSerializer,
)

def create_order(data, user):
    """Create an order for the current user."""
    return Order.objects.create(owner=user, **data)

create_order_spec = ServiceSpec(
    service=create_order,
    input_serializer=OrderInputSerializer,
    output_selector_spec=SelectorSpec(
        kind=SelectorKind.RETRIEVE,
        output_serializer=OrderSerializer,
    ),
)
```

## 2. Build the toolset

```python
from drf_pydantic_ai import SpecToolset

toolset = SpecToolset({
    "list_orders": list_orders_spec,
    "create_order": create_order_spec,
})
```

Each key is the tool name. The description comes from the selector/service
docstring, the parameter schema from the spec's input serializer, and the
`readOnlyHint` annotation from the spec kind (selectors read, services mutate).
List selectors additionally accept `page`, `limit`, and `order` tool args.

## 3. Run an agent

The acting user flows through `RunContext.deps`. The default
[`AgentDeps`](reference.md#drf_pydantic_ai.AgentDeps) carries it:

```python
from pydantic_ai import Agent
from drf_pydantic_ai import AgentDeps

agent = Agent("anthropic:claude-opus-4-8", deps_type=AgentDeps, toolsets=[toolset])

result = await agent.run(
    "show me my last 5 orders, newest first",
    deps=AgentDeps(user=request.user),
)
```

For that request the model can call `list_orders` with
`{"limit": 5, "order": "-created_at"}` and the toolset enforces permissions,
runs the selector as `request.user`, slices the result, and renders it through
`OrderSerializer`.

## Custom identity

If your project carries identity on a richer deps object, hand the toolset a
`get_user` extractor instead of using `AgentDeps`:

```python
toolset = SpecToolset(specs, get_user=lambda ctx: ctx.deps.principal.user)
```

## Unexpected arguments

By default the toolset **rejects** tool args outside a spec's declared input set
— a key the model invented — surfacing them as a `ModelRetry` so the model
self-corrects. Specs whose declared set is open (a `filter_set` or `**kwargs`
selector) are unaffected. Pass `unknown_arguments=` to change this:

```python
from rest_framework_services import UnknownArguments

# silently drop unexpected keys instead of rejecting them
toolset = SpecToolset(specs, unknown_arguments=UnknownArguments.IGNORE)
```

## Error handling

The toolset maps drf-services' failure kinds onto the Pydantic-AI model loop:

| drf-services outcome | What the agent sees |
| --- | --- |
| `ServiceValidationError` (bad input) | `ModelRetry` with the field errors — the model self-corrects |
| `ServiceError` (business rule) | `{"error": "..."}` — model-readable content |
| Unresolved instance | `{"error": "not found"}` |
| Unexpected argument (default `REJECT`) | `ModelRetry` naming the unknown key |
| Non-integer `page` / `limit`, non-string `order` | `ModelRetry` — the model corrects the argument type |
| Bad `order` field | `ModelRetry` — the model picked a column that doesn't exist |
| Denied `permission_classes` (class-level `has_permission` **or** object-level `has_object_permission`) | `PermissionDenied` is raised and aborts the run |
