# Declaring specs once, across transports

If the agent is the only place your project exposes its specs, keep passing a
plain dict — this page buys you nothing.

It pays off when the **same specs** are also exposed elsewhere: over MCP with
[`djangorestframework-mcp-server`](https://github.com/Artui/djangorestframework-mcp-server),
or as ordinary HTTP views. Each transport has to be told which specs it exposes,
so the list gets written once per transport and the copies drift — a spec is
added to the MCP wiring and forgotten in the agent's, or the same operation is
named `create_order` in one place and something else in the other.

`SpecRegistry` (`djangorestframework-services` 0.27+) is the one declaration
site. `SpecToolset` and `SpecCapability` accept one **anywhere they accept a
mapping** — there is no separate constructor to learn.

## Declare once

```python
# orders/registry.py
from rest_framework_services import SpecRegistry

registry = SpecRegistry()


# orders/apps.py
from django.apps import AppConfig


class OrdersConfig(AppConfig):
    name = "orders"

    def ready(self) -> None:
        from orders import specs
        from orders.registry import registry

        registry.register("list_orders", specs.list_orders, tags=("read", "public"))
        registry.register("refund_order", specs.refund_order, tags=("write", "admin"))
```

## Hand it to the agent

```python
from pydantic_ai import Agent

from rest_framework_pydantic_ai import AgentDeps, SpecToolset
from orders.registry import registry

agent = Agent(model, deps_type=AgentDeps, toolsets=[SpecToolset(registry)])
```

That is the whole change. Each entry becomes one tool, named as it is in the
registry, in registration order.

Pass the **registry**, not `registry.specs()`. The mapping is what the registry
holds minus everything else the entry carries, and one of those things — the
[agent contract](#what-the-entry-carries-for-an-agent) — is read here.

The same goes for the capability:

```python
from rest_framework_pydantic_ai import SpecCapability

AgentConfig(capabilities=[SpecCapability(registry)])
```

## Project several toolsets from one declaration

`by_tag` and `subset` each return a **new** registry holding a snapshot, so one
declaration site can feed several toolsets with no shared state:

```python
reads = SpecToolset(registry.by_tag("read"), id="reads")
admin = SpecToolset(registry.by_tag("admin"), id="admin")

agent = Agent(model, deps_type=AgentDeps, toolsets=[reads, admin])
```

With capabilities, give each its own `id` — the id keys `defer_loading`'s
catalog entry, so two capabilities sharing one would collide:

```python
AgentConfig(
    capabilities=[
        SpecCapability(
            registry.by_tag("read"),
            id="reads",
            description="Read-only order and customer lookups.",
        ),
        SpecCapability(
            registry.by_tag("admin"),
            id="admin",
            description="Account administration: suspend, refund, reassign.",
            defer_loading=True,
        ),
    ]
)
```

Here `defer_loading` hides the admin tools behind Pydantic-AI's native
`load_capability` tool until the model asks for them — worth doing when the
admin surface is large and rarely needed.

**Give every deferred capability a `description`.** It is the line the model
picks from: the catalog renders `- {id}: {description}` when there is one and a
bare `- {id}` when there is not, so a few undescribed capabilities leave the
model choosing between names alone — and it will either guess or load all of
them, which is the cost deferring was meant to avoid. `description` names the
*capability*; the separate `descriptions` mapping relabels individual tools.

## What the entry carries for an agent

An entry may hold an
[`OfflineContract`][rest_framework_services.types.offline_contract.OfflineContract]:
what a caller with **no HTTP request** has to be told, because the URLconf and
query string tell an HTTP one for free.

```python
registry.register(
    "list_orders",
    specs.list_orders,
    tags=("read", "public"),
    agent_contract=OfflineContract(
        url_kwargs=(UrlKwarg(name="tenant_pk"),),
        query_params=(QueryParam(name="fields"),),
    ),
)
```

`SpecToolset` reads it, and so does an MCP server's `register_specs`. That is
the point of putting it there: an operation needs the *identical* description of
the absent request whichever agent transport calls it, so declaring it per mount
is how two mounts come to disagree. `field_audiences` rides on the same object,
for the one tool that must return what its siblings hide.

**Pass the registry, not `registry.specs()`.** The mapping is the flattened
`name -> spec` view, and a contract is on the entry rather than the spec — so a
toolset built from the mapping silently has none of it.

## What stays per toolset

The rest of `SpecToolset`'s signature, because it is genuinely this mount's:
`get_user`, `unknown_arguments`, `max_retries`, the result-size and page
ceilings. A publicly exposed MCP endpoint and an in-process toolset carry
different risk, so one shared number would be a regression rather than a
simplification.

The [`QueryParam`](reference.md#rest_framework_pydantic_ai.QueryParam) /
[`UrlKwarg`](reference.md#rest_framework_pydantic_ai.UrlKwarg) constructor
arguments remain, and override the entry's contract by name:

```python
SpecToolset(
    registry.by_tag("read"),
    query_params=[QueryParam(name="fields")],
    tool_url_kwargs={"list_orders": [UrlKwarg(name="tenant_pk")]},
)
```

They are the right tool for a toolset with no registry behind it, and for a
mount that genuinely differs. Where a project runs a second agent transport,
prefer the contract.

Per-tool maps like `tool_query_params` / `tool_url_kwargs` key off the registry's
names, and an unknown key still raises — a typo is a configuration error, not a
silent no-op.

## Names

Registry names are free-form, but tool names are not: model providers constrain
them to `[a-zA-Z0-9_-]{1,64}`. A registry name outside that shape raises when the
toolset is built, rather than failing later at the provider boundary. If you
share a registry with a transport that has looser naming, keep the names
tool-safe.
