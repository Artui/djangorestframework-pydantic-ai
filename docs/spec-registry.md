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
from django_service_specs import SpecRegistry

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

The registry is django-service-specs', and it accepts any spec that compiles to
one of the kernel's. The toolset dispatches through drf-services, so every entry
it is handed must be a `rest_framework_services.ServiceSpec` or `SelectorSpec`.
A django-service-specs spec raises `TypeError` naming the tool when the toolset
is built; call one through `django_service_specs.dispatch` instead.

Pass the **registry**, not `registry.specs()`. The mapping is what the registry
holds minus everything else the entry carries, and one of those things — the
[agent contract](#what-the-spec-declares-for-an-agent) — is read here.

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

## What the spec declares for an agent

A spec declares what a caller with **no HTTP request** has to be told, because
the URLconf and query string tell an HTTP one for free: `reads=` names the route
captures and query values it reads, and `field_audiences=` marks output fields
for an agent. Its registry entry answers both as one
[`OfflineContract`][rest_framework_services.types.offline_contract.OfflineContract]
(`entry.agent_contract`).

```python
from dataclasses import replace

from rest_framework_pydantic_ai import QueryParam, UrlKwarg

registry.register(
    "list_orders",
    replace(
        specs.list_orders,
        reads=[UrlKwarg(name="tenant_pk"), QueryParam(name="fields")],
    ),
    tags=("read", "public"),
)
```

`replace` is for a spec declared elsewhere; a spec you write declares `reads=`
in its own constructor. `SpecRegistry.register` takes no contract of its own:
the spec is the one place it is said.

`SpecToolset` reads the contract off the entry, and so does an MCP server's
`register_specs`. That is the point of putting it there: an operation needs the
*identical* description of the absent request whichever agent transport calls
it, so declaring it per mount is how two mounts come to disagree.
`field_audiences` is part of the spec's output declaration, so every projection
of the spec carries it, whichever transport or mapping it is reached through.

**Pass the registry, not `registry.specs()`.** The mapping is the flattened
`name -> spec` view, and the toolset reads channels off the entry, so a toolset
built from the mapping advertises none of the spec's `reads`. The spec still
carries them, so a value the model sends under one of those names is admitted
and routed onto the request as drf-services routes any caller's, but a model is
not told it may send one.

**A read naming an input declares where the input comes from.** A `QueryParam`
read naming a parameter the spec's selector takes, such as
`reads=[QueryParam("status")]` on a selector taking `status`, a `filter_set`
field, or a parameter of a service's target lookup, is not a second argument
beside it. The tool
advertises the parameter once, as the selector declares it, and leaves its
value in the call, where drf-services hands it to the parameter and puts it on
`request.query_params` both. `SpecToolset(registry)` and
`SpecToolset(registry.specs())` build it and serve it alike.

A `QueryParam` of that name passed to the toolset's own `query_params=` or
`tool_query_params=` is still refused when the toolset is built, a re-declaration
of the spec's read included: the toolset takes such a value out of the call and
routes it to the query string alone, so the parameter would never receive it.
A `UrlKwarg` of that name with no `default`, passed to `url_kwargs=` or
`tool_url_kwargs=`, is built: it carries the input's value onto `view.kwargs`,
and the spec's read puts it on `request.query_params`, from the registry and from
`registry.specs()` alike. Two pairings still differ between the routes. A
`UrlKwarg` with a `default` fills the parameter itself, so the tool no longer
takes the name as an input, and a `UrlKwarg` may name a read the tool never took
as one; in both, the registry refuses the spec's read beside the `UrlKwarg` with
`ValueError`, as one name on two channels, while `registry.specs()`, which
advertises none of the spec's reads, builds the toolset.

## Arguments the server supplies

Some arguments are not the model's to choose. An agent run scoped to one
project should not be able to list another project's rows by sending a
different `project_pk`, and a model asked for an identifier it cannot know
tends to invent one or send the parameter's name. `tool_server_arguments`
answers for such names per tool, with django-service-specs'
[`ServerArguments`][django_service_specs.types.server_arguments.ServerArguments]:

```python
from django_service_specs import ServerArguments

toolset = SpecToolset(
    registry,
    tool_server_arguments={
        "list_document_vendors": ServerArguments(
            values={"project_pk": str(document.project_id)},
            withheld=frozenset({"draw_request_pk"}),
        ),
    },
)
```

- **A bound name** (`values`) carries the server's value on every call. It
  satisfies a `required` parameter, and every reader of the call sees the one
  value: a bound `UrlKwarg` read is on `view.kwargs`, where a permission class
  or a `kwargs=` provider finds it, a bound `QueryParam` read on
  `request.query_params`, and the selector receives it as its parameter.
- **A withheld name** is absent from every call, so the read's declared default,
  or the callable's own, answers. Withholding the other names of an either-or
  scope is how a tool taking a project *or* a draw request is held to the
  project.
- **Neither is advertised.** Both leave the tool's `properties` and `required`,
  and the instructions name neither. If the model sends one anyway, its value
  is dropped before the call is checked, never refused, so a guess costs no
  retry, and the toolset logs the tool and the names at `WARNING`, never the
  values.

Each name must be one the tool's spec declares: a selector or target-lookup
parameter, or one of its `reads`. The toolset checks every binding when it is
built, with django-service-specs' `check_server_arguments` and the toolset's
own `pool_seeds`, and raises `ImproperlyConfigured` naming the tool for a name
the spec does not declare, a name a pool seed or a typed `kwargs=` provider
fills, a withheld name no call could do without, or a bound value its
parameter refuses. A key naming a tool the toolset does not expose raises
`ValueError`, as every per-tool mapping does.

The bindings are fixed when the toolset is built, like its tool definitions.
For a per-run scope, build a toolset per run, from the run's scope. A tool
counts as scoped by the run only when the run binds one of the scope names it
reads: its other optional scope reads are then withheld too, so the model
cannot widen the scope through a sibling, while a required one the run does
not bind stays the model's to send, since withheld no call could supply it. A
tool reading only scope names the run does not bind gets no binding and keeps
them all:

```python
SCOPE_NAMES = frozenset({"project_pk", "draw_request_pk", "document_pk"})


def scope_bindings(registry, scope):
    """Each tool the run's scope reaches: the bound names, and optional siblings withheld."""
    bindings = {}
    for entry in registry.all():
        reads = [read for read in entry.spec.reads if read.name in SCOPE_NAMES]
        values = {read.name: scope[read.name] for read in reads if read.name in scope}
        if not values:
            continue
        bindings[entry.name] = ServerArguments(
            values=values,
            # A QueryParam read has no ``required``; a UrlKwarg read may.
            withheld=frozenset(
                read.name
                for read in reads
                if read.name not in values and not getattr(read, "required", False)
            ),
        )
    return bindings


toolset = SpecToolset(registry, tool_server_arguments=scope_bindings(registry, scope))
```

`SpecCapability` takes the same keyword.

## What stays per toolset

The rest of `SpecToolset`'s signature, because it is genuinely this mount's:
`get_user`, `unknown_arguments`, `max_retries`, the result-size and page
ceilings. A publicly exposed MCP endpoint and an in-process toolset carry
different risk, so one shared number would be a regression rather than a
simplification.

### Deprecated: channel declarations on the toolset

The [`QueryParam`](reference.md#rest_framework_pydantic_ai.QueryParam) /
[`UrlKwarg`](reference.md#rest_framework_pydantic_ai.UrlKwarg) constructor
arguments, `query_params=`, `tool_query_params=`, `url_kwargs=` and
`tool_url_kwargs=`, are deprecated. Passing any of them warns with
`DeprecationWarning`, and they will be removed in the next minor release. They
declare at one mount what the spec's `reads` declare for every transport, which
is how two mounts come to disagree about one operation. Move each onto the spec
and build the toolset from the registry:

```python
# Before
SpecToolset(
    registry.by_tag("read"),
    query_params=[QueryParam(name="fields")],
    tool_url_kwargs={"list_orders": [UrlKwarg(name="tenant_pk")]},
)

# After
registry.register(
    "list_orders",
    replace(specs.list_orders, reads=[UrlKwarg(name="tenant_pk"), QueryParam(name="fields")]),
)
SpecToolset(registry.by_tag("read"))
```

A `UrlKwarg(default=...)` used to keep a value the model should not choose was
only ever a default: the model could override it, and it stayed advertised.
That use moves to [`tool_server_arguments`](#arguments-the-server-supplies).

Until they go, they behave as before. Each overrides the entry's declaration of
the same name, and the override is true of what the toolset advertises, routes
and defaults, and of nothing drf-services reads: the spec's own declaration
still checks the call. A mount loosening a spec's `required` read to optional
advertises an argument the call is then refused without. Per-tool maps key off
the registry's names, and an unknown key still raises.

## Names

Registry names are free-form, but tool names are not: model providers constrain
them to `[a-zA-Z0-9_-]{1,64}`. A registry name outside that shape raises when the
toolset is built, rather than failing later at the provider boundary. If you
share a registry with a transport that has looser naming, keep the names
tool-safe.
