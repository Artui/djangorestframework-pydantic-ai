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
from rest_framework_pydantic_ai import SpecToolset

toolset = SpecToolset(
    {
        "list_orders": list_orders_spec,
        "create_order": create_order_spec,
    }
)
```

Each key is the tool name. The description comes from the selector/service
docstring, the parameter schema from the selector's parameters or the service's
input serializer and target lookup (see
[What a tool asks the model for](#what-a-tool-asks-the-model-for)), the
`return_schema` from its output serializer, and the `readOnlyHint` annotation
from the spec kind (selectors read, services mutate). List selectors
additionally accept `page` and `limit` tool args, plus `ordering` where the
selector's [`filter_set` declares one](#ordering).

## 3. Run an agent

The acting user flows through `RunContext.deps`. The default
[`AgentDeps`](reference.md#rest_framework_pydantic_ai.AgentDeps) carries it:

```python
from pydantic_ai import Agent
from rest_framework_pydantic_ai import AgentDeps

agent = Agent("anthropic:claude-opus-4-8", deps_type=AgentDeps, toolsets=[toolset])

result = await agent.run(
    "show me my last 5 orders, newest first",
    deps=AgentDeps(user=request.user),
)
```

!!! tip "No request in sight?"

    `request.user` is the HTTP shape. A Celery task, a management command or a
    scheduled job resolves the acting user itself and passes it the same way —
    see [Running from a worker](background-runs.md), which also covers the one
    setting a fan-out genuinely needs.

For that request the model can call `list_orders` with
`{"limit": 5, "ordering": "-created"}` and the toolset enforces permissions,
runs the selector as `request.user`, hands `ordering` to the selector's
[`filter_set`](#ordering), slices the result, and renders it through
`OrderSerializer`.

## Every list result is a page

A list selector answers with the pagination envelope, never a bare array:

```python
{
    "items": [{"id": 12, "total": "48.00"}, ...],
    "page": 1,
    "totalPages": 4,
    "hasNext": True,
}
```

`limit` defaults to 100 rows and `page` to 1, so a tool that used to return an
entire table now returns its first hundred rows **and says so**. That is the
point: `page` and `limit` were advertised on every list tool from the start
while the payload was a bare slice, so a model asking for a collection received
50 of 51 rows with nothing in the answer telling it more existed. `hasNext` is
what was missing — the model can ask for `page: 2`, or narrow the request with a
filter, instead of answering from a page it took for the whole set.

`max_page_size` caps `limit` and advertises itself as JSON-Schema `maximum`
on it. A call naming no `limit` is served the smaller of 100 and the ceiling, and
that is the default both the `limit` description and the toolset's instructions
state, so with the toolset below the model reads "Defaults to 25" beside
`maximum: 25`. A ceiling above 100 raises what a call may ask for, not what it
gets by asking for nothing: the stated default stays 100.

```python
toolset = SpecToolset(specs, max_page_size=25)
```

The same envelope is what the tool's `return_schema` describes, generated from
the same spec — so the schema and the payload cannot disagree about it.

## The output schema

Each tool definition carries a `return_schema` derived from the spec's output
serializer, projected the same way the payload is: a field marked
[hidden](agent-audience.md) is absent from both, and a marked handle carries its
description in both.

It is populated but **not sent** by default, because a return schema costs
context on every turn of every run and only your model and your serializers say
whether that trade is worth it. Pydantic-AI owns the opt-in, at either scope:

```python
agent = Agent(model, toolsets=[SpecToolset(specs).include_return_schemas()])
```

A spec with no `output_serializer` gets `None` rather than a guessed shape.

A retrieve selector declaring `allow_none=True` returns `None` when nothing
matches, so its `return_schema` admits it: the root `type` is
`["object", "null"]`. A service's `return_schema` stays an object whatever its
`output_selector_spec` declares, because dispatch ignores `allow_none` on a
nested spec.

## What a tool asks the model for

A tool's parameter schema is the model's only account of what to send, and a
selector's signature cannot say on its own which parameters the model sends and
which the toolset fills. So the toolset says so when it builds the schema
(through drf-services' `spec_to_json_schema(supplied=...)`):

- **A name the toolset fills is not advertised.** That is drf-services' own
  pool seeds (`request`, `user`, `progress` and the rest), every name registered
  in [`pool_seeds`](#project-pool-seeds), and the keys a `kwargs=` provider
  returns when its return annotation is a `TypedDict`, less any annotated to
  admit `UnsetType` (below). A [`UrlKwarg`](#url-derived-values-route-captures)
  that declares a `default` fills its name too, so the selector's parameter of
  that name is not required, but the `UrlKwarg` itself stays advertised, as the
  optional argument it always is.
- **Every other parameter without a default is required.** `get_widget(user, pk)`
  is advertised as `{"properties": {"pk": {}}, "required": ["pk"]}`. A parameter
  with a default stays optional, and `InputRequired` still makes one required.
- **A `kwargs=` provider without a `TypedDict` return annotation may fill any
  parameter**, a plain `dict` or a lambda included, so for its spec nothing is
  required merely for lacking a default, though `InputRequired` still requires
  one. So may a provider whose annotations do not resolve, the parameters'
  as well as the return's, and a `TypedDict` whose own key annotations do not
  resolve: a name imported only under `TYPE_CHECKING` leaves the provider
  untyped. Annotate the provider to have the rest required:

```python
from typing import TypedDict


class ProjectScope(TypedDict):
    ceiling: int


def scope(view) -> ProjectScope:
    # ceiling is filled here, so the model is never asked for it.
    return {"ceiling": ceiling_for(view.kwargs["project_pk"])}


list_spec = SelectorSpec(kind=SelectorKind.LIST, selector=priced_under, kwargs=scope)
```

- **A key annotated to admit `UnsetType` is not one the toolset fills.** A
  provider may decline a key by returning drf-services' `UNSET` for it, which
  drops the key from the pool and lets the model's value through, so
  `ceiling: int | UnsetType` keeps `ceiling` advertised for the model to send,
  and optional, since the provider may fill it instead. The provider's other
  keys are still not asked for. If the provider declines the key and the model
  has not sent it either, nothing fills the parameter and the selector raises
  `TypeError` out of the run, as it does when an untyped provider leaves a
  parameter unfilled, since a call is not checked for a name a provider may
  fill. Where a provider may decline a key the model may also leave out, give
  the selector's parameter a default.
- **A name a
  [`build_context`](reference.md#rest_framework_pydantic_ai.SpecToolset)
  override fills has to be declared.** An override is code the schema cannot
  read, so a selector parameter it fills through `kwargs` (which become
  `view.kwargs`) is advertised as required like any other parameter without a
  default, and a call leaving it out is handed back before the override runs.
  Mark the selector parameter with drf-services' `NotClientInput`: the name is
  left out of the schema and the override fills it. The marker hides the name
  rather than blocking it. A call that sends it anyway is handed back as an
  [unexpected argument](#unexpected-arguments) by default, but only where the
  selector's input set is closed (no `filter_set`, no `**kwargs`); otherwise
  the value reaches the selector's pool, so the override has to write the key
  on every call.
  Where the value can be resolved from what a seed resolver receives, register
  it as a [pool seed](#project-pool-seeds) and resolve it there instead:

```python
from typing import Annotated

from rest_framework_services import NotClientInput, SelectorKind, SelectorSpec


def list_tasks(user, project_pk: Annotated[int, NotClientInput]):
    """List the user's tasks in the run's project."""
    return Task.objects.filter(project_id=project_pk, assignee=user)


class ProjectScopedToolset(SpecToolset):
    def build_context(self, user, params, *, ctx, kwargs=None, **rest):
        scoped = {**(kwargs or {}), "project_pk": current_project_pk(ctx)}
        return super().build_context(user, params, ctx=ctx, kwargs=scoped, **rest)


# project_pk is marked NotClientInput and filled by the override, so the model
# is never asked for it, and a call that sends one is refused.
toolset = ProjectScopedToolset(
    {
        "list_tasks": SelectorSpec(
            kind=SelectorKind.LIST, selector=list_tasks, output_serializer=TaskSerializer
        )
    }
)
```

A `UrlKwarg` with a `default` also keeps a call that leaves the name out from
being refused, but it does not keep the name from the model. It stays
advertised, as an optional argument the model can see and send, and the
override's value replaces whatever the model sent, with nothing said to the
model: a model that asks for one project is served another's rows.

**A service tool also advertises its target lookup.** drf-services hands the
arguments it validates against the input serializer to the selector that
resolves the row or the set, too, so that selector's parameters are reflected
beside the serializer's fields by the same rules. It is the
`collection_selector_spec` when the service declares one, and the
`instance_selector_spec` otherwise: dispatch never runs the instance lookup
beside a collection one, so a service declaring both is not asked for the
instance lookup's `pk`. A rename tool whose instance selector is
`task_by_pk(user, *, pk)` asks for `pk` as well as the new title, and requires
it. Where a lookup parameter and a serializer field share a name, the
serializer's property is the one advertised, since the serializer validates
the value, and the name is required if either requires it, since the lookup
cannot run without it whatever a `partial` serializer says. A `many=True`
service reads no target, so its schema stays the list alone.

**A call that leaves out a required selector parameter is handed back**, the
tool's own or its target lookup's, as `ModelRetry` naming each one left out,
as in ``Missing required argument(s): `pk`.``, the same sentence a required
`UrlKwarg` gets, so the model corrects the call on its next turn. It used to
reach the selector, which raised `TypeError` out of the run. A required
serializer field is checked by the serializer once the call runs, so a service
call missing both `pk` and a field is told about `pk` first, and about the
field on the turn after. A name a provider may fill is not checked before the
call, since only the pool the call assembles can say whether it arrived.

## Custom identity

If your project carries identity on a richer deps object, hand the toolset a
`get_user` extractor instead of using `AgentDeps`:

```python
toolset = SpecToolset(specs, get_user=lambda ctx: ctx.deps.principal.user)
```

## Project pool seeds

Over HTTP a service reads its tenant, locale or clock off `request`. Off HTTP
there is no request carrying any of that, and drf-services'
[`PoolSeeds`][rest_framework_services.types.pool_seeds.PoolSeeds] is the channel
for it. Hand the registry to the toolset as `pool_seeds=`:

```python
from rest_framework_services import DEFAULT_POOL_SEEDS

seeds = DEFAULT_POOL_SEEDS.extend(tenant=lambda user: user.profile.tenant)
toolset = SpecToolset(specs, pool_seeds=seeds)
```

Every dispatch the toolset makes is handed the registry. A service, a selector
or an affordance's condition that declares `tenant` therefore receives the
resolver's value, as it would from `dispatch_spec(pool_seeds=seeds)` called
directly. The per-step check that
[leaves out an unavailable operation](#an-operation-that-is-unavailable-right-now-is-left-out)
asks its conditions against the same seeds, so a step's catalog and the call's
own refusal answer alike.

A registered name is reserved the way `user` and `request` are:

- **The model cannot supply it.** An argument of that name is not passed on and
  is not refused as unknown either, so the callable always sees the resolver's
  value.
- **A channel cannot declare it.** A `QueryParam` or `UrlKwarg` with that name
  raises `ImproperlyConfigured` when the toolset is built, as one named `user`
  does. Dispatch strips a reserved name from the route captures it hands a
  selector, so an accepted `UrlKwarg` would offer the model an argument that is
  then dropped on every call.
- **No schema advertises it.** A selector or target lookup declaring the seed
  as a parameter does not offer it to the model, so the model is not asked for
  a value the call would ignore. See
  [What a tool asks the model for](#what-a-tool-asks-the-model-for).

The registry applies to the whole toolset, with no per-tool or per-call form. A
seed is ambient to the deployment, and what varies from call to call belongs in
the resolver, which declares `user` or `request` to receive them.

## What a permission class sees

Every call is authorized by `spec.permission_classes`, run against a synthetic
request and view. drf-services supports a permission class reading `request`,
`view.action` and `view.kwargs` off HTTP — anything beyond those three (a
`view.queryset`, as `DjangoModelPermissions` wants) is not available. Here is
what this package puts in each:

- **`request.user`** is the acting identity — `deps.user`, or whatever
  `get_user` returns. A configured `http_request` never contributes an identity.
- **`request.query_params`** holds exactly the [query
  params](#read-shaping-query-params) declared for that tool, and nothing else.
  A tool that declares none dispatches with an empty query string even when the
  toolset was given an `http_request`, so the ambient endpoint's own query
  string can never reach a serializer or a `filter_set`.
- **`view.action`** is the **tool name** — the key the spec is registered under
  in the mapping you passed. Off HTTP there is no router to name an action, and
  the tool name is the identity the model called, so it is the honest answer;
  it is also what the MCP transport reports for the same spec. A permission
  class branching on viewset action names (`"create"`, `"retrieve"`) will not
  match one of those unless a tool happens to be named that — check the `else`
  branch of such a class before exposing its spec, and rewrite `action` in a
  `build_context` override if you need a specific one.
- **`view.kwargs`** holds the [URL kwargs](#url-derived-values-route-captures)
  declared for that tool, the off-HTTP counterpart of a route's captures.

### The tool catalog is not permission-filtered

`get_tools` advertises every spec to every run, except a tool whose [operation
condition is unmet](#an-operation-that-is-unavailable-right-now-is-left-out) at
that step. A tool whose permissions will deny this caller is still listed — the
denial happens on the call. That is
deliberate: a permission whose answer depends on the arguments has none to read
at listing time and would hide a tool the caller can actually use, the listing
runs once per model step so a database-backed check would cost a query per spec
per step, and a model that cannot see a tool cannot ask about it. A listing
carries a name, a description and an input schema; no row data.

If a deployment does want a narrower catalog, override `is_tool_listed`:

```python
from asgiref.sync import sync_to_async


class OpsOnlyToolset(SpecToolset):
    async def is_tool_listed(self, name, ctx):
        if name != "suspend_account":
            return True
        # Django refuses ORM access on the event loop, so anything that
        # queries has to go through a thread — as dispatch itself does.
        return await sync_to_async(ctx.deps.user.groups.filter(name="ops").exists)()
```

Hiding a tool is a disclosure decision, never an authorization one: the call is
gated by `permission_classes` whatever this returns.

### An operation that is unavailable right now is left out

A spec's `Affordance` answered without a row, a callable `when` such as
`lambda: Period.current().is_open` (see [below](#a-refused-affordance-names-its-code)),
is asked on every model step, and a tool whose condition is unmet is left out of
that step's catalog. The arguments against filtering by permission do not apply
to it. The condition reads only the seeds (the user, the request and any
[registered seed](#project-pool-seeds)), never an argument, so it cannot hide a tool the caller could
have invoked with other arguments. Only the specs declaring one are asked, and a
toolset declaring none pays nothing. And the model can still tell the user about
the missing tool, because the instructions for that step end by naming each tool
left out with its reason:

```text
- These operations exist but cannot be performed right now, so they are not among your tools. If the user asks for one, say it is unavailable at the moment and give the reason listed for it, rather than guessing why:
  - `post_invoice`: The books are closed.
```

The rest of the instructions are derived from the tools that step offers, so no
line describes a tool it lacks. An `instructions=` override replaces those
conventions but not this list, which is appended after the override: which
tools a given step lacks is not something an override written in advance can
say. The heading is the `unavailable_heading` field of
[`conventions=`](#changing-what-the-model-is-told), and setting it to `None`
drops the heading and the list, with an override or without one.

A few things follow from how it is asked:

- **An `is_tool_listed` override cannot put the tool back.** The omission is
  applied beside it rather than inside the default, so an override returning
  `True` does not need to call `super()` to keep it.
- **Conditions run where a dispatch runs**, in a thread under the toolset's
  `thread_sensitive` and `executor`, so one that queries is fine. The connection
  it opens is closed after it, as a dispatch's is.
- **The request a condition reads is built through `build_context`**, so a
  condition reading `request` sees the same object when the catalog is listed as
  at the call, including anything an override adds to it. That listing call
  carries no arguments, no `action` and an empty query string.
- **A condition on the row (an ORM expression) is never asked here.** There is
  no row when the catalog is listed, so the call answers it for its own row.
- **`get_tools` and `get_instructions` each ask once per step.** A shared answer
  would need a key scoping it to one run, and the run context has none that is
  safe: `run_id` can be supplied by the caller and is reused when a failed run
  is retried. A condition that flips between the two reads can leave one step's
  catalog and instructions disagreeing about one tool.

None of this changes the call. `dispatch_spec` enforces every affordance
whatever was listed, a call against a listing that went stale is refused as
[below](#a-refused-affordance-names-its-code), and a model calling a tool that
was left out gets pydantic-ai's unknown-tool retry, which names the tools that do
exist.

## Changing what the model is told

Every line of the instructions block, the heading of the unavailable list
included, is a field of
[`AgentConventions`](reference.md#rest_framework_pydantic_ai.AgentConventions),
and so are three sentences outside it: the description a handle field gets when
it declares none, the sentence scoping a paged tool's read-shaping parameters,
and the retry for a missing argument. `conventions=` changes them one line at a
time. Each field defaults to the toolset's own wording, so `AgentConventions()`
changes nothing, and leaving `conventions=` unset is the same as passing it.

Nothing else the toolset writes is a field. The descriptions of the arguments it
adds, such as `limit` and `page`, stay fixed, and so do its other retries: a
`limit` or `page` that is not a positive integer, an `ordering` the tool does not
take, a service asking for more input, and the opening of a
[render retry](#a-selection-the-serializer-rejects), whose closing scope
sentence is the one part that is a field.

```python
from rest_framework_pydantic_ai import AgentConventions, SpecToolset

toolset = SpecToolset(
    specs,
    conventions=AgentConventions(
        # {page_size} is the page an omitted `limit` is served; a literal brace is doubled.
        pagination=(
            "- Collections come back one page at a time, as "
            '{{"items": [...], "page": 1, "totalPages": N, "hasNext": true|false}}, '
            "{page_size} items unless you pass `limit`. Ask for the next `page` while "
            "`hasNext` is true."
        ),
        handles=None,  # this audience never sees identifiers, so say nothing about them
    ),
)
```

**A field changes what a line says, never whether it appears.** The toolset still
decides that, exactly as before: the `pagination` line above is still absent from
a toolset with no list tool, and the block is still re-derived on each step an
[unmet condition](#an-operation-that-is-unavailable-right-now-is-left-out) leaves
a tool out, so an overridden line goes with the tool it describes. `None` drops a
line wherever it would have been said.

| Field | Where it lands | Said when | Placeholders |
| --- | --- | --- | --- |
| `base` | instructions, first | always | none |
| `pagination` | instructions | some tool returns a page | `{page_size}` |
| `ordering` | instructions | some tool advertises a sort argument | `{names}` |
| `handles` | instructions | some tool's output marks a handle | none |
| `read_shaping` | instructions | some tool declares a `QueryParam` | `{names}` |
| `read_shaping_on_pages` | instructions, after `read_shaping` | that tool also returns a page | none |
| `unavailable_heading` | instructions, last, above the tools left out | a condition left a tool out this step | none |
| `handle_field_description` | a handle field's output schema | the field declares no description of its own | none |
| `query_param_on_pages` | a `QueryParam`'s description, and the [render retry](#a-selection-the-serializer-rejects) | the tool returns a page | none |
| `missing_arguments` | the retry for a call missing an argument | a selector parameter or a required `UrlKwarg` is left out | `{names}` |

A few rules follow from what each field is:

- **Every field is a `str.format` template**, placeholders or not, so a literal
  brace is written twice everywhere. `{names}` is each name in backticks, joined
  with `, ` (sorted, except `ordering`'s, which follow the tools).
- **A typo fails at startup.** A placeholder the field does not accept, an
  unbalanced brace, or a format spec its value cannot take raises
  `ImproperlyConfigured` naming the field when the `AgentConventions` is built.
- **`None` on `read_shaping` drops `read_shaping_on_pages` too**, because that
  sentence continues it; `None` on `unavailable_heading` drops the list beneath
  it. To drop a line, use `None`, not `""`, which leaves a blank line.
  `missing_arguments` cannot be `None`, empty or only whitespace: it is the
  retry's whole text.
- **`missing_arguments` covers the two checks made before the input serializer
  runs.** A field the input serializer requires is still reported in the
  serializer's own words.

**Some lines state facts about behaviour, and an override owns keeping them
true.** `pagination` and `query_param_on_pages` name the page envelope's keys, and
`base` says which failures come back as a retry and which as a failed call. The
toolset goes on behaving as the defaults describe whatever the text says, so a
rewording that drops or contradicts one of those facts tells the model something
false.

**`conventions=` and `instructions=`.** `instructions=` replaces the whole
derived block, so changing one of its lines (`base` through
`read_shaping_on_pages`) beside it would be ignored, and the toolset refuses the
pair with `ImproperlyConfigured` naming the fields. The other four land outside
the block and apply either way, `unavailable_heading` included, since the list of
unavailable operations is appended after an override too. Prefer `conventions=`
where it can say what you need: an override is a copy of the block, so it keeps
advising about pagination on a step with no list tool, and it misses every
correction a later release makes to the wording.

`SpecCapability` takes the same keyword, and
[`SpecCapability.from_toolset`](reference.md#rest_framework_pydantic_ai.SpecCapability.from_toolset)
keeps the conventions of the toolset it wraps. They reach this toolset's tools
only: the MCP transport, djangorestframework-mcp-server, writes its own schemas
and retries, so a model reading the same specs over MCP is told what that
transport is configured to say.

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

`IGNORE` drops a key no parameter declares. A parameter marked `NotClientInput`
is still declared, only left out of the schema, so a value the model sends for
it reaches the selector under `IGNORE`.

## A list as input

A `ServiceSpec` declaring `many=True` becomes a tool that takes its list as **one
named argument**. Its input validates as a JSON array, and a model's tool
arguments are always a JSON object, so the array travels under the argument the
spec's `many_argument` names, `items` unless it names another:

```python
def create_widgets(*, data, user):
    """Create several widgets in one call."""
    return [Widget.objects.create(owner=user, **item) for item in data]


create_widgets_spec = ServiceSpec(
    service=create_widgets,
    input_serializer=WidgetInputSerializer,
    many=True,
    many_argument="widgets",  # optional; the default is "items"
    output_selector_spec=SelectorSpec(
        kind=SelectorKind.RETRIEVE, output_serializer=WidgetSerializer
    ),
)
```

The model sends `{"widgets": [{...}, {...}]}` and the service receives the list,
exactly as it would from a REST view's bare array body. The tool's parameter
schema says so: an object with that one required property, an array of the item
schema, carrying the list serializer's `allow_empty`, `min_length` and
`max_length` as `minItems` / `maxItems`, and `additionalProperties: false`. A
declared `QueryParam` or `UrlKwarg` is advertised beside it, since the toolset
takes those out of the arguments before dispatch. The result is the rendered
list, and the tool's `return_schema` describes it as an array even though the
`output_selector_spec` is `RETRIEVE`.

What the model is told when a call is wrong:

- **An invalid item** comes back as a `ModelRetry` naming the argument, then the
  invalid item's index, then the field, on every Django REST framework version
  this package supports. The model reads `widgets[1].price: This field is
  required.`, the same rendering every validation error gets, and corrects that
  one item.
- **The argument missing, `null` or not a list** comes back under the argument
  in the words that field would use: `widgets: This field is required.`
- **Any other argument sent beside the list** comes back as
  `Unexpected argument(s): 'note'.` under every `unknown_arguments` policy,
  because the service receives only the list and the argument would have
  nowhere to go. The policy still decides what happens to an undeclared key
  inside an item.

A `QueryParam` or `UrlKwarg` sharing the list's argument name is refused with
`ImproperlyConfigured` when the toolset is built, whether it is declared
toolset-wide, per tool or on the registry entry's `OfflineContract`: it would be
taken out of the arguments first and carry the list away with it.

A tool that needs arguments **beside** its list cannot be `many=True`. Name the
list as a field of the input serializer instead, and loop over it:

```python
class BulkWidgetInput(serializers.Serializer):
    items = WidgetInputSerializer(many=True)
    dry_run = serializers.BooleanField(default=False)
```

The model sends `{"items": [...], "dry_run": true}`, and an invalid item comes
back placing its errors at its index, as `items[1].price: ...`. Django REST
framework keys them by index from 3.18 and lists them below it, with an empty
entry for each valid item; the model reads the same line either way.

## Ordering

**The `filter_set` owns ordering.** Declare a django-filter `OrderingFilter`
named `ordering` on the selector's FilterSet and you are done:

```python
import django_filters


class OrderFilterSet(django_filters.FilterSet):
    ordering = django_filters.OrderingFilter(
        fields=(("created_at", "created"), ("total_cents", "total")),
    )

    class Meta:
        model = Order
        fields = ["status"]
```

drf-services reflects that filter into the tool's input schema as its public
choices (`created`, `-created`, `total`, `-total`) — `OrderingFilter` subclasses
`ChoiceFilter`, which the schema generator maps to a set of `const` options
carrying the filter's own labels — so the model is told exactly what it may sort
by, in the words the FilterSet uses. At call time the value is handed to
the FilterSet as filter data: it validates the choice, applies its own
`param_map`, and a value outside the enum comes back as a `ModelRetry`. The
toolset contributes nothing and takes nothing away.

One vocabulary, one declaration site, and the same ordering your HTTP views
already serve.

The filter's name is yours to pick — an `OrderingFilter` declared as `sorting`
is found and used the same way. What the schema advertises is what the model may
send, under whatever it is called.

### A list selector with no `filter_set`

A selector that takes its own sort argument works too: declare it on the
callable and it is reflected into the tool schema like any other parameter, and
handed to the callable to apply.

```python
def list_orders(user, ordering: str = "-created_at"):
    """List the acting user's orders."""
    return Order.objects.filter(customer=user).order_by(ordering)
```

Prefer the `FilterSet` where there is one: it validates the value against a
published set of choices before anything reaches the ORM, while a bare parameter
is only as safe as what the selector does with it.

### Migrating from `ordering_fields`

`SpecToolset(specs, ordering_fields=[...])` and its per-tool
`tool_ordering_fields` form were deprecated in 0.16.0 and have now been removed;
passing either raises `TypeError` at construction. Move the names onto an
`OrderingFilter` as `(orm_path, public_name)` pairs — the FilterSet at the top of
this section is exactly `ordering_fields=["created_at", "total_cents"]`
rewritten — and drop the argument.

The vocabularies are not the same, and that is the point of the move: the knob's
values were raw **ORM paths**, because the toolset applied them with
`queryset.order_by` itself, while a FilterSet's choices are public names it maps
through its own `param_map`. Picking public names is the migration's one
decision — they are what the model sees, so give them the words a reader would
use.

## Read-shaping query params

`page` / `limit` are built in for list selectors and `ordering` comes from the
`filter_set`, but you can register your own request-level params with
[`QueryParam`](reference.md#rest_framework_pydantic_ai.QueryParam). Each is
advertised as a tool arg, then — instead of reaching the spec as an input — seeded
into `request.query_params` over the off-HTTP path. That is for whatever reads
`request.query_params` **directly**: django-restql field selection, or a custom
serializer that branches on the query string.

!!! note "You don't need this for `filter_set`"
    A `SelectorSpec.filter_set`'s fields are already generated into the tool's
    input schema (the `[filter]` extra) and flow through as ordinary `params` —
    which `dispatch_spec` hands the FilterSet as its `filter_data`. So the model
    can filter a list selector with no `QueryParam` declaration at all, and the
    same goes for [ordering](#ordering); `QueryParam` is only for params a
    serializer reads off `request.query_params`.

```python
from rest_framework_pydantic_ai import QueryParam

toolset = SpecToolset(
    specs,
    # applies to every tool
    query_params=[QueryParam("query", description="django-restql field selection")],
    # or scope params to one tool
    tool_query_params={"list_orders": [QueryParam("status", default="open")]},
)
```

A registered param is popped before dispatch, so `unknown_arguments` never flags
it; a declared `default` is seeded when the model omits the arg or sends it as
`null`, which never reaches the query string itself. (Names can't be
`page` / `limit` / `ordering` — those are reserved transport keys. `ordering` is
reserved even when a `filter_set` owns it: a registered channel pops the value at
call time, so the FilterSet would never see it.)
Requires `djangorestframework-services>=0.23`, which added the
`build_offline_context(query_params=…)` seam.

### On a list tool, a param shapes each row

Every list tool returns a page, `{"items": [...], "page": 1, "totalPages": N,
"hasNext": ...}`, but the serializer that reads a read-shaping param renders one
row at a time and never sees the envelope. So a selection written against the
documented shape, `fields=items` or django-restql's `{items{id, name}}`, asks
each row for an `items` field it does not have. The toolset tells the model this in two places:

- each `QueryParam` on a list tool has "On a paged result it applies to each item
  in `items`, never to the page envelope (`items`, `page`, `totalPages`,
  `hasNext`)." appended to its description, or as its description when it
  declares none; a retrieve tool's param is left as declared;
- the read-shaping line of the instructions the toolset derives says the same,
  "On a tool that returns a page, they apply to each item in `items`, never to
  the page itself.", when some list tool declares a `QueryParam`.

An `instructions=` override replaces the derived block, and this line with it:
an override written before this sentence existed does not gain it, so add the
advice to your own text if your tools page and take a selection. Changing the
lines you need with [`conventions=`](#changing-what-the-model-is-told) instead
keeps this one, and both sentences are fields there (`read_shaping_on_pages` and
`query_param_on_pages`).

### A selection the serializer rejects

A read-shaping param is the one argument used **while the result is rendered**
rather than while the call runs, so it is the one that can fail after the work
is done. When the output serializer raises a `ValidationError` (or a
`ServiceValidationError`) while rendering, and the model supplied a value for at
least one of the tool's read-shaping params, the model gets a retry naming the
argument:

```text
`fields` was rejected while rendering the result: Unknown field `items`. On a
paged result it applies to each item in `items`, never to the page envelope
(`items`, `page`, `totalPages`, `hasNext`).
```

What follows the colon is the serializer's own message, verbatim. The toolset
never reads a param's value, so this works the same for a `fields` your own
serializer parses, for django-restql's `query`, or for anything else that raises.
The last sentence is there only on a list tool. With several params supplied,
all of them are named, since the serializer does not say which one it refused.
`exception_map=` / `translate_exception` sees the error first, as on the dispatch
path.

The same error **stays loud**, and ends the run as before, when nothing the model
sent shaped the render: no read-shaping value supplied, an explicit `null`, or a
value that came from a declared `default`. A retry cannot fix a serializer that
fails on its own or a default that is wrong, and would hide the bug behind the
retry budget. Only validation errors are converted; an `AttributeError` in a
serializer is a bug whatever the model sent.

!!! tip "Use strict selection on tools an agent calls"
    This only works if the serializer says no. Make a name it does not know
    raise a `ValidationError` (django-restql does by default), and the model
    corrects itself in one retry.
    A tolerant selection, one that drops fields it cannot find, turns the same
    mistake into a page of empty rows, `"items": [{}, {}]`, which is a successful
    result as far as anything downstream can tell. The toolset cannot tell that
    apart from a real answer without knowing the param's grammar, so it does not
    try.

## URL-derived values (route captures)

Over HTTP a nested route (`/projects/{project_pk}/widgets/`) supplies
`project_pk` from the URL, and a selector reads it from `view.kwargs` — directly,
or through a `spec.kwargs` provider that scopes by it (a tenant/role lookup). Off
the HTTP path there is no route, so register the value with
[`UrlKwarg`](reference.md#rest_framework_pydantic_ai.UrlKwarg). It is advertised
as a tool arg, then popped and seeded into `build_offline_context(kwargs=…)`,
from where drf-services spreads it into the selector / target pools —
authoritative over the spec `params`, below a `spec.kwargs` provider (mirroring
HTTP precedence exactly).

```python
from rest_framework_pydantic_ai import UrlKwarg

toolset = SpecToolset(
    specs,
    url_kwargs=[UrlKwarg("project_pk", type="integer", description="owning project")],
    # or scope to one tool: tool_url_kwargs={"list_widgets": [UrlKwarg("project_pk")]}
)
```

Reach for `UrlKwarg` when the value is **request state** rather than an ordinary
argument to the callable — the axis is where the value has to land, not whether
it is advertised:

- a scoping `spec.kwargs` provider that reads `view.kwargs` — the case ordinary
  `params` cannot cover, because the provider reads `view.kwargs`, not `params`;
- a closed-surface spec whose route capture must be model-suppliable.

Like `QueryParam`, a registered kwarg is popped before dispatch (so
`unknown_arguments` never flags it) and its `default` is seeded when the model
omits it. A kwarg with a `default` therefore fills a selector parameter of the
same name, and the tool does not require it; one without a default reaches the
selector only when the model sends it, so the selector's own signature still
decides whether the tool requires it. A name can't be `page` / `limit` / `ordering`, nor one of drf-services'
pool seeds (`request` / `user` / `data` / `instance` / `serializer` /
`collection` / `progress` — a caller must not be able to route a value onto
those) or a name
registered in [`pool_seeds`](#project-pool-seeds), nor be registered as both a
`QueryParam` and a `UrlKwarg` on the same tool.

A capture the spec genuinely cannot run without takes `required=True`:

```python
UrlKwarg("project_pk", type="integer", required=True)
```

The name joins the tool's `required` list, so the model is told up front. Because
a schema hint is only a hint — models omit required arguments routinely — a call
that omits it raises `ModelRetry` naming the argument, giving the model a turn to
supply it rather than failing deeper in. `required` can't be combined with a
`default` (a default always satisfies the argument, so requiring it would be a
no-op); that raises at construction.

### A reflected `**extras` key is not a route capture

A selector typed `def list_widgets(user, **extras: Unpack[WidgetExtras])` that
reads `extras["project_pk"]` already has that key reflected into the tool schema
by drf-services (0.26+) — no `UrlKwarg` needed **for the selector itself**, which
receives it through `params`. Marking it `InputRequired` makes the model supply
it; that is a *schema* statement and changes nothing about where the value lands.

The two declarations answer different questions, and only one of them puts a
value on the request:

| | reflected `**extras` key (± `InputRequired`) | registered `UrlKwarg` |
| --- | --- | --- |
| In the tool schema | yes | yes |
| Can be required | yes (`InputRequired`) | yes (`required=True`, plus a `ModelRetry` when omitted) |
| Reaches the selector | yes, via `params` | yes, via the `view.kwargs` spread |
| Reaches `view.kwargs` | **no** | yes |
| Ranks above caller-supplied `params` | no — it *is* caller input | yes |

So anything that reads request state rather than its own arguments — a
`spec.kwargs` provider, `extend_queryset`, a permission class, an
`output_serializer_context` provider — sees nothing for a reflected-only key. A
scoping provider doing `view.kwargs.get("project_pk")` returns `None` and
**mis-scopes every call** instead of failing, which is the failure mode worth
naming: it is silent.

Register the `UrlKwarg` as well when the value is scope. It is a strict superset
— the selector still receives it in `**extras`, the schema keeps one property and
one `required` entry (an explicit `UrlKwarg` wins the merge over a reflected key
of the same name), and the provider gets its value:

```python
# project_pk reflected from WidgetExtras *and* registered here:
#   selector's extras -> 7      view.kwargs -> {"project_pk": 7}
toolset = SpecToolset(
    specs,
    tool_url_kwargs={"list_widgets": [UrlKwarg("project_pk", type="integer", required=True)]},
)
```

That split mirrors HTTP, where a route capture arrives in the URL and never in
the body — which is what makes it unspoofable. Off the HTTP path, `params` are
whatever the model chose; a `UrlKwarg` value outranks them. If a provider scopes
by it, it has to come through the channel that carries that precedence.

`UrlKwarg` and `QueryParam` are
[drf-services' types](https://github.com/Artui/djangorestframework-services/blob/main/rest_framework_services/types/url_kwarg.py),
re-exported here — the declaration is the same whichever transport carries it,
and this package's copy had drifted from the MCP transport's on which names each
reserved. `from rest_framework_pydantic_ai import UrlKwarg, QueryParam` keeps
working. Requires `djangorestframework-services>=0.28.1`.

## Absolute URLs (file and hyperlinked fields)

Off the HTTP path there is no ambient request, so there is no origin to build
absolute URLs from — and DRF's `FileField`, `HyperlinkedIdentityField`, and
`HyperlinkedRelatedField` call `request.build_absolute_uri()` for every value.
Name your origin and they resolve:

```python
toolset = SpecToolset(specs, host="https://app.example.com")
```

`host` accepts `"example.com"`, `"example.com:8000"`, or a full origin whose
scheme decides whether links are https. It is toolset-wide, with no per-tool
variant: an origin is a property of the deployment, not of a tool.

Left unset, those fields produce **relative** URLs (`/media/doc.pdf`) — usable,
and exactly what they fall back to on their own when no request is in the
serializer context. Nothing is inferred: only your project knows its public
origin, and a guess would emit confidently-wrong links that look valid.
Requires `djangorestframework-services>=0.29.1`.

## Error handling

The toolset maps drf-services' failure kinds onto the Pydantic-AI model loop:

| drf-services outcome | What the agent sees |
| --- | --- |
| `ServiceValidationError` (bad input) | `ModelRetry` with the field errors, one `field: message` line each — the model self-corrects |
| A read-shaping value the output serializer rejects while rendering | `ModelRetry` naming the argument — see [A selection the serializer rejects](#a-selection-the-serializer-rejects) |
| `ActionUnavailable` (an `Affordance` condition not met) | `ToolFailed` with the reason followed by the rule's code — `The books are closed. (code: books_closed)`; see below |
| `ServiceError` (business rule) | `ToolFailed` with the rule's own message — a failed result the model reads and reports |
| Unresolved instance | `ToolFailed("not found")`, unless the spec sets `allow_none=True`, when the tool returns `None` |
| A dispatch past `dispatch_timeout` | `ToolFailed` — abandoned, with the sentence telling the model to narrow and call again |
| A rendered result over `max_result_bytes` | `ToolFailed` — refused rather than truncated, since a partial payload looks complete |
| Unexpected argument (default `REJECT`) | `ModelRetry` naming the unknown key |
| A required argument left out: a selector parameter with no default, or a service's target lookup such as `pk` | `ModelRetry` naming each such parameter left out; a missing serializer field is reported by the serializer when the call runs — see [What a tool asks the model for](#what-a-tool-asks-the-model-for) |
| An invalid item in a `many=True` list | `ModelRetry` with the errors keyed under the list's argument, then the item's index — see [A list as input](#a-list-as-input) |
| An argument sent beside a `many=True` list | `ModelRetry` naming it, whatever `unknown_arguments` says |
| Non-integer `page` / `limit` | `ModelRetry` — naming what is accepted |
| An `ordering` sent to a list tool that advertises no sort at all | `ModelRetry` — saying the tool has none, rather than letting it fall through as an unknown key |
| A `limit` over `max_page_size`, or a `page` past the last one | Clamped, not refused — the envelope reports the `page` and `totalPages` actually served, so the clamp is visible rather than silent |
| An `ordering` outside a `filter_set`'s `OrderingFilter` choices | `ModelRetry` — the FilterSet rejects it, which arrives as the `ValidationError` row above |
| A FilterSet `param_map` target that isn't a real column | Django's `FieldError` propagates — an author's error the model cannot correct by picking a different sort, and one that can't be checked at construction without a queryset |
| Denied `permission_classes` (class-level `has_permission` **or** object-level `has_object_permission`) | `PermissionDenied` is raised and aborts the run — see the caveat below |

!!! warning "A tool-failure policy does *not* change the last row"
    "Aborts the run" is what a plain `pydantic_ai.Agent` does: nothing catches
    the exception, so it propagates out of `agent.run`. A tool-failure policy
    normally converts an escaping exception into a failed-tool result and lets
    the run continue — but `django-pydantic-agent`'s exempts an authorization
    refusal from exactly that, by default, and `django-ag-ui` inherits the
    exemption. So a denial aborts the run under those hosts too.

    That exemption is deliberate and worth knowing rather than working around:
    converting a denial would leave the run alive with the model free to try the
    next row, and a refusal a model can tell apart from a missing row turns a
    permission boundary into an existence oracle over rows the acting user
    cannot read — inside one turn, spending no retry budget.

    The denial itself is unaffected on every host: nothing is dispatched and no
    data is rendered.

!!! danger "Your own exception has to *be* a `ServiceError`"
    Every row above is matched by type. A service that raises its own error
    class gets none of this unless that class derives from drf-services'
    `ServiceError`:

    ```python
    from rest_framework_services import ServiceError


    class BookingError(ServiceError):  # not Exception
        """Something the booking rules refuse."""
    ```

    Deriving from `Exception` is the ordinary thing to reach for, and the
    failure is silent in a way worth spelling out, because it is not confined to
    the agent. The same exception escapes the DRF view as a **500 for what is
    plainly a 409**, escapes MCP as a protocol error, and aborts an agent run
    rather than settling the call as failed. Nothing warns — not at
    declaration, not at mount, not at dispatch.

    Two separate projects made this exact mistake and neither test suite could
    see it, because every write test asserted a path that succeeds. If a project
    genuinely cannot change its exception's base class, `exception_map=` is the
    other door: it takes the type and returns what the model should be told.

### A refused affordance names its code

drf-services raises `ActionUnavailable` when one of a spec's `Affordance`
conditions is not met, with the affordance's `reason` as the message and its
`code` as an attribute. `ToolFailed` carries a message and nothing else, and that
message is also what a transport forwards as the tool result, so the toolset puts
both into it — the reason first, then the code, labelled:

```python
from rest_framework_services import Affordance, ServiceSpec


def post_invoice(data, user):
    """Post an invoice to the current period."""
    ...


post_invoice_spec = ServiceSpec(
    service=post_invoice,
    input_serializer=InvoiceInputSerializer,
    affordances=[
        Affordance(
            code="books_closed",
            reason="The books are closed.",
            when=lambda: Period.current().is_open,
        ),
    ],
)
```

While the period is closed the tool is [left out of the
catalog](#an-operation-that-is-unavailable-right-now-is-left-out), so a run reaches
this refusal only through a listing that went stale: the step was offered the tool,
and the period closed before the call arrived. A condition on the row is refused
the same way, and it is never asked when the catalog is listed. The failed tool
result's content, which is what the model reads and what a transport streams, is:

```text
The books are closed. (code: books_closed)
```

The code is the part that connects the refusal to what the model may already have
read. A selector naming the same spec in its own `affordances`
(`affordances={"post_invoice": post_invoice_spec}`) answers it on each row it
renders, under `affordances.post_invoice`, as
`{"available": false, "code": "books_closed", "reason": "The books are closed."}`
— and the reason is a sentence a project may reword while the code stays put. The
conventions block `get_instructions` returns tells the model a refusal may end
this way. A `ServiceError` or `ServiceConflict` raised by hand has
no code and keeps its message exactly as written.

**The suffix is written for the model and for a person reading a tool card, not
for a program.** It is meant to read the same on every agent transport serving
these specs, and its wording is not an interface. A program that branches on the
code reads `.code` off the exception instead: the `ActionUnavailable` is the
`ToolFailed`'s `__cause__`, and `translate_exception` receives it before the
toolset's own handling runs. A handler returned from there, or registered in
`exception_map` for `ActionUnavailable` or any class above it, replaces the
sentence along with the rest of the default.

### Why the failed rows raise instead of returning

Every `ToolFailed` row above used to be a returned `{"error": "..."}` dict. The
model read the same sentence either way, so the change is not about what it
sees — it is about what everything *else* sees. Pydantic-AI marks an ordinary
return `outcome="success"` on the resulting `ToolReturnPart`; only a raised
`ToolFailed` marks it `"failed"`. Returning the dict therefore made a refusal
indistinguishable from an answer to a log, an audit record, or a transport
streaming the call to a browser, all of which had nothing but the payload's
wording to go on.

`ToolFailed` keeps the two properties the dict was chosen for: it spends none of
the tool's retry budget, and it does not end the run. It also prepends no
correction instructions, which `ModelRetry` does — right for a bad argument, and
wrong for a conflict the model cannot argue with.

If a failure is one your project would rather express as an ordinary result,
`exception_map` still takes a handler that **returns** a value, and a returned
value is still marked `"success"`.

Each `ModelRetry` row consumes one unit of the tool's retry budget: after
`max_retries` failed attempts (default `1`, pydantic-ai's function-tool
default) the run aborts with `UnexpectedModelBehavior`. Raise it for models
that need more attempts to converge:

```python
toolset = SpecToolset(specs, max_retries=3)
```
