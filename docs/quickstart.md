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
docstring, the parameter schema from the spec's input serializer, the
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

`max_page_size` lowers the default and advertises itself as JSON-Schema
`maximum` on `limit`:

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

## Custom identity

If your project carries identity on a richer deps object, hand the toolset a
`get_user` extractor instead of using `AgentDeps`:

```python
toolset = SpecToolset(specs, get_user=lambda ctx: ctx.deps.principal.user)
```

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

`get_tools` advertises every spec to every run. A tool whose permissions will
deny this caller is still listed — the denial happens on the call. That is
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
it; a declared `default` is seeded when the model omits the arg. (Names can't be
`page` / `limit` / `ordering` — those are reserved transport keys. `ordering` is
reserved even when a `filter_set` owns it: a registered channel pops the value at
call time, so the FilterSet would never see it.)
Requires `djangorestframework-services>=0.23`, which added the
`build_offline_context(query_params=…)` seam.

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
omits it. A name can't be `page` / `limit` / `ordering`, nor one of drf-services'
pool seeds (`request` / `user` / `data` / `instance` / `serializer` /
`collection` — a caller must not be able to route a value onto those), nor be
registered as both a `QueryParam` and a `UrlKwarg` on the same tool.

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
| `ServiceValidationError` (bad input) | `ModelRetry` with the field errors — the model self-corrects |
| `ServiceError` (business rule) | `ToolFailed` with the rule's own message — a failed result the model reads and reports |
| Unresolved instance | `ToolFailed("not found")` |
| A dispatch past `dispatch_timeout` | `ToolFailed` — abandoned, with the sentence telling the model to narrow and call again |
| A rendered result over `max_result_bytes` | `ToolFailed` — refused rather than truncated, since a partial payload looks complete |
| Unexpected argument (default `REJECT`) | `ModelRetry` naming the unknown key |
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
