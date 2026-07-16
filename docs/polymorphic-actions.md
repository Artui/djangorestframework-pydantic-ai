# Polymorphic actions as tools

drf-services 0.25 added `PolymorphicServiceSpec` — one action that accepts
*N* mutually exclusive payload shapes and picks a variant at dispatch time
from a `discriminator` callable reading the request body. It earns its keep on
HTTP, where a URL is a scarce, addressable resource and one endpoint accepting
a tagged-union body can beat *N* endpoints.

An agent toolset has no such scarcity, and a model calls tools far more
reliably when each is a single-purpose tool with a flat parameter schema than
when it has to choose a union arm *and* set a discriminator field correctly.
So don't hand a `PolymorphicServiceSpec` to `SpecToolset` as one union tool —
**expand it into one tool per variant.** The tool *name* selects the variant,
so the discriminator never runs and the model never sees a union.

`SpecToolset` already takes a plain `Mapping[str, Spec]` (tool name → spec),
and `PolymorphicServiceSpec.specs` is a public `Mapping[str, ServiceSpec]`
(variant key → full `ServiceSpec`, each with its own `input_serializer`,
`service`, and output pipeline). So the expansion is a dict spread:

```python
from rest_framework_services import PolymorphicServiceSpec, ServiceSpec
from rest_framework_pydantic_ai import SpecToolset

moderate = PolymorphicServiceSpec(
    discriminator=lambda *, data: data["op"],       # HTTP-only; unused here
    specs={
        "approve": ServiceSpec(service=approve_document, output_selector_spec=DOC_OUT),
        "reject": ServiceSpec(service=reject_document, output_selector_spec=DOC_OUT),
    },
)

toolset = SpecToolset({
    "list_orders": orders_selector_spec,
    # one tool per variant → moderate_document_approve, moderate_document_reject
    **{f"moderate_document_{key}": variant for key, variant in moderate.specs.items()},
})
```

Each expanded entry is an ordinary service tool: its parameter schema comes
from that variant's `input_serializer`, and a call dispatches through the usual
`input_serializer → run_service(atomic) → output` pipeline with the same
`permission_classes` checks. Nothing about `PolymorphicServiceSpec` reaches the
model — the discriminator is a server-side HTTP concern, and the agent sees a
flat menu of clear tools.

See the [Quickstart](quickstart.md) for wiring a toolset to an agent.
