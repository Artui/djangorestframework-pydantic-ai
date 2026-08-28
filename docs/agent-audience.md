# Hide plumbing from the model

A serializer written for your REST API is handed to the model verbatim when the
same spec becomes a tool. Everything in it is equally visible, so records get
referred to by primary key, a status reads as `IN_STOCK` rather than "In stock",
and internal fields get narrated as if they were content.

**This bites harder in-process than it does over MCP.** A `ToolDefinition`
carries a parameter schema and no output schema, so the payload is the model's
only view of a result — a nicer label that lived only in a schema would never
reach it.

## Mark the fields

The marking is `AgentField`, from
[djangorestframework-services](https://artui.github.io/djangorestframework-services/),
in DRF's per-field `style` bag. It reaches a `ModelSerializer` through
`Meta.extra_kwargs`, so nothing has to be redeclared:

```python
from rest_framework import serializers
from rest_framework_services import AGENT, AgentField


class WidgetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Widget
        fields = ["id", "name", "price", "status"]
        extra_kwargs = {
            "id": {"style": {AGENT: AgentField.handle("Widget handle.")}},
            "price": {"style": {AGENT: AgentField.hidden()}},
            "name": {"style": {AGENT: AgentField.label()}},
        }
```

Nothing changes at the toolset. A tool rendering through that serializer returns:

```python
[{"id": 1, "name": "Sprocket", "status": "In stock"}]
```

`price` is gone. `status` reads as a person would say it. `id` is untouched — a
handle is another tool's input, so its value is never re-spelled.

The derived instructions gain one line, and only when some tool in the toolset
actually returns a handle:

> Some tools return opaque identifier fields, described as such in the tool's
> output. Pass them to other tools that ask for one; refer to records by their
> name in anything you say, never by the identifier.

## One tool that needs what its siblings hide

The serializer is the declaration and stays authoritative. The exception is a
tool whose whole job is handing back something the others drop — a lookup
returning the identifier a list view hides. That is an override, and it goes on
the registry entry:

```python
from rest_framework_services import AgentContract, AgentField

registry.register(
    "lookup_widget",
    specs.lookup_widget,
    agent_contract=AgentContract(field_audiences={"price": AgentField()}),
)
```

`SpecToolset(registry)` reads it, and so does an MCP server registering the same
entry — which is the reason it lives there rather than in either constructor.
`AgentField`'s axis is *audience*, not transport: an in-process toolset and an
MCP server want the same thing as each other and something different from a
browser, so a field hidden from one agent caller and visible to another is a bug
you find in a transcript rather than in a test.

A project that genuinely wants two agent audiences with different visibility
does not want this. It wants two serializers.

Two fields left claiming `AgentField.label()` raises `ImproperlyConfigured`
naming the tool, at construction: a record has one name.

## What it costs

Nothing per call. The markings are pure in the serializer class, like the tool
schemas, so `SpecToolset` resolves them once at construction and reuses them.

## Your REST API is unaffected

DRF consults `style` only in `HTMLFormRenderer`, and only for its own keys. A
marked-up serializer renders byte-identically behind a viewset — which is the
point: one serializer, whichever consumer is reading.
