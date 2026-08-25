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

## What it costs

Nothing per call. The markings are pure in the serializer class, like the tool
schemas, so `SpecToolset` resolves them once at construction and reuses them.

## Your REST API is unaffected

DRF consults `style` only in `HTMLFormRenderer`, and only for its own keys. A
marked-up serializer renders byte-identically behind a viewset — which is the
point: one serializer, whichever consumer is reading.
