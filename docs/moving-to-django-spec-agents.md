# Moving to django-spec-agents

`djangorestframework-pydantic-ai` is no longer developed. Version 0.36.2 is its
final release, and its code is identical to 0.36.1. Development continues as
[django-spec-agents](https://artui.github.io/django-spec-agents/), which merges
this package with `django-pydantic-agent`. What this package did is now
`django_spec_agents.spec_tools`.

Nothing here warns, breaks or re-exports: an install that keeps this package
keeps working exactly as 0.36.1 did. This page is for the day you move.

## Install

```bash
pip install "django-spec-agents[specs]"
```

The `[specs]` extra is what brings in the dependencies this package declared
(`djangorestframework-services`, `django-service-specs` and the DRF
requirement). The `[filter]` extra keeps its name and its meaning, so a project
that installed `djangorestframework-pydantic-ai[filter]` installs
`django-spec-agents[specs,filter]`.

If you reach this package through `django-ag-ui[spec-tools]`, that extra still
installs this package for one more minor release. Its replacement is
`django-ag-ui[specs]`.

## Where each name went

Every name in this package's `__all__` at 0.36.1:

| Was | Now |
| --- | --- |
| `rest_framework_pydantic_ai.SpecToolset` | `django_spec_agents.spec_tools.SpecToolset` |
| `rest_framework_pydantic_ai.SpecCapability` | `django_spec_agents.spec_tools.SpecCapability` |
| `rest_framework_pydantic_ai.AgentConventions` | `django_spec_agents.spec_tools.AgentConventions` |
| `rest_framework_pydantic_ai.AgentDeps` | `django_spec_agents.AgentDeps`, the agent host's own deps type |
| `rest_framework_pydantic_ai.QueryParam` | `rest_framework_services.QueryParam` |
| `rest_framework_pydantic_ai.UrlKwarg` | `rest_framework_services.UrlKwarg` |
| `rest_framework_pydantic_ai.__version__` | `django_spec_agents.__version__`, the version of the merged package |

`django_spec_agents.spec_tools` exports `SpecToolset`, `SpecCapability` and
`AgentConventions` and nothing else. `QueryParam` and `UrlKwarg` were only ever
re-exports of `djangorestframework-services`, so they are imported from there:

```python
# Before
from rest_framework_pydantic_ai import AgentDeps, QueryParam, SpecToolset, UrlKwarg

# After
from django_spec_agents import AgentDeps
from django_spec_agents.spec_tools import SpecToolset
from rest_framework_services import QueryParam, UrlKwarg
```

### Modules

Module by module, for an import that reaches past the package root:

| Was | Now |
| --- | --- |
| `rest_framework_pydantic_ai.spec_toolset` | `django_spec_agents.spec_tools.spec_toolset` |
| `rest_framework_pydantic_ai.spec_capability` | `django_spec_agents.spec_tools.spec_capability` |
| `rest_framework_pydantic_ai.types.agent_conventions` | `django_spec_agents.spec_tools.types.agent_conventions` |
| `rest_framework_pydantic_ai.types.agent_deps` | `django_spec_agents.agent.types.agent_deps`, or import `AgentDeps` from `django_spec_agents` |
| `rest_framework_pydantic_ai.types.query_param` | removed; import `QueryParam` from `rest_framework_services` |
| `rest_framework_pydantic_ai.types.url_kwarg` | removed; import `UrlKwarg` from `rest_framework_services` |
| `rest_framework_pydantic_ai.version` | removed; read `django_spec_agents.__version__` |
| `rest_framework_pydantic_ai.testing` | `django_spec_agents.testing`, with the same three doubles: `tool_calling_model`, `streaming_tool_calling_model` and `instruction_capturing_model` |

## Mount keywords that are gone

`SpecToolset` and `SpecCapability` no longer take `query_params=`,
`tool_query_params=`, `url_kwargs=` or `tool_url_kwargs=`; passing one raises
`TypeError`. They declared at one mount what the spec's `reads` declare for every
transport, which is how two mounts came to disagree about one operation. This
package already warned with `DeprecationWarning` when you passed one; the new
package removes them. Declare what a spec reads on the spec itself, with `reads=`, and build the
toolset from the registry:

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

A value the server supplies rather than the model, which is what a mount-level
`url_kwargs=` binding was often used for, goes through `tool_server_arguments=`
and the kernel's `ServerArguments`. Both are unchanged from this package's 0.36
series, so existing uses move over as they are:

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

Every other keyword of `SpecToolset` and `SpecCapability` is unchanged.

## Names that change without an import error

- **The logger** is `django_spec_agents.spec_tools`, where it was
  `rest_framework_pydantic_ai`. A `LOGGING` entry under the old name stops
  receiving this toolset's records after the move, and nothing fails to tell you.
  Rename the entry, or add one for the new name.
- **The `[filter]` extra** keeps its name in django-spec-agents.

## Where the rest of the documentation went

The pages of this site live on in the django-spec-agents documentation, under
[Specs as tools](https://artui.github.io/django-spec-agents/spec-tools/):
the quickstart, the shared spec registry, polymorphic actions, hiding plumbing
from the model and running from a worker.
