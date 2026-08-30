# Running from a worker

Every worked example in these docs is HTTP-shaped, down to
`deps=AgentDeps(user=request.user)`. The larger surface for a lot of projects is
the other one: a Celery task, a management command, a scheduled job — an agent
run with nobody waiting on it.

**There is no headless mode to switch on, and this page is not describing one.**
The same `SpecToolset` runs either way; a spec dispatched from a worker takes
exactly the path it takes behind a view, because that path was never an HTTP one
to begin with. What actually differs is two things: **there is no request**, and
**there is more load**. Everything below follows from one of those.

## The smallest run

`Agent.run_sync` works today and needs nothing from this package:

```python title="tasks.py"
from celery import shared_task
from django.contrib.auth import get_user_model
from pydantic_ai import Agent

from rest_framework_pydantic_ai import AgentDeps, SpecToolset

toolset = SpecToolset({"list_orders": list_orders_spec})
agent = Agent("anthropic:claude-opus-4-8", deps_type=AgentDeps, toolsets=[toolset])


@shared_task
def summarise_account(user_id: int) -> str:
    user = get_user_model().objects.get(pk=user_id)
    result = agent.run_sync(
        "Summarise this account's open orders.",
        deps=AgentDeps(user=user),
    )
    return result.output
```

Build the toolset and the agent **once, at module scope**. Both are reusable
across runs — the toolset resolves its specs, validates its declarations and
derives each tool's schema in the constructor, and repeating that per task is
pure waste. Nothing per-request is closed over in either, which is what makes
that safe.

## The acting user is the whole authorization story

`AgentDeps.user` is not a convenience. It is the identity every
`permission_class`, `extend_queryset` and scoping provider resolves against, and
off HTTP it is the **only** thing that carries it.

Resolve it server-side, from something you trust: a task argument that is a
primary key you put there, a row you already own. Never from the model's own
output, and never from text the model was given — a run that can talk its way
into a different `user` has no authorization boundary at all.

!!! warning "`http_request=` is not an identity"

    [`http_request=` / `get_http_request=`](reference.md#rest_framework_pydantic_ai.SpecToolset)
    exist so a serializer or scoping provider reading `request.META` finds
    something plausible. Nothing downstream re-derives the acting user from it.
    Passing an authenticated request authorizes nothing, and leaning on it would
    put a second, invisible identity in the call.

### Nobody is anonymous by accident

A view refuses an unauthenticated caller before the agent is reached. A worker
has no such gate, so a task whose `user_id` is `None` runs with `user=None` and
whatever `permission_classes` make of that. Decide it at the top of the task,
where the answer is obvious, rather than in a permission class three layers down.

## Concurrency: the one setting a worker genuinely needs

Every tool call goes through `sync_to_async`, which defaults
`thread_sensitive=True` — and asgiref's `single_thread_executor` is a **class**
attribute, so that is *one thread for the whole process*, shared by every toolset
instance and every concurrent run in it.

Pydantic-AI runs function tools in parallel within a segment. Four 0.30s tool
calls under one model step therefore take about **1.2s rather than 0.3s**. In a
chat turn calling one tool this is invisible. In a fan-out it is the ceiling.

```python
from concurrent.futures import ThreadPoolExecutor

toolset = SpecToolset(
    SPECS,
    thread_sensitive=False,
    executor=ThreadPoolExecutor(max_workers=8),
)
```

!!! danger "Read this before flipping it"

    `thread_sensitive=True` is what keeps Django's thread-local database
    connections coherent. Setting it `False` moves dispatch onto a pool, and
    **each of those threads opens and owns its own connection**. That is fine —
    and it is what you want — provided you have accounted for it:

    - Your database can take `max_workers` more connections per worker process,
      or you front it with a pooler.
    - Nothing in your specs relies on being in the caller's transaction. An
      outer `atomic()` block in the task does **not** wrap work that ran on
      another thread.
    - You are not inside `pytest.mark.django_db` without `transaction=True`,
      where the test's wrapping transaction is invisible to other threads.

    `executor=` is only consulted when `thread_sensitive` is `False` — that is
    asgiref's rule, not this package's. Setting it alone does nothing, silently.

## Reading the logs afterwards

Nobody is watching a background run, so the log line is the whole record. Every
tool call logs at `DEBUG` (`WARNING` for a permission denial, which is otherwise
traceless — the run loop absorbs it into a message) on the
`rest_framework_pydantic_ai` logger, and carries correlation fields in `extra`:

| Field | |
| --- | --- |
| `run_id`, `conversation_id` | which run this line belongs to |
| `run_step`, `tool_call_id` | where in that run |
| `run_input_tokens`, `run_output_tokens`, `run_requests`, `run_tool_calls` | the run's usage **so far** |

Concurrent runs interleave, and without these the lines are indistinguishable.
The usage fields are cumulative for the run rather than attributable to the call
— a tool call spends no tokens itself — which is exactly what makes them useful
here: they say where the budget stood when a long run went wrong. They ride on
the timing line; the `run_id` / `conversation_id` / `run_step` / `tool_call_id`
four are on every line the package emits, including the two `WARNING`s a
misbehaving call produces (a dispatch timeout, a result over its byte ceiling).

```python title="settings.py"
LOGGING = {
    "version": 1,
    "loggers": {
        "rest_framework_pydantic_ai": {"level": "DEBUG", "handlers": ["json"]},
    },
}
```

**Bounding the run is `Agent.run`'s job, not the toolset's.** Pass
[`UsageLimits`](https://ai.pydantic.dev/agents/#usage-limits) there. A toolset
can only refuse the *next* tool call, which is the wrong instrument and a second
place for the limit to live.

The limits are still *readable* from a tool call, though, and one thing is worth
doing with them: `ctx.usage_limits` reaches an
[`enforce_result_bytes`](reference.md#rest_framework_pydantic_ai.SpecToolset.enforce_result_bytes)
override, so a project can taper how much a call is allowed to return as the run
consumes its budget. That is shaping a result, not enforcing a limit — the run
still stops where `Agent.run` says it does.

## Bounding what comes back

A model reading a 40,000-row selector is a cost and latency problem before it is
a correctness one, and in a fan-out it is multiplied by every concurrent run.

- `max_page_size=` caps what one list call can return, however large a `limit`
  the model asks for. The clamp is reported back in the envelope, so the model
  is told it received less than it asked for rather than silently truncated.
- `max_result_bytes=` (and `tool_max_result_bytes=` per tool) bounds the
  serialized payload, answering with a refusal the model can act on.
- `dispatch_timeout=` bounds how long a call blocks the run. **It does not stop
  the work** — the thread runs to completion; the run stops waiting.

## Changing the pipeline

Six overridable methods cover a tool call end to end. The front half —
`build_context` and `translate_exception` — is documented under
[`SpecToolset`](reference.md#rest_framework_pydantic_ai.SpecToolset); the back
half is `shape_page`, `render_output`, `output_extras` and
`enforce_result_bytes`. All six receive the live `RunContext`, which is the
point: a run's typed deps are usually what needs to reach them.

The one worth knowing about before you need it is
[`render_output`](reference.md#rest_framework_pydantic_ai.SpecToolset.render_output).
By default a result is rendered through the spec's serializer **and projected**
for the agent audience. If a background job feeds one spec's output into the
next, project the handles away and the next step has nothing to key on. Rendering
with `render_spec_output` instead is the opt-out — and passing `projection=None`
is *not* it, since that means "derive one from the spec".

## Testing it

The run loop is where the interesting failures live, and it needs a model.
[`rest_framework_pydantic_ai.testing`](reference.md#testing) ships the two
doubles this package's own suite uses — no provider, no network, no key:

```python
from pydantic_ai import Agent

from rest_framework_pydantic_ai import AgentDeps, SpecToolset
from rest_framework_pydantic_ai.testing import tool_calling_model


async def test_the_task_calls_the_tool():
    toolset = SpecToolset({"list_orders": list_orders_spec})
    agent = Agent(tool_calling_model("list_orders"), deps_type=AgentDeps, toolsets=[toolset])

    result = await agent.run("go", deps=AgentDeps(user=user))

    assert result.output == "done"
```

Test the task as a whole this way rather than calling `call_tool` directly.
Asserting on `call_tool` answers *does the tool work*; only a run answers whether
the tools execute in-process, whether a `ModelRetry` is fed back for another
attempt, and whether the toolset's instructions reached the model.

If your worker code is sync, drive it with `agent.run_sync` inside
`django_db(transaction=True)` — the same shape production uses.

## Progress, when there is no browser

`AgentDeps.progress` takes a
[`ProgressReporter`](https://artui.github.io/djangorestframework-services/),
and a spec that reports progress calls it wherever it runs. This package never
constructs one: picking a sink would be picking a transport it does not own.
Off HTTP, the useful sinks are the ones a worker already has — a task-state
update, a row, a log line.
