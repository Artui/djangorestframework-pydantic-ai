# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Tool calls whose serializer reads `self.context["request"]` no longer raise
  `KeyError`.** Over HTTP DRF hands every serializer
  `get_serializer_context()` — `request` / `format` / `view` — so serializers
  read those keys unguarded (`request.user`, an ownership check in a
  `SerializerMethodField`). Off the HTTP path there is no view to ask, and
  drf-services passed only what a spec's `*_serializer_context` provider
  returned — nothing at all when none was declared — so a serializer that
  renders behind a view failed the tool call here. Fixed in
  `djangorestframework-services` 0.29.0, which this release requires: input
  validation and output rendering both start from that baseline, with the
  spec's provider merged over it. `request` is the synthetic one
  `build_offline_context` builds, so `request.user` is the acting user.

### Changed

- Requires `djangorestframework-services>=0.29.0` (was `>=0.28.1,<0.29`).

## [0.9.0] — 2026-07-28

### Added

- **`UrlKwarg(required=True)` — advertise a route capture the spec can't run
  without.** The name joins the tool's `required` list, and a call that omits it
  raises `ModelRetry` naming the argument, so the model gets a turn to supply it
  instead of the run failing somewhere less legible. Schema `required` is only a
  hint; advertising it without enforcing it would have changed nothing.
- **The reflected `required` list is now extended rather than merely preserved.**
  drf-services 0.28 contributes keys a selector's own extras `TypedDict` marks
  `InputRequired`; registered `UrlKwarg`s add theirs. A key that is both appears
  once — the same statement made in two places is one requirement.

### Changed

- **`UrlKwarg` and `QueryParam` now come from `djangorestframework-services`**
  (0.28), which owns the single definitions. This package and
  `djangorestframework-mcp-server` each carried a `UrlKwarg`, and the copies had
  **already drifted**: this one reserved only `page` / `limit` / `order` while
  the MCP transport also reserved the dispatcher's pool seeds, so
  `UrlKwarg("user")` was legal here and rejected there, and `UrlKwarg("order")`
  the reverse. `QueryParam` had no second copy yet — lifting it now prevents the
  fork rather than repairing one. Both import paths are preserved permanently.
- **⚠ Bad channel declarations now raise `ImproperlyConfigured`, not
  `ValueError`.** Reserved-name checking moved to drf-services'
  `validate_channel_names`, so one exception type covers every bad declaration
  instead of `ValueError` for pagination names and something else for pool seeds.
  The unknown-per-tool-key check still raises `ValueError` — it is about the
  mapping's keys, not a declaration.
- **A pool-seed name is now rejected.** Previously only the pagination names were
  checked here, so `UrlKwarg("user")` — a name that overrides a
  dispatcher-controlled seed — passed construction in this toolset while the MCP
  transport rejected it.
- **Declaration checks run after the toolset-wide/per-tool merge**, which is what
  actually reaches the schema. Validating the pre-merge concatenation would have
  read an intentional per-tool override as a duplicate.
- **Requires `djangorestframework-services>=0.28.1,<0.29`** (was `>=0.27,<0.28`).

## [0.8.0] — 2026-07-27

### Added

- **`SpecToolset` and `SpecCapability` accept a `SpecRegistry`** wherever they
  accept a `name -> spec` mapping (`djangorestframework-services` 0.27+). A
  project exposing the same specs over more than one transport — this toolset
  plus an MCP server, plus HTTP views — otherwise writes the list once per
  transport, and the copies drift. The registry is the one declaration site:

  ```python
  agent = Agent(model, deps_type=AgentDeps, toolsets=[SpecToolset(registry)])
  ```

  Deliberately **not** a `from_registry` constructor. `registry.specs()` is
  already the dict these accept, so the only thing a classmethod would add is a
  restatement of all nine keyword arguments it forwards — in two more places,
  free to drift from the signature they mirror. Widening the parameter gets the
  same ergonomics with none of that, and works on `SpecCapability` for free
  since it forwards `specs` straight through.
  - **Filtered views project several toolsets from one declaration.**
    `registry.by_tag("read")` returns a new registry, so
    `SpecToolset(registry.by_tag("read"), id="reads")` and its admin sibling
    share no state. Give each capability its own `id` — the id keys
    `defer_loading`'s catalog entry.
  - **Nothing else moves.** The registry carries only the invariant part of an
    operation (which spec, its name, its tags); `get_user`,
    `unknown_arguments`, `max_retries` and the `QueryParam` / `UrlKwarg`
    registrations stay per-toolset, because they are transport-specific.
    Per-tool maps key off the registry's names, and an unknown key still
    raises.
  - **Name validation is unchanged**, which matters here: registry names are
    free-form but model providers constrain tool names to
    `[a-zA-Z0-9_-]{1,64}`, so a registry name outside that shape still fails
    fast at construction rather than at the provider boundary.

### Changed

- **`djangorestframework-services` floor raised to `>=0.27,<0.28`** (from
  `>=0.26,<0.27`) — `SpecRegistry` is imported at module level.

## [0.7.0] — 2026-07-24

### Added

- **`UrlKwarg` — expose a URL route capture as a tool arg.** The off-HTTP
  counterpart of a nested route's URL captures (the `project_pk` of
  `/projects/{project_pk}/widgets/`). Register them on `SpecToolset` /
  `SpecCapability` toolset-wide (`url_kwargs=`) or per-tool
  (`tool_url_kwargs=`), the same shape as `QueryParam` / `tool_query_params`.
  Each is advertised as a tool arg, then popped at call time and seeded into
  `build_offline_context(kwargs=…)` — from where drf-services spreads it into
  the selector / target pools, authoritative over the spec `params` and below a
  `spec.kwargs` provider (mirroring HTTP's `extra_url_kwargs=view.kwargs`
  precedence). Because it is popped, it never reaches the spec as an ordinary
  input, so `unknown_arguments` never flags it — which is what makes the
  provider-read case work: a value a scoping `spec.kwargs` provider reads off
  `view.kwargs` (a tenant/role lookup keyed on `project_pk`), which ordinary
  `params` cannot deliver. A selector that reads the value from its
  `**extras: Unpack[TypedDict]` needs no `UrlKwarg` (drf-services reflects the
  key and delivers it through `params`), but a key may be **both** reflected and
  `UrlKwarg`-registered — the explicit `UrlKwarg` schema wins the merge, and the
  authoritative `kwargs=` spread still reaches the selector. Names can't be
  `page` / `limit` / `order`, nor be registered as both a `QueryParam` and a
  `UrlKwarg` on the same tool.

### Changed

- **Raise the `djangorestframework-services` floor from `>=0.23` to `>=0.26`**
  (ceiling `<0.26` → `<0.27`). `UrlKwarg` relies on drf-services 0.26 spreading
  the off-HTTP view's `kwargs` into the dispatch pools (and reflecting a
  selector's `Unpack[TypedDict]` extras into the tool schema), so the floor
  moves up with the feature. Tested ceiling raised to 0.26.x.

## [0.6.1] — 2026-07-16

### Changed

- Raise the `djangorestframework-services` ceiling from `<0.25` to `<0.26`
  (floor unchanged at `>=0.23`) so the toolset installs alongside drf-services
  0.25.x. 0.25.0 is additive for this adapter; the one relevant change is the
  bugfix giving `collection_selector_spec` selectors the view's URL kwargs on
  the bulk path, which flows through `dispatch_spec` automatically. Tested
  ceiling raised to 0.25.x.

### Added

- Docs: [Polymorphic actions as tools](polymorphic-actions.md) — expand a
  drf-services 0.25 `PolymorphicServiceSpec` into one flat tool per variant
  rather than handing `SpecToolset` a union tool.

## [0.6.0] — 2026-07-13

### Added

- **`SpecToolset` now teaches the model its conventions directly**, via a
  `get_instructions()` override (the native `AbstractToolset` hook). The
  pagination (`page` / `limit` / `order`) and error-contract text that
  `SpecCapability` introduced in 0.5.0 now reaches the model **whether the toolset
  is attached directly** (`Agent(toolsets=[SpecToolset(...)])`) **or wrapped by a
  capability** — previously a plain-toolset consumer got the tools but not the
  conventions. `SpecToolset(..., instructions=...)` overrides the derived block.

### Changed (breaking)

- **`SpecCapability` no longer emits its own instructions** — it delegates to the
  wrapped `SpecToolset`, since pydantic-ai already collects an owned toolset's
  `get_instructions()`. Model-facing behaviour is **identical** (the same
  conventions block reaches the system prompt, exactly once), but two direct-call
  surfaces changed:
  - `SpecCapability.get_instructions()` now returns `None` (the toolset provides
    the text). Read `capability.get_toolset().get_instructions(ctx)` if you need
    it programmatically.
  - `SpecCapability.from_toolset()` **no longer accepts `instructions=`** — set
    the override on the `SpecToolset` (`SpecToolset(..., instructions=...)`)
    before wrapping. The `SpecCapability(specs, instructions=...)` ctor still
    accepts it and forwards it to the toolset it builds.

## [0.5.0] — 2026-07-13

### Added

- **`SpecCapability`** — a Pydantic-AI v2 capability wrapping `SpecToolset`. It
  exposes the same tools, and additionally carries the toolset's conventions to
  the model through `get_instructions()`: that list tools accept `page` / `limit`
  / `order`, that a business failure returns a readable `{"error": …}` result (a
  final answer, not a reason to retry) while a bad argument returns a retry
  request, and that a permission error is final. Those conventions previously
  lived only in human docs, so the model relearned them per run or discovered
  them by failing a call. Construct it exactly like `SpecToolset` (it forwards the
  toolset knobs) and pass it to `Agent(capabilities=[...])`, or wrap an existing
  toolset with `SpecCapability.from_toolset(...)`. The instructions are derived
  from the specs (pagination text only when a list selector is present,
  read-shaping text only when a `QueryParam` is declared); pass `instructions=`
  to override, or `defer_loading=True` to hide the whole toolset behind
  pydantic-ai's native `load_capability` tool for large spec maps.

### Changed

- **Raise the `pydantic-ai-slim` floor from `>=1.0` to `>=2` (kept `<3`).** The
  capability API (`pydantic_ai.capabilities`) `SpecCapability` builds on is
  v2-only, and `SpecToolset` already imports v2's `pydantic_ai.toolsets`, so this
  formalises the existing requirement and drops 1.x.

## [0.4.0] — 2026-07-10

### Added

- `SpecToolset(max_retries=...)` — each tool's retry budget: how many times a
  `ModelRetry` (a validation failure, a bad `order` field) is fed back to the
  model before the run aborts. Defaults to `1`, matching pydantic-ai's own
  function-tool default.

### Changed

- `SpecToolset` now subclasses `pydantic_ai.toolsets.AbstractToolset` directly
  (the documented extension point) instead of `ExternalToolset`, building its
  tool definitions `kind="function"` from the start. Previously it inherited
  from a base class that models the opposite of in-process execution (external
  tools are *deferred* to the client) and re-stamped every tool definition back
  to `kind="function"` per run — a seam that depended on `ExternalToolset`
  internals. Public API and tool behaviour are unchanged.

### Fixed

- A tool's `ModelRetry` (an input-validation failure, a bad `order` field) now
  actually reaches the model to self-correct, as documented. `ExternalToolset`
  pinned every tool's retry budget to `0`, so in a real agent run the first
  `ModelRetry` aborted the run with `UnexpectedModelBehavior` instead of
  retrying. Both behaviours are now pinned by full agent-run integration tests.

## [0.3.2] — 2026-07-08

### Changed

- Widen the `djangorestframework-services` constraint from `>=0.23,<0.24` to
  `>=0.23,<0.25`, so the adapter installs against drf-services 0.24.x. Selector
  tool schemas transparently gain the 0.24 selector-input-schema fidelity (a
  selector's own callable parameters are now reflected) with no code change.
  Verified against drf-services 0.24.0.

## [0.3.1] — 2026-07-08

### Changed

- Widen the `pydantic-ai-slim` dependency constraint from `>=1.0,<2` to
  `>=1.0,<3`, so the adapter installs against Pydantic-AI 2.x (verified against
  `pydantic-ai-slim` 2.6.0). The 1.x line remains supported. Refreshed the
  pinned dependency set at the same time.

## [0.3.0] — 2026-07-08

### Added

- **`QueryParam` — register read-shaping request-level params on `SpecToolset`.**
  The extensible generalization of the built-in `page` / `limit` / `order` list
  args: declare a `QueryParam(name, type=…, description=…, default=…)` toolset-wide
  (`SpecToolset(specs, query_params=[…])`) or per-tool
  (`tool_query_params={"tool": [...]}`). Each is advertised as a tool arg, then —
  instead of reaching the spec as an input — popped and seeded into
  `request.query_params` over the off-HTTP path via
  `build_offline_context(query_params=…)`. This is for whatever reads
  `request.query_params` **directly** — django-restql field selection, or a
  custom serializer branching on the query string. (A `SelectorSpec.filter_set`
  needs none of this: its fields are already generated into the tool schema and
  flow through as ordinary `params`.) A registered param is popped before dispatch (so
  `unknown_arguments` never flags it); a declared `default` is seeded when the
  model omits the arg; reserved names (`page`/`limit`/`order`) and unknown
  per-tool keys are rejected at construction. (QP-2.)

### Changed

- Bumped the `djangorestframework-services` floor to `>=0.23,<0.24` for the
  `build_offline_context(query_params=…)` seam `QueryParam` builds on.

## [0.2.2] — 2026-07-03

### Changed

- Widened the `djangorestframework-services` dependency to `>=0.21.1,<0.23` to
  allow the published 0.22.x line.

## [0.2.1] — 2026-07-02

### Documentation

- README now describes the model-loop error mapping (invalid input / pagination
  args / unexpected arguments → `ModelRetry`; permission denials abort) and the
  `unknown_arguments` knob added in 0.2.0. No code change — a docs-only patch so
  the updated README ships to PyPI.

## [0.2.0] — 2026-07-02

### Changed (breaking)

- **The importable package is renamed `drf_pydantic_ai` → `rest_framework_pydantic_ai`.**
  This matches the sibling packages (`djangorestframework-services` →
  `rest_framework_services`, `djangorestframework-mcp-server` →
  `rest_framework_mcp`); the PyPI name is unchanged
  (`djangorestframework-pydantic-ai`). Update imports:
  `from rest_framework_pydantic_ai import SpecToolset, AgentDeps`.

### Added

- **`unknown_arguments` knob on `SpecToolset`.** Controls what happens
  to tool args outside a spec's declared input set — a key the model
  hallucinated. Defaults to `REJECT`, surfacing the unexpected key as a
  `ModelRetry` so the model self-corrects (specs with an open declared set — a
  `filter_set` / `**kwargs` selector — are unaffected). Pass `IGNORE` to drop
  them silently or `PASSTHROUGH` to forward them.

### Fixed

- **Object-level permissions are now enforced.** `SpecToolset` ran only
  a spec's class-level `has_permission`; the `on_target_resolved` object-level
  hook was never wired, so a mutation guarded by the standard DRF ownership
  pattern (`IsOwner.has_object_permission`) let an agent acting as user A
  update/delete user B's row. The dispatch call now passes
  `on_target_resolved=enforce_permissions`, so object-level checks run on the
  resolved row and a denial raises `PermissionDenied` (aborting the run, not a
  `ModelRetry`), exactly as over HTTP. The README / docs parity wording is
  corrected to state precisely what runs.
- **Model-supplied pagination args are validated.** `page` / `limit` /
  `order` reach the toolset untyped (`ExternalToolset` installs a no-op argument
  validator), so `limit="2"` or `order=["a"]` previously raised a `TypeError` /
  `AttributeError` that aborted the run. They are now coerced and validated
  (positive integers; a string `order`), mapping a bad value to `ModelRetry` so
  the model corrects it.
- **Tool names are validated at construction.** A `SpecToolset` mapping
  key that violates the model provider's function-name constraint
  (`^[a-zA-Z0-9_-]{1,64}$`) now raises `ValueError` at construction instead of
  failing opaquely at the provider boundary.

### Changed

- **`djangorestframework-services` floor raised to `>=0.21.1,<0.22`.** Required
  for the object-permission guard to fire on selector dispatch and for
  collection-safe enforcement.

## [0.1.0] — 2026-06-24

### Added
- `SpecToolset` — a Pydantic-AI toolset that exposes
  `djangorestframework-services` services and selectors as agent tools. Each
  call routes through drf-services' transport-neutral surface (`dispatch_spec`
  plus its off-HTTP helpers `build_offline_context` / `enforce_permissions` /
  `spec_to_json_schema` / `render_spec_output`) — **no MCP server and no AG-UI
  bridge** in the path. The toolset enforces `spec.permission_classes` (which
  `dispatch_spec` deliberately does not), builds the off-HTTP request/view
  context, derives each tool's description and `readOnlyHint` annotation from
  the spec, and exposes `page` / `limit` / `order` tool args for list selectors.
  Validation errors map to `ModelRetry`, business errors and unresolved
  instances to a model-readable `{"error": ...}`.
- `AgentDeps` — the default `user`-carrying dependency the toolset reads off
  `RunContext.deps`; override with a `get_user` extractor for a custom identity
  shape.

[Unreleased]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.0.0...v0.1.0
