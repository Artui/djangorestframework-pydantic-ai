# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.0.0...v0.1.0
