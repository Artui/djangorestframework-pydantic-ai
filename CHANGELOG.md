# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- **`unknown_arguments` knob on `SpecToolset` (CONF-6).** Controls what happens
  to tool args outside a spec's declared input set — a key the model
  hallucinated. Defaults to `REJECT`, surfacing the unexpected key as a
  `ModelRetry` so the model self-corrects (specs with an open declared set — a
  `filter_set` / `**kwargs` selector — are unaffected). Pass `IGNORE` to drop
  them silently or `PASSTHROUGH` to forward them.

### Fixed

- **Object-level permissions are now enforced (AUTHZ-3).** `SpecToolset` ran only
  a spec's class-level `has_permission`; the `on_target_resolved` object-level
  hook was never wired, so a mutation guarded by the standard DRF ownership
  pattern (`IsOwner.has_object_permission`) let an agent acting as user A
  update/delete user B's row. The dispatch call now passes
  `on_target_resolved=enforce_permissions`, so object-level checks run on the
  resolved row and a denial raises `PermissionDenied` (aborting the run, not a
  `ModelRetry`), exactly as over HTTP. The README / docs parity wording is
  corrected to state precisely what runs.
- **Model-supplied pagination args are validated (CONF-6).** `page` / `limit` /
  `order` reach the toolset untyped (`ExternalToolset` installs a no-op argument
  validator), so `limit="2"` or `order=["a"]` previously raised a `TypeError` /
  `AttributeError` that aborted the run. They are now coerced and validated
  (positive integers; a string `order`), mapping a bad value to `ModelRetry` so
  the model corrects it.
- **Tool names are validated at construction (CONF-6).** A `SpecToolset` mapping
  key that violates the model provider's function-name constraint
  (`^[a-zA-Z0-9_-]{1,64}$`) now raises `ValueError` at construction instead of
  failing opaquely at the provider boundary.

### Changed

- **`djangorestframework-services` floor raised to `>=0.21.1,<0.22`.** Required
  for the object-permission guard to fire on selector dispatch (AUTHZ-1b) and for
  collection-safe enforcement (AUTHZ-1a).

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

[Unreleased]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Artui/djangorestframework-pydantic-ai/compare/v0.0.0...v0.1.0
