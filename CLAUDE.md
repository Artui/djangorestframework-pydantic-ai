# CLAUDE.md

Conventions for writing code in this repo. For *what* the library does, see
[`README.md`](README.md) and the [docs](docs/index.md).

- Package on PyPI: `djangorestframework-pydantic-ai`
- Importable name: `drf_pydantic_ai`
- Build backend: hatchling (version is dynamic, read from `drf_pydantic_ai/version.py`)

## What this package is

A single thin adapter — `SpecToolset` — that exposes
[`djangorestframework-services`](https://github.com/Artui/djangorestframework-services)
`ServiceSpec` / `SelectorSpec` objects as a [Pydantic-AI](https://ai.pydantic.dev)
toolset, so a plain `pydantic_ai.Agent` can call them as tools. There is **no MCP
server and no AG-UI bridge** in the path — that is the whole point. Every call
routes through drf-services' transport-neutral surface (`dispatch_spec` + its
off-HTTP helpers `build_offline_context` / `enforce_permissions` /
`spec_to_json_schema` / `render_spec_output`).

**Do not rebuild the dispatch engine.** Validation, instance resolution, output
rendering, schema generation, and permission enforcement all live in
drf-services — import them, don't reimplement. If a needed concern is missing
from drf-services, the fix is to lift it *down* into drf-services, not to grow a
parallel copy here.

The reference toolset to match is `django-ag-ui`'s `DrfMcpToolset` (same
`ExternalToolset` shape, re-stamped `kind="function"`); the difference is that
`SpecToolset` skips the MCP hop and calls drf-services directly.

## Commands

Use the Makefile targets; CI and pre-commit both call them.

```bash
make init           # uv sync --all-groups + pre-commit install
make test           # uv run pytest (--cov=drf_pydantic_ai --cov-fail-under=100)
make lint           # ruff check + ty check drf_pydantic_ai
make lint-fix       # ruff check --fix
make format         # ruff format
make format-check   # ruff format --check --diff
make type-check     # ty check drf_pydantic_ai
make docs-serve     # live-reload docs at http://localhost:8000
make docs-build     # mkdocs build --strict
make release-bump VERSION=X.Y.Z   # rewrite version.py + promote CHANGELOG [Unreleased]
make release-publish              # prepare → uv publish → finalize (workstation release)
```

## Structural rules

Non-negotiable. They keep the package navigable.

1. **One exported class or function per file.** The file is named after the
   exported symbol in `snake_case`. `SpecToolset` → `spec_toolset.py`;
   `AgentDeps` → `types/agent_deps.py`.
2. **Private helpers used only in one file** stay in that file with a `_name`
   prefix (e.g. the schema/pagination helpers in `spec_toolset.py`).
3. **Non-exported helpers shared across files** go in that package's `utils.py`.
4. **Top-level imports only.** Lazy / function-local imports are forbidden
   unless a circular import is proven, *or* the import targets a declared
   optional dependency gated behind an `enable_*()` opt-in. Document the reason
   inline.
5. **Full type annotations on every function and method signature.** `Any` is
   allowed only at the Django / Pydantic-AI boundaries where the type genuinely
   is `Any`.
6. **`__init__.py` is the only re-export point.** Each `__init__.py` lists the
   public surface in `__all__`. Internal modules import from leaf paths, never
   from the package's `__init__`.
7. **Types live in `types/`.** Value-shape carriers (`AgentDeps`) live under
   `types/`; behavioural code (`SpecToolset`) lives at the package root.

## Adding a feature

**Always work on a dedicated branch** — never on `main`. A typical change
touches three places:

1. The source file (one new `.py` per new class/function).
2. The package's `__init__.py` (add to imports + `__all__`) if it's public.
3. A test file under `tests/` mirroring the source path
   (`drf_pydantic_ai/foo/bar.py` → `tests/foo/test_bar.py`).

Then `make lint-fix && make format && make test`. If the symbol is public, also
add it to `CHANGELOG.md` under `[Unreleased]`.

## Tests

- Live in `tests/`, mirroring the source tree. Async tests rely on
  `pytest-asyncio` (`asyncio_mode = "auto"`); just write `async def test_...`.
- DB-touching tests use `@pytest.mark.django_db`. The minimal Django app lives
  at `tests/testapp/`; `tests/conftest_settings.py` is the settings module.
- 100% line + branch coverage is enforced via `--cov-fail-under=100`. If a
  branch is genuinely unreachable, **restructure rather than
  `# pragma: no cover`**.

## Type checking

`ty` is scoped to `drf_pydantic_ai/` only (Django's dynamic descriptors trip the
checker when it walks `tests/`). Fix the source rather than narrowing the
checker. Use `# ty: ignore[<rule>]`, not the mypy-style comment.

## Linting and formatting

- `ruff check` enforces `E`, `F`, `UP`, `B`, `SIM`, `I`, `TID`.
- `ruff format` is the source of truth for layout.
- **Use `...` (Ellipsis) instead of `pass` for empty bodies.**

## Imports inside the package

- Always absolute, fully qualified:
  `from drf_pydantic_ai.spec_toolset import SpecToolset`. **Never** relative
  imports anywhere, including `__init__.py`.
- isort via ruff (`I` rules). Order: stdlib → third-party → first-party.

## Compatibility floor

| Axis | Floor | Tested ceiling |
|---|---|---|
| Python | 3.10 | 3.14 |
| Django | 4.2 | 6.0 |
| DRF | 3.14 | latest |
| drf-services | 0.20 | 0.20.x |
| Pydantic-AI | 1.0 (`pydantic-ai-slim`) | latest |

`from __future__ import annotations` at the top of every `.py` with annotations.

## CI and pre-commit

`.github/workflows/tests.yml` runs `lint`, `docs build`, and the
Python × Django `test` matrix on every push to `main` and every PR.
`.github/workflows/release.yml` is **merge-to-main triggered**: it runs
`make release-publish-prepare`, which no-ops unless the version was bumped past
the latest `vX.Y.Z` tag, then publishes to PyPI via OIDC and deploys docs.

`.pre-commit-config.yaml` runs `make lint-fix`, `make format`,
`make type-check`, and a `forbid-local-paths` guard on every commit. Fix the
underlying issue and make a new commit — never `--no-verify`.

## Releasing

Merge-to-main triggered, via `scripts/release-publish.sh` (byte-identical across
the services / mcp-server / pydantic-ai repos).

```bash
make release-bump VERSION=0.2.0   # rewrites version.py + promotes CHANGELOG
git diff && git commit -am "Release 0.2.0"
git push -u origin release/0.2.0 && gh pr create
# Merge to main; release.yml fires on the merge commit, tags + publishes v0.2.0.
```

### One-time setup (manual, by the repo owner)

1. **PyPI Trusted Publisher** — add a publisher for
   `Artui/djangorestframework-pydantic-ai`, workflow `release.yml`, environment
   `pypi` (use a "Pending" publisher before the first release).
2. **GitHub Environment** — create a `pypi` environment (no secrets; OIDC).
3. **GitHub Pages** — `Settings → Pages → Deploy from branch → gh-pages`. The
   first tag push with a `mkdocs.yml` creates that branch.
