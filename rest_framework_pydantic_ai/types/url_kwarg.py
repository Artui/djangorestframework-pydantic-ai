"""``UrlKwarg`` — a URL route capture a ``SpecToolset`` tool advertises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UrlKwarg:
    """A URL-derived value exposed as a tool arg and seeded into the off-HTTP
    request's view ``kwargs``.

    The off-HTTP counterpart of a nested route's URL captures (``parent_pk`` of
    ``/parents/{parent_pk}/children/``). Over HTTP such a value comes from the
    route and reaches a selector through ``view.kwargs``; off-HTTP the model
    supplies it as a tool arg, the toolset pops it, and hands it to
    ``build_offline_context(kwargs=…)`` — from where drf-services spreads it into
    the selector / target pools (authoritative over the spec ``params``, below a
    ``spec.kwargs`` provider). It never reaches the spec as an ordinary input, so
    it is unaffected by ``unknown_arguments``.

    Use it for a URL-derived value a spec depends on that is **not** already in
    the tool schema:

    - a scoping ``spec.kwargs`` provider that reads ``view.kwargs`` (e.g. a
      ``project_pk`` behind a tenant/role lookup) — the case ``params`` alone
      cannot cover, because the provider reads ``view.kwargs``, not ``params``;
    - a closed-surface spec whose route capture must be model-suppliable.

    A selector that already reads the value from its ``**extras: Unpack[TypedDict]``
    needs **no** ``UrlKwarg`` — drf-services reflects the TypedDict key into the
    tool schema and delivers it through ``params``. A key can, however, be *both*
    reflected and ``UrlKwarg``-registered (a ``project_pk`` the selector reads
    *and* a scoping provider reads off ``view.kwargs``): the explicit ``UrlKwarg``
    schema wins the merge, registration pops the arg into ``kwargs=``, and
    drf-services' authoritative spread still delivers it to the selector pool.

    Register them on :class:`~rest_framework_pydantic_ai.SpecToolset` toolset-wide
    (``url_kwargs=``) or per-tool (``tool_url_kwargs=``).

    - ``name`` — the tool-arg / view-kwarg key. Must not be ``page`` / ``limit``
      / ``order`` (reserved for list-selector pagination), and must not also be
      registered as a :class:`~rest_framework_pydantic_ai.QueryParam` on the same
      tool (a value cannot route to two channels).
    - ``type`` — the JSON-Schema type advertised to the model (``"string"`` by
      default; ``"integer"`` / ``"number"`` / ``"boolean"`` …).
    - ``description`` — optional help text shown to the model.
    - ``default`` — optional value seeded when the model omits the arg; also
      surfaced as the schema ``default``.
    """

    name: str
    type: str = "string"
    description: str | None = None
    default: Any = None

    def json_schema(self) -> dict[str, Any]:
        """The JSON-Schema property this kwarg contributes to a tool's input."""
        schema: dict[str, Any] = {"type": self.type}
        if self.description is not None:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        return schema


__all__ = ["UrlKwarg"]
