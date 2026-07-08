"""``QueryParam`` — a request-level query param a ``SpecToolset`` tool advertises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QueryParam:
    """A request-level query param exposed as a tool arg and seeded into the
    off-HTTP request's ``query_params``.

    Generalizes the built-in ``page`` / ``limit`` / ``order`` list-selector args
    to any read-shaping param a serializer reads off ``request.query_params`` —
    django-restql field selection (``?query=`` / ``?fields=``) or a custom
    serializer that branches on the query string. (A ``SelectorSpec.filter_set``
    does **not** need this — its fields are already generated into the tool schema
    and flow through as ordinary ``params``.) Register them on
    :class:`~rest_framework_pydantic_ai.SpecToolset` toolset-wide
    (``query_params=``) or per-tool (``tool_query_params=``). On a call the model
    supplies the value as a tool arg (or the declared ``default`` is used); the
    toolset pops it from the args and hands it to
    ``build_offline_context(query_params=…)`` — it never reaches the spec as an
    input, so it is unaffected by ``unknown_arguments``.

    - ``name`` — the tool-arg / query-string key. Must not be ``page`` / ``limit``
      / ``order`` (reserved for list-selector pagination).
    - ``type`` — the JSON-Schema type advertised to the model (``"string"`` by
      default; ``"integer"`` / ``"number"`` / ``"boolean"`` / ``"array"`` …).
    - ``description`` — optional help text shown to the model.
    - ``default`` — optional value seeded when the model omits the arg; also
      surfaced as the schema ``default``.
    """

    name: str
    type: str = "string"
    description: str | None = None
    default: Any = None

    def json_schema(self) -> dict[str, Any]:
        """The JSON-Schema property this param contributes to a tool's input."""
        schema: dict[str, Any] = {"type": self.type}
        if self.description is not None:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        return schema


__all__ = ["QueryParam"]
