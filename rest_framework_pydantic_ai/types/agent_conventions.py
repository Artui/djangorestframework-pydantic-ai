"""``AgentConventions`` -- the sentences a ``SpecToolset`` tells the model, one field each."""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

from django.core.exceptions import ImproperlyConfigured


@dataclass(frozen=True)
class AgentConventions:
    """The model-facing wording of a toolset, overridable one line at a time.

    Every field defaults to the sentence the toolset says today, so
    ``AgentConventions()`` changes nothing and ``AgentConventions(pagination=...)``
    changes exactly one line. Pass it as ``conventions=`` to
    [`SpecToolset`][rest_framework_pydantic_ai.SpecToolset] or
    [`SpecCapability`][rest_framework_pydantic_ai.SpecCapability]; leaving that
    unset is the same as passing ``AgentConventions()``.

    **A field changes what a line says, never whether it appears.** The toolset
    still decides that: the pagination line only when some tool returns a page,
    the ordering line only when some tool can sort, and so on, re-derived per step
    when an operation condition leaves a tool out. An override keeps those
    conditions, which is what ``instructions=`` loses by replacing the whole
    block, and a later release that corrects one default reaches every field you
    did not override.

    **``None`` drops the line** wherever the toolset would have said it. The one
    exception is ``missing_arguments``, which is the whole text of a retry and so
    cannot be dropped.

    **Each field is a ``str.format`` template.** The placeholders a field accepts
    are the ones listed on it, and nothing else; anything outside them raises
    ``ImproperlyConfigured`` naming the field when the instance is built, so a typo
    fails at startup rather than inside a prompt. A literal brace is written twice,
    ``{{`` or ``}}``, in every field, as the default ``pagination`` does.

    **Some lines state facts about behaviour,** not just advice: ``pagination``
    names the keys of the page envelope, ``base`` says which failures come back as
    a retry and which as a failed call, and ``query_param_on_pages`` lists the
    envelope's keys again. The toolset keeps behaving as the defaults describe
    whatever the text says, so an override that rewords one of those owns keeping
    it true.

    Each transport words these for its own reader, which is why they live here and
    not in drf-services, and why changing them here changes nothing the MCP
    transport says about the same specs.

    Raises:
        ImproperlyConfigured: A field uses a placeholder it does not accept, is not
            a valid format string, is neither a string nor ``None``, or is
            ``missing_arguments`` set to ``None``.
    """

    base: str | None = (
        "The following tools call Django REST Framework services and selectors.\n"
        "- A successful call returns the tool's data. A business-rule failure comes back as a "
        "failed call whose content is a sentence explaining why — that is a final answer, not a "
        "reason to retry; read it and report it, do not call the same tool the same way again. "
        "The sentence may end with `(code: <name>)`, naming the rule that refused; it is the "
        "same code an item's `affordances` answer carries when a tool advertised one.\n"
        "- An invalid or missing argument comes back as a retry request naming the problem; "
        "correct the argument and call again.\n"
        "- A permission error is final: the current user may not perform that call — do not "
        "retry it.\n"
        "- Only pass documented parameters; unknown arguments are rejected."
    )
    """The opening of the instructions block, said whatever the tools are.

    Unconditional because the failure contract it states holds for every tool, and
    because no spec can answer whether a refusal will carry a ``(code: <name>)``:
    a declared ``Affordance`` produces one, and so does a service raising
    ``ActionUnavailable`` by hand, which nothing declares. No placeholders.
    """

    pagination: str | None = (
        "- Read-only tools that return a collection always return one page, shaped "
        '{{"items": [...], "page": 1, "totalPages": N, "hasNext": true|false}}. They accept '
        "optional `limit` (items per page, default {page_size}) and `page` (1-based). When "
        "`hasNext` is true there are more items than you were shown: ask for the next `page`, "
        "or narrow the request with a filter — never answer as if the page were the whole "
        "collection."
    )
    """The block's line on pages, when some tool returns one.

    ``{page_size}`` is the page an omitted ``limit`` is served, the same number
    each tool's ``limit`` description states, so the model is not told one default
    here and another on the tool.
    """

    ordering: str | None = (
        "- Some collection tools also accept {names}. It takes exactly one of the values "
        "listed in that tool's schema (a sortable name, or the same name prefixed with `-` for "
        "descending) — not a comma-separated list, and not an arbitrary column."
    )
    """The block's line on sorting, when some tool advertises a sort argument.

    ``{names}`` is every sort argument the tools advertise, each in backticks and
    joined with ``", "``: the name is the project's to choose, and one toolset can
    carry tools whose sorts are declared under different names.
    """

    handles: str | None = (
        "- Some tools return opaque identifier fields, described as such in the tool's "
        "output. Pass them to other tools that ask for one; refer to records by their "
        "name in anything you say, never by the identifier."
    )
    """The block's line on opaque identifiers, when some tool's output marks one.

    The block-level half of ``handle_field_description``. No placeholders.
    """

    read_shaping: str | None = (
        "- Some tools accept read-shaping parameters ({names}) that adjust the shape "
        "of the returned data without filtering it."
    )
    """The block's line on read-shaping parameters, when some tool declares a ``QueryParam``.

    ``{names}`` is every declared name, sorted, each in backticks and joined with
    ``", "``. ``None`` drops the line and ``read_shaping_on_pages`` with it, since
    that sentence continues this one.
    """

    read_shaping_on_pages: str | None = (
        "On a tool that returns a page, they apply to each item in `items`, never to the page "
        "itself."
    )
    """Appended to ``read_shaping``, after a space, when some tool that returns a page
    declares a ``QueryParam``.

    The block-level half of ``query_param_on_pages``. No placeholders.
    """

    unavailable_heading: str | None = (
        "- These operations exist but cannot be performed right now, so they are not among your "
        "tools. If the user asks for one, say it is unavailable at the moment and give the reason "
        "listed for it, rather than guessing why:"
    )
    """The heading of the per-step list of operations an unmet condition left out.

    Followed by one line per such tool, its name and its ``reason``, and said only
    on a step that left one out. It is appended after an ``instructions=``
    override as well, so it is the one block line that override does not replace.
    ``None`` drops the heading and the list beneath it. No placeholders.
    """

    handle_field_description: str | None = (
        "An opaque identifier. Pass it to other tools that ask for one; refer to the record "
        "by its name in anything you say, never by this value."
    )
    """The description of a handle field in a tool's output schema, when the field's
    own marking declares none.

    drf-services supplies no wording for a handle on purpose: what a reader should
    do with an identifier depends on the reader. No placeholders.
    """

    query_param_on_pages: str | None = (
        "On a paged result it applies to each item in `items`, never to the page envelope "
        "(`items`, `page`, `totalPages`, `hasNext`)."
    )
    """What a read-shaping parameter applies to on a tool that returns a page.

    Appended after a space to each ``QueryParam``'s description on such a tool, or
    its whole description when it declares none, and to the retry a render-time
    rejection on such a tool produces. No placeholders.

    The envelope is the likeliest target for a bad selection because it is exactly
    the shape the tool documents returning -- ``{items{id, name}}`` is a natural
    reading of it -- while the serializer that reads the param only ever sees one
    row. Saying so at the parameter, where the model is choosing a value, turns
    that mistake into one retry.
    """

    missing_arguments: str = "Missing required argument(s): {names}."
    """The retry for a call that left out an argument it cannot run without.

    Covers both checks made before the input serializer runs: a parameter the
    selector requires, and a ``UrlKwarg`` declared ``required``. A field the input
    serializer requires is reported by the serializer, in its own words.
    ``{names}`` is every missing name, sorted, each in backticks and joined with
    ``", "``. Never ``None``: it is the retry's whole text.
    """

    def __post_init__(self) -> None:
        for field in fields(self):
            _validate(field.name, getattr(self, field.name))


_PLACEHOLDERS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "base": frozenset(),
        "pagination": frozenset({"page_size"}),
        "ordering": frozenset({"names"}),
        "handles": frozenset(),
        "read_shaping": frozenset({"names"}),
        "read_shaping_on_pages": frozenset(),
        "unavailable_heading": frozenset(),
        "handle_field_description": frozenset(),
        "query_param_on_pages": frozenset(),
        "missing_arguments": frozenset({"names"}),
    }
)
"""The placeholders each field accepts.

Keyed by every field, so a field added without an entry fails every construction
(``test_every_field_declares_its_placeholders``) rather than accepting anything.
"""

_SAMPLES: Mapping[str, Any] = MappingProxyType({"page_size": 1, "names": "`name`"})
"""A value of the type each placeholder is rendered with, for the trial render."""

_NEVER_NONE = frozenset({"missing_arguments"})


def _validate(name: str, value: Any) -> None:
    """Refuse a field the toolset could not render, naming it.

    The names are read with ``string.Formatter().parse`` and checked against the
    field's own set; then the template is rendered once with sample values of the
    types it will receive, which is what catches a conversion or a format spec the
    values cannot take (``{page_size:q}``). Each step is held by its own case of
    ``test_a_template_the_toolset_could_not_render_is_refused``.
    """
    if value is None:
        if name in _NEVER_NONE:
            raise ImproperlyConfigured(
                f"AgentConventions.{name} cannot be None: it is the whole text of a retry, and "
                "a retry with nothing in it tells the model nothing."
            )
        return
    if not isinstance(value, str):
        raise ImproperlyConfigured(
            f"AgentConventions.{name} must be a string or None, not {type(value).__name__}."
        )
    accepted = _PLACEHOLDERS[name]
    try:
        used = {field for _, field, _, _ in string.Formatter().parse(value) if field is not None}
    except ValueError as exc:
        raise ImproperlyConfigured(
            f"AgentConventions.{name} is not a valid format string ({exc}); write a literal "
            "brace twice, as {{ or }}."
        ) from exc
    unknown = sorted(used - accepted)
    if unknown:
        listed = ", ".join(f"{{{placeholder}}}" for placeholder in unknown)
        allowed = ", ".join(f"{{{placeholder}}}" for placeholder in sorted(accepted)) or "none"
        raise ImproperlyConfigured(
            f"AgentConventions.{name} uses {listed}, which it does not accept (accepted: "
            f"{allowed}); write a literal brace twice, as {{{{ or }}}}."
        )
    try:
        value.format(**{placeholder: _SAMPLES[placeholder] for placeholder in accepted})
    # Whatever the reason, a template that cannot be rendered with the values it
    # will be given is a configuration error, and this is the moment to say so.
    except Exception as exc:
        raise ImproperlyConfigured(f"AgentConventions.{name} cannot be rendered: {exc!r}.") from exc
