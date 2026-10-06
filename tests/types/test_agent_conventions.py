from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import pytest
from django.core.exceptions import ImproperlyConfigured

from rest_framework_pydantic_ai import AgentConventions
from rest_framework_pydantic_ai.types import agent_conventions

_FIELDS = [field.name for field in fields(AgentConventions)]

# Each field's own placeholders, written out: the table the docs give a reader.
_ACCEPTED = {
    "base": (),
    "pagination": ("page_size",),
    "ordering": ("names",),
    "handles": (),
    "read_shaping": ("names",),
    "read_shaping_on_pages": (),
    "unavailable_heading": (),
    "handle_field_description": (),
    "query_param_on_pages": (),
    "missing_arguments": ("names",),
}


def test_every_field_declares_its_placeholders():
    """A field with no entry would fail every construction, so this is the reminder."""
    assert set(agent_conventions._PLACEHOLDERS) == set(_FIELDS)
    assert {name: tuple(sorted(accepted)) for name, accepted in _ACCEPTED.items()} == {
        name: tuple(sorted(accepted)) for name, accepted in agent_conventions._PLACEHOLDERS.items()
    }


@pytest.mark.parametrize("name", _FIELDS)
def test_a_field_using_every_placeholder_it_accepts_is_accepted(name):
    template = " ".join(f"{{{placeholder}}}" for placeholder in _ACCEPTED[name]) or "Plain."

    assert getattr(AgentConventions(**{name: template}), name) == template


@pytest.mark.parametrize("name", [name for name in _FIELDS if name != "missing_arguments"])
def test_none_is_accepted_for_every_line(name):
    assert getattr(AgentConventions(**{name: None}), name) is None


def test_the_missing_argument_retry_cannot_be_none():
    with pytest.raises(ImproperlyConfigured, match=r"AgentConventions\.missing_arguments cannot"):
        AgentConventions(missing_arguments=None)


@pytest.mark.parametrize(
    ("name", "template", "named"),
    [
        ("pagination", "Pages of {limit}.", "{limit}"),
        ("ordering", "Sort with {page_size}.", "{page_size}"),
        ("base", "Hello {user}.", "{user}"),
        ("missing_arguments", "Missing {name}.", "{name}"),
        ("handle_field_description", "An id, {}.", "{}"),
        ("read_shaping", "Shape with {names[0]}.", "{names[0]}"),
    ],
    ids=["other-fields-placeholder", "swapped", "none-accepted", "typo", "positional", "indexed"],
)
def test_a_placeholder_a_field_does_not_accept_is_refused_naming_the_field(name, template, named):
    with pytest.raises(ImproperlyConfigured) as refused:
        AgentConventions(**{name: template})

    message = str(refused.value)
    assert message.startswith(f"AgentConventions.{name} uses {named},")
    assert "which it does not accept" in message


@pytest.mark.parametrize(
    ("name", "template", "says"),
    [
        ("handles", "An id looks like {this.", "is not a valid format string"),
        ("unavailable_heading", "Closed }", "is not a valid format string"),
        ("pagination", "Pages of {page_size:q}.", "cannot be rendered"),
        ("pagination", "Pages of {page_size:{width}}.", "cannot be rendered"),
        ("ordering", "Sort with {names!z}.", "cannot be rendered"),
        ("base", 42, "must be a string or None, not int"),
    ],
    ids=["unclosed", "unopened", "format-spec", "nested-field", "conversion", "not-a-string"],
)
def test_a_template_the_toolset_could_not_render_is_refused(name, template, says):
    """Each case is the one step of ``_validate`` that answers it.

    ``unclosed`` / ``unopened`` are refused by the parse; the next three name only
    placeholders the field accepts, so the name check passes them and the trial
    render is what refuses -- the negative assertion says the name check did not
    answer first.
    """
    with pytest.raises(ImproperlyConfigured) as refused:
        AgentConventions(**{name: template})

    message = str(refused.value)
    assert message.startswith(f"AgentConventions.{name} ")
    assert says in message
    assert "which it does not accept" not in message


def test_a_doubled_brace_is_a_literal_brace_and_not_a_placeholder():
    conventions = AgentConventions(handles="- Ids look like {{this}}.")

    assert conventions.handles == "- Ids look like {{this}}."


def test_it_is_frozen():
    with pytest.raises(FrozenInstanceError):
        AgentConventions().base = "Hello."


def test_two_default_instances_are_equal():
    """Equality is what the ``instructions=`` refusal compares a field against."""
    assert AgentConventions() == AgentConventions()
    assert AgentConventions(base=None) != AgentConventions()
