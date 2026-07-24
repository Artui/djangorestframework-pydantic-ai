from __future__ import annotations

from rest_framework_pydantic_ai import UrlKwarg


def test_defaults_are_a_plain_string_kwarg():
    kwarg = UrlKwarg("parent_pk")
    assert kwarg.name == "parent_pk"
    assert kwarg.type == "string"
    assert kwarg.description is None
    assert kwarg.default is None
    assert kwarg.json_schema() == {"type": "string"}


def test_json_schema_includes_description_and_default_when_set():
    kwarg = UrlKwarg("project_pk", type="integer", description="owning project", default=1)
    assert kwarg.json_schema() == {
        "type": "integer",
        "description": "owning project",
        "default": 1,
    }
