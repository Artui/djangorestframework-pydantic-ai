from __future__ import annotations

from rest_framework_services import UNSET

from rest_framework_pydantic_ai import QueryParam


def test_defaults_are_a_plain_string_param():
    param = QueryParam("fields")
    assert param.name == "fields"
    assert param.type == "string"
    assert param.description is None
    # The declared floor's "no default" marker. Plain ``None`` became a
    # *declarable* value there, so a param that declares nothing carries the
    # sentinel and never reaches the spec as an argument.
    assert param.default is UNSET
    assert param.json_schema() == {"type": "string"}


def test_json_schema_includes_description_and_default_when_set():
    param = QueryParam("page_size", type="integer", description="rows per page", default=20)
    assert param.json_schema() == {
        "type": "integer",
        "description": "rows per page",
        "default": 20,
    }
