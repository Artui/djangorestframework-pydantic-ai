from __future__ import annotations

import re

import rest_framework_pydantic_ai


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", rest_framework_pydantic_ai.__version__)
