from __future__ import annotations

import re

import drf_pydantic_ai


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", drf_pydantic_ai.__version__)
