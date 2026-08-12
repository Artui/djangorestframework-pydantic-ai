"""``UrlKwarg`` — re-exported from drf-services, which owns the type.

The declaration is identical whichever transport carries it, and a per-package
copy drifts: two copies once validated the same declaration against different
reserved-name sets, so ``UrlKwarg("user")`` was legal for one and rejected by
the other.

The import path here is preserved permanently — ``from rest_framework_pydantic_ai
import UrlKwarg`` keeps working. New code may import from either location; they
are the same class.
"""

from __future__ import annotations

from rest_framework_services.types.url_kwarg import UrlKwarg

__all__ = ["UrlKwarg"]
