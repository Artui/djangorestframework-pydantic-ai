"""``QueryParam`` — re-exported from drf-services, which owns the type.

The declaration is the same whichever transport carries it. Unlike ``UrlKwarg``
it has no ``required`` flag, deliberately: a query param is *read-shaping*, so
omitting one is legitimate by construction.

The import path here is preserved permanently.
"""

from __future__ import annotations

from rest_framework_services.types.query_param import QueryParam

__all__ = ["QueryParam"]
