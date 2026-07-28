"""``QueryParam`` — re-exported from drf-services, which now owns the type.

Lifted alongside :class:`~rest_framework_pydantic_ai.UrlKwarg` for the same
reason: the declaration is the same whichever transport carries it. This one had
no second copy to drift from yet — the MCP transport's request-level query-param
registration is still unbuilt — so lifting it now is what prevents the fork
rather than repairing one.

Unlike ``UrlKwarg`` it has no ``required`` flag, deliberately: a query param is
*read-shaping*, so omitting one is legitimate by construction.

The import path here is preserved permanently.
"""

from __future__ import annotations

from rest_framework_services.types.query_param import QueryParam

__all__ = ["QueryParam"]
