"""Django settings used by the test suite."""

from __future__ import annotations

import os
import tempfile

SECRET_KEY = "test-secret-key"

DEBUG = False

# ``TEST["NAME"]`` is what makes a connection genuinely closable, and it is load
# bearing rather than a preference. Django's SQLite backend *ignores* ``close()``
# on an in-memory database, to avoid destroying it -- and for SQLite the test
# database is in-memory whatever ``NAME`` says unless ``TEST["NAME"]`` names a
# file. Under the default, any assertion about a connection being released
# passes for a working fix and for a no-op alike, which is how an attempt at the
# off-HTTP connection cleanup in ``spec_toolset`` came out green without having
# been verified at all. Two tests assert ``not connection.is_in_memory_db()``
# before asserting anything else, so reverting this line fails loudly instead of
# quietly making them vacuous.
#
# Built rather than written down: no committed file in this family may carry an
# absolute local path. The pid keeps two concurrent runs off one file; Django
# deletes it at the end of the session.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
        "TEST": {"NAME": os.path.join(tempfile.gettempdir(), f"pai_test_{os.getpid()}.sqlite3")},
    },
}

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "rest_framework",
    "rest_framework_services",
    "tests.testapp",
]

# Skip migrations for the test app — pytest-django builds the schema from models.
MIGRATION_MODULES = {"testapp": None}

USE_TZ = True

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [],
    "TEST_REQUEST_DEFAULT_FORMAT": "json",
}
