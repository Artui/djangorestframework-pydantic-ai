"""Suite-wide guard: no test may leave rows committed in the test database.

``pytest-django``'s rollback is per *connection*, and ``django.db.connections``
is thread-local -- so it covers the connection on the thread running the test and
nothing else. Every dispatch this package makes crosses a thread: ``call_tool``
runs the whole ORM pipeline through ``sync_to_async`` because Django forbids the
ORM on the event loop, and asgiref's shared thread has no transaction on it. A
write that lands there is in autocommit, commits, and outlives the test that made
it.

The cost is paid somewhere else. A stray row is invisible to the test that wrote
it -- every assertion there still passes -- and lands on whichever later test
counts rows, which then reads as broken for a reason nowhere near itself. It is
order-dependent twice over: which test pays depends on the run order, and the
evidence is *erased* by the next ``transaction=True`` test, whose teardown
truncates every table.

That last part is why this check is per-test rather than one assertion at the end
of the session. At the end of the session this database is clean: the leak this
file was written for had already been wiped by a truncation sixteen tests later,
so a session-scoped assertion would have passed while sixteen tests ran against a
polluted database.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest
from django.db import connections

# Stashed on ``config`` rather than kept in module globals, so the state belongs
# to the run that produced it and no test can reach it.
_DB_PATH = pytest.StashKey[str]()
# What a clean database holds, captured once. Not empty: migrations leave content
# types and permissions behind.
_PRISTINE = pytest.StashKey[dict[str, int]]()
# What the *next* test is entitled to find. Tracks ``_PRISTINE`` until something
# leaks; see ``_assert_nothing_outlived`` for why it then moves.
_EXPECTED = pytest.StashKey[dict[str, int]]()
_CURRENT_TEST = pytest.StashKey[str]()
_LAST_TEST_CHECKED = pytest.StashKey[bool]()


def _committed_row_counts(path: str) -> dict[str, int]:
    """Row counts for every table, read over a connection of this check's own.

    Two reasons it is raw ``sqlite3`` rather than the ORM. It must see
    *committed* state and only that -- Django's own connection is inside the
    test's atomic block for much of the run, so it would report rows that are
    about to be rolled back and accuse every well-behaved test of the one thing
    this file is looking for. And this suite asserts directly on Django
    connection state -- which thread holds one, whether a dispatch closed what it
    opened -- so a check that opened or closed one of those would be perturbing
    the thing it measures. A separate read-only handle touches neither.
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
    finally:
        connection.close()


def _assert_nothing_outlived(config: pytest.Config, nodeid: str) -> None:
    """Fail ``nodeid`` if the committed contents of the database moved under it.

    Two states are legitimate. Unchanged since the previous test is the ordinary
    one. Back to pristine is the other: a ``transaction=True`` test truncates
    every table on teardown, which is also how a leak gets erased, and blaming
    the test that cleaned up would point at the wrong end of the problem.

    Anything else is a leak, and ``_EXPECTED`` moves to what was actually found
    *before* raising. That is what keeps the accusation on one test: without it
    the sixteen tests that merely ran after the original leak would each fail
    for a row none of them wrote -- the exact confusion this guard exists to
    stop, reproduced by the guard itself.
    """
    counts = _committed_row_counts(config.stash[_DB_PATH])
    expected = config.stash[_EXPECTED]
    pristine = config.stash[_PRISTINE]
    config.stash[_EXPECTED] = counts
    if counts in (expected, pristine):
        return
    changed = {
        table: (expected.get(table, 0), count)
        for table, count in counts.items()
        if count != expected.get(table, 0)
    }
    raise AssertionError(
        f"{nodeid} left rows committed in the test database -- table: (before, after) "
        f"{changed}. pytest-django rolls back the connection belonging to the test's own "
        "thread and nothing else, so a write that ran anywhere else -- anything reached "
        "through sync_to_async, which is every dispatch this package makes -- committed, "
        "and a later test will read it as though that test were broken. Mark this test "
        "@pytest.mark.django_db(transaction=True), whose teardown truncates."
    )


@pytest.fixture(scope="session", autouse=True)
def _committed_row_baseline(request: pytest.FixtureRequest, django_db_setup: Any) -> Iterator[None]:
    """Capture the pristine contents, and check the run's final test.

    Depending on ``django_db_setup`` is what makes the baseline exist before the
    first teardown asks for it, instead of being captured off whichever test ran
    first -- which would fold that test's own leak into the baseline and exempt
    it forever.

    The same dependency is what lets this fixture check the *last* test of the
    run. Session fixtures are finalized during that test's teardown, in reverse
    order of setup, so this one runs while ``django_db_setup`` has yet to destroy
    the database -- and by the time the teardown hook below is called there is no
    database left to read. Nothing else in the run is positioned to ask.
    """
    path = str(connections["default"].settings_dict["NAME"])
    # Deliberately unguarded. A settings change that made this unreadable would
    # otherwise turn the whole guard into a silent no-op, which is the failure
    # mode ``conftest_settings.py`` already carries a comment about.
    request.config.stash[_DB_PATH] = path
    request.config.stash[_PRISTINE] = _committed_row_counts(path)
    request.config.stash[_EXPECTED] = dict(request.config.stash[_PRISTINE])
    yield
    request.config.stash[_LAST_TEST_CHECKED] = True
    _assert_nothing_outlived(request.config, request.config.stash[_CURRENT_TEST])


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Remember which test is running, so the fixture above can name it.

    Read only on the final-test path: every other test is named by the teardown
    hook, which is handed its own item.
    """
    item.config.stash[_CURRENT_TEST] = item.nodeid


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Check after every test, and blame the test that actually wrote the row.

    ``trylast`` so this runs after the runner's own teardown implementation --
    that is what finalizes the fixtures, so by the time this reads the database
    pytest-django's rollback (or a ``transaction=True`` flush) has already
    happened and a well-behaved test genuinely owes nothing.
    """
    if item.config.stash.get(_EXPECTED, None) is None:
        return  # the session fixture never ran, so there is no baseline to compare against
    if item.config.stash.get(_LAST_TEST_CHECKED, False):
        return  # the run's final test, already checked while its database existed
    _assert_nothing_outlived(item.config, item.nodeid)
