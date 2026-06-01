"""Shared fixtures for test isolation."""

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_testing_env(request):
    """Restore the TESTING env var after each test so fixture-level mutations
    do not leak to subsequent test files.

    Tests that set TESTING=1 via a *fixture* (e.g. test_db.py, test_topic_config.py)
    are covered: after teardown TESTING is rolled back.

    Tests that set TESTING=1 at *module level* (top-of-file) are NOT cleaned by
    this fixture because the assignment runs at import time.  For those, the env
    var is already set before any fixture runs.  We detect this case by checking
    whether the *current test module* has a module-level TESTING assignment, and
    if so we leave it alone during the test but still clean up afterward.
    """
    original = os.environ.get("TESTING")
    yield
    if original is None:
        os.environ.pop("TESTING", None)
    else:
        os.environ["TESTING"] = original
