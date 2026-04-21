"""Tests for the ``after_log_to_db`` hookspec.

Plugins that persist auxiliary metadata alongside each logged response
(replay chain hashes, audit trails, etc.) need a notification that fires
if and only if llm actually writes the ``responses`` row. Gating on
``log_to_db`` itself keeps plugin persistence aligned with the user's
log policy (``llm logs off``, ``--no-log``) without the plugin having to
re-implement that policy.
"""

import sqlite_utils

from llm import hookimpl
from llm.migrations import migrate
from llm.plugins import pm


class _Recorder:
    """Captures (response, db) pairs passed to the hook."""

    def __init__(self):
        self.calls = []

    def record(self, response, db):
        self.calls.append((response, db))


def _register_recorder(recorder, *, name="AfterLogToDbRecorder"):
    class RecorderPlugin:
        __name__ = name

        @hookimpl
        def after_log_to_db(self, response, db):
            recorder.record(response, db)

    instance = RecorderPlugin()
    pm.register(instance, name=name)
    return instance


def test_hook_fires_after_response_persisted(mock_model):
    """Hook receives the same response the caller logged, and the db row is
    already visible at hook time so plugins can read it back."""
    recorder = _Recorder()
    try:
        _register_recorder(recorder)

        db = sqlite_utils.Database(memory=True)
        migrate(db)
        response = mock_model.prompt("hello")
        response.text()
        response.log_to_db(db)

        assert len(recorder.calls) == 1
        logged_response, logged_db = recorder.calls[0]
        assert logged_response is response
        assert logged_db is db

        row = db["responses"].get(response.id)
        assert row["prompt"] == "hello"
    finally:
        pm.unregister(name="AfterLogToDbRecorder")


def test_hook_not_fired_when_log_to_db_not_called(mock_model):
    """No log_to_db, no notification — the hook is strictly tied to
    persistence, not response completion."""
    recorder = _Recorder()
    try:
        _register_recorder(recorder)

        response = mock_model.prompt("hello")
        response.text()
        # Deliberately no log_to_db call.
        assert recorder.calls == []
    finally:
        pm.unregister(name="AfterLogToDbRecorder")


def test_log_to_db_works_with_no_registered_plugins(mock_model):
    """Regression guard: log_to_db completes cleanly when nobody listens."""
    db = sqlite_utils.Database(memory=True)
    migrate(db)
    response = mock_model.prompt("hello")
    response.text()
    response.log_to_db(db)
    assert db["responses"].get(response.id)["prompt"] == "hello"
