"""Tests for the ``register_prompt_gates`` hookspec.

A prompt gate is a duck-typed object with a ``check(prompt, model,
conversation=None)`` method (and optionally ``acheck`` for async
responses). Gates are consulted before ``model.execute`` runs; a gate
raising :class:`llm.CancelPrompt` aborts the prompt before any upstream
API call, which is what plugins such as ``llm-confirm-tokens`` use to
interpose a "proceed?" prompt keyed on token count.
"""

from llm import CancelPrompt, hookimpl
from llm.plugins import pm
import pytest


class StubGate:
    """Minimal PromptGate used to verify the register_prompt_gates hookspec.

    Records the (prompt, model, conversation) tuples it has seen and
    optionally raises :class:`CancelPrompt` to exercise the cancellation
    path.
    """

    def __init__(self, *, cancel=False, reason="stub cancel"):
        self.cancel = cancel
        self.reason = reason
        self.check_calls = []
        self.acheck_calls = []

    def check(self, prompt, model, conversation=None):
        self.check_calls.append((prompt, model, conversation))
        if self.cancel:
            raise CancelPrompt(self.reason)

    async def acheck(self, prompt, model, conversation=None):
        self.acheck_calls.append((prompt, model, conversation))
        if self.cancel:
            raise CancelPrompt(self.reason)


class SyncOnlyGate:
    """Gate without acheck — async path must fall back to sync check()."""

    def __init__(self):
        self.check_calls = []

    def check(self, prompt, model, conversation=None):
        self.check_calls.append((prompt, model, conversation))


class LegacyGate:
    """Gate pinned to the original ``(prompt, model)`` signature.

    Used to verify core's TypeError fallback — old gates that don't
    accept the new ``conversation`` kwarg must still be invoked in the
    two-arg shape.
    """

    def __init__(self):
        self.check_calls = []

    def check(self, prompt, model):
        self.check_calls.append((prompt, model))


def _register_gate(gate, *, name="PromptGatePlugin"):
    class PromptGatePlugin:
        __name__ = name

        @hookimpl
        def register_prompt_gates(self, register):
            register(gate)

    instance = PromptGatePlugin()
    pm.register(instance, name=name)
    return instance


def test_gate_check_runs_on_live_path(mock_model):
    """Every registered gate's ``check`` is invoked before model.execute."""
    gate = StubGate()
    try:
        _register_gate(gate)
        mock_model.enqueue(["hello"])
        response = mock_model.prompt("hi")
        assert response.text() == "hello"
        assert len(gate.check_calls) == 1
        prompt, model, conversation = gate.check_calls[0]
        assert prompt is response.prompt
        assert model is mock_model
        # One-shot prompt — no conversation attached.
        assert conversation is None
    finally:
        pm.unregister(name="PromptGatePlugin")


def test_gate_receives_conversation_on_continue(mock_model):
    """When the prompt runs inside a Conversation, the gate sees it."""
    gate = StubGate()
    try:
        _register_gate(gate)
        conversation = mock_model.conversation()
        mock_model.enqueue(["first"])
        conversation.prompt("hello").text()
        mock_model.enqueue(["second"])
        response = conversation.prompt("again")
        assert response.text() == "second"
        # Two checks (one per turn), each with the same conversation.
        assert len(gate.check_calls) == 2
        assert gate.check_calls[0][2] is conversation
        assert gate.check_calls[1][2] is conversation
    finally:
        pm.unregister(name="PromptGatePlugin")


def test_legacy_gate_still_invoked_with_two_args(mock_model):
    """Gates pinned to ``(prompt, model)`` are invoked in the old shape."""
    gate = LegacyGate()
    try:
        _register_gate(gate)
        mock_model.enqueue(["ok"])
        response = mock_model.prompt("hi")
        assert response.text() == "ok"
        assert len(gate.check_calls) == 1
    finally:
        pm.unregister(name="PromptGatePlugin")


def test_cancelprompt_aborts_before_execute(mock_model):
    """CancelPrompt from a gate propagates and model.execute is not called."""
    gate = StubGate(cancel=True, reason="user said no")
    try:
        _register_gate(gate)
        response = mock_model.prompt("hi")
        with pytest.raises(CancelPrompt, match="user said no"):
            response.text()
        assert mock_model.history == []
    finally:
        pm.unregister(name="PromptGatePlugin")


def test_cancelprompt_short_circuits_remaining_gates(mock_model):
    """Once any gate raises, subsequent gates in dispatch order are not consulted.

    pluggy dispatches registered plugins in LIFO order, so we don't assert on
    *which* gate fires first — only that exactly one of the two is consulted
    before the CancelPrompt propagates.
    """
    first = StubGate(cancel=True, reason="blocked")
    second = StubGate(cancel=True, reason="also blocked")
    try:
        _register_gate(first, name="FirstGate")
        _register_gate(second, name="SecondGate")
        response = mock_model.prompt("hi")
        with pytest.raises(CancelPrompt):
            response.text()
        total_checks = len(first.check_calls) + len(second.check_calls)
        assert total_checks == 1
    finally:
        pm.unregister(name="FirstGate")
        pm.unregister(name="SecondGate")


def test_no_gates_registered_runs_live_path(mock_model):
    """With no registered gates, the live execute path runs unchanged."""
    mock_model.enqueue(["live response"])
    response = mock_model.prompt("hi")
    assert response.text() == "live response"


@pytest.mark.asyncio
async def test_async_gate_uses_acheck(async_mock_model):
    """AsyncResponse prefers ``acheck`` when the gate exposes it."""
    gate = StubGate()
    try:
        _register_gate(gate)
        async_mock_model.enqueue(["async live"])
        response = async_mock_model.prompt("hi")
        assert await response.text() == "async live"
        assert len(gate.acheck_calls) == 1
        assert gate.check_calls == []
    finally:
        pm.unregister(name="PromptGatePlugin")


@pytest.mark.asyncio
async def test_async_gate_falls_back_to_check(async_mock_model):
    """A gate without ``acheck`` has its sync ``check`` called by the async path."""
    gate = SyncOnlyGate()
    try:
        _register_gate(gate)
        async_mock_model.enqueue(["async live"])
        response = async_mock_model.prompt("hi")
        assert await response.text() == "async live"
        assert len(gate.check_calls) == 1
    finally:
        pm.unregister(name="PromptGatePlugin")


@pytest.mark.asyncio
async def test_async_cancelprompt_aborts_before_execute(async_mock_model):
    """CancelPrompt from a gate propagates on the async path too."""
    gate = StubGate(cancel=True, reason="nope")
    try:
        _register_gate(gate)
        response = async_mock_model.prompt("hi")
        with pytest.raises(CancelPrompt, match="nope"):
            await response.text()
        assert async_mock_model.history == []
    finally:
        pm.unregister(name="PromptGatePlugin")


def test_gate_fires_once_per_response(mock_model):
    """Repeated forcing of a response must not re-consult gates."""
    gate = StubGate()
    try:
        _register_gate(gate)
        mock_model.enqueue(["hi"])
        response = mock_model.prompt("hi")
        response.text()
        response.text()
        response.text()
        assert len(gate.check_calls) == 1
    finally:
        pm.unregister(name="PromptGatePlugin")
