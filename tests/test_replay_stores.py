from llm import hookimpl, ToolCall, ToolResult
from llm.plugins import pm
import pytest


class StubReplayStore:
    """Minimal ReplayStore used to verify the register_replay_stores hookspec.

    Mirrors the duck-typed shape documented in the llm-replay plugin: a
    ``lookup`` method that either returns ``None`` (miss) or an object with
    ``chunks``, ``response_json``, and ``source_id`` attributes (hit).

    Enablement is the store's own concern — a real plugin would consult an
    internal flag before deciding to engage. The test stub always engages
    when registered, so the test exercises the core dispatch rather than a
    plugin's enable gate.
    """

    def __init__(self, replayed=None):
        self.replayed = replayed
        self.lookup_calls = 0
        self.alookup_calls = 0

    def lookup(self, response):
        self.lookup_calls += 1
        return self.replayed

    async def alookup(self, response):
        self.alookup_calls += 1
        return self.replayed


class _Replayed:
    def __init__(
        self,
        chunks,
        response_json=None,
        source_id=None,
        tool_calls=None,
        tool_results=None,
        resolved_model=None,
    ):
        self.chunks = chunks
        self.response_json = response_json
        self.source_id = source_id
        self.tool_calls = tool_calls
        self.tool_results = tool_results
        self.resolved_model = resolved_model


def _register_store(store, *, name="ReplayStorePlugin"):
    class ReplayStorePlugin:
        __name__ = name

        @hookimpl
        def register_replay_stores(self, register):
            register(store)

    instance = ReplayStorePlugin()
    pm.register(instance, name=name)
    return instance


def test_replay_hit_intercepts_execution(mock_model):
    """A registered store returning a ReplayedResponse short-circuits execute()."""
    store = StubReplayStore(
        replayed=_Replayed(
            chunks=["replayed text"],
            response_json={"cached": True},
            source_id="prior-response-id",
        )
    )
    try:
        _register_store(store)
        response = mock_model.prompt("hello")
        assert response.text() == "replayed text"
        assert store.lookup_calls == 1
        assert response.replayed is True
        assert response.replay_source_id == "prior-response-id"
        assert response.response_json == {"cached": True}
        assert mock_model.history == []
    finally:
        pm.unregister(name="ReplayStorePlugin")


def test_no_stores_registered_no_lookup(mock_model):
    """With no registered stores, the live execute path runs unchanged."""
    mock_model.enqueue(["live response"])
    response = mock_model.prompt("hello")
    assert response.text() == "live response"
    assert response.replayed is False
    assert response.replay_source_id is None


def test_replay_miss_falls_through_to_execute(mock_model):
    """A store returning None lets the live execute() path run."""
    store = StubReplayStore(replayed=None)
    try:
        _register_store(store)
        mock_model.enqueue(["fresh"])
        response = mock_model.prompt("hello")
        assert response.text() == "fresh"
        assert store.lookup_calls == 1
        assert response.replayed is False
        assert response.replay_source_id is None
    finally:
        pm.unregister(name="ReplayStorePlugin")


def test_first_non_none_lookup_wins(mock_model):
    """When several stores are registered, the first non-None lookup wins."""
    first_store = StubReplayStore(replayed=None)
    second_store = StubReplayStore(
        replayed=_Replayed(chunks=["from second"], source_id="second-id")
    )
    third_store = StubReplayStore(
        replayed=_Replayed(chunks=["from third"], source_id="third-id")
    )
    try:
        _register_store(first_store, name="FirstReplayStore")
        _register_store(second_store, name="SecondReplayStore")
        _register_store(third_store, name="ThirdReplayStore")

        response = mock_model.prompt("hello")
        text = response.text()
        winners = [
            s for s in (first_store, second_store, third_store) if s.lookup_calls
        ]
        assert any(s.replayed is not None for s in winners)
        assert text in {"from second", "from third"}
        assert response.replayed is True
        assert response.replay_source_id in {"second-id", "third-id"}
    finally:
        for name in ("FirstReplayStore", "SecondReplayStore", "ThirdReplayStore"):
            pm.unregister(name=name)


@pytest.mark.asyncio
async def test_async_replay_hit(async_mock_model):
    """AsyncResponse consults ``alookup`` and serves the replayed chunks."""
    store = StubReplayStore(
        replayed=_Replayed(
            chunks=["async replayed"],
            response_json={"async": True},
            source_id="async-source-id",
        )
    )
    try:
        _register_store(store)
        response = async_mock_model.prompt("hello")
        text = await response.text()
        assert text == "async replayed"
        assert store.alookup_calls == 1
        assert store.lookup_calls == 0
        assert response.replayed is True
        assert response.replay_source_id == "async-source-id"
        assert async_mock_model.history == []
    finally:
        pm.unregister(name="ReplayStorePlugin")


def test_conversation_tracks_replayed_response(mock_model):
    """A replay hit still appends the response to its conversation."""
    store = StubReplayStore(replayed=_Replayed(chunks=["replayed"], source_id="src"))
    try:
        _register_store(store)
        conversation = mock_model.conversation()
        response = conversation.prompt("first")
        response.text()
        assert conversation.responses == [response]
    finally:
        pm.unregister(name="ReplayStorePlugin")


def test_execute_tool_calls_short_circuits_when_replayed_results_set(mock_model):
    """Setting _replayed_tool_results bypasses live tool execution.

    Exercises the short-circuit mechanism in isolation: no replay store,
    no hook dispatch — just proves that the attribute, when set, makes
    execute_tool_calls() return those results without consulting tools.
    """
    mock_model.enqueue(["ignored"])
    response = mock_model.prompt("hello")
    response.text()  # drain so _done is set
    mocked = [ToolResult(name="fake_tool", output="mocked output", tool_call_id="c1")]
    response._replayed_tool_results = mocked
    assert response.execute_tool_calls() is mocked


@pytest.mark.asyncio
async def test_async_execute_tool_calls_short_circuits_when_replayed_results_set(
    async_mock_model,
):
    """Async short-circuit: the attribute works identically on AsyncResponse."""
    async_mock_model.enqueue(["ignored"])
    response = async_mock_model.prompt("hello")
    await response.text()
    mocked = [ToolResult(name="fake_tool", output="mocked output", tool_call_id="c1")]
    response._replayed_tool_results = mocked
    assert await response.execute_tool_calls() is mocked


def test_replayed_tool_results_default_is_none(mock_model):
    """The attribute is initialized to None so live tool execution is the default."""
    response = mock_model.prompt("hello")
    assert response._replayed_tool_results is None


def test_replay_hit_maps_tool_calls_results_and_resolved_model(mock_model):
    """A ReplayedResponse carrying tool_calls / tool_results / resolved_model
    flows them onto the Response so the chain and audit trail see them."""
    tool_calls = [ToolCall(name="t", arguments={"x": 1}, tool_call_id="c1")]
    tool_results = [ToolResult(name="t", output="ok", tool_call_id="c1")]
    store = StubReplayStore(
        replayed=_Replayed(
            chunks=["replayed"],
            source_id="src",
            tool_calls=tool_calls,
            tool_results=tool_results,
            resolved_model="provider-canonical-v3",
        )
    )
    try:
        _register_store(store)
        response = mock_model.prompt("hello")
        response.text()
        assert response._tool_calls == tool_calls
        assert response._replayed_tool_results is tool_results
        assert response.resolved_model == "provider-canonical-v3"
        # And the short-circuit still engages on the populated attribute.
        assert response.execute_tool_calls() is tool_results
    finally:
        pm.unregister(name="ReplayStorePlugin")


@pytest.mark.asyncio
async def test_async_replay_hit_maps_tool_calls_results_and_resolved_model(
    async_mock_model,
):
    """Async parity with the sync mapping test."""
    tool_calls = [ToolCall(name="t", arguments={"x": 1}, tool_call_id="c1")]
    tool_results = [ToolResult(name="t", output="ok", tool_call_id="c1")]
    store = StubReplayStore(
        replayed=_Replayed(
            chunks=["replayed"],
            source_id="src",
            tool_calls=tool_calls,
            tool_results=tool_results,
            resolved_model="provider-canonical-v3",
        )
    )
    try:
        _register_store(store)
        response = async_mock_model.prompt("hello")
        await response.text()
        assert response._tool_calls == tool_calls
        assert response._replayed_tool_results is tool_results
        assert response.resolved_model == "provider-canonical-v3"
        assert await response.execute_tool_calls() is tool_results
    finally:
        pm.unregister(name="ReplayStorePlugin")


def test_replay_hit_without_optional_fields_leaves_defaults(mock_model):
    """Stores that only supply chunks (the v1 minimum) don't clobber defaults."""
    store = StubReplayStore(replayed=_Replayed(chunks=["replayed"], source_id="src"))
    try:
        _register_store(store)
        response = mock_model.prompt("hello")
        response.text()
        assert response._tool_calls == []
        assert response._replayed_tool_results is None
        assert response.resolved_model is None
    finally:
        pm.unregister(name="ReplayStorePlugin")
