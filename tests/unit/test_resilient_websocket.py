"""Tests for `swarm_common.resilient_websocket` — `ResilientWebSocket` and heartbeat."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from time import monotonic
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swarm_common.channel import ReliableChannel
from swarm_common.resilient_websocket import (
    AppLevelHeartbeat,
    ConnectionState,
    ResilientWebSocket,
)

_WS_CONNECT = "swarm_common.resilient_websocket.websockets_connect"


def _mock_ws_connect(mock_ws: AsyncMock | None = None) -> AsyncMock:
    """Build an `AsyncMock` for `websockets_connect` whose result resolves to `mock_ws`."""
    if mock_ws is None:
        mock_ws = AsyncMock()
    return AsyncMock(return_value=mock_ws)


class TestConnectionState:
    """Tests for the `ConnectionState` enum."""

    def test_values(self) -> None:
        assert ConnectionState.DISCONNECTED == "DISCONNECTED"
        assert ConnectionState.CONNECTING == "CONNECTING"
        assert ConnectionState.CONNECTED == "CONNECTED"

    def test_is_str_enum(self) -> None:
        for state in ConnectionState:
            assert isinstance(state, str)


class TestAppLevelHeartbeat:
    """Tests for the `AppLevelHeartbeat` strategy."""

    def test_intercept_recv_recognizes_pong(self) -> None:
        """`intercept_recv` returns `True` for the matching pong payload."""
        hb = AppLevelHeartbeat(
            ping_payload=b"ping", pong_payload=b"pong",
        )
        assert hb.intercept_recv(b"pong") is True

    def test_intercept_recv_ignores_non_pong(self) -> None:
        """`intercept_recv` returns `False` for non-pong data."""
        hb = AppLevelHeartbeat(
            ping_payload=b"ping", pong_payload=b"pong",
        )
        assert hb.intercept_recv(b"some other data") is False

    def test_intercept_recv_resets_timer(self) -> None:
        """Receiving a pong resets `_last_pong_time`."""
        hb = AppLevelHeartbeat(
            ping_payload=b"ping", pong_payload=b"pong",
        )
        old_time = hb._last_pong_time
        hb.intercept_recv(b"pong")
        assert hb._last_pong_time >= old_time

    def test_on_connected_resets_timer(self) -> None:
        """`on_connected` resets the pong timer."""
        hb = AppLevelHeartbeat(
            ping_payload=b"ping", pong_payload=b"pong",
        )
        hb._last_pong_time = 0.0
        hb.on_connected()
        assert hb._last_pong_time > 0.0

    async def test_run_sends_ping(self) -> None:
        """The heartbeat loop sends ping payloads."""
        hb = AppLevelHeartbeat(
            ping_payload=b"test-ping", pong_payload=b"test-pong",
            interval_s=0.05, timeout_s=5.0,
        )
        hb.on_connected()

        send = AsyncMock()
        mark_dead = AsyncMock()

        task = asyncio.create_task(hb.run(send, mark_dead))
        await asyncio.sleep(0.15)
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

        assert send.await_count >= 1
        send.assert_awaited_with(b"test-ping")

    async def test_run_detects_timeout(self) -> None:
        """The heartbeat loop calls `mark_dead` on pong timeout."""
        # timeout < interval forces immediate detection on first tick
        hb = AppLevelHeartbeat(
            ping_payload=b"ping", pong_payload=b"pong",
            interval_s=0.05, timeout_s=0.01,
        )
        hb._last_pong_time = monotonic() - 10.0

        send = AsyncMock()
        mark_dead = AsyncMock()

        await hb.run(send, mark_dead)

        mark_dead.assert_awaited_once()

    async def test_run_exits_on_send_failure(self) -> None:
        """The heartbeat loop exits when send raises `ConnectionError`."""
        hb = AppLevelHeartbeat(
            ping_payload=b"ping", pong_payload=b"pong",
            interval_s=0.05, timeout_s=5.0,
        )
        hb.on_connected()

        send = AsyncMock(side_effect=ConnectionError("disconnected"))
        mark_dead = AsyncMock()

        await hb.run(send, mark_dead)

        mark_dead.assert_not_awaited()


class TestResilientWebSocketInit:
    """Tests for `ResilientWebSocket` initialization."""

    def test_default_state(self) -> None:
        """New instance starts `DISCONNECTED` with the event unset."""
        rws = ResilientWebSocket(url="ws://localhost:8100/test")
        assert rws.state == ConnectionState.DISCONNECTED
        assert not rws.is_connected
        assert not rws.disconnected_event.is_set()

    def test_url_stored(self) -> None:
        """URL is accessible via property."""
        rws = ResilientWebSocket(url="ws://host:1234/path")
        assert rws.url == "ws://host:1234/path"

    def test_callbacks_settable(self) -> None:
        """`on_connect` and `on_disconnect` are settable properties."""
        rws = ResilientWebSocket(url="ws://localhost")
        cb = AsyncMock()
        rws.on_connect = cb
        rws.on_disconnect = cb
        assert rws.on_connect is cb
        assert rws.on_disconnect is cb

    def test_heartbeat_none_by_default(self) -> None:
        """No heartbeat strategy by default."""
        rws = ResilientWebSocket(url="ws://localhost")
        assert rws._heartbeat is None


class TestResilientWebSocketConnect:
    """Tests for the `connect()` method."""

    async def test_connect_success_transitions_to_connected(self) -> None:
        """Successful `connect` transitions state to `CONNECTED`."""
        rws = ResilientWebSocket(url="ws://localhost:8100/test")

        with patch(_WS_CONNECT, _mock_ws_connect()):
            await rws.connect()

        assert rws.state == ConnectionState.CONNECTED
        assert rws.is_connected

    async def test_connect_failure_raises_connection_error(self) -> None:
        """Failed `connect` raises `ConnectionError` and stays `DISCONNECTED`."""
        rws = ResilientWebSocket(url="ws://localhost:8100/test")

        with (
            patch(_WS_CONNECT, AsyncMock(side_effect=OSError("Connection refused"))),
            pytest.raises(ConnectionError),
        ):
            await rws.connect()

        assert rws.state == ConnectionState.DISCONNECTED

    async def test_connect_fires_on_connect_callback(self) -> None:
        """`on_connect` callback is called after successful connection."""
        callback = AsyncMock()
        rws = ResilientWebSocket(url="ws://localhost", on_connect=callback)

        with patch(_WS_CONNECT, _mock_ws_connect()):
            await rws.connect()

        callback.assert_awaited_once()

    async def test_connect_passes_headers(self) -> None:
        """Headers are passed to `websockets_connect`."""
        headers = {"Authorization": "Bearer test-key"}
        rws = ResilientWebSocket(url="ws://localhost", headers=headers)

        mock_connect = _mock_ws_connect()
        with patch(_WS_CONNECT, mock_connect):
            await rws.connect()

        _, kwargs = mock_connect.call_args
        assert kwargs["additional_headers"] == headers

    async def test_connect_passes_ssl_context(self) -> None:
        """SSL context is passed to `websockets_connect`."""
        ssl_ctx = MagicMock()
        rws = ResilientWebSocket(url="wss://localhost", ssl_context=ssl_ctx)

        mock_connect = _mock_ws_connect()
        with patch(_WS_CONNECT, mock_connect):
            await rws.connect()

        _, kwargs = mock_connect.call_args
        assert kwargs["ssl"] is ssl_ctx

    async def test_connect_passes_connect_kwargs(self) -> None:
        """Extra `connect_kwargs` are forwarded to `websockets_connect`."""
        rws = ResilientWebSocket(
            url="ws://localhost",
            connect_kwargs={"ping_interval": 20, "ping_timeout": 30},
        )

        mock_connect = _mock_ws_connect()
        with patch(_WS_CONNECT, mock_connect):
            await rws.connect()

        _, kwargs = mock_connect.call_args
        assert kwargs["ping_interval"] == 20
        assert kwargs["ping_timeout"] == 30

    async def test_connect_clears_disconnected_event(self) -> None:
        """`disconnected_event` is cleared on successful connect."""
        rws = ResilientWebSocket(url="ws://localhost")
        rws._disconnected_event.set()

        with patch(_WS_CONNECT, _mock_ws_connect()):
            await rws.connect()

        assert not rws.disconnected_event.is_set()

    async def test_connect_resets_heartbeat(self) -> None:
        """Heartbeat strategy `on_connected` is called after connect."""
        hb = AppLevelHeartbeat(
            ping_payload=b"ping", pong_payload=b"pong",
        )
        hb._last_pong_time = 0.0
        rws = ResilientWebSocket(url="ws://localhost", heartbeat=hb)

        with patch(_WS_CONNECT, _mock_ws_connect()):
            await rws.connect()

        assert hb._last_pong_time > 0.0


class TestResilientWebSocketSendRecv:
    """Tests for `send_raw()` and `recv_raw()`."""

    async def test_send_delegates_to_ws(self) -> None:
        """`send_raw` delegates to the underlying WebSocket."""
        mock_ws = AsyncMock()
        rws = ResilientWebSocket(url="ws://localhost")

        with patch(_WS_CONNECT, _mock_ws_connect(mock_ws)):
            await rws.connect()

        await rws.send_raw(b"hello")
        mock_ws.send.assert_awaited_once_with(b"hello")

    async def test_send_when_disconnected_raises(self) -> None:
        """`send_raw` raises `ConnectionError` when not connected."""
        rws = ResilientWebSocket(url="ws://localhost")

        with pytest.raises(ConnectionError):
            await rws.send_raw(b"hello")

    async def test_send_failure_marks_disconnected(self) -> None:
        """`send_raw` marks disconnected and raises on send failure."""
        mock_ws = AsyncMock()
        mock_ws.send.side_effect = OSError("broken pipe")
        rws = ResilientWebSocket(url="ws://localhost")

        with patch(_WS_CONNECT, _mock_ws_connect(mock_ws)):
            await rws.connect()

        with pytest.raises(ConnectionError):
            await rws.send_raw(b"hello")

        assert rws.state == ConnectionState.DISCONNECTED
        assert rws.disconnected_event.is_set()

    async def test_recv_delegates_to_ws(self) -> None:
        """`recv_raw` returns data from the underlying WebSocket."""
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = b"response"
        rws = ResilientWebSocket(url="ws://localhost")

        with patch(_WS_CONNECT, _mock_ws_connect(mock_ws)):
            await rws.connect()

        data = await rws.recv_raw()
        assert data == b"response"

    async def test_recv_handles_string_data(self) -> None:
        """`recv_raw` handles string data by encoding to bytes."""
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = '{"type": "test"}'
        rws = ResilientWebSocket(url="ws://localhost")

        with patch(_WS_CONNECT, _mock_ws_connect(mock_ws)):
            await rws.connect()

        data = await rws.recv_raw()
        assert data == b'{"type": "test"}'

    async def test_recv_when_disconnected_raises(self) -> None:
        """`recv_raw` raises `ConnectionError` when not connected."""
        rws = ResilientWebSocket(url="ws://localhost")

        with pytest.raises(ConnectionError):
            await rws.recv_raw()

    async def test_recv_intercepts_pongs_with_heartbeat(self) -> None:
        """`recv_raw` intercepts pong payloads when a heartbeat is configured."""
        hb = AppLevelHeartbeat(
            ping_payload=b"ping", pong_payload=b"pong",
        )
        mock_ws = AsyncMock()
        mock_ws.recv.side_effect = [b"pong", b"real-data"]
        rws = ResilientWebSocket(url="ws://localhost", heartbeat=hb)

        with patch(_WS_CONNECT, _mock_ws_connect(mock_ws)):
            await rws.connect()

        data = await rws.recv_raw()
        assert data == b"real-data"
        assert mock_ws.recv.await_count == 2

    async def test_recv_passes_through_without_heartbeat(self) -> None:
        """`recv_raw` returns all data when no heartbeat is configured."""
        mock_ws = AsyncMock()
        mock_ws.recv.return_value = b"pong"
        rws = ResilientWebSocket(url="ws://localhost")

        with patch(_WS_CONNECT, _mock_ws_connect(mock_ws)):
            await rws.connect()

        data = await rws.recv_raw()
        assert data == b"pong"


class TestResilientWebSocketDisconnect:
    """Tests for the `disconnect()` method."""

    async def test_disconnect_closes_ws(self) -> None:
        """`disconnect` closes the underlying WebSocket."""
        mock_ws = AsyncMock()
        rws = ResilientWebSocket(url="ws://localhost")

        with patch(_WS_CONNECT, _mock_ws_connect(mock_ws)):
            await rws.connect()

        await rws.disconnect()
        mock_ws.close.assert_awaited_once()
        assert rws.state == ConnectionState.DISCONNECTED

    async def test_disconnect_fires_callback(self) -> None:
        """`disconnect` fires the `on_disconnect` callback."""
        callback = AsyncMock()
        rws = ResilientWebSocket(url="ws://localhost", on_disconnect=callback)

        with patch(_WS_CONNECT, _mock_ws_connect()):
            await rws.connect()

        await rws.disconnect()
        callback.assert_awaited_once()

    async def test_disconnect_idempotent(self) -> None:
        """`disconnect` is safe to call when already disconnected."""
        rws = ResilientWebSocket(url="ws://localhost")
        await rws.disconnect()
        assert rws.state == ConnectionState.DISCONNECTED

    async def test_disconnect_sets_event(self) -> None:
        """`disconnect` sets `disconnected_event`."""
        rws = ResilientWebSocket(url="ws://localhost")

        with patch(_WS_CONNECT, _mock_ws_connect()):
            await rws.connect()

        assert not rws.disconnected_event.is_set()
        await rws.disconnect()
        assert rws.disconnected_event.is_set()

    async def test_mark_disconnected_fires_once(self) -> None:
        """`_mark_disconnected` only fires the callback on the first call."""
        callback = AsyncMock()
        rws = ResilientWebSocket(url="ws://localhost", on_disconnect=callback)

        with patch(_WS_CONNECT, _mock_ws_connect()):
            await rws.connect()

        await rws._mark_disconnected()
        await rws._mark_disconnected()
        callback.assert_awaited_once()


class TestResilientWebSocketBackoff:
    """Tests for `calc_backoff()`."""

    def test_first_attempt_is_base(self) -> None:
        """Attempt 0 returns base + jitter."""
        # jitter_factor=0 gives a deterministic delay
        rws = ResilientWebSocket(
            url="ws://localhost",
            backoff_base_s=1.0,
            backoff_max_s=60.0,
            backoff_jitter_factor=0.0,
        )
        delay = rws.calc_backoff(0)
        assert delay == 1.0

    def test_exponential_increase(self) -> None:
        """Delay doubles with each attempt."""
        rws = ResilientWebSocket(
            url="ws://localhost",
            backoff_base_s=1.0,
            backoff_max_s=60.0,
            backoff_jitter_factor=0.0,
        )
        assert rws.calc_backoff(0) == 1.0
        assert rws.calc_backoff(1) == 2.0
        assert rws.calc_backoff(2) == 4.0
        assert rws.calc_backoff(3) == 8.0

    def test_caps_at_max(self) -> None:
        """Delay is capped at `backoff_max_s`."""
        rws = ResilientWebSocket(
            url="ws://localhost",
            backoff_base_s=1.0,
            backoff_max_s=10.0,
            backoff_jitter_factor=0.0,
        )
        assert rws.calc_backoff(100) == 10.0

    def test_jitter_within_range(self) -> None:
        """Jitter adds between 0 and `base_delay * jitter_factor`."""
        rws = ResilientWebSocket(
            url="ws://localhost",
            backoff_base_s=10.0,
            backoff_max_s=60.0,
            backoff_jitter_factor=0.25,
        )
        for _ in range(100):
            delay = rws.calc_backoff(0)
            assert 10.0 <= delay <= 12.5


class TestResilientWebSocketNoHeartbeat:
    """Tests for `run_heartbeat()` without a heartbeat strategy."""

    async def test_run_heartbeat_noop_without_strategy(self) -> None:
        """`run_heartbeat` returns immediately when `heartbeat=None`."""
        rws = ResilientWebSocket(url="ws://localhost")
        await rws.run_heartbeat()


class TestRawTransportCompliance:
    """Verify `ResilientWebSocket` satisfies the `RawTransport` protocol."""

    def test_has_send_raw(self) -> None:
        rws = ResilientWebSocket(url="ws://localhost")
        assert hasattr(rws, "send_raw")
        assert callable(rws.send_raw)

    def test_has_recv_raw(self) -> None:
        rws = ResilientWebSocket(url="ws://localhost")
        assert hasattr(rws, "recv_raw")
        assert callable(rws.recv_raw)

    def test_satisfies_protocol(self) -> None:
        """`ResilientWebSocket` can be used where `RawTransport` is expected."""
        rws = ResilientWebSocket(url="ws://localhost")
        channel = ReliableChannel(rws)
        assert channel._connection is rws
