"""Client-mode WebSocket transport with reconnection, optional heartbeat, and backoff."""

from __future__ import annotations

from asyncio import Event, sleep
from collections.abc import Awaitable, Callable
from contextlib import suppress
from enum import StrEnum
from logging import getLogger
from random import uniform
from ssl import SSLContext
from time import monotonic
from typing import Any, Protocol

from websockets import connect as websockets_connect
from websockets.asyncio.client import ClientConnection

log = getLogger(__name__)


class ConnectionState(StrEnum):
    """WebSocket connection lifecycle states."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"


class HeartbeatStrategy(Protocol):
    """Pluggable heartbeat strategy — falls back to library-level ping/pong when omitted."""

    def on_connected(self) -> None:
        """Reset internal state when a new connection is established."""
        ...

    async def run(
        self,
        send: Callable[[bytes], Awaitable[None]],
        mark_dead: Callable[[], Awaitable[None]],
    ) -> None:
        """Run the heartbeat loop until the connection is lost.

        Args:
            send: Callable that sends raw bytes over the WebSocket.
            mark_dead: Callable that marks the connection as dead.
        """
        ...

    def intercept_recv(self, data: bytes) -> bool:
        """Return `True` if `data` is a heartbeat response and should be consumed."""
        ...


class AppLevelHeartbeat:
    """Heartbeat over custom ping/pong payloads — requires the peer to respond to the configured bytes."""

    def __init__(
        self,
        ping_payload: bytes,
        pong_payload: bytes,
        interval_s: float = 15.0,
        timeout_s: float = 30.0,
    ) -> None:
        self._ping_payload = ping_payload
        self._pong_payload = pong_payload
        self._interval_s = interval_s
        self._timeout_s = timeout_s
        self._last_pong_time = monotonic()
        log.debug(
            f"AppLevelHeartbeat initialized: ping={ping_payload!r}, pong={pong_payload!r}, "
            f"{interval_s=:.1f}s, {timeout_s=:.1f}s"
        )

    def on_connected(self) -> None:
        """Reset the pong timer when a new connection is established."""
        self._last_pong_time = monotonic()
        log.log(5, f"AppLevelHeartbeat.on_connected: pong timer reset to {self._last_pong_time:.3f}")
        log.debug("AppLevelHeartbeat.on_connected: pong timer reset")

    async def run(
        self,
        send: Callable[[bytes], Awaitable[None]],
        mark_dead: Callable[[], Awaitable[None]],
    ) -> None:
        """Send pings at the configured interval and call `mark_dead` if no pong arrives in time."""
        interval = self._interval_s
        timeout = self._timeout_s
        log.info(f"AppLevelHeartbeat.run: starting ({interval=:.1f}s, {timeout=:.1f}s)")

        while True:
            await sleep(interval)
            elapsed = monotonic() - self._last_pong_time
            log.log(5, f"AppLevelHeartbeat.run: tick — {elapsed=:.2f}s since last pong, {timeout=:.1f}s threshold")
            log.debug(f"AppLevelHeartbeat.run: {elapsed=:.1f}s since last pong")

            if elapsed > timeout:
                log.warning(f"AppLevelHeartbeat.run: TIMEOUT — {elapsed=:.1f}s > {timeout=:.1f}s")
                await mark_dead()
                break

            try:
                log.log(5, f"AppLevelHeartbeat.run: sending ping ({len(self._ping_payload)} bytes)")
                await send(self._ping_payload)
                log.debug("AppLevelHeartbeat.run: ping sent")
            except ConnectionError:
                log.debug("AppLevelHeartbeat.run: send failed, exiting")
                break

        log.info("AppLevelHeartbeat.run: heartbeat loop exited")

    def intercept_recv(self, data: bytes) -> bool:
        """Reset the pong timer and consume the message when `data` matches the configured pong payload."""
        if data == self._pong_payload:
            prev = self._last_pong_time
            self._last_pong_time = monotonic()
            log.log(
                5,
                f"AppLevelHeartbeat.intercept_recv: pong received, "
                f"prev={prev:.3f}, new={self._last_pong_time:.3f}, gap={self._last_pong_time - prev:.3f}s",
            )
            log.debug("AppLevelHeartbeat.intercept_recv: pong received, timer reset")
            return True
        return False


class ResilientWebSocket:
    """Client-mode WebSocket that exposes lifecycle primitives — callers compose their own reconnect loop."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        ssl_context: SSLContext | None = None,
        heartbeat: HeartbeatStrategy | None = None,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 60.0,
        backoff_jitter_factor: float = 0.25,
        on_connect: Callable[[], Awaitable[None]] | None = None,
        on_disconnect: Callable[[], Awaitable[None]] | None = None,
        connect_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._ssl_context = ssl_context
        self._heartbeat = heartbeat
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._backoff_jitter_factor = backoff_jitter_factor
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._connect_kwargs = connect_kwargs or {}
        self._state = ConnectionState.DISCONNECTED
        self._ws: ClientConnection | None = None
        self._disconnected_event = Event()
        log.debug(
            f"ResilientWebSocket initialized: {url=}, heartbeat={'on' if heartbeat else 'off'}, "
            f"{backoff_base_s=:.1f}s, {backoff_max_s=:.1f}s, {backoff_jitter_factor=}"
        )

    @property
    def url(self) -> str:
        """The WebSocket URL."""
        return self._url

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """`True` while the connection is in the `CONNECTED` state."""
        return self._state == ConnectionState.CONNECTED

    @property
    def disconnected_event(self) -> Event:
        """`Event` set when the connection is lost; callers can await this."""
        return self._disconnected_event

    @property
    def on_connect(self) -> Callable[[], Awaitable[None]] | None:
        """Callback fired after successful connection."""
        return self._on_connect

    @on_connect.setter
    def on_connect(self, callback: Callable[[], Awaitable[None]] | None) -> None:
        self._on_connect = callback

    @property
    def on_disconnect(self) -> Callable[[], Awaitable[None]] | None:
        """Callback fired when the connection is lost."""
        return self._on_disconnect

    @on_disconnect.setter
    def on_disconnect(self, callback: Callable[[], Awaitable[None]] | None) -> None:
        self._on_disconnect = callback

    async def connect(self) -> None:
        """Attempt a single WebSocket connection; fires `on_connect` and resets the heartbeat on success.

        Raises:
            ConnectionError: If the connection cannot be established.
        """
        log.info(f"connecting to {self._url}")
        self._state = ConnectionState.CONNECTING

        try:
            kwargs: dict[str, Any] = {
                "additional_headers": self._headers,
            }
            if self._ssl_context is not None:
                kwargs["ssl"] = self._ssl_context
            kwargs.update(self._connect_kwargs)

            log.log(5, f"connect kwargs={{{', '.join(f'{k}=...' for k in kwargs)}}}")

            ws = await websockets_connect(self._url, **kwargs)
            self._ws = ws
            self._state = ConnectionState.CONNECTED
            self._disconnected_event.clear()

            if self._heartbeat is not None:
                self._heartbeat.on_connected()

            log.info(f"connected to {self._url}")

            if self._on_connect is not None:
                log.debug("firing on_connect callback")
                await self._on_connect()

        except Exception as exc:
            log.warning(f"failed — {type(exc).__name__}: {exc}")
            self._state = ConnectionState.DISCONNECTED
            raise ConnectionError(f"Connect failed: {exc}") from exc

    async def disconnect(self) -> None:
        """Gracefully close the connection; idempotent, and fires `on_disconnect` on real transition."""
        if self._state == ConnectionState.DISCONNECTED:
            log.debug("already DISCONNECTED, skipping")
            return

        log.info("closing connection")
        prev = self._state.value
        self._state = ConnectionState.DISCONNECTED
        self._disconnected_event.set()

        if self._ws is not None:
            with suppress(Exception):
                await self._ws.close()
            self._ws = None

        log.debug(f"{prev} -> DISCONNECTED")

        if self._on_disconnect is not None:
            log.debug("firing on_disconnect callback")
            with suppress(Exception):
                await self._on_disconnect()

    async def send_raw(self, data: bytes) -> None:
        """Send raw bytes over the WebSocket.

        Raises:
            ConnectionError: If not connected or the send fails.
        """
        if not self.is_connected or self._ws is None:
            log.debug("not connected, raising ConnectionError")
            raise ConnectionError("Cannot send: not connected")

        size = len(data)
        log.log(5, f"{size=}, first_50_hex={data[:50].hex()}")
        log.debug(f"sending {size=} bytes")

        try:
            await self._ws.send(data)
            log.log(5, f"sent {size=} bytes successfully")
            log.debug(f"sent {size=} bytes")
        except Exception as exc:
            log.warning(f"failed — {type(exc).__name__}: {exc}")
            await self._mark_disconnected()
            raise ConnectionError(f"Send failed: {exc}") from exc

    async def recv_raw(self) -> bytes:
        """Receive raw bytes from the WebSocket; pong payloads are intercepted by the heartbeat strategy.

        Raises:
            ConnectionError: If not connected or the receive fails.
        """
        if not self.is_connected or self._ws is None:
            log.debug("not connected, raising ConnectionError")
            raise ConnectionError("Cannot receive: not connected")

        log.debug("waiting for data...")

        try:
            while True:
                raw = await self._ws.recv()
                data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)

                if self._heartbeat is not None and self._heartbeat.intercept_recv(data):
                    log.log(5, f"heartbeat intercepted {len(data)} bytes")
                    continue

                size = len(data)
                log.log(5, f"{size=}, first_50_hex={data[:50].hex()}")
                log.debug(f"received {size=} bytes")
                return data

        except Exception as exc:
            log.warning(f"failed — {type(exc).__name__}: {exc}")
            await self._mark_disconnected()
            raise ConnectionError(f"Receive failed: {exc}") from exc

    async def run_heartbeat(self) -> None:
        """Block on the heartbeat strategy loop until disconnect or timeout; no-op when no strategy is set."""
        if self._heartbeat is None:
            log.debug("no heartbeat strategy configured, no-op")
            return

        log.debug("delegating to heartbeat strategy")
        await self._heartbeat.run(
            send=self.send_raw,
            mark_dead=self._mark_disconnected,
        )

    def calc_backoff(self, attempt: int) -> float:
        """Calculate the reconnection delay using exponential backoff with jitter.

        Args:
            attempt: Zero-indexed reconnection attempt number.

        Returns:
            Delay in seconds before the next reconnection attempt.
        """
        base_delay = min(self._backoff_base_s * (2 ** attempt), self._backoff_max_s)
        jitter_amount = uniform(0.0, base_delay * self._backoff_jitter_factor)
        delay = float(base_delay + jitter_amount)
        log.log(
            5,
            f"{attempt=}, base_s={self._backoff_base_s}, max_s={self._backoff_max_s}, "
            f"{base_delay=:.3f}s, jitter={jitter_amount:.3f}s, {delay=:.3f}s",
        )
        log.debug(f"{attempt=}, {base_delay=:.2f}s, jitter={jitter_amount:.2f}s, {delay=:.2f}s")
        return delay

    async def _mark_disconnected(self) -> None:
        """Transition to `DISCONNECTED`, set the event, and fire `on_disconnect` exactly once."""
        if self._state == ConnectionState.DISCONNECTED:
            return

        prev = self._state.value
        log.info(f"{prev} -> DISCONNECTED")
        self._state = ConnectionState.DISCONNECTED
        self._disconnected_event.set()

        if self._on_disconnect is not None:
            with suppress(Exception):
                await self._on_disconnect()
