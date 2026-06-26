"""Layer 2: ReliableChannel — at-least-once delivery over a connection.

Provides:
- Outbound: msg_id assignment, send buffer, ACK tracking
- Inbound: auto-ACK, deduplication via bounded seen-set
- Backpressure: drop oldest NORMAL messages when buffer full, never CRITICAL
- Reconnect: re-send all unacked messages in order

Works with any connection that implements the RawTransport protocol
(send_raw/recv_raw). Both the hive's server-mode ConnectionManager and
the drone's client-mode ConnectionManager satisfy this interface.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import StrEnum
from logging import getLogger
from typing import Protocol

from swarm_common.models import MessageType, WireMessage
from swarm_common.protocol import deserialize, make_ack, serialize

log = getLogger(__name__)


class RawTransport(Protocol):
    """Protocol for any transport that sends/receives raw bytes."""

    async def send_raw(self, data: bytes) -> None:
        """Send raw bytes over the transport."""
        ...

    async def recv_raw(self) -> bytes:
        """Receive raw bytes from the transport."""
        ...


class MessagePriority(StrEnum):
    """Priority levels for outbound messages.

    CRITICAL messages (commands, status, state_sync) are never dropped.
    NORMAL messages (event_batch, log_batch) may be evicted under pressure.
    """

    CRITICAL = "critical"
    NORMAL = "normal"


class ReliableChannel:
    """Layer 2: Reliable message delivery over an unreliable WebSocket.

    Outbound messages are buffered until ACK'd. On reconnect, all unacked
    messages are re-sent. Inbound messages are deduplicated by msg_id.
    """

    def __init__(
        self,
        connection: RawTransport,
        buffer_size: int = 10_000,
        dedup_max_size: int = 100_000,
    ) -> None:
        self._connection = connection
        self._buffer_size = buffer_size
        self._dedup_max_size = dedup_max_size
        self._send_buffer: OrderedDict[str, tuple[WireMessage, MessagePriority]] = OrderedDict()
        self._seen_ids: OrderedDict[str, None] = OrderedDict()
        log.debug(f"ReliableChannel initialized: {buffer_size=}, {dedup_max_size=}")

    @property
    def send_buffer(self) -> OrderedDict[str, tuple[WireMessage, MessagePriority]]:
        """The current outbound send buffer."""
        return self._send_buffer

    async def send(
        self,
        msg: WireMessage,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> None:
        """Send a message reliably.

        The message is buffered until an ACK is received. If the buffer
        exceeds its size limit, the oldest NORMAL-priority messages are
        evicted (CRITICAL messages are never dropped).

        Args:
            msg: The WireMessage to send.
            priority: CRITICAL or NORMAL. CRITICAL messages are never evicted.
        """
        msg_id = msg.msg_id
        msg_type = msg.type.value
        buf_size = len(self._send_buffer)
        log.log(
            5,
            f"send: enqueue to buffer — {msg_id=}, {msg_type=}, priority={priority.value}, "
            f"buffer_before={buf_size}, buffer_limit={self._buffer_size}",
        )
        log.debug(f"send: buffering {msg_id=}, {msg_type=}, priority={priority.value}, {buf_size=}")

        self._send_buffer[msg.msg_id] = (msg, priority)
        self._enforce_backpressure()

        data = serialize(msg)
        try:
            await self._connection.send_raw(data)
            buf_after = len(self._send_buffer)
            log.log(5, f"send: {msg_id=} sent ({len(data)} bytes), buffer_after={buf_after}, awaiting ACK")
            log.debug(f"send: {msg_id=} sent ({len(data)} bytes), awaiting ACK")
        except ConnectionError:
            buf_after = len(self._send_buffer)
            log.log(5, f"send: {msg_id=} buffered but send failed (disconnected), buffer_after={buf_after}, will replay on reconnect")
            log.warning(f"send: {msg_id=} buffered, send failed (disconnected) — will replay on reconnect")

    async def recv(self) -> WireMessage | None:
        """Receive the next deduplicated inbound message.

        ACK messages are processed internally (removing from send buffer)
        and not returned. Duplicate messages (same msg_id) are dropped.

        Returns:
            The next WireMessage, or None if the message was a duplicate or ACK.
        """
        data = await self._connection.recv_raw()
        msg = deserialize(data)
        msg_id = msg.msg_id
        msg_type = msg.type.value
        log.debug(f"recv: received {msg_id=}, {msg_type=}")

        if msg.type == MessageType.ACK:
            ack_id = msg.payload.get("ack_id", "")
            log.debug(f"recv: processing ACK for {ack_id=}")
            self.process_ack(ack_id)
            return None

        if msg_id in self._seen_ids:
            log.debug(f"recv: duplicate {msg_id=}, dropping (still sending ACK)")
            await self._send_ack(msg_id)
            return None

        self._mark_seen(msg_id)
        await self._send_ack(msg_id)
        log.debug(f"recv: {msg_id=} accepted and ACK'd")
        return msg

    def process_ack(self, msg_id: str) -> None:
        """Remove an ACK'd message from the send buffer.

        Safe to call with unknown msg_ids (no-op).
        """
        if msg_id in self._send_buffer:
            _msg, _prio = self._send_buffer[msg_id]
            log.log(
                5,
                f"process_ack: removing {msg_id=} (type={_msg.type.value}, priority={_prio.value}) "
                f"from buffer, buffer_before={len(self._send_buffer)}",
            )
            self._send_buffer.pop(msg_id)
            buf_size = len(self._send_buffer)
            log.log(5, f"process_ack: {msg_id=} removed, buffer_after={buf_size}")
            log.debug(f"process_ack: {msg_id=} removed from buffer ({buf_size=})")
        else:
            log.log(5, f"process_ack: {msg_id=} not found in buffer (buffer_size={len(self._send_buffer)})")
            log.debug(f"process_ack: {msg_id=} not in buffer, ignoring")

    async def on_reconnect(self) -> None:
        """Re-send all unacked messages in the send buffer (insertion order)."""
        count = len(self._send_buffer)
        if count == 0:
            log.log(5, "on_reconnect: buffer empty, nothing to replay")
            log.debug("on_reconnect: no unacked messages to re-send")
            return
        log.log(5, f"on_reconnect: replaying {count} unacked messages from buffer")
        log.info(f"on_reconnect: re-sending {count} unacked messages")
        replayed = 0
        for _msg_id, (msg, _priority) in self._send_buffer.items():
            data = serialize(msg)
            await self._connection.send_raw(data)
            replayed += 1
            msg_id = msg.msg_id
            msg_type = msg.type.value
            log.log(
                5,
                f"on_reconnect: replayed {replayed}/{count} — {msg_id=}, {msg_type=}, "
                f"priority={_priority.value}, wire_size={len(data)}",
            )
            log.debug(f"on_reconnect: re-sent {msg_id=}, {msg_type=}")
        log.info(f"on_reconnect: finished re-sending {count} messages")

    def _mark_seen(self, msg_id: str) -> None:
        """Record a msg_id as seen, evicting the oldest if at capacity."""
        if msg_id in self._seen_ids:
            return
        self._seen_ids[msg_id] = None
        evicted = 0
        while len(self._seen_ids) > self._dedup_max_size:
            self._seen_ids.popitem(last=False)
            evicted += 1
        if evicted > 0:
            log.debug(f"_mark_seen: {evicted=} old entries from dedup set")

    def _enforce_backpressure(self) -> None:
        """Evict oldest NORMAL messages if the buffer exceeds max size.

        CRITICAL messages are never evicted. If the buffer is entirely
        CRITICAL, no eviction occurs (buffer grows beyond limit).
        """
        while len(self._send_buffer) > self._buffer_size:
            evicted = False
            for msg_id, (_msg, priority) in self._send_buffer.items():
                if priority == MessagePriority.NORMAL:
                    self._send_buffer.pop(msg_id)
                    buf_size = len(self._send_buffer)
                    log.warning(f"backpressure: evicted NORMAL {msg_id=} ({buf_size=})")
                    evicted = True
                    break
            if not evicted:
                buf_size = len(self._send_buffer)
                limit = self._buffer_size
                log.warning(f"backpressure: {buf_size=} > {limit=} but all CRITICAL — cannot evict")
                break

    async def _send_ack(self, msg_id: str) -> None:
        """Send an ACK for the given msg_id."""
        ack = make_ack(msg_id)
        data = serialize(ack)
        await self._connection.send_raw(data)
        log.debug(f"_send_ack: sent ACK for {msg_id=}")
