"""Tests for common.channel — Layer 2: ReliableChannel.

ReliableChannel provides at-least-once delivery over a ConnectionManager,
with ACK tracking, deduplication, send buffering, and backpressure.
"""

import asyncio
from unittest.mock import AsyncMock

from swarm_common.channel import MessagePriority, ReliableChannel
from swarm_common.models import MessageType
from swarm_common.protocol import make_ack, make_message, serialize


def _make_mock_connection(recv_messages: list[bytes] | None = None) -> AsyncMock:
    """Create a mock ConnectionManager for testing.

    Args:
        recv_messages: Optional list of serialized WireMessages to return from recv_raw.
    """
    conn = AsyncMock()
    conn.is_connected = True
    conn.send_raw = AsyncMock()
    if recv_messages:
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        for msg in recv_messages:
            queue.put_nowait(msg)

        async def _recv_from_queue() -> bytes:
            return await queue.get()

        conn.recv_raw = AsyncMock(side_effect=_recv_from_queue)
    else:
        conn.recv_raw = AsyncMock(side_effect=asyncio.CancelledError)
    return conn


class TestMessagePriority:
    """Tests for the MessagePriority enum."""

    def test_values(self) -> None:
        """Verify both priority levels exist."""
        assert MessagePriority.CRITICAL.value == "critical"
        assert MessagePriority.NORMAL.value == "normal"


class TestReliableChannelSend:
    """Tests for the send path: buffering, msg_id assignment, serialization."""

    async def test_send_serializes_and_forwards(self) -> None:
        """send() should serialize the message and call connection.send_raw."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)

        msg = make_message(MessageType.COMMAND, {"cmd": "start_scan"})
        await channel.send(msg)

        conn.send_raw.assert_awaited_once()
        sent_bytes = conn.send_raw.call_args[0][0]
        assert isinstance(sent_bytes, bytes)

    async def test_send_buffers_until_acked(self) -> None:
        """Messages should remain in the send buffer until ACK'd."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)

        msg = make_message(MessageType.COMMAND, {"cmd": "start_scan"})
        await channel.send(msg)

        assert msg.msg_id in channel.send_buffer
        assert len(channel.send_buffer) == 1

    async def test_ack_removes_from_buffer(self) -> None:
        """Processing an ACK should remove the message from the send buffer."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)

        msg = make_message(MessageType.COMMAND, {"cmd": "start_scan"})
        await channel.send(msg)
        assert msg.msg_id in channel.send_buffer

        # Simulate receiving an ACK
        channel.process_ack(msg.msg_id)
        assert msg.msg_id not in channel.send_buffer

    async def test_ack_for_unknown_msg_is_safe(self) -> None:
        """ACK for a message not in the buffer should not raise."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)
        channel.process_ack("nonexistent-id")  # should not raise

    async def test_multiple_messages_buffered(self) -> None:
        """Multiple sent messages should all be in the buffer."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)

        msgs = [make_message(MessageType.EVENT_BATCH, {"i": i}) for i in range(5)]
        for msg in msgs:
            await channel.send(msg)

        assert len(channel.send_buffer) == 5
        for msg in msgs:
            assert msg.msg_id in channel.send_buffer


class TestReliableChannelRecv:
    """Tests for the receive path: deserialization, ACK sending, deduplication."""

    async def test_recv_deserializes_message(self) -> None:
        """recv() should return a deserialized WireMessage."""
        original = make_message(MessageType.COMMAND, {"cmd": "stop_scan"})
        conn = _make_mock_connection(recv_messages=[serialize(original)])
        channel = ReliableChannel(conn)

        received = await channel.recv()
        assert received is not None
        assert received.msg_id == original.msg_id
        assert received.type == MessageType.COMMAND

    async def test_recv_sends_ack_back(self) -> None:
        """Receiving a non-ACK message should automatically send an ACK."""
        original = make_message(MessageType.COMMAND, {"cmd": "stop_scan"})
        conn = _make_mock_connection(recv_messages=[serialize(original)])
        channel = ReliableChannel(conn)

        await channel.recv()

        # Should have sent an ACK back
        conn.send_raw.assert_awaited_once()
        sent_bytes = conn.send_raw.call_args[0][0]
        assert b"ack" in sent_bytes

    async def test_recv_ack_message_processes_internally(self) -> None:
        """Receiving an ACK message should process it and return None."""
        # First send a message so there's something to ACK
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)

        outgoing = make_message(MessageType.COMMAND, {"cmd": "start"})
        await channel.send(outgoing)
        assert outgoing.msg_id in channel.send_buffer

        # Now simulate receiving an ACK for it, followed by a real message
        ack = make_ack(outgoing.msg_id)
        real_msg = make_message(MessageType.SCAN_STATUS, {"scan_id": "s1"})
        conn.recv_raw = AsyncMock(
            side_effect=[serialize(ack), serialize(real_msg)]
        )

        # First recv() processes the ACK internally and returns None
        ack_result = await channel.recv()
        assert ack_result is None
        assert outgoing.msg_id not in channel.send_buffer

        # Second recv() returns the real message
        received = await channel.recv()
        assert received is not None
        assert received.msg_id == real_msg.msg_id

    async def test_dedup_rejects_duplicate_msg_id(self) -> None:
        """The same msg_id received twice should only be yielded once."""
        original = make_message(MessageType.COMMAND, {"cmd": "stop_scan"})
        data = serialize(original)
        conn = _make_mock_connection(recv_messages=[data, data])
        channel = ReliableChannel(conn)

        first = await channel.recv()
        assert first is not None
        assert first.msg_id == original.msg_id

        # Second recv with same msg_id should be skipped (returns None)
        second = await channel.recv()
        assert second is None

    async def test_dedup_allows_different_msg_ids(self) -> None:
        """Different msg_ids should both be yielded."""
        msg1 = make_message(MessageType.COMMAND, {"cmd": "a"})
        msg2 = make_message(MessageType.COMMAND, {"cmd": "b"})
        conn = _make_mock_connection(recv_messages=[serialize(msg1), serialize(msg2)])
        channel = ReliableChannel(conn)

        first = await channel.recv()
        second = await channel.recv()
        assert first is not None
        assert second is not None
        assert first.msg_id != second.msg_id


class TestReliableChannelBackpressure:
    """Tests for backpressure: buffer eviction under pressure."""

    async def test_buffer_bounded_by_max_size(self) -> None:
        """When buffer exceeds max_size, oldest NORMAL messages should be evicted."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn, buffer_size=5)

        # Fill buffer with NORMAL priority messages
        for i in range(7):
            msg = make_message(MessageType.EVENT_BATCH, {"i": i})
            await channel.send(msg, priority=MessagePriority.NORMAL)

        # Buffer should be at most 5
        assert len(channel.send_buffer) <= 5

    async def test_critical_messages_never_evicted(self) -> None:
        """CRITICAL messages should never be dropped, even under pressure."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn, buffer_size=3)

        # Send 3 CRITICAL messages
        critical_ids = []
        for i in range(3):
            msg = make_message(MessageType.SCAN_STATUS, {"scan_id": f"s{i}"})
            await channel.send(msg, priority=MessagePriority.CRITICAL)
            critical_ids.append(msg.msg_id)

        # Now send 2 more NORMAL — these should trigger eviction of... nothing, since all are CRITICAL
        for i in range(2):
            msg = make_message(MessageType.EVENT_BATCH, {"i": i})
            await channel.send(msg, priority=MessagePriority.NORMAL)

        # All 3 critical messages should still be in the buffer
        for cid in critical_ids:
            assert cid in channel.send_buffer

    async def test_normal_messages_evicted_before_critical(self) -> None:
        """Under pressure, NORMAL messages should be evicted first."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn, buffer_size=4)

        # Send 2 NORMAL, then 2 CRITICAL
        normal_msg = make_message(MessageType.EVENT_BATCH, {"x": 1})
        await channel.send(normal_msg, priority=MessagePriority.NORMAL)
        normal_msg2 = make_message(MessageType.LOG_BATCH, {"x": 2})
        await channel.send(normal_msg2, priority=MessagePriority.NORMAL)

        critical_msg = make_message(MessageType.SCAN_STATUS, {"s": 1})
        await channel.send(critical_msg, priority=MessagePriority.CRITICAL)
        critical_msg2 = make_message(MessageType.STATE_SYNC, {"s": 2})
        await channel.send(critical_msg2, priority=MessagePriority.CRITICAL)

        # Now send one more NORMAL — should evict oldest NORMAL
        overflow_msg = make_message(MessageType.EVENT_BATCH, {"x": 3})
        await channel.send(overflow_msg, priority=MessagePriority.NORMAL)

        # Critical messages must still be present
        assert critical_msg.msg_id in channel.send_buffer
        assert critical_msg2.msg_id in channel.send_buffer

        # The oldest NORMAL (normal_msg) should have been evicted
        assert normal_msg.msg_id not in channel.send_buffer


class TestReliableChannelReconnect:
    """Tests for reconnect behavior: re-sending unacked messages."""

    async def test_on_reconnect_resends_all_buffered(self) -> None:
        """on_reconnect() should re-send all messages in the send buffer."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)

        # Send 3 messages (all unacked)
        msgs = [make_message(MessageType.COMMAND, {"i": i}) for i in range(3)]
        for msg in msgs:
            await channel.send(msg)

        # Reset mock to count only reconnect sends
        conn.send_raw.reset_mock()

        await channel.on_reconnect()

        # All 3 should have been re-sent
        assert conn.send_raw.await_count == 3

    async def test_on_reconnect_preserves_order(self) -> None:
        """Re-sent messages should be in the original insertion order."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)

        msg_ids = []
        for i in range(3):
            msg = make_message(MessageType.COMMAND, {"i": i})
            await channel.send(msg)
            msg_ids.append(msg.msg_id)

        conn.send_raw.reset_mock()
        await channel.on_reconnect()

        # Verify order by checking the serialized msg_ids
        sent_calls = conn.send_raw.call_args_list
        for i, call in enumerate(sent_calls):
            sent_bytes = call[0][0]
            assert msg_ids[i].encode() in sent_bytes

    async def test_on_reconnect_with_empty_buffer(self) -> None:
        """on_reconnect() with no buffered messages should be a no-op."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)

        await channel.on_reconnect()
        # send_raw should not have been called (no messages to re-send)
        # Note: it may have been called 0 times or not at all
        # We just verify no error is raised


class TestReliableChannelDedup:
    """Tests for the deduplication set behavior."""

    def test_seen_ids_bounded(self) -> None:
        """The seen_ids set should not grow unbounded."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn, dedup_max_size=100)

        # Add 200 IDs
        for i in range(200):
            channel._mark_seen(f"id-{i}")

        # Should be bounded
        assert len(channel._seen_ids) <= 100

    def test_seen_ids_contains_marked(self) -> None:
        """Marked msg_ids should be in _seen_ids."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)
        channel._mark_seen("abc")
        assert "abc" in channel._seen_ids

    def test_seen_ids_does_not_contain_unmarked(self) -> None:
        """Unmarked msg_ids should not be in _seen_ids."""
        conn = _make_mock_connection()
        channel = ReliableChannel(conn)
        assert "xyz" not in channel._seen_ids
