"""Tests for common.protocol — wire format serialization and helpers."""

from time import time

import orjson
import pytest
from pydantic import ValidationError

from swarm_common.models import MessageType, WireMessage
from swarm_common.protocol import deserialize, make_ack, make_message, serialize


class TestMakeMessage:
    """Tests for the make_message factory function."""

    def test_creates_wire_message_with_uuid(self) -> None:
        """Verify make_message returns a WireMessage with a UUID msg_id."""
        msg = make_message(MessageType.COMMAND, {"cmd": "start_scan"})
        assert isinstance(msg, WireMessage)
        assert msg.type == MessageType.COMMAND
        assert msg.payload == {"cmd": "start_scan"}
        assert msg.reply_to is None
        assert isinstance(msg.msg_id, str)
        assert len(msg.msg_id) > 0

    def test_unique_msg_ids(self) -> None:
        """Every call should produce a unique msg_id."""
        msg1 = make_message(MessageType.COMMAND, {"a": 1})
        msg2 = make_message(MessageType.COMMAND, {"a": 1})
        assert msg1.msg_id != msg2.msg_id

    def test_timestamp_is_recent(self) -> None:
        """Timestamp should be between before and after the call."""
        before = time()
        msg = make_message(MessageType.EVENT_BATCH, {})
        after = time()
        assert before <= msg.timestamp <= after

    def test_with_reply_to(self) -> None:
        """Verify reply_to is preserved when provided."""
        msg = make_message(MessageType.ACK, {}, reply_to="orig-123")
        assert msg.reply_to == "orig-123"


class TestMakeAck:
    """Tests for the make_ack helper."""

    def test_creates_ack_message(self) -> None:
        """Verify ACK message has correct type and payload."""
        ack = make_ack("target-msg-id")
        assert ack.type == MessageType.ACK
        assert ack.payload == {"ack_id": "target-msg-id"}
        assert isinstance(ack.msg_id, str)
        assert len(ack.msg_id) > 0

    def test_ack_has_no_reply_to(self) -> None:
        """ACK messages should not have reply_to set."""
        ack = make_ack("x")
        assert ack.reply_to is None


class TestSerializeDeserialize:
    """Tests for serialize/deserialize roundtrip and error handling."""

    def test_roundtrip(self) -> None:
        """Basic serialize -> deserialize should produce an equal message."""
        original = make_message(
            MessageType.SCAN_STATUS,
            {"scan_id": "s1", "status": "RUNNING", "status_code": 3},
        )
        data = serialize(original)
        assert isinstance(data, bytes)
        restored = deserialize(data)
        assert restored == original

    def test_roundtrip_with_nested_payload(self) -> None:
        """Nested dicts and lists in payload should survive the roundtrip."""
        original = make_message(
            MessageType.EVENT_BATCH,
            {
                "scan_id": "s1",
                "events": [
                    {"type": "DNS_NAME", "data": "example.com", "tags": ["in-scope"]},
                    {"type": "IP_ADDRESS", "data": "1.2.3.4"},
                ],
            },
        )
        data = serialize(original)
        restored = deserialize(data)
        assert restored.payload["events"][0]["data"] == "example.com"
        assert len(restored.payload["events"]) == 2

    def test_roundtrip_with_reply_to(self) -> None:
        """reply_to field should survive the roundtrip."""
        original = make_message(MessageType.ACK, {"ack_id": "abc"}, reply_to="abc")
        data = serialize(original)
        restored = deserialize(data)
        assert restored.reply_to == "abc"

    def test_serialize_returns_bytes(self) -> None:
        """serialize() must return bytes."""
        msg = make_message(MessageType.COMMAND, {})
        result = serialize(msg)
        assert isinstance(result, bytes)

    def test_deserialize_invalid_bytes_raises(self) -> None:
        """Invalid JSON bytes should raise orjson.JSONDecodeError."""
        with pytest.raises(orjson.JSONDecodeError):
            deserialize(b"not valid json at all {{{")

    def test_deserialize_valid_json_but_wrong_schema_raises(self) -> None:
        """Valid JSON that doesn't match WireMessage schema should raise ValidationError."""
        with pytest.raises(ValidationError):
            deserialize(b'{"foo": "bar"}')

    def test_roundtrip_empty_payload(self) -> None:
        """Empty payload should roundtrip correctly."""
        original = make_message(MessageType.STATE_SYNC, {})
        data = serialize(original)
        restored = deserialize(data)
        assert restored.payload == {}

    def test_large_payload(self) -> None:
        """Large payloads (1000 events) should serialize/deserialize correctly."""
        events = [{"type": "DNS_NAME", "data": f"host-{i}.example.com"} for i in range(1000)]
        original = make_message(MessageType.EVENT_BATCH, {"scan_id": "s1", "events": events})
        data = serialize(original)
        restored = deserialize(data)
        assert len(restored.payload["events"]) == 1000
