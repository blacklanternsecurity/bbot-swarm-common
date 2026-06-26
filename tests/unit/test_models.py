"""Tests for common.models — all Pydantic models, enums, and wire format."""

from time import time

import pytest
from pydantic import ValidationError

from swarm_common.models import (
    BeeStatus,
    EventBatchPayload,
    LogBatchPayload,
    MessageType,
    ScanInfo,
    ScanStatus,
    ScanStatusPayload,
    StartScanPayload,
    StateSyncPayload,
    StopScanPayload,
    WireMessage,
    scan_status_code,
    scan_status_name,
)

# ---------------------------------------------------------------------------
# ScanStatus enum
# ---------------------------------------------------------------------------


class TestScanStatus:
    """Tests for ScanStatus enum and status code conversion helpers."""

    def test_all_statuses_present(self) -> None:
        expected = {
            "QUEUED",
            "NOT_STARTED",
            "STARTING",
            "RUNNING",
            "FINISHING",
            "ABORTING",
            "FINISHED",
            "FAILED",
            "ABORTED",
        }
        assert {s.value for s in ScanStatus} == expected

    def test_status_is_str_enum(self) -> None:
        assert isinstance(ScanStatus.QUEUED, str)
        assert ScanStatus.QUEUED == "QUEUED"

    def test_status_codes_monotonically_increase(self) -> None:
        """Verify ordering matches bbot.constants: QUEUED=0 through ABORTED=8."""
        ordered = [
            ScanStatus.QUEUED,
            ScanStatus.NOT_STARTED,
            ScanStatus.STARTING,
            ScanStatus.RUNNING,
            ScanStatus.FINISHING,
            ScanStatus.ABORTING,
            ScanStatus.FINISHED,
            ScanStatus.FAILED,
            ScanStatus.ABORTED,
        ]
        codes = [scan_status_code(s) for s in ordered]
        assert codes == sorted(codes)

    def test_scan_status_code_roundtrip(self) -> None:
        """Converting status -> code -> name should roundtrip correctly."""
        for status in ScanStatus:
            code = scan_status_code(status)
            assert isinstance(code, int)
            name = scan_status_name(code)
            assert name == status.value

    def test_is_terminal_positive(self) -> None:
        """FINISHED, FAILED, ABORTED are the terminal states."""
        assert ScanStatus.FINISHED.is_terminal
        assert ScanStatus.FAILED.is_terminal
        assert ScanStatus.ABORTED.is_terminal

    def test_is_terminal_negative(self) -> None:
        """In-flight states must NOT report as terminal."""
        for status in (
            ScanStatus.QUEUED,
            ScanStatus.NOT_STARTED,
            ScanStatus.STARTING,
            ScanStatus.RUNNING,
            ScanStatus.FINISHING,
            ScanStatus.ABORTING,
        ):
            assert not status.is_terminal, f"{status.value} must not be terminal"


# ---------------------------------------------------------------------------
# BeeStatus enum
# ---------------------------------------------------------------------------


class TestBeeStatus:
    """Tests for BeeStatus enum."""

    def test_all_statuses_present(self) -> None:
        assert {s.value for s in BeeStatus} == {"ONLINE", "DEGRADED", "OFFLINE"}

    def test_is_str_enum(self) -> None:
        assert isinstance(BeeStatus.ONLINE, str)


# ---------------------------------------------------------------------------
# MessageType enum
# ---------------------------------------------------------------------------


class TestMessageType:
    """Tests for MessageType enum."""

    def test_all_types_present(self) -> None:
        expected = {"command", "cmd_result", "event_batch", "log_batch", "scan_status", "state_sync", "ack"}
        assert {t.value for t in MessageType} == expected


# ---------------------------------------------------------------------------
# WireMessage
# ---------------------------------------------------------------------------


class TestWireMessage:
    """Tests for the WireMessage envelope model."""

    def test_create_minimal(self) -> None:
        msg = WireMessage(
            msg_id="abc-123",
            type=MessageType.COMMAND,
            timestamp=1234567890.0,
            payload={"cmd": "start_scan"},
        )
        assert msg.msg_id == "abc-123"
        assert msg.type == MessageType.COMMAND
        assert msg.timestamp == 1234567890.0
        assert msg.payload == {"cmd": "start_scan"}
        assert msg.reply_to is None

    def test_create_with_reply_to(self) -> None:
        msg = WireMessage(
            msg_id="abc-123",
            type=MessageType.ACK,
            timestamp=1234567890.0,
            payload={},
            reply_to="orig-456",
        )
        assert msg.reply_to == "orig-456"

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WireMessage(
                msg_id="abc",
                type="not_a_real_type",  # type: ignore[arg-type]
                timestamp=1234567890.0,
                payload={},
            )

    def test_serialization_roundtrip(self) -> None:
        msg = WireMessage(
            msg_id="test-id",
            type=MessageType.EVENT_BATCH,
            timestamp=1711843200.0,
            payload={"scan_id": "scan-1", "events": [{"type": "DNS_NAME"}]},
        )
        data = msg.model_dump()
        restored = WireMessage(**data)
        assert restored == msg

    def test_json_roundtrip(self) -> None:
        msg = WireMessage(
            msg_id="test-id",
            type=MessageType.SCAN_STATUS,
            timestamp=1711843200.0,
            payload={"scan_id": "s1", "status": "RUNNING"},
            reply_to="r1",
        )
        json_str = msg.model_dump_json()
        restored = WireMessage.model_validate_json(json_str)
        assert restored == msg


# ---------------------------------------------------------------------------
# Command payloads
# ---------------------------------------------------------------------------


class TestStartScanPayload:
    """Tests for StartScanPayload command model."""

    def test_create_full(self) -> None:
        p = StartScanPayload(
            scan_id="scan-001",
            preset={"target": ["example.com"], "modules": ["httpx"]},
            name="My Scan",
        )
        assert p.scan_id == "scan-001"
        assert p.preset["target"] == ["example.com"]
        assert p.name == "My Scan"

    def test_name_optional(self) -> None:
        p = StartScanPayload(scan_id="s1", preset={})
        assert p.name is None

    def test_missing_scan_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StartScanPayload(preset={})  # type: ignore[call-arg]

    def test_missing_preset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StartScanPayload(scan_id="s1")  # type: ignore[call-arg]


class TestStopScanPayload:
    """Tests for StopScanPayload command model."""

    def test_create(self) -> None:
        p = StopScanPayload(scan_id="scan-001")
        assert p.scan_id == "scan-001"
        assert p.force is False

    def test_force_flag(self) -> None:
        p = StopScanPayload(scan_id="scan-001", force=True)
        assert p.force is True


# ---------------------------------------------------------------------------
# Drone -> Hive payloads
# ---------------------------------------------------------------------------


class TestScanStatusPayload:
    """Tests for ScanStatusPayload report model."""

    def test_create(self) -> None:
        p = ScanStatusPayload(
            scan_id="scan-001",
            status=ScanStatus.RUNNING,
            status_code=3,
        )
        assert p.scan_id == "scan-001"
        assert p.status == ScanStatus.RUNNING
        assert p.status_code == 3
        assert p.detail is None

    def test_with_detail(self) -> None:
        p = ScanStatusPayload(
            scan_id="s1",
            status=ScanStatus.FAILED,
            status_code=7,
            detail="container exited with code 1",
        )
        assert p.detail == "container exited with code 1"

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScanStatusPayload(
                scan_id="s1",
                status="INVALID",  # type: ignore[arg-type]
                status_code=99,
            )


class TestEventBatchPayload:
    """Tests for EventBatchPayload report model."""

    def test_create(self) -> None:
        events = [{"type": "DNS_NAME", "data": "example.com"}, {"type": "IP_ADDRESS", "data": "1.2.3.4"}]
        p = EventBatchPayload(scan_id="s1", events=events)
        assert p.scan_id == "s1"
        assert len(p.events) == 2

    def test_empty_events(self) -> None:
        p = EventBatchPayload(scan_id="s1", events=[])
        assert p.events == []


class TestLogBatchPayload:
    """Tests for LogBatchPayload report model."""

    def test_create(self) -> None:
        p = LogBatchPayload(scan_id="s1", lines=["line1", "line2"])
        assert len(p.lines) == 2

    def test_empty_lines(self) -> None:
        p = LogBatchPayload(scan_id="s1", lines=[])
        assert p.lines == []


class TestScanInfo:
    """Tests for ScanInfo summary model used in state_sync."""

    def test_create_minimal(self) -> None:
        info = ScanInfo(scan_id="s1", status=ScanStatus.RUNNING)
        assert info.events_sent == 0
        assert info.started_at is None

    def test_create_full(self) -> None:
        now = time()
        info = ScanInfo(
            scan_id="s1",
            status=ScanStatus.FINISHED,
            events_sent=4523,
            started_at=now,
        )
        assert info.events_sent == 4523
        assert info.started_at == now


class TestStateSyncPayload:
    """Tests for StateSyncPayload — full drone state on connect/reconnect."""

    def test_create(self) -> None:
        payload = StateSyncPayload(
            bee_id="drone-001",
            status=BeeStatus.ONLINE,
            active_scans={
                "scan-1": ScanInfo(scan_id="scan-1", status=ScanStatus.RUNNING, events_sent=100),
            },
            capacity={"max_scans": 3, "available": 2},
        )
        assert payload.bee_id == "drone-001"
        assert payload.status == BeeStatus.ONLINE
        assert "scan-1" in payload.active_scans
        assert payload.capacity["max_scans"] == 3

    def test_empty_scans(self) -> None:
        payload = StateSyncPayload(
            bee_id="d1",
            status=BeeStatus.ONLINE,
            active_scans={},
            capacity={"max_scans": 3, "available": 3},
        )
        assert len(payload.active_scans) == 0

    def test_serialization_roundtrip(self) -> None:
        payload = StateSyncPayload(
            bee_id="d1",
            status=BeeStatus.DEGRADED,
            active_scans={
                "s1": ScanInfo(scan_id="s1", status=ScanStatus.FINISHING, events_sent=50),
            },
            capacity={"max_scans": 2, "available": 1},
        )
        data = payload.model_dump()
        restored = StateSyncPayload(**data)
        assert restored.bee_id == payload.bee_id
        assert restored.active_scans["s1"].events_sent == 50
