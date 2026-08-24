"""Pydantic models, enums, and wire format for the BBOT Swarm protocol."""

from __future__ import annotations

from enum import StrEnum
from logging import getLogger
from typing import Any

from bbot.constants import get_scan_status_code, get_scan_status_name
from pydantic import BaseModel

log = getLogger(__name__)


class ScanStatus(StrEnum):
    """Scan lifecycle statuses, mirroring bbot.constants."""

    QUEUED = "QUEUED"
    NOT_STARTED = "NOT_STARTED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    FINISHING = "FINISHING"
    ABORTING = "ABORTING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"

    @property
    def is_terminal(self) -> bool:
        """`True` iff the scan has reached `FINISHED`, `FAILED`, or `ABORTED`."""
        return self in _TERMINAL_SCAN_STATUSES


_TERMINAL_SCAN_STATUSES: frozenset[ScanStatus] = frozenset(
    {ScanStatus.FINISHED, ScanStatus.FAILED, ScanStatus.ABORTED}
)


class BeeStatus(StrEnum):
    """Drone connection/health statuses."""

    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class MessageType(StrEnum):
    """Wire protocol message types."""

    COMMAND = "command"
    CMD_RESULT = "cmd_result"
    EVENT_BATCH = "event_batch"
    LOG_BATCH = "log_batch"
    SCAN_STATUS = "scan_status"
    STATE_SYNC = "state_sync"
    ACK = "ack"


def scan_status_code(status: ScanStatus | str) -> int:
    """Convert a `ScanStatus` to its integer code via `bbot.constants`."""
    log.debug(f"{status=}")
    value = status.value if isinstance(status, ScanStatus) else status
    return int(get_scan_status_code(value))


def scan_status_name(code: int) -> str:
    """Convert an integer status code to its string name via `bbot.constants`."""
    log.debug(f"{code=}")
    return str(get_scan_status_name(code))


class WireMessage(BaseModel):
    """Envelope for all messages on the WebSocket wire protocol."""

    msg_id: str
    type: MessageType
    timestamp: float
    payload: dict[str, Any]
    reply_to: str | None = None


class StartScanPayload(BaseModel):
    """Payload for `start_scan` commands."""

    scan_id: str
    preset: dict[str, Any]
    name: str | None = None


class StopScanPayload(BaseModel):
    """Payload for `stop_scan` commands."""

    scan_id: str
    force: bool = False


class ScanStatusPayload(BaseModel):
    """Payload for scan status updates from drone."""

    scan_id: str
    status: ScanStatus
    status_code: int
    detail: str | None = None


class EventBatchPayload(BaseModel):
    """Payload for batched scan events from drone."""

    scan_id: str
    events: list[dict[str, Any]]


class LogBatchPayload(BaseModel):
    """Payload for batched log lines from drone."""

    scan_id: str
    lines: list[str]


class ScanInfo(BaseModel):
    """Summary of a single scan, used in state_sync."""

    scan_id: str
    status: ScanStatus
    events_sent: int = 0
    started_at: float | None = None


class StateSyncPayload(BaseModel):
    """Full drone state, sent on connect/reconnect."""

    bee_id: str
    status: BeeStatus
    active_scans: dict[str, ScanInfo]
    capacity: dict[str, int]
    # Per-process nonce; a change tells the hive the queen restarted (scans lost)
    # rather than merely reconnected. None from a pre-upgrade bee.
    boot_id: str | None = None
