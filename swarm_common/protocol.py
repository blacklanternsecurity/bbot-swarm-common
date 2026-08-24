"""Wire protocol serialization and message helpers for BBOT Swarm."""

from __future__ import annotations

from logging import getLogger
from time import time
from typing import Any
from uuid import uuid4

from orjson import dumps, loads

from swarm_common.models import MessageType, WireMessage

log = getLogger(__name__)


def make_message(
    msg_type: MessageType,
    payload: dict[str, Any],
    reply_to: str | None = None,
) -> WireMessage:
    """Create a `WireMessage` with an auto-generated `msg_id` and current timestamp."""
    msg_id = str(uuid4())
    ts = time()
    msg = WireMessage(
        msg_id=msg_id,
        type=msg_type,
        timestamp=ts,
        payload=payload,
        reply_to=reply_to,
    )
    log.debug(f"{msg_id=}, type={msg_type.value}, {reply_to=}")
    return msg


def make_ack(msg_id: str) -> WireMessage:
    """Create an ACK message for `msg_id`."""
    log.debug(f"creating ACK for {msg_id=}")
    return make_message(MessageType.ACK, {"ack_id": msg_id})


def serialize(msg: WireMessage) -> bytes:
    """Serialize a `WireMessage` to bytes using `orjson`."""
    data = dumps(msg.model_dump())
    size = len(data)
    payload_size = len(str(msg.payload))
    log.log(5, f"type={msg.type.value}, msg_id={msg.msg_id}, payload_size={payload_size}, wire_size={size}")
    log.debug(f"msg_id={msg.msg_id}, type={msg.type.value}, {size=} bytes")
    return data


def deserialize(data: bytes) -> WireMessage:
    """Deserialize bytes into a `WireMessage`."""
    parsed = loads(data)
    msg = WireMessage(**parsed)
    size = len(data)
    log.log(
        5,
        f"wire_size={size}, type={msg.type.value}, msg_id={msg.msg_id}, "
        f"payload_keys={list(msg.payload.keys())}",
    )
    log.debug(f"msg_id={msg.msg_id}, type={msg.type.value}, {size=} bytes")
    return msg
