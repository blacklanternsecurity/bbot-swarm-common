"""Ed25519 wrapper around `cryptography.hazmat` — the rest of the codebase never imports hazmat directly."""

from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as BinAsciiError
from dataclasses import dataclass
from logging import getLogger

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

log = getLogger(__name__)


class CryptoError(Exception):
    """Base exception for all cryptographic operation failures."""


class InvalidSignatureError(CryptoError):
    """Raised when signature verification fails."""


# Fail fast if OpenSSL is in FIPS mode (Ed25519 isn't a FIPS-approved algorithm).
if not backend.ed25519_supported():
    msg = "Ed25519 is not available — OpenSSL is running in FIPS mode"
    raise CryptoError(msg)

log.debug("crypto: Ed25519 available (OpenSSL FIPS mode not active)")


@dataclass(frozen=True)
class Ed25519KeyPair:
    """An Ed25519 key pair (private + derived public key)."""

    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey


def generate_keypair() -> Ed25519KeyPair:
    """Generate a new Ed25519 key pair.

    Returns:
        An `Ed25519KeyPair` with a fresh private key and its derived public key.

    Raises:
        CryptoError: If OpenSSL key generation fails.
    """
    log.debug("generating new Ed25519 key pair")
    try:
        private_key = Ed25519PrivateKey.generate()
    except Exception as exc:
        log.error(f"key generation failed: {type(exc).__name__}: {exc}")
        raise CryptoError(f"Ed25519 key generation failed: {exc}") from exc
    public_key = private_key.public_key()
    log.debug("key pair generated successfully")
    return Ed25519KeyPair(private_key=private_key, public_key=public_key)


def sign(private_key: Ed25519PrivateKey, data: bytes) -> bytes:
    """Sign data with an Ed25519 private key.

    Args:
        private_key: The signing key.
        data: The message to sign — any length, including empty.

    Returns:
        A 64-byte Ed25519 signature.

    Raises:
        CryptoError: If `data` is not `bytes` or signing fails.
    """
    if not isinstance(data, bytes):
        raise CryptoError(f"sign: data must be bytes, got {type(data).__name__}")
    try:
        signature = private_key.sign(data)
    except TypeError as exc:
        raise CryptoError(f"sign: invalid input type: {exc}") from exc
    except Exception as exc:
        log.error(f"signing failed: {type(exc).__name__}: {exc}")
        raise CryptoError(f"Ed25519 signing failed: {exc}") from exc
    log.debug(f"produced {len(signature)}-byte signature")
    return signature


def verify(public_key: Ed25519PublicKey, data: bytes, signature: bytes) -> None:
    """Verify an Ed25519 signature.

    Args:
        public_key: The verification key.
        data: The original signed message.
        signature: The 64-byte signature to verify.

    Raises:
        InvalidSignatureError: If the signature is invalid — wrong data, wrong key,
            wrong length, corrupted, etc.
        CryptoError: If inputs are not `bytes`.
    """
    if not isinstance(data, bytes):
        raise CryptoError(f"verify: data must be bytes, got {type(data).__name__}")
    if not isinstance(signature, bytes):
        raise CryptoError(f"verify: signature must be bytes, got {type(signature).__name__}")
    try:
        public_key.verify(signature, data)
    except InvalidSignature as exc:
        raise InvalidSignatureError("Ed25519 signature verification failed") from exc
    except TypeError as exc:
        raise CryptoError(f"verify: invalid input type: {exc}") from exc
    log.debug("signature valid")


def serialize_public_key(public_key: Ed25519PublicKey) -> str:
    """Serialize an Ed25519 public key to a base64 string.

    Args:
        public_key: The key to serialize.

    Returns:
        Base64-encoded 44-char string (32 raw bytes).
    """
    raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    encoded = b64encode(raw).decode()
    log.debug(f"{len(raw)} bytes -> {len(encoded)} chars b64")
    return encoded


def serialize_private_key(private_key: Ed25519PrivateKey) -> str:
    """Serialize an Ed25519 private key to a base64 string.

    Args:
        private_key: The key to serialize.

    Returns:
        Base64-encoded 44-char string (32 raw bytes).
    """
    raw = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    encoded = b64encode(raw).decode()
    log.debug(f"{len(raw)} bytes -> {len(encoded)} chars b64")
    return encoded


def deserialize_public_key(data: str) -> Ed25519PublicKey:
    """Deserialize an Ed25519 public key from a base64 string.

    Args:
        data: Base64-encoded public key (44 chars / 32 raw bytes).

    Returns:
        An `Ed25519PublicKey` instance.

    Raises:
        CryptoError: If `data` is not `str`, not valid base64, or not a valid
            32-byte Ed25519 public key.
    """
    if not isinstance(data, str):
        raise CryptoError(f"deserialize_public_key: data must be str, got {type(data).__name__}")
    try:
        raw = b64decode(data)
    except (BinAsciiError, ValueError) as exc:
        raise CryptoError(f"deserialize_public_key: invalid base64: {exc}") from exc
    try:
        key = Ed25519PublicKey.from_public_bytes(raw)
    except (ValueError, UnsupportedAlgorithm) as exc:
        raise CryptoError(f"Invalid Ed25519 public key: {exc}") from exc
    log.debug("key loaded")
    return key


def deserialize_private_key(data: str) -> Ed25519PrivateKey:
    """Deserialize an Ed25519 private key from a base64 string.

    Args:
        data: Base64-encoded private key (44 chars / 32 raw bytes).

    Returns:
        An `Ed25519PrivateKey` instance.

    Raises:
        CryptoError: If `data` is not `str`, not valid base64, or not a valid
            32-byte Ed25519 private key.
    """
    if not isinstance(data, str):
        raise CryptoError(f"deserialize_private_key: data must be str, got {type(data).__name__}")
    try:
        raw = b64decode(data)
    except (BinAsciiError, ValueError) as exc:
        raise CryptoError(f"deserialize_private_key: invalid base64: {exc}") from exc
    try:
        key = Ed25519PrivateKey.from_private_bytes(raw)
    except (ValueError, UnsupportedAlgorithm) as exc:
        raise CryptoError(f"Invalid Ed25519 private key: {exc}") from exc
    log.debug("key loaded")
    return key
