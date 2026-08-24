"""Tests for swarm_common.crypto — Ed25519 wrapper behavior."""

from __future__ import annotations

from base64 import b64encode
from dataclasses import FrozenInstanceError

import pytest

from swarm_common.crypto import (
    CryptoError,
    Ed25519KeyPair,
    InvalidSignatureError,
    deserialize_private_key,
    deserialize_public_key,
    generate_keypair,
    serialize_private_key,
    serialize_public_key,
    sign,
    verify,
)


class TestGenerateKeypair:
    """Tests for generate_keypair wrapper."""

    def test_returns_keypair_dataclass(self) -> None:
        """Should return an Ed25519KeyPair with both keys populated."""
        kp = generate_keypair()
        assert isinstance(kp, Ed25519KeyPair)
        assert kp.private_key is not None
        assert kp.public_key is not None

    def test_keypair_is_frozen(self) -> None:
        """KeyPair should be immutable — assignment must raise."""
        kp = generate_keypair()
        with pytest.raises(FrozenInstanceError):
            # intentional frozen-field assignment to test immutability
            kp.private_key = None  # type: ignore[misc]

    def test_each_call_produces_unique_keys(self) -> None:
        """Two calls should never produce the same public key."""
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        assert serialize_public_key(kp1.public_key) != serialize_public_key(kp2.public_key)
        assert serialize_private_key(kp1.private_key) != serialize_private_key(kp2.private_key)

    def test_public_key_derives_from_private_key(self) -> None:
        """The keypair's public key should match the one derived from its private key."""
        kp = generate_keypair()
        # Deserialize the private key independently and derive a public key
        raw_priv = serialize_private_key(kp.private_key)
        restored_priv = deserialize_private_key(raw_priv)
        derived_pub = serialize_public_key(restored_priv.public_key())
        original_pub = serialize_public_key(kp.public_key)
        assert derived_pub == original_pub


class TestInputValidation:
    """Type validation in sign/verify/deserialize — wrapper rejects bad types before FFI."""

    def test_sign_rejects_str(self) -> None:
        """sign() must reject str data."""
        kp = generate_keypair()
        with pytest.raises(CryptoError, match="must be bytes"):
            # intentional wrong type to test rejection
            sign(kp.private_key, "string data")  # type: ignore[arg-type]

    def test_sign_rejects_int(self) -> None:
        """sign() must reject int data."""
        kp = generate_keypair()
        with pytest.raises(CryptoError, match="must be bytes"):
            # intentional wrong type to test rejection
            sign(kp.private_key, 42)  # type: ignore[arg-type]

    def test_sign_rejects_none(self) -> None:
        """sign() must reject None data."""
        kp = generate_keypair()
        with pytest.raises(CryptoError, match="must be bytes"):
            # intentional wrong type to test rejection
            sign(kp.private_key, None)  # type: ignore[arg-type]

    def test_sign_rejects_bytearray(self) -> None:
        """sign() must reject bytearray (not concurrency-safe at FFI boundary)."""
        kp = generate_keypair()
        with pytest.raises(CryptoError, match="must be bytes"):
            # intentional wrong type to test rejection
            sign(kp.private_key, bytearray(b"hello"))  # type: ignore[arg-type]

    def test_verify_rejects_str_data(self) -> None:
        """verify() must reject str data."""
        kp = generate_keypair()
        sig = sign(kp.private_key, b"hello")
        with pytest.raises(CryptoError, match="data must be bytes"):
            # intentional wrong type to test rejection
            verify(kp.public_key, "string", sig)  # type: ignore[arg-type]

    def test_verify_rejects_str_signature(self) -> None:
        """verify() must reject str signature."""
        kp = generate_keypair()
        with pytest.raises(CryptoError, match="signature must be bytes"):
            # intentional wrong type to test rejection
            verify(kp.public_key, b"hello", "not bytes")  # type: ignore[arg-type]

    def test_verify_rejects_bytearray_data(self) -> None:
        """verify() must reject bytearray data."""
        kp = generate_keypair()
        sig = sign(kp.private_key, b"hello")
        with pytest.raises(CryptoError, match="data must be bytes"):
            # intentional wrong type to test rejection
            verify(kp.public_key, bytearray(b"hello"), sig)  # type: ignore[arg-type]

    def test_verify_rejects_bytearray_signature(self) -> None:
        """verify() must reject bytearray signature."""
        kp = generate_keypair()
        with pytest.raises(CryptoError, match="signature must be bytes"):
            # intentional wrong type to test rejection
            verify(kp.public_key, b"hello", bytearray(b"\x00" * 64))  # type: ignore[arg-type]

    def test_deserialize_public_key_rejects_bytes(self) -> None:
        """deserialize_public_key must reject bytes (expects str)."""
        with pytest.raises(CryptoError, match="must be str"):
            # intentional wrong type to test rejection
            deserialize_public_key(b"\x00" * 32)  # type: ignore[arg-type]

    def test_deserialize_private_key_rejects_bytes(self) -> None:
        """deserialize_private_key must reject bytes (expects str)."""
        with pytest.raises(CryptoError, match="must be str"):
            # intentional wrong type to test rejection
            deserialize_private_key(b"\x00" * 32)  # type: ignore[arg-type]

    def test_deserialize_public_key_rejects_int(self) -> None:
        """deserialize_public_key must reject int."""
        with pytest.raises(CryptoError, match="must be str"):
            # intentional wrong type to test rejection
            deserialize_public_key(42)  # type: ignore[arg-type]

    def test_deserialize_private_key_rejects_int(self) -> None:
        """deserialize_private_key must reject int."""
        with pytest.raises(CryptoError, match="must be str"):
            # intentional wrong type to test rejection
            deserialize_private_key(42)  # type: ignore[arg-type]

    def test_deserialize_public_key_rejects_invalid_base64(self) -> None:
        """deserialize_public_key must reject invalid base64."""
        with pytest.raises(CryptoError, match="invalid base64"):
            deserialize_public_key("not!valid!base64!!!")

    def test_deserialize_private_key_rejects_invalid_base64(self) -> None:
        """deserialize_private_key must reject invalid base64."""
        with pytest.raises(CryptoError, match="invalid base64"):
            deserialize_private_key("not!valid!base64!!!")


class TestExceptionWrapping:
    """Library exceptions must be wrapped — callers only see CryptoError or InvalidSignatureError."""

    def test_invalid_signature_wraps_to_our_exception(self) -> None:
        """Wrong signature must raise InvalidSignatureError, not cryptography's InvalidSignature."""
        kp = generate_keypair()
        sig = sign(kp.private_key, b"hello")
        with pytest.raises(InvalidSignatureError):
            verify(kp.public_key, b"wrong message", sig)

    def test_wrong_length_signature_wraps_to_invalid_signature(self) -> None:
        """Malformed signature (wrong length) must raise InvalidSignatureError."""
        kp = generate_keypair()
        for length in (0, 1, 32, 63, 65, 128):
            with pytest.raises(InvalidSignatureError):
                verify(kp.public_key, b"hello", b"\x00" * length)

    def test_wrong_key_length_wraps_to_crypto_error(self) -> None:
        """Wrong-length key data must raise CryptoError, not ValueError."""
        for length in (0, 1, 16, 31, 33, 64):
            bad_b64 = b64encode(b"\x00" * length).decode()
            with pytest.raises(CryptoError):
                deserialize_public_key(bad_b64)
            with pytest.raises(CryptoError):
                deserialize_private_key(bad_b64)

    def test_invalid_signature_is_subclass_of_crypto_error(self) -> None:
        """InvalidSignatureError must be catchable as CryptoError."""
        assert issubclass(InvalidSignatureError, CryptoError)
        kp = generate_keypair()
        with pytest.raises(CryptoError):
            verify(kp.public_key, b"hello", b"\x00" * 64)

    def test_crypto_error_is_base_exception(self) -> None:
        """CryptoError must be a plain Exception subclass."""
        assert issubclass(CryptoError, Exception)
        assert not issubclass(CryptoError, BaseException.__subclasses__()[0]
                              if BaseException.__subclasses__() else type)


class TestSerializationRoundTrip:
    """Tests for key serialization/deserialization wrapper correctness."""

    def test_public_key_round_trip_verifies(self) -> None:
        """A deserialized public key must verify signatures from the original private key."""
        kp = generate_keypair()
        raw = serialize_public_key(kp.public_key)
        restored = deserialize_public_key(raw)
        sig = sign(kp.private_key, b"round-trip test")
        verify(restored, b"round-trip test", sig)

    def test_private_key_round_trip_signs_identically(self) -> None:
        """A deserialized private key must produce identical signatures (Ed25519 is deterministic)."""
        kp = generate_keypair()
        raw = serialize_private_key(kp.private_key)
        restored = deserialize_private_key(raw)
        sig1 = sign(kp.private_key, b"round-trip test")
        sig2 = sign(restored, b"round-trip test")
        assert sig1 == sig2

    def test_public_key_serializes_to_b64_string(self) -> None:
        """Serialized public key must be a 44-char base64 string (32 raw bytes)."""
        kp = generate_keypair()
        result = serialize_public_key(kp.public_key)
        assert isinstance(result, str)
        assert len(result) == 44  # b64 of 32 bytes = 44 chars

    def test_private_key_serializes_to_b64_string(self) -> None:
        """Serialized private key must be a 44-char base64 string (32 raw bytes)."""
        kp = generate_keypair()
        result = serialize_private_key(kp.private_key)
        assert isinstance(result, str)
        assert len(result) == 44  # b64 of 32 bytes = 44 chars

    def test_cross_key_rejection(self) -> None:
        """A signature from key A must not verify with the deserialized public key of key B."""
        kp_a = generate_keypair()
        kp_b = generate_keypair()
        sig = sign(kp_a.private_key, b"cross key test")
        restored_b = deserialize_public_key(serialize_public_key(kp_b.public_key))
        with pytest.raises(InvalidSignatureError):
            verify(restored_b, b"cross key test", sig)

    def test_double_round_trip(self) -> None:
        """Serialize -> deserialize -> serialize must produce identical bytes."""
        kp = generate_keypair()
        raw1 = serialize_public_key(kp.public_key)
        restored = deserialize_public_key(raw1)
        raw2 = serialize_public_key(restored)
        assert raw1 == raw2

        raw_priv1 = serialize_private_key(kp.private_key)
        restored_priv = deserialize_private_key(raw_priv1)
        raw_priv2 = serialize_private_key(restored_priv)
        assert raw_priv1 == raw_priv2


class TestEdgeCases:
    """Edge cases for the wrapper's handling of boundary inputs."""

    def test_sign_empty_message(self) -> None:
        """Signing empty bytes must succeed (valid per Ed25519 spec)."""
        kp = generate_keypair()
        sig = sign(kp.private_key, b"")
        assert len(sig) == 64
        verify(kp.public_key, b"", sig)

    def test_sign_single_byte(self) -> None:
        """Signing a single byte must succeed."""
        kp = generate_keypair()
        sig = sign(kp.private_key, b"\x00")
        assert len(sig) == 64
        verify(kp.public_key, b"\x00", sig)

    def test_signature_always_64_bytes(self) -> None:
        """Regardless of message size, signature must always be exactly 64 bytes."""
        kp = generate_keypair()
        for size in (0, 1, 32, 64, 256, 1024, 100_000):
            sig = sign(kp.private_key, b"\xAB" * size)
            assert len(sig) == 64

    def test_corrupted_signature_first_byte(self) -> None:
        """Flipping the first byte of a valid signature must fail verification."""
        kp = generate_keypair()
        sig = sign(kp.private_key, b"test")
        corrupted = bytes([sig[0] ^ 0xFF]) + sig[1:]
        with pytest.raises(InvalidSignatureError):
            verify(kp.public_key, b"test", corrupted)

    def test_corrupted_signature_last_byte(self) -> None:
        """Flipping the last byte of a valid signature must fail verification."""
        kp = generate_keypair()
        sig = sign(kp.private_key, b"test")
        corrupted = sig[:-1] + bytes([sig[-1] ^ 0xFF])
        with pytest.raises(InvalidSignatureError):
            verify(kp.public_key, b"test", corrupted)

    def test_all_zeros_signature(self) -> None:
        """An all-zeros 64-byte signature must fail verification."""
        kp = generate_keypair()
        with pytest.raises(InvalidSignatureError):
            verify(kp.public_key, b"test", b"\x00" * 64)

    def test_all_ones_signature(self) -> None:
        """An all-0xFF 64-byte signature must fail verification."""
        kp = generate_keypair()
        with pytest.raises(InvalidSignatureError):
            verify(kp.public_key, b"test", b"\xFF" * 64)
