"""Generate an Ed25519 keypair for BBOT Swarm.

Usage:
    python -m swarm_common
    uv run bbot-swarm-keygen
"""

from swarm_common.crypto import generate_keypair, serialize_private_key, serialize_public_key


def main() -> None:
    """Generate and print an Ed25519 keypair."""
    keypair = generate_keypair()
    print(f"Private key: {serialize_private_key(keypair.private_key)}")
    print(f"Public key:  {serialize_public_key(keypair.public_key)}")


if __name__ == "__main__":
    main()
