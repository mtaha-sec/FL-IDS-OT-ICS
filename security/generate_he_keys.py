"""
security/generate_he_keys.py
=============================
One-time key generation script.  Run this ONCE before starting the FL loop.

Outputs
-------
  security/keys/he_context_full.seal    — full context (secret + public key)
                                          Distribute SECURELY to every client.
  security/keys/he_context_public.seal  — public-only context (no secret key)
                                          Give this to the server only.

Security model
--------------
  • The server only ever receives he_context_public.seal.
  • It can aggregate ciphertexts homomorphically but CANNOT decrypt any
    individual client's parameters — not even the aggregated global model.
  • Only clients (holders of he_context_full.seal) can decrypt w_global.

Usage
-----
    python -m security.generate_he_keys
    # or
    python security/generate_he_keys.py
"""

import logging
import os
import sys

# Allow running as a top-level script from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from security.he_crypto import (
    generate_he_context,
    make_public_context,
    save_context,
    POLY_MOD_DEGREE,
    MAX_SLOTS,
    FL_MODEL_PARAM_COUNT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

KEYS_DIR        = os.path.join("security", "keys")
FULL_KEY_PATH   = os.path.join(KEYS_DIR, "he_context_full.seal")
PUBLIC_KEY_PATH = os.path.join(KEYS_DIR, "he_context_public.seal")


def main() -> None:
    logger.info("=" * 65)
    logger.info("FL-IDS-OT-ICS — HE Key Generation")
    logger.info("  Scheme              : CKKS (TenSEAL)")
    logger.info("  poly_modulus_degree : 16384")
    logger.info("  CKKS slots          : %d  (model params: %d ✓)",
                MAX_SLOTS, FL_MODEL_PARAM_COUNT)
    logger.info("  Scale               : 2^40  (~12 decimal digits)")
    logger.info("=" * 65)

    # Generate the CKKS context (public + secret key)
    logger.info("Generating CKKS context…  (may take a few seconds)")
    full_ctx = generate_he_context()

    # Save full context (for clients)
    save_context(full_ctx, FULL_KEY_PATH, save_secret_key=True)

    # Build and save public-only context (for server)
    pub_ctx = make_public_context(full_ctx)
    save_context(pub_ctx, PUBLIC_KEY_PATH, save_secret_key=False)

    logger.info("")
    logger.info("=" * 65)
    logger.info("Key generation complete!")
    logger.info("")
    logger.info("  %-38s  ← clients", FULL_KEY_PATH)
    logger.info("  %-38s  ← server only", PUBLIC_KEY_PATH)
    logger.info("")
    logger.info("IMPORTANT — Security instructions:")
    logger.info("  1. Distribute '%s' to every client", FULL_KEY_PATH)
    logger.info("     via a SECURE channel (TLS / encrypted copy).")
    logger.info("  2. The server receives ONLY '%s'.", PUBLIC_KEY_PATH)
    logger.info("  3. Delete or protect '%s' on the server — if an", FULL_KEY_PATH)
    logger.info("     adversary captures it, they can decrypt client params.")
    logger.info("  4. All 6 clients MUST use the SAME he_context_full.seal")
    logger.info("     so the server can aggregate their ciphertexts correctly.")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
