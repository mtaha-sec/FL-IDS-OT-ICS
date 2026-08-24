"""
security/he_crypto.py
=====================
Homomorphic Encryption helpers using TenSEAL (CKKS scheme).

Core privacy guarantee
-----------------------
The server aggregates model parameters **in the encrypted domain** — it holds
only the PUBLIC key and can never decrypt any individual client's weights.
Only the clients (holders of the SECRET key) can decrypt the aggregated result.

Encryption flow
---------------
  Client  →  encrypt(w_k, ctx_full)   →  ciphertext_k  ──────────►  Server
  Server  →  Σ_k (n_k/N)·ciphertext_k  (NO decryption, pure HE ops) →  agg_ct
  Server  →  agg_ct ──────────────────────────────────────────────►  Client
  Client  →  decrypt(agg_ct, ctx_full)  →  w_global

CKKS parameters for FLIDSModel (3 905 trainable parameters)
-----------------------------------------------------------
  poly_modulus_degree = 8 192   →  4 096 CKKS slots  (3 905 < 4 096 ✓)
  coeff_mod_bit_sizes = [60, 40, 40, 60]  →  3 mult levels, 128-bit security
  scale = 2^40  →  ~12 decimal digits of precision (more than enough for FL)

The BN running statistics (running_mean, running_var, num_batches_tracked) are
NOT trainable and are sent in plaintext; they reveal some data distribution info
but are far less sensitive than the learned weights/biases.
"""

import os
import logging
import numpy as np
from typing import List, Tuple

try:
    import tenseal as ts
    HE_AVAILABLE = True
except ImportError:
    HE_AVAILABLE = False
    ts = None  # type: ignore

logger = logging.getLogger(__name__)

# ─── CKKS Hyper-parameters ────────────────────────────────────────────────────
POLY_MOD_DEGREE: int       = 16_384
COEFF_MOD_BITS:  List[int] = [60, 40, 40, 60]
GLOBAL_SCALE:    float     = 2 ** 40
MAX_SLOTS:       int       = POLY_MOD_DEGREE // 2   # = 8 192

# FLIDSModel trainable param count  (19→64→32→16→1 MLP with BatchNorm affine)
# Must stay ≤ MAX_SLOTS to fit in a single CKKS ciphertext vector.
FL_MODEL_PARAM_COUNT: int = 5_961   # verified < 8 192


# ─── Internal guard ───────────────────────────────────────────────────────────

def _require_tenseal() -> None:
    if not HE_AVAILABLE:
        raise ImportError(
            "TenSEAL is required for homomorphic encryption.\n"
            "Install it with:  pip install tenseal"
        )


# ─── Key / Context Management ─────────────────────────────────────────────────

def generate_he_context() -> "ts.Context":
    """
    Generate a fresh CKKS context containing **both** the public key and the
    secret key.  Call this ONCE before starting the FL training loop.

    Workflow after generation
    -------------------------
    1. Save the FULL context  → distribute securely to every client.
    2. Call make_public_context() → save the PUBLIC-ONLY context for the server.
       The server can aggregate ciphertexts but **cannot** decrypt them.
    """
    _require_tenseal()
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MOD_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BITS,
    )
    ctx.generate_galois_keys()
    ctx.global_scale = GLOBAL_SCALE
    logger.info(
        "HE context generated — poly_mod_degree=%d  slots=%d  scale=2^40",
        POLY_MOD_DEGREE, MAX_SLOTS,
    )
    return ctx


def make_public_context(full_ctx: "ts.Context") -> "ts.Context":
    """
    Strip the secret key from *full_ctx* and return a **public-only** context.
    Give this to the server; the server can homomorphically aggregate
    ciphertexts with it but cannot decrypt any individual result.
    """
    _require_tenseal()
    pub_bytes = full_ctx.serialize(save_secret_key=False)
    return ts.context_from(pub_bytes)


def save_context(ctx: "ts.Context", path: str, save_secret_key: bool = True) -> None:
    """Persist a TenSEAL context to *path*."""
    _require_tenseal()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(ctx.serialize(save_secret_key=save_secret_key))
    key_tag = "full (secret+public)" if save_secret_key else "public-only"
    logger.info("HE context saved [%s] → %s", key_tag, path)


def load_context(path: str) -> "ts.Context":
    """Load a serialized TenSEAL context from *path*."""
    _require_tenseal()
    with open(path, "rb") as fh:
        data = fh.read()
    ctx = ts.context_from(data)
    key_tag = "full" if ctx.is_private() else "public-only"
    logger.info("HE context loaded [%s] ← %s", key_tag, path)
    return ctx


# ─── Encrypt / Decrypt ────────────────────────────────────────────────────────

def _flatten_params(params: List[np.ndarray]) -> List[float]:
    """Concatenate all parameter arrays into a single Python list of float64."""
    return np.concatenate(
        [p.flatten().astype(np.float64) for p in params]
    ).tolist()


def encrypt_parameters(
    params: List[np.ndarray],
    ctx: "ts.Context",
) -> np.ndarray:
    """
    Encrypt a list of numpy parameter arrays as a **single CKKS vector**.

    Returns
    -------
    np.ndarray dtype=uint8
        Serialized ciphertext encoded as a uint8 array so it can be transported
        through Flower's numpy parameter system without modification.

    Notes
    -----
    *ctx* may be either the full context or the public context — both can
    encrypt.  Only the holder of the secret key can decrypt.
    """
    _require_tenseal()
    flat = _flatten_params(params)
    if len(flat) > MAX_SLOTS:
        raise ValueError(
            f"Model has {len(flat)} trainable parameters but CKKS context "
            f"supports only {MAX_SLOTS} slots.  "
            f"Increase poly_modulus_degree to 16 384 and rerun key generation."
        )
    enc_vec   = ts.ckks_vector(ctx, flat)
    enc_bytes = enc_vec.serialize()
    return np.frombuffer(enc_bytes, dtype=np.uint8).copy()


def decrypt_parameters(
    enc_uint8: np.ndarray,
    ctx: "ts.Context",
    shapes: List[Tuple],
) -> List[np.ndarray]:
    """
    Decrypt a uint8-encoded ciphertext and reconstruct the original list of
    parameter arrays.

    Parameters
    ----------
    enc_uint8 : np.ndarray dtype=uint8
        Serialized ciphertext (output of encrypt_parameters or
        aggregate_encrypted_params).
    ctx : ts.Context
        **Full** context containing the secret key.
    shapes : list of tuple
        Shape of each parameter tensor, in the same order as model.parameters().

    Returns
    -------
    list of np.ndarray dtype=float32
    """
    _require_tenseal()
    if not ctx.is_private():
        raise ValueError(
            "Cannot decrypt: the loaded context does not contain the secret key. "
            "Use the full context file (he_context_full.seal), not the public one."
        )
    enc_bytes = enc_uint8.tobytes()
    enc_vec   = ts.ckks_vector_from(ctx, enc_bytes)
    flat      = np.array(enc_vec.decrypt(), dtype=np.float32)

    result, offset = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        result.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return result


# ─── Server-side Homomorphic Aggregation (no decryption!) ─────────────────────

def aggregate_encrypted_params(
    enc_list:  List[np.ndarray],
    weights:   List[float],
    pub_ctx:   "ts.Context",
) -> np.ndarray:
    """
    FedProx / FedAvg aggregation **entirely in the encrypted domain**.

    Computes:
        Enc(w_global) = Σ_k  (n_k / N) · Enc(w_k)

    using only homomorphic scalar multiplication and ciphertext addition.
    The server **never** holds the plaintext weights of any individual client.

    Parameters
    ----------
    enc_list : list of np.ndarray dtype=uint8
        One serialized ciphertext per client (output of encrypt_parameters).
    weights : list of float
        Per-client aggregation weights.  Must sum to 1.0.
        Typically  n_k / Σ_j n_j  (sample-count weighted).
    pub_ctx : ts.Context
        Server's **public-only** context (no secret key required for HE ops).

    Returns
    -------
    np.ndarray dtype=uint8
        Serialized aggregated ciphertext.  Send to clients; they decrypt it
        with their full context to obtain the new global model.
    """
    _require_tenseal()
    if abs(sum(weights) - 1.0) > 1e-5:
        raise ValueError(
            f"Aggregation weights must sum to 1.0, got {sum(weights):.6f}"
        )

    result_enc = None
    for enc_uint8, w in zip(enc_list, weights):
        enc_bytes = enc_uint8.tobytes()
        enc_vec   = ts.ckks_vector_from(pub_ctx, enc_bytes)
        weighted  = enc_vec * w     # HE: scalar × ciphertext  (no decryption)

        if result_enc is None:
            result_enc = weighted
        else:
            result_enc += weighted  # HE: ciphertext + ciphertext (no decryption)

    agg_bytes = result_enc.serialize()
    logger.debug(
        "HE aggregation complete — %d clients, weights=%s",
        len(enc_list), [f"{w:.4f}" for w in weights],
    )
    return np.frombuffer(agg_bytes, dtype=np.uint8).copy()


# ─── Architecture helpers ──────────────────────────────────────────────────────

def get_param_shapes() -> List[Tuple]:
    """
    Return the shapes of ALL trainable parameters of FLIDSModel
    (fixed architecture 19→64→32→16→1, identical on every client and server).
    """
    from models.global_model.fl_model import build_fl_model
    model = build_fl_model()
    return [tuple(p.shape) for p in model.parameters()]


def get_buffer_shapes() -> List[Tuple]:
    """
    Return the shapes of the non-trainable buffers of FLIDSModel
    (BatchNorm running_mean, running_var, num_batches_tracked).
    """
    from models.global_model.fl_model import build_fl_model
    model = build_fl_model()
    return [tuple(b.shape) for b in model.buffers()]
