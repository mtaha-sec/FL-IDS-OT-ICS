"""
central_server/server.py
=========================
FL Global Server — wires up HEFedProxStrategy and starts the Flower gRPC server.

Pre-requisites (run once before any training)
---------------------------------------------
    python -m security.generate_he_keys
    # Outputs:
    #   security/keys/he_context_full.seal    → distribute to all clients
    #   security/keys/he_context_public.seal  → used by this server

Usage
-----
    # Minimal (default 10 rounds, μ=0.01, wait for 6 clients)
    python -m central_server.server

    # Full options
    python -m central_server.server \\
        --rounds       10 \\
        --mu           0.01 \\
        --min-clients  6 \\
        --server-address 0.0.0.0:8080 \\
        --he-context   security/keys/he_context_public.seal \\
        --save-dir     checkpoints/fl

Then in 6 separate terminals (or via docker-compose):
    python clients/client_app.py --client-name power       --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name utilities   --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name sap         --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name pap         --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name beneficiation --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name granulation --server-address 127.0.0.1:8080
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import flwr as fl

from central_server.strategy import HEFedProxStrategy
from models.global_model.fl_model import build_fl_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ALL_CLIENTS = ["power", "utilities", "sap", "pap", "beneficiation", "granulation"]


# ── Initial global model ───────────────────────────────────────────────────────

def build_initial_parameters() -> fl.common.Parameters:
    """
    Build plaintext initial parameters for round 0.

    Encoding (mirrors the HE-aware client encoding, but plaintext for round 0):
      ndarrays[0]      = trainable params concatenated as float32 flat vector
      ndarrays[1 .. M] = BN buffer arrays (running_mean, running_var, etc.)

    Why plaintext for round 0?
      The server holds only the public key → it cannot encrypt.
      Clients receive these plaintext params, load them into the model,
      then immediately ENCRYPT their own updated weights when returning
      the first fit() result.  From round 1 onward, all inter-party
      parameter transfers are ciphertext only.
    """
    model = build_fl_model()

    # Flatten all trainable parameters into one float32 vector
    trainable_flat = (
        torch.cat([p.data.view(-1) for p in model.parameters()])
        .detach()
        .cpu()
        .numpy()
    )   # shape: (3905,), dtype: float32

    # Collect non-trainable buffers (BN running_mean/var/count) as-is
    buffer_arrays = [b.cpu().numpy() for b in model.buffers()]

    all_arrays = [trainable_flat] + buffer_arrays
    logger.info(
        "Initial global model built — trainable params: %d  BN buffers: %d",
        trainable_flat.shape[0], len(buffer_arrays),
    )
    return fl.common.ndarrays_to_parameters(all_arrays)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> fl.server.history.History:
    parser = argparse.ArgumentParser(
        description="FL-IDS-OT-ICS — Global Server (HE + FedProx)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rounds", type=int, default=10,
        help="Number of federated learning rounds.",
    )
    parser.add_argument(
        "--mu", type=float, default=0.01,
        help="FedProx proximal coefficient μ (0.0 = standard FedAvg).",
    )
    parser.add_argument(
        "--min-clients", type=int, default=6,
        help="Minimum clients required per round (defaults to all 6 sites).",
    )
    parser.add_argument(
        "--server-address", default="0.0.0.0:8080",
        help="gRPC server address.",
    )
    parser.add_argument(
    "--tls-ca",
    default=os.path.join("security", "certificates", "ca.crt"),
    help="TLS CA certificate.",
    )

    parser.add_argument(
    "--tls-cert",
    default=os.path.join("security", "certificates", "server.crt"),
    help="TLS server certificate.",
    )

    parser.add_argument(
    "--tls-key",
    default=os.path.join("security", "certificates", "server.key"),
    help="TLS server private key.",
    )
    parser.add_argument(
        "--he-context",
        default=os.path.join("security", "keys", "he_context_public.seal"),
        help="Path to the server's PUBLIC HE context (no secret key).",
    )
    parser.add_argument(
        "--save-dir", default=os.path.join("checkpoints", "fl"),
        help="Directory to save global model checkpoints each round.",
    )
    args = parser.parse_args()
    # ── Validate TLS certificates ─────────────────────────────────────────────
    for tls_file in [args.tls_ca, args.tls_cert, args.tls_key]:
          if not os.path.exists(tls_file):
            logger.error("TLS file not found: %s", tls_file)
            raise FileNotFoundError(f"TLS file missing: {tls_file}")

    # ── Validate the HE context file ──────────────────────────────────────────
    if not os.path.exists(args.he_context):
        logger.error(
            "Public HE context not found at: %s\n"
            "Run 'python -m security.generate_he_keys' first to create it.",
            args.he_context,
        )
        raise FileNotFoundError(f"HE public context missing: {args.he_context}")

    # ── Banner ────────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("FL-IDS-OT-ICS — Global Server")
    logger.info("  Rounds             : %d", args.rounds)
    logger.info("  FedProx μ          : %.4f  (%s)",
                args.mu, "FedAvg" if args.mu == 0 else "FedProx")
    logger.info("  Min clients / round: %d", args.min_clients)
    logger.info("  Server address     : %s", args.server_address)
    logger.info("  TLS                : ENABLED")
    logger.info("  TLS CA             : %s", args.tls_ca)
    logger.info("  TLS certificate    : %s", args.tls_cert)
    logger.info("  HE context (public): %s", args.he_context)
    logger.info("  Checkpoint dir     : %s", args.save_dir)
    logger.info("  Privacy guarantee  : server aggregates IN ENCRYPTED DOMAIN,")
    logger.info("                       individual client params NEVER decrypted.")
    logger.info("=" * 65)

    # ── Build initial global model ─────────────────────────────────────────────
    initial_params = build_initial_parameters()

    # ── Instantiate strategy ───────────────────────────────────────────────────
    strategy = HEFedProxStrategy(
        public_context_path    = args.he_context,
        mu                     = args.mu,
        min_fit_clients        = args.min_clients,
        min_evaluate_clients   = args.min_clients,
        min_available_clients  = args.min_clients,
        initial_parameters     = initial_params,
        save_dir               = args.save_dir,
        fraction_fit           = 1.0,
        fraction_evaluate      = 1.0,
    )

    # ── Start Flower server ────────────────────────────────────────────────────
    logger.info(
        "Waiting for at least %d client(s) on %s …",
        args.min_clients, args.server_address,
    )
    history = fl.server.start_server(
        server_address = args.server_address,
        strategy       = strategy,
        config         = fl.server.ServerConfig(num_rounds=args.rounds),
        certificates   = (
        Path(args.tls_ca).read_bytes(),
        Path(args.tls_cert).read_bytes(),
        Path(args.tls_key).read_bytes(),
    ),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("FL training complete — %d rounds", args.rounds)
    if history.losses_distributed:
        rounds_log = history.losses_distributed
        logger.info("Distributed loss per round:")
        for rnd, loss in rounds_log:
            logger.info("  Round %2d: %.6f", rnd, loss)
        logger.info("Final loss: %.6f", rounds_log[-1][1])
    if history.metrics_distributed:
        acc_history = history.metrics_distributed.get("accuracy", [])
        if acc_history:
            logger.info("Final accuracy (distributed): %.4f", acc_history[-1][1])
    logger.info("=" * 65)
    return history


if __name__ == "__main__":
    main()
