import argparse
import logging
import flwr as fl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CustomFedProx(fl.server.strategy.FedProx):
    def aggregate_evaluate(self, server_round, results, failures):
        # Appel de la méthode parente pour calculer la loss globale
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if not results:
            return loss, metrics
            
        # Agrégation personnalisée de toutes nos métriques (pondération par le nombre d'exemples de test)
        total_examples = sum([res.num_examples for _, res in results])
        aggregated_metrics = {}
        
        if total_examples > 0:
            for metric_name in ["accuracy", "precision", "recall", "f1"]:
                weighted_sum = sum([
                    res.num_examples * res.metrics.get(metric_name, 0.0) 
                    for _, res in results
                ])
                aggregated_metrics[metric_name] = weighted_sum / total_examples
                
        # Impression formatée pour que FLRunnerWorker la parse facilement
        logger.info(
            f"[GLOBAL_METRICS] round={server_round} "
            f"loss={loss:.4f} "
            f"accuracy={aggregated_metrics.get('accuracy', 0.0):.4f} "
            f"precision={aggregated_metrics.get('precision', 0.0):.4f} "
            f"recall={aggregated_metrics.get('recall', 0.0):.4f} "
            f"f1={aggregated_metrics.get('f1', 0.0):.4f}"
        )
        
        return loss, aggregated_metrics

def main():
    parser = argparse.ArgumentParser(description="Serveur Central Flower (FedProx)")
    parser.add_argument("--num-rounds", type=int, default=10, help="Nombre de rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Nombre minimum de clients pour commencer")
    parser.add_argument("--mu", type=float, default=0.1, help="Paramètre FedProx µ")
    args = parser.parse_args()

    # Création de la stratégie FedProx modifiée
    strategy = CustomFedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        proximal_mu=args.mu,
    )

    logger.info(f"Démarrage du serveur Flower avec FedProx (Rounds={args.num_rounds}, MinClients={args.min_clients}, µ={args.mu})")

    fl.server.start_server(
        server_address="127.0.0.1:8085",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()

"""
central_server/server.py
=========================

FL Global Server — FL-IDS-OT-ICS

Features
--------
- Flower gRPC server
- Homomorphic Encryption (HE)
- FedProx
- TLS
- Centralized server logging
- Round-level monitoring
- Client-level metrics aggregated by the strategy
- Global model checkpoints

Usage
-----
    python -m security.generate_he_keys

    python -m central_server.server

Or:

    python -m central_server.server \
        --rounds 10 \
        --mu 0.01 \
        --min-clients 6 \
        --server-address 0.0.0.0:8080 \
        --he-context security/keys/he_context_public.seal \
        --save-dir checkpoints/fl
"""

import argparse
import logging
import os
import time
from pathlib import Path

import flwr as fl
import torch

from central_server.strategy import HEFedProxStrategy
from models.global_model.fl_model import build_fl_model

from monitoring.monitor import (
    get_server_logger,
    record_round_metrics,
)


# =============================================================================
# SERVER LOGGER
# =============================================================================

logger = get_server_logger()


# =============================================================================
# CONSTANTS
# =============================================================================

ALL_CLIENTS = [
    "power",
    "utilities",
    "sap",
    "pap",
    "beneficiation",
    "granulation",
]


# =============================================================================
# INITIAL GLOBAL MODEL
# =============================================================================

def build_initial_parameters() -> fl.common.Parameters:
    """
    Build the initial plaintext global model.

    Round 0:
        parameters[0]     -> flattened trainable parameters
        parameters[1..M]  -> BatchNorm buffers

    From round 1 onward, trainable parameters are transported
    as encrypted CKKS ciphertexts.
    """

    model = build_fl_model()

    # -------------------------------------------------------------------------
    # Flatten trainable parameters
    # -------------------------------------------------------------------------

    trainable_flat = (
        torch.cat(
            [
                p.data.view(-1)
                for p in model.parameters()
            ]
        )
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    # -------------------------------------------------------------------------
    # BatchNorm / non-trainable buffers
    # -------------------------------------------------------------------------

    buffer_arrays = [
        buffer.cpu().numpy()
        for buffer in model.buffers()
    ]

    all_arrays = [
        trainable_flat,
        *buffer_arrays,
    ]

    logger.info(
        "Initial global model built | trainable_params=%d | BN_buffers=%d",
        trainable_flat.shape[0],
        len(buffer_arrays),
    )

    return fl.common.ndarrays_to_parameters(all_arrays)


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="FL-IDS-OT-ICS Global Server (HE + FedProx + TLS + Monitoring)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Federated Learning
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds.",
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=0.01,
        help="FedProx proximal coefficient μ.",
    )

    parser.add_argument(
        "--min-clients",
        type=int,
        default=6,
        help="Minimum number of clients required per round.",
    )

    # -------------------------------------------------------------------------
    # Flower server
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--server-address",
        default="0.0.0.0:8080",
        help="Flower gRPC server address.",
    )

    # -------------------------------------------------------------------------
    # TLS
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--tls-ca",
        default=os.path.join(
            "security",
            "certificates",
            "ca.crt",
        ),
        help="TLS CA certificate.",
    )

    parser.add_argument(
        "--tls-cert",
        default=os.path.join(
            "security",
            "certificates",
            "server.crt",
        ),
        help="TLS server certificate.",
    )

    parser.add_argument(
        "--tls-key",
        default=os.path.join(
            "security",
            "certificates",
            "server.key",
        ),
        help="TLS server private key.",
    )

    # -------------------------------------------------------------------------
    # Homomorphic Encryption
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--he-context",
        default=os.path.join(
            "security",
            "keys",
            "he_context_public.seal",
        ),
        help=(
            "Path to the PUBLIC HE context. "
            "The server must never receive the secret key."
        ),
    )

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--save-dir",
        default=os.path.join(
            "checkpoints",
            "fl",
        ),
        help="Directory used to save FL checkpoints.",
    )

    return parser.parse_args()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_files(args: argparse.Namespace) -> None:
    """
    Validate required TLS and HE files before starting the server.
    """

    # -------------------------------------------------------------------------
    # TLS files
    # -------------------------------------------------------------------------

    tls_files = {
        "TLS CA": args.tls_ca,
        "TLS certificate": args.tls_cert,
        "TLS private key": args.tls_key,
    }

    for name, path in tls_files.items():

        if not os.path.exists(path):

            logger.error(
                "%s not found: %s",
                name,
                path,
            )

            raise FileNotFoundError(
                f"{name} missing: {path}"
            )

    # -------------------------------------------------------------------------
    # HE public context
    # -------------------------------------------------------------------------

    if not os.path.exists(args.he_context):

        logger.error(
            "Public HE context not found: %s",
            args.he_context,
        )

        logger.error(
            "Run: python -m security.generate_he_keys"
        )

        raise FileNotFoundError(
            f"Public HE context missing: {args.he_context}"
        )


# =============================================================================
# SERVER BANNER
# =============================================================================

def log_server_configuration(args: argparse.Namespace) -> None:
    """
    Print the server configuration into server.log.
    """

    logger.info("=" * 70)

    logger.info(
        "FL-IDS-OT-ICS — Global Federated Learning Server"
    )

    logger.info("-" * 70)

    logger.info(
        "Rounds              : %d",
        args.rounds,
    )

    logger.info(
        "FedProx μ           : %.4f (%s)",
        args.mu,
        "FedAvg" if args.mu == 0 else "FedProx",
    )

    logger.info(
        "Minimum clients     : %d",
        args.min_clients,
    )

    logger.info(
        "Expected clients    : %s",
        ", ".join(ALL_CLIENTS),
    )

    logger.info(
        "Server address      : %s",
        args.server_address,
    )

    logger.info(
        "TLS                 : ENABLED",
    )

    logger.info(
        "TLS CA              : %s",
        args.tls_ca,
    )

    logger.info(
        "TLS certificate     : %s",
        args.tls_cert,
    )

    logger.info(
        "HE public context   : %s",
        args.he_context,
    )

    logger.info(
        "Checkpoint directory: %s",
        args.save_dir,
    )

    logger.info(
        "Monitoring logs     : monitoring/logs/server.log",
    )

    logger.info(
        "Monitoring metrics  : monitoring/metrics/rounds.csv",
    )

    logger.info(
        "Privacy guarantee   : encrypted-domain aggregation",
    )

    logger.info(
        "Server secret key   : NOT PRESENT",
    )

    logger.info("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main() -> fl.server.history.History:

    args = parse_arguments()

    # -------------------------------------------------------------------------
    # Validate configuration
    # -------------------------------------------------------------------------

    validate_files(args)

    # -------------------------------------------------------------------------
    # Server configuration
    # -------------------------------------------------------------------------

    log_server_configuration(args)

    # -------------------------------------------------------------------------
    # Build initial global model
    # -------------------------------------------------------------------------

    logger.info(
        "Building initial global model..."
    )

    initial_params = build_initial_parameters()

    logger.info(
        "Initial global model ready."
    )

    # -------------------------------------------------------------------------
    # Instantiate HE + FedProx strategy
    # -------------------------------------------------------------------------

    logger.info(
        "Initializing HEFedProxStrategy..."
    )

    strategy = HEFedProxStrategy(
        public_context_path=args.he_context,

        mu=args.mu,

        min_fit_clients=args.min_clients,

        min_evaluate_clients=args.min_clients,

        min_available_clients=args.min_clients,

        initial_parameters=initial_params,

        save_dir=args.save_dir,

        fraction_fit=1.0,

        fraction_evaluate=1.0,
    )

    logger.info(
        "HEFedProxStrategy initialized successfully."
    )

    # -------------------------------------------------------------------------
    # TLS certificates
    # -------------------------------------------------------------------------

    tls_ca = Path(args.tls_ca).read_bytes()
    tls_cert = Path(args.tls_cert).read_bytes()
    tls_key = Path(args.tls_key).read_bytes()

    logger.info(
        "TLS certificates loaded successfully."
    )

    # -------------------------------------------------------------------------
    # Start server
    # -------------------------------------------------------------------------

    logger.info(
        "Waiting for at least %d client(s)...",
        args.min_clients,
    )

    logger.info(
        "Flower server listening on %s",
        args.server_address,
    )

    logger.info(
        "Expected clients: %s",
        ", ".join(ALL_CLIENTS),
    )

    logger.info(
        "Monitoring is ENABLED."
    )

    # -------------------------------------------------------------------------
    # FL training timer
    # -------------------------------------------------------------------------

    training_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Start Flower server
    # -------------------------------------------------------------------------

    history = fl.server.start_server(

        server_address=args.server_address,

        strategy=strategy,

        config=fl.server.ServerConfig(
            num_rounds=args.rounds,
        ),

        certificates=(
            tls_ca,
            tls_cert,
            tls_key,
        ),
    )

    # -------------------------------------------------------------------------
    # Training duration
    # -------------------------------------------------------------------------

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    # -------------------------------------------------------------------------
    # Training summary
    # -------------------------------------------------------------------------

    logger.info("=" * 70)

    logger.info(
        "FL training complete."
    )

    logger.info(
        "Completed rounds: %d",
        args.rounds,
    )

    logger.info(
        "Total training duration: %.3f seconds",
        total_training_time,
    )

    # -------------------------------------------------------------------------
    # Distributed losses
    # -------------------------------------------------------------------------

    if history.losses_distributed:

        logger.info(
            "Distributed evaluation loss:"
        )

        for round_number, loss in history.losses_distributed:

            logger.info(
                "Round %d | loss=%.6f",
                round_number,
                loss,
            )

    # -------------------------------------------------------------------------
    # Distributed metrics
    # -------------------------------------------------------------------------

    if history.metrics_distributed:

        logger.info(
            "Distributed evaluation metrics:"
        )

        for metric_name, values in history.metrics_distributed.items():

            logger.info(
                "Metric: %s",
                metric_name,
            )

            for round_number, value in values:

                logger.info(
                    "Round %d | %s=%.6f",
                    round_number,
                    metric_name,
                    value,
                )

    # -------------------------------------------------------------------------
    # Final accuracy
    # -------------------------------------------------------------------------

    final_accuracy = None

    if history.metrics_distributed:

        accuracy_history = (
            history.metrics_distributed.get(
                "accuracy",
                [],
            )
        )

        if accuracy_history:

            final_accuracy = accuracy_history[-1][1]

            logger.info(
                "Final distributed accuracy: %.6f",
                final_accuracy,
            )

    # -------------------------------------------------------------------------
    # Record final monitoring information
    # -------------------------------------------------------------------------

    record_round_metrics(
        round_number=args.rounds,
        accuracy=final_accuracy,
        duration_seconds=total_training_time,
        num_clients=args.min_clients,
    )

    logger.info(
        "Final training summary recorded in monitoring/metrics/rounds.csv"
    )

    logger.info(
        "Server log available at monitoring/logs/server.log"
    )

    logger.info(
        "Client metrics available at monitoring/metrics/clients.csv"
    )

    logger.info("=" * 70)

    return history


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()