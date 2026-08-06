"""
models/autoencoder/main.py
===========================
Point d'entree CLI pour l'entrainement et l'evaluation de l'IDSAutoencoder.

Usage :
    python -m models.autoencoder.main --client power
    python -m models.autoencoder.main --client power --epochs 30 --save
    python -m models.autoencoder.main --all --epochs 20 --save
"""

import argparse
import json
import logging
import os

import torch

from models.autoencoder.model import IDSAutoencoder
from models.autoencoder.train import train_autoencoder, evaluate_autoencoder

ALL_CLIENTS = ["power", "utilities", "sap", "pap", "beneficiation", "granulation"]
CHECKPOINTS_DIR = "checkpoints/autoencoder"
METRICS_DIR     = "results/metrics"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def run_client(client_name: str, args) -> dict:
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Client : %s", client_name)
    logger.info("=" * 60)

    model = IDSAutoencoder(
        input_dim=19,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    )
    logger.info("IDSAutoencoder — %d parametres", model.count_parameters())

    # ── Entrainement ────────────────────────────────────────────────────────
    train_result = train_autoencoder(
        client_name    = client_name,
        model          = model,
        epochs         = args.epochs,
        batch_size     = args.batch,
        lr             = args.lr,
        partitions_dir = args.partitions,
    )
    threshold = train_result["threshold"]

    # ── Sauvegarde checkpoint ────────────────────────────────────────────────
    if args.save:
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"ae_{client_name}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "threshold":        threshold,
            "input_dim":        19,
            "latent_dim":       args.latent_dim,
            "dropout":          args.dropout,
            "train_history":    train_result["history"],
        }, ckpt_path)
        logger.info("Checkpoint sauvegarde : %s", ckpt_path)

    # ── Evaluation ──────────────────────────────────────────────────────────
    metrics = evaluate_autoencoder(
        client_name    = client_name,
        model          = model,
        threshold      = threshold,
        partitions_dir = args.partitions,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Entrainement et evaluation de l'IDSAutoencoder (local, sans federation)"
    )
    parser.add_argument("--client",      choices=ALL_CLIENTS,
                        help="Un seul client a traiter")
    parser.add_argument("--all",         action="store_true",
                        help="Traiter les 6 clients successivement")
    parser.add_argument("--epochs",      type=int,   default=20,
                        help="Nombre d'epochs d'entrainement (defaut: 20)")
    parser.add_argument("--batch",       type=int,   default=256,
                        help="Taille du batch (defaut: 256)")
    parser.add_argument("--lr",          type=float, default=1e-3,
                        help="Learning rate fixe (defaut: 0.001)")
    parser.add_argument("--latent-dim",  type=int,   default=16,
                        help="Dimension de l'espace latent (defaut: 16)")
    parser.add_argument("--dropout",     type=float, default=0.2,
                        help="Taux de dropout dans l'encodeur (defaut: 0.2)")
    parser.add_argument("--partitions",  default="datasets/partitions",
                        help="Dossier racine des partitions")
    parser.add_argument("--save",        action="store_true",
                        help="Sauvegarder le checkpoint et les metriques")
    parser.add_argument("--verbose",     action="store_true",
                        help="Active les logs DEBUG")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if not args.client and not args.all:
        parser.error("Specifier --client <nom> ou --all")

    clients = ALL_CLIENTS if args.all else [args.client]
    all_metrics = {}

    for client in clients:
        metrics = run_client(client, args)
        all_metrics[client] = metrics

    # ── Tableau recap si plusieurs clients ──────────────────────────────────
    if len(clients) > 1:
        logger.info("=" * 70)
        logger.info("RECAP AUTOENCODER — tous les clients")
        logger.info("=" * 70)
        logger.info("%-15s %8s %8s %8s %8s %8s",
                    "Client", "Acc", "Prec", "Recall", "F1", "AUC")
        logger.info("-" * 70)
        for c, m in all_metrics.items():
            logger.info("%-15s %8.4f %8.4f %8.4f %8.4f %8s",
                        c, m["accuracy"], m["precision"],
                        m["recall"], m["f1_score"],
                        f"{m['roc_auc']:.4f}" if m["roc_auc"] else "  N/A ")

    # ── Sauvegarde metriques JSON ────────────────────────────────────────────
    if args.save:
        os.makedirs(METRICS_DIR, exist_ok=True)
        tag = "all" if args.all else args.client
        out = os.path.join(METRICS_DIR, f"ae_metrics_{tag}.json")
        with open(out, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info("Metriques sauvegardees : %s", out)


if __name__ == "__main__":
    main()
