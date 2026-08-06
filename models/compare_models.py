"""
models/compare_models.py
=========================
Script de comparaison complete : MLP vs Autoencoder vs Hybride.

Lance les trois modeles sur tous les clients (ou un seul) et affiche
un tableau comparatif final par client et par metrique.

Usage :
    python -m models.compare_models --client power --mlp-epochs 10 --ae-epochs 20
    python -m models.compare_models --all --mlp-epochs 10 --ae-epochs 20 --save
"""

import argparse
import json
import logging
import os

import numpy as np
import torch

from models.local_ids.model       import IDSMLP
from models.local_ids.train_local_baseline import train_and_evaluate as mlp_train_eval
from models.autoencoder.model     import IDSAutoencoder
from models.autoencoder.train     import train_autoencoder, evaluate_autoencoder
from models.hybrid_detector       import HybridDetector, evaluate_hybrid

ALL_CLIENTS = ["power", "utilities", "sap", "pap", "beneficiation", "granulation"]
CHECKPOINTS_DIR = "checkpoints"
METRICS_DIR     = "results/metrics"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def run_comparison(client_name: str, args) -> dict:
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("COMPARAISON — Client : %s", client_name)
    logger.info("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. MLP
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[%s] >>> Entrainement MLP ...", client_name)
    mlp_metrics = mlp_train_eval(
        client_name,
        model_type      = "mlp",
        epochs          = args.mlp_epochs,
        batch_size      = args.batch,
        lr              = args.lr,
        save_checkpoint = args.save,
        checkpoints_dir = f"{CHECKPOINTS_DIR}/mlp",
    )
    # Recuperer le modele entraine directement (evite de recharger depuis disque)
    mlp_model = mlp_metrics.pop("_model")


    # ─────────────────────────────────────────────────────────────────────────
    # 2. Autoencoder
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[%s] >>> Entrainement Autoencoder ...", client_name)
    ae_model = IDSAutoencoder(input_dim=19, latent_dim=16, dropout=0.2)
    train_result = train_autoencoder(
        client_name    = client_name,
        model          = ae_model,
        epochs         = args.ae_epochs,
        batch_size     = args.batch,
        lr             = args.lr,
        partitions_dir = args.partitions,
    )
    ae_threshold = train_result["threshold"]
    ae_metrics   = evaluate_autoencoder(
        client_name    = client_name,
        model          = ae_model,
        threshold      = ae_threshold,
        partitions_dir = args.partitions,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Hybride
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("[%s] >>> Evaluation Detecteur Hybride ...", client_name)
    detector = HybridDetector(
        mlp           = mlp_model,
        autoencoder   = ae_model,
        ae_threshold  = ae_threshold,
        mlp_threshold = 0.5,
    )
    hybrid_metrics = evaluate_hybrid(
        detector       = detector,
        client_name    = client_name,
        partitions_dir = args.partitions,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Sauvegarde optionnelle des checkpoints
    # ─────────────────────────────────────────────────────────────────────────
    if args.save:
        os.makedirs(f"{CHECKPOINTS_DIR}/mlp",         exist_ok=True)
        os.makedirs(f"{CHECKPOINTS_DIR}/autoencoder",  exist_ok=True)
        torch.save({"model_state_dict": mlp_model.state_dict()},
                   f"{CHECKPOINTS_DIR}/mlp/mlp_{client_name}.pt")
        torch.save({
            "model_state_dict": ae_model.state_dict(),
            "threshold":        ae_threshold,
            "input_dim":        19, "latent_dim": 16, "dropout": 0.2,
        }, f"{CHECKPOINTS_DIR}/autoencoder/ae_{client_name}.pt")
        logger.info("[%s] Checkpoints MLP et AE sauvegardes.", client_name)

    return {
        "mlp":    mlp_metrics,
        "ae":     ae_metrics,
        "hybrid": hybrid_metrics,
    }


def print_comparison_table(all_results: dict) -> None:
    logger = logging.getLogger(__name__)
    METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score"]
    MODELS = ["mlp", "ae", "hybrid"]

    logger.info("")
    logger.info("=" * 90)
    logger.info("TABLEAU COMPARATIF FINAL — MLP vs Autoencoder vs Hybride")
    logger.info("=" * 90)
    header = f"{'Client':15s} {'Modele':10s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}"
    logger.info(header)
    logger.info("-" * 90)

    for client, results in all_results.items():
        for i, model_name in enumerate(MODELS):
            m = results[model_name]
            row = (
                f"{client if i == 0 else '':15s}"
                f"{model_name:10s}"
                f"{m.get('accuracy', 0):>10.4f}"
                f"{m.get('precision', 0):>10.4f}"
                f"{m.get('recall', 0):>10.4f}"
                f"{m.get('f1_score', 0):>10.4f}"
            )
            logger.info(row)
        logger.info("-" * 90)

    # Synthese : qui gagne sur F1 ?
    logger.info("")
    logger.info("SYNTHESE : Meilleur modele par F1-Score")
    for client, results in all_results.items():
        f1s = {m: results[m].get("f1_score", 0) for m in MODELS}
        winner = max(f1s, key=f1s.get)
        scores = "  |  ".join(f"{m}={v:.4f}" for m, v in f1s.items())
        logger.info("  %-15s -> %s  [%s]", client, winner.upper(), scores)


def main():
    parser = argparse.ArgumentParser(
        description="Comparaison MLP vs Autoencoder vs Hybride sur les partitions FL-IDS"
    )
    parser.add_argument("--client",      choices=ALL_CLIENTS,
                        help="Un seul client a comparer")
    parser.add_argument("--all",         action="store_true",
                        help="Comparer sur tous les clients")
    parser.add_argument("--mlp-epochs",  type=int, default=10)
    parser.add_argument("--ae-epochs",   type=int, default=20)
    parser.add_argument("--batch",       type=int, default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--partitions",  default="datasets/partitions")
    parser.add_argument("--save",        action="store_true",
                        help="Sauvegarder checkpoints et metriques")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    if not args.client and not args.all:
        parser.error("Specifier --client <nom> ou --all")

    clients     = ALL_CLIENTS if args.all else [args.client]
    all_results = {}

    for client in clients:
        all_results[client] = run_comparison(client, args)

    print_comparison_table(all_results)

    if args.save:
        os.makedirs(METRICS_DIR, exist_ok=True)
        tag = "all" if args.all else args.client
        out = f"{METRICS_DIR}/comparison_{tag}.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Resultats sauvegardes : %s", out)


if __name__ == "__main__":
    main()
