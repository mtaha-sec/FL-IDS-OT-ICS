"""
models/hybrid_detector.py
==========================
Pipeline de detection hybride : MLP + Autoencoder.

Architecture de decision :
  ┌──────────────────────────────────────────────────────────────────┐
  │  flux x (19 features)                                            │
  │       │                                                          │
  │  Autoencoder ──> reconstruction_error                           │
  │       │                                                          │
  │       ├─ error < threshold  ──>  NORMAL (confiance AE)          │
  │       │                                                          │
  │       └─ error >= threshold ──>  ANOMALIE                        │
  │               │                                                  │
  │          MLP ──> proba [0,1]                                     │
  │               │                                                  │
  │               ├─ proba >= mlp_threshold  ──>  Attaque CONNUE    │
  │               │                                                  │
  │               └─ proba < mlp_threshold   ──>  ZERO-DAY SUSPECT  │
  └──────────────────────────────────────────────────────────────────┘

Comment le FL combine les deux modeles :
  - MLP global  : agregation FedAvg des poids du classificateur binaire.
                  Detecte et classe les attaques CONNUES (patterns vus
                  par au moins un client lors des rounds FL).
  - AE global   : agregation FedAvg des poids de l'autoencoder.
                  Reconstruit le trafic NORMAL de l'ensemble du reseau.
                  Toute deviation du profil normal global => anomalie,
                  meme si le MLP ne reconnait pas le pattern => ZERO-DAY.

Usage CLI :
    python -m models.hybrid_detector \\
        --client power \\
        --mlp-ckpt checkpoints/mlp/mlp_power.pt \\
        --ae-ckpt  checkpoints/autoencoder/ae_power.pt

    python -m models.hybrid_detector --all \\
        --mlp-ckpt checkpoints/mlp/mlp_{client}.pt \\
        --ae-ckpt  checkpoints/autoencoder/ae_{client}.pt
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

from models.autoencoder.model import IDSAutoencoder
from models.local_ids.model   import IDSMLP

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "syn_flag", "ack_flag", "fin_flag", "rst_flag", "synack_flag",
    "byte_rate", "pkt_rate", "bytes_ratio", "pkts_ratio", "with_payload",
    "proto_tcp", "proto_udp", "proto_icmp", "proto_other",
]

ALL_CLIENTS = ["power", "utilities", "sap", "pap", "beneficiation", "granulation"]


# ─────────────────────────────────────────────────────────────────────────────
# Classe principale
# ─────────────────────────────────────────────────────────────────────────────
class HybridDetector:
    """
    Pipeline de detection hybride MLP + Autoencoder.

    Parametres
    ----------
    mlp           : IDSMLP deja charge et en mode eval
    autoencoder   : IDSAutoencoder deja charge et en mode eval
    ae_threshold  : seuil d'erreur de reconstruction (float)
    mlp_threshold : seuil de probabilite MLP pour valider une attaque connue
                    (float, defaut 0.5 ; abaisser pour etre plus sensible)
    """

    def __init__(self,
                 mlp: IDSMLP,
                 autoencoder: IDSAutoencoder,
                 ae_threshold: float,
                 mlp_threshold: float = 0.5):
        self.mlp           = mlp.to(DEVICE).eval()
        self.autoencoder   = autoencoder.to(DEVICE).eval()
        self.ae_threshold  = ae_threshold
        self.mlp_threshold = mlp_threshold

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Retourne pour chaque echantillon :
          - label      : 0=normal, 1=attaque connue, 2=zero-day suspect
          - ae_error   : erreur de reconstruction de l'AE
          - mlp_proba  : probabilite d'attaque selon le MLP
          - confidence : score de confiance global [0,1]

        Algorithme :
          1. AE calcule l'erreur de reconstruction.
          2. Si error < ae_threshold -> normal (label=0).
          3. Sinon -> anomalie -> MLP calcule la proba d'attaque.
             a. proba >= mlp_threshold -> attaque connue (label=1)
             b. proba <  mlp_threshold -> zero-day suspect (label=2)
        """
        x = x.to(DEVICE)
        batch_size = x.size(0)

        # ── Autoencoder ────────────────────────────────────────────────────
        ae_errors  = self.autoencoder.reconstruction_error(x).cpu().numpy()
        ae_anomaly = ae_errors >= self.ae_threshold

        # ── MLP (sur tout le batch pour efficacite) ────────────────────────
        logits     = self.mlp(x).squeeze(1)
        mlp_proba  = torch.sigmoid(logits).cpu().numpy()

        # ── Decision hybride ──────────────────────────────────────────────
        labels     = np.zeros(batch_size, dtype=int)         # defaut : normal
        confidence = np.ones(batch_size, dtype=float)

        for i in range(batch_size):
            if ae_anomaly[i]:
                if mlp_proba[i] >= self.mlp_threshold:
                    labels[i]     = 1                         # attaque connue
                    confidence[i] = float(mlp_proba[i])
                else:
                    labels[i]     = 2                         # zero-day suspect
                    # confiance basee sur l'amplitude de l'erreur AE
                    confidence[i] = min(1.0, ae_errors[i] / (self.ae_threshold * 3))
            else:
                labels[i]     = 0                             # normal
                confidence[i] = 1.0 - float(ae_errors[i] / self.ae_threshold)

        return {
            "labels":     labels,
            "ae_error":   ae_errors,
            "mlp_proba":  mlp_proba,
            "confidence": confidence,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation complete
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_hybrid(detector: HybridDetector,
                    client_name: str,
                    partitions_dir: str = "datasets/partitions") -> dict:
    """
    Evalue le detecteur hybride sur le jeu de test d'un client.
    Les labels reels sont binaires (0=normal, 1=attaque).
    Les labels predits peuvent etre 0, 1 ou 2 (zero-day).
    Pour les metriques binaires, on regroupe 1 et 2 comme "attaque".
    """
    test_df = pd.read_csv(os.path.join(partitions_dir, client_name, "test.csv"))
    X_test  = torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
    y_true  = test_df["label"].values.astype(int)

    result  = detector.predict(X_test)
    y_pred_3class = result["labels"]

    # Binarisation pour metriques standard (0=normal, 1 ou 2=attaque)
    y_pred_bin = (y_pred_3class > 0).astype(int)

    # Flux classes comme zero-day
    zero_day_mask = (y_pred_3class == 2)
    n_zero_day    = int(zero_day_mask.sum())

    # Zero-day vrais positifs = flux zero-day qui sont bien des attaques
    if n_zero_day > 0:
        zd_tp = int((y_true[zero_day_mask] == 1).sum())
    else:
        zd_tp = 0

    metrics = {
        "client":            client_name,
        "n_test":            len(y_true),
        "n_normal":          int((y_true == 0).sum()),
        "n_attack":          int((y_true == 1).sum()),
        "n_pred_normal":     int((y_pred_bin == 0).sum()),
        "n_pred_known_atk":  int((y_pred_3class == 1).sum()),
        "n_pred_zero_day":   n_zero_day,
        "zero_day_real_atk": zd_tp,
        # Metriques binaires (normal vs tout-attaque)
        "accuracy":          float(accuracy_score(y_true, y_pred_bin)),
        "precision":         float(precision_score(y_true, y_pred_bin, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred_bin, zero_division=0)),
        "f1_score":          float(f1_score(y_true, y_pred_bin, zero_division=0)),
        "roc_auc":           float(roc_auc_score(y_true, result["mlp_proba"])) if len(set(y_true)) > 1 else None,
        "confusion_matrix":  confusion_matrix(y_true, y_pred_bin).tolist(),
        "ae_threshold":      detector.ae_threshold,
        "mlp_threshold":     detector.mlp_threshold,
    }

    logger.info("=== Evaluation Hybride [%s] ===", client_name)
    logger.info(
        "Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f | ROC-AUC=%s",
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1_score"],
        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "N/A",
    )
    logger.info(
        "Predictions : Normal=%d | Attaque connue=%d | Zero-day suspect=%d (dont %d vrais positifs)",
        metrics["n_pred_normal"], metrics["n_pred_known_atk"],
        metrics["n_pred_zero_day"], metrics["zero_day_real_atk"],
    )
    print(classification_report(y_true, y_pred_bin,
                                target_names=["normal", "attaque"],
                                zero_division=0))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def load_mlp_from_checkpoint(path: str) -> IDSMLP:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = IDSMLP(input_dim=19)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def load_ae_from_checkpoint(path: str):
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    ae    = IDSAutoencoder(
        input_dim=ckpt.get("input_dim", 19),
        latent_dim=ckpt.get("latent_dim", 16),
        dropout=ckpt.get("dropout", 0.2),
    )
    ae.load_state_dict(ckpt["model_state_dict"])
    threshold = ckpt["threshold"]
    return ae, threshold


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation du detecteur hybride MLP + Autoencoder"
    )
    parser.add_argument("--client",      choices=ALL_CLIENTS,
                        help="Un seul client a evaluer")
    parser.add_argument("--all",         action="store_true",
                        help="Evaluer tous les clients")
    parser.add_argument("--mlp-ckpt",    required=True,
                        help="Chemin checkpoint MLP (.pt). Utiliser {client} comme placeholder.")
    parser.add_argument("--ae-ckpt",     required=True,
                        help="Chemin checkpoint AE (.pt). Utiliser {client} comme placeholder.")
    parser.add_argument("--mlp-threshold", type=float, default=0.5,
                        help="Seuil de probabilite MLP (defaut: 0.5)")
    parser.add_argument("--partitions",  default="datasets/partitions")
    parser.add_argument("--save",        action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.client and not args.all:
        parser.error("Specifier --client <nom> ou --all")

    clients     = ALL_CLIENTS if args.all else [args.client]
    all_metrics = {}

    for client in clients:
        mlp_path = args.mlp_ckpt.format(client=client)
        ae_path  = args.ae_ckpt.format(client=client)

        if not os.path.exists(mlp_path):
            logger.error("Checkpoint MLP introuvable : %s", mlp_path)
            continue
        if not os.path.exists(ae_path):
            logger.error("Checkpoint AE introuvable : %s", ae_path)
            continue

        mlp             = load_mlp_from_checkpoint(mlp_path)
        ae, ae_threshold = load_ae_from_checkpoint(ae_path)

        detector = HybridDetector(
            mlp=mlp, autoencoder=ae,
            ae_threshold=ae_threshold,
            mlp_threshold=args.mlp_threshold,
        )
        metrics = evaluate_hybrid(detector, client, args.partitions)
        all_metrics[client] = metrics

    # ── Tableau comparatif ───────────────────────────────────────────────────
    if len(all_metrics) > 1:
        logger.info("=" * 75)
        logger.info("RECAP HYBRIDE — tous les clients")
        logger.info("=" * 75)
        logger.info("%-15s %8s %8s %8s %8s %10s",
                    "Client", "Acc", "Prec", "Recall", "F1", "Zero-day")
        logger.info("-" * 75)
        for c, m in all_metrics.items():
            logger.info("%-15s %8.4f %8.4f %8.4f %8.4f %10d",
                        c, m["accuracy"], m["precision"],
                        m["recall"], m["f1_score"],
                        m["n_pred_zero_day"])

    if args.save:
        os.makedirs("results/metrics", exist_ok=True)
        tag = "all" if args.all else args.client
        out = f"results/metrics/hybrid_metrics_{tag}.json"
        with open(out, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info("Metriques sauvegardees : %s", out)


if __name__ == "__main__":
    main()
