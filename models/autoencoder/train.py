"""
models/autoencoder/train.py
============================
Fonctions d'entrainement et d'evaluation de l'IDSAutoencoder.

Points cles :
  - L'autoencoder est entraine UNIQUEMENT sur le trafic NORMAL (label=0).
    Les flux d'attaque sont exclus du training set : le modele apprend
    le profil du trafic legitime, et toute deviation est une anomalie.
  - Le seuil de detection (threshold) est calcule sur l'ensemble de
    validation normal : threshold = mean(erreurs) + k * std(erreurs).
    k=3 par defaut (bonne balance TPR/FPR empiriquement).
  - Module bibliotheque uniquement -- le point d'entree CLI est main.py.
"""

import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from models.autoencoder.model import IDSAutoencoder

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
    "syn_flag", "ack_flag", "fin_flag", "rst_flag", "synack_flag",
    "byte_rate", "pkt_rate", "bytes_ratio", "pkts_ratio", "with_payload",
    "proto_tcp", "proto_udp", "proto_icmp", "proto_other",
]


# ─────────────────────────────────────────────────────────────────────────────
# Chargement des donnees
# ─────────────────────────────────────────────────────────────────────────────
def load_client_data(client_name: str,
                     partitions_dir: str = "datasets/partitions") -> dict:
    """
    Charge train.csv et test.csv pour un client.
    Retourne un dict avec les tenseurs X et les labels y.
    """
    base = os.path.join(partitions_dir, client_name)
    train_df = pd.read_csv(os.path.join(base, "train.csv"))
    test_df  = pd.read_csv(os.path.join(base, "test.csv"))

    missing = [c for c in FEATURE_COLS if c not in train_df.columns]
    if missing:
        raise ValueError(f"[{client_name}] Colonnes manquantes : {missing}")

    X_train = torch.tensor(train_df[FEATURE_COLS].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["label"].values,      dtype=torch.long)
    X_test  = torch.tensor(test_df[FEATURE_COLS].values,  dtype=torch.float32)
    y_test  = torch.tensor(test_df["label"].values,       dtype=torch.long)

    logger.info(
        "[%s] Donnees chargees : train=%d (normal=%d | attaque=%d)  test=%d",
        client_name, len(X_train),
        (y_train == 0).sum().item(), (y_train == 1).sum().item(),
        len(X_test),
    )
    return {"X_train": X_train, "y_train": y_train,
            "X_test":  X_test,  "y_test":  y_test}


# ─────────────────────────────────────────────────────────────────────────────
# Entrainement
# ─────────────────────────────────────────────────────────────────────────────
def train_autoencoder(client_name: str,
                      model: IDSAutoencoder,
                      epochs: int = 20,
                      batch_size: int = 256,
                      lr: float = 1e-3,
                      partitions_dir: str = "datasets/partitions") -> dict:
    """
    Entraine l'autoencoder sur le trafic NORMAL uniquement.

    Retourne un dict avec l'historique de loss et le seuil de detection
    calcule sur le jeu de validation normal (80/20 split interne).
    """
    data = load_client_data(client_name, partitions_dir)
    X_train, y_train = data["X_train"], data["y_train"]

    # ── Isoler UNIQUEMENT le trafic normal pour l'entrainement ───────────────
    normal_mask = (y_train == 0)
    X_normal = X_train[normal_mask]
    n_normal = len(X_normal)

    if n_normal < 2:
        raise RuntimeError(f"[{client_name}] Trop peu de flux normaux ({n_normal}) pour entrainer l'AE.")

    logger.info(
        "[%s] Entrainement AE sur %d flux normaux (%.1f%% du train total)",
        client_name, n_normal, 100 * n_normal / len(X_train),
    )

    # ── Split validation interne (20%) pour calculer le seuil ────────────────
    val_size = max(1, int(0.2 * n_normal))
    idx = torch.randperm(n_normal)
    X_val_normal  = X_normal[idx[:val_size]]
    X_fit_normal  = X_normal[idx[val_size:]]

    loader = DataLoader(
        TensorDataset(X_fit_normal),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    logger.info(
        "[%s] Optimiseur Adam — lr=%.4f  weight_decay=1e-5  lr_fixe=True",
        client_name, lr,
    )

    history = []
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches  = 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            x_hat = model(xb)
            loss  = criterion(x_hat, xb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)
        logger.info("[%s] AE Epoch %d/%d — loss=%.6f", client_name, epoch + 1, epochs, avg_loss)

    # ── Calcul du seuil sur le jeu de validation normal ──────────────────────
    model.eval()
    with torch.no_grad():
        val_errors = model.reconstruction_error(X_val_normal.to(DEVICE)).cpu().numpy()

    threshold = float(val_errors.mean() + 3.0 * val_errors.std())
    logger.info(
        "[%s] Seuil de detection calcule : %.6f  (mean=%.6f  std=%.6f)",
        client_name, threshold, val_errors.mean(), val_errors.std(),
    )

    return {"history": history, "threshold": threshold, "n_normal_train": n_normal}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_autoencoder(client_name: str,
                         model: IDSAutoencoder,
                         threshold: float,
                         partitions_dir: str = "datasets/partitions") -> dict:
    """
    Evalue l'autoencoder sur le jeu de test complet (normal + attaque).
    Rapporte : Accuracy, Precision, Recall, F1, ROC-AUC, matrice de confusion,
               et la distribution des erreurs de reconstruction par classe.
    """
    data = load_client_data(client_name, partitions_dir)
    X_test, y_test = data["X_test"].to(DEVICE), data["y_test"]

    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        errors = model.reconstruction_error(X_test).cpu().numpy()

    y_true = y_test.numpy()
    y_pred = (errors >= threshold).astype(int)

    # Erreurs par classe (utile pour ajuster le seuil)
    err_normal  = errors[y_true == 0]
    err_attack  = errors[y_true == 1]

    metrics = {
        "client":           client_name,
        "threshold":        threshold,
        "n_test":           len(y_true),
        "n_normal_test":    int((y_true == 0).sum()),
        "n_attack_test":    int((y_true == 1).sum()),
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score":         float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":          float(roc_auc_score(y_true, errors)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "recon_error_normal_mean":  float(err_normal.mean()) if len(err_normal) > 0 else None,
        "recon_error_attack_mean":  float(err_attack.mean()) if len(err_attack) > 0 else None,
    }

    logger.info("=== Evaluation AE [%s] ===", client_name)
    logger.info(
        "Threshold=%.6f | Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f | ROC-AUC=%s",
        threshold, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1_score"],
        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A",
    )
    logger.info(
        "Erreur de reconstruction — Normal: %.6f | Attaque: %.6f",
        metrics["recon_error_normal_mean"] or 0,
        metrics["recon_error_attack_mean"] or 0,
    )
    cm = np.array(metrics["confusion_matrix"])
    logger.info("Matrice de confusion [[TN,FP],[FN,TP]] :\n%s", cm)

    return metrics
