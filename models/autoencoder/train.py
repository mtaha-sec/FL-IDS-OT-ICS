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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
# ── Configs par client ────────────────────────────────────────────────────────────
EPOCHS_CONFIG   = {"power": 40, "utilities": 40, "sap": 50,
                   "pap": 50, "beneficiation": 50, "granulation": 40}
BATCH_CONFIG    = {"power": 128, "utilities": 128, "sap": 256,
                   "pap": 512, "beneficiation": 512, "granulation": 1024}
PATIENCE_CONFIG = {"power": 15, "utilities": 15, "sap": 10,
                   "pap": 10, "beneficiation": 20, "granulation": 12}
# Optimisation cle : latent_dim reduit pour forcer une compression plus agressive
# sur les clients dont le trafic normal est difficile a distinguer des attaques.
# Ratio cible : latent_dim/input_dim < 50% pour un vrai goulot d'etranglement.
LATENT_DIM_CONFIG = {
    "power":         8,   # 8/19 = 42% compression  (v1=16, F1=0.666)
    "utilities":    16,   # 16/19 = 84% (deja bon, ROC-AUC=0.90)
    "sap":          16,   # 16/19 = 84% (excellent, F1=0.966)
    "pap":          16,   # 16/19 = 84% (excellent, F1=0.955)
    "beneficiation": 6,   # 6/19 = 31% compression  (v1=16, ROC-AUC=0.499 !)
    "granulation":   8,   # 8/19 = 42% compression  (v1=16, F1=0.773)
}


def _find_optimal_threshold(errors: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Cherche le seuil qui maximise le F1-Score sur un ensemble de validation
    mixte (normal + attaque).

    Commence par le 95e percentile des erreurs normales comme point de depart,
    puis balaye [min_attack, max_attack] pour trouver le maximum F1.
    Robuste aux outliers (pas de std qui explose).

    Retourne : (best_threshold, best_f1, percentile_threshold)
    """
    err_normal = errors[labels == 0]
    err_attack = errors[labels == 1]

    # Percentile robuste sur les flux normaux
    p95_normal = float(np.percentile(err_normal, 95)) if len(err_normal) else 0.01

    # Sweep sur une grille entre le 5e percentile attaque et le 99e percentile attaque
    attack_lo = float(np.percentile(err_attack, 5))  if len(err_attack) else p95_normal * 0.5
    attack_hi = float(np.percentile(err_attack, 99)) if len(err_attack) else p95_normal * 3
    candidates = np.unique(np.concatenate([
        np.linspace(attack_lo, attack_hi, 100),
        [p95_normal],
    ]))

    best_thr, best_f1 = p95_normal, 0.0
    for thr in candidates:
        preds = (errors >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    return best_thr, best_f1, p95_normal


# ───────────────────────────────────────────────────────────────────────────────
# Entrainement
# ───────────────────────────────────────────────────────────────────────────────
def train_autoencoder(client_name: str,
                      model: IDSAutoencoder,
                      epochs: int = None,
                      batch_size: int = None,
                      lr: float = 1e-3,
                      patience: int = None,
                      partitions_dir: str = "datasets/partitions") -> dict:
    """
    Entraine l'autoencoder sur le trafic NORMAL uniquement.

    Ameliorations v3 :
      - latent_dim adaptatif par client (beneficiation 16->6, power/granulation 16->8)
      - ReduceLROnPlateau : reduit le LR x10 si val_loss stagne 5 epochs
      - Loss combinee MSE + MAE : moins sensible aux valeurs aberrantes
      - Early stopping, gradient clipping, sweep F1 (conserves de v2)
    """
    # ── Configs par defaut par client ──
    c_epochs   = epochs     or EPOCHS_CONFIG.get(client_name, 30)
    c_batch    = batch_size or BATCH_CONFIG.get(client_name, 256)
    c_patience = patience   or PATIENCE_CONFIG.get(client_name, 8)

    data = load_client_data(client_name, partitions_dir)
    X_all, y_all = data["X_train"], data["y_train"]

    # ── Isoler le trafic normal ─────────────────────────────────────────────────
    normal_mask  = (y_all == 0)
    attack_mask  = (y_all == 1)
    X_normal     = X_all[normal_mask]
    X_attack_val = X_all[attack_mask]       # attaques du train -> val calibration
    n_normal     = len(X_normal)

    if n_normal < 10:
        raise RuntimeError(f"[{client_name}] Trop peu de flux normaux ({n_normal}).")

    logger.info(
        "[%s] AE v2 — %d flux normaux pour entrainement | epochs=%d | batch=%d | patience=%d",
        client_name, n_normal, c_epochs, c_batch, c_patience,
    )

    # ── Split normal : 80% train, 20% val ─────────────────────────────────────────
    X_norm_np = X_normal.numpy()
    idx_tr, idx_va = train_test_split(
        np.arange(n_normal), test_size=0.2, random_state=42
    )
    X_fit_normal = torch.tensor(X_norm_np[idx_tr], dtype=torch.float32)
    X_val_normal = torch.tensor(X_norm_np[idx_va], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_fit_normal),
        batch_size=c_batch, shuffle=True, drop_last=False,
    )

    model     = model.to(DEVICE)
    # Loss combinee MSE + MAE : MSE penalise les grandes erreurs,
    # MAE est robuste aux outliers -> meilleure separation normal/attaque
    mse_loss  = nn.MSELoss()
    mae_loss  = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # ReduceLROnPlateau : divise le LR par 10 si val_loss ne baisse pas
    # pendant 5 epochs consecutives. Evite les oscillations de pap/beneficiation.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, min_lr=1e-6
    )

    logger.info("[%s] Adam lr=%.4f | weight_decay=1e-5 | grad_clip=1.0 | ReduceLROnPlateau", client_name, lr)

    # ── Entrainement avec Early Stopping ─────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_ctr   = 0
    best_state     = None
    history        = []

    for epoch in range(c_epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            x_hat = model(xb)
            # Loss combinee : 0.7*MSE + 0.3*MAE
            loss  = 0.7 * mse_loss(x_hat, xb) + 0.3 * mae_loss(x_hat, xb)
            loss.backward()
            # Gradient clipping : evite les spikes (observes sur beneficiation ep.26)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        history.append(avg_train_loss)

        # ── Validation loss ──
        model.eval()
        with torch.no_grad():
            val_out  = model(X_val_normal.to(DEVICE))
            val_loss = (0.7 * mse_loss(val_out, X_val_normal.to(DEVICE))
                        + 0.3 * mae_loss(val_out, X_val_normal.to(DEVICE))).item()

        # ReduceLROnPlateau step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        log_every = max(1, c_epochs // 10)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            logger.info("[%s] AE Epoch %d/%d  train=%.6f  val=%.6f  lr=%.2e",
                        client_name, epoch + 1, c_epochs, avg_train_loss, val_loss, current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= c_patience:
                logger.info("[%s] Early stopping @ epoch %d (best_val=%.6f)",
                            client_name, epoch + 1, best_val_loss)
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── Calibration du seuil par sweep F1 ────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        err_val_normal = model.reconstruction_error(X_val_normal.to(DEVICE)).cpu().numpy()

    # Prendre un sous-ensemble d'attaques du train pour calibrer le seuil
    n_atk_cal = min(len(X_attack_val), max(500, len(X_attack_val) // 5))
    atk_idx   = np.random.choice(len(X_attack_val), n_atk_cal, replace=False)
    X_atk_cal = X_attack_val[atk_idx]
    with torch.no_grad():
        err_val_attack = model.reconstruction_error(X_atk_cal.to(DEVICE)).cpu().numpy()

    # Val mixte pour le sweep
    err_cal    = np.concatenate([err_val_normal, err_val_attack])
    labels_cal = np.concatenate([np.zeros(len(err_val_normal)), np.ones(len(err_val_attack))])

    best_thr, best_f1_cal, p95 = _find_optimal_threshold(err_cal, labels_cal)

    logger.info(
        "[%s] Seuil : p95_normal=%.6f | optimal_F1=%.6f (F1_cal=%.4f) | "
        "mean_err_normal=%.6f | std=%.6f",
        client_name, p95, best_thr, best_f1_cal,
        err_val_normal.mean(), err_val_normal.std(),
    )

    return {
        "history":        history,
        "threshold":      best_thr,
        "p95_threshold":  p95,
        "n_normal_train": n_normal,
        "best_val_loss":  best_val_loss,
        "cal_f1":         best_f1_cal,
    }


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
