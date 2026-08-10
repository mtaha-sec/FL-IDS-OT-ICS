"""
models/global_model/fl_train_client.py
========================================
Script d'entrainement local pour UN ROUND de Federated Learning.

Nouveaute : support FedProx
----------------------------
Chaque client minimise desormais :

    L(w) = F_k(w ; D_k) + (μ/2) · ‖w − w_global‖²

  - F_k      : loss de tache (BCEWithLogitsLoss pondere)
  - μ        : coefficient proximal (0 = FedAvg standard)
  - w_global : poids du modele global recu du serveur (fige pendant le round)

Le terme proximal empeche la derive excessive des clients sur des donnees
Non-IID (chaque site industriel a un profil de trafic tres different).

Workflow d'un round FL :
  1. Le serveur envoie le modele global (fl_model.pt) a chaque client.
  2. Chaque client charge ses donnees locales et fait N epochs
     d'entrainement avec le terme proximal FedProx.
  3. Chaque client chiffre ses poids avec la cle HE et les renvoie au serveur.
  4. Le serveur effectue l'agregation DANS LE DOMAINE CHIFFRE (HE) et
     produit le nouveau modele global chiffre.
  5. Les clients dechiffrent le modele global avec leur cle secrete.

Usage :
    # Round 0 (initialisation) : sans modele global
    python -m models.global_model.fl_train_client --client power --save

    # Rounds suivants : avec modele global recu du serveur
    python -m models.global_model.fl_train_client \\
        --client power \\
        --global-model checkpoints/fl/global_round_3.pt \\
        --mu 0.01 \\
        --save

    # Tous les clients en un seul appel (simulation locale)
    python -m models.global_model.fl_train_client --all --mu 0.01 --save
"""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.global_model.fl_model import build_fl_model, FL_INPUT_DIM
from preprocessing.feature_engineering import get_model_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_CLIENTS = ["power", "utilities", "sap", "pap", "beneficiation", "granulation"]

# ─── Configs par client ──────────────────────────────────────────────────────
EPOCHS_CONFIG   = {"power": 25, "utilities": 25, "sap": 15,
                   "pap":   20, "beneficiation": 30, "granulation": 20}
LR_CONFIG       = {"power": 1e-3, "utilities": 1e-3, "sap": 1e-3,
                   "pap":   1e-3, "beneficiation": 5e-4, "granulation": 1e-3}
BATCH_CONFIG    = {"power": 64, "utilities": 64,  "sap": 128,
                   "pap":   256, "beneficiation": 512, "granulation": 1024}
PATIENCE_CONFIG = {"power": 8, "utilities": 8, "sap": 5,
                   "pap":   5, "beneficiation": 10, "granulation": 5}


def load_client_data(
    client_name: str,
    partitions_dir: str = "datasets/partitions",
):
    client_dir   = os.path.join(partitions_dir, client_name)
    train_df     = pd.read_csv(os.path.join(client_dir, "train.csv"))
    test_df      = pd.read_csv(os.path.join(client_dir, "test.csv"))
    feature_cols = get_model_feature_columns()

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour '{client_name}': {missing}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.float32)
    return X_train, y_train, X_test, y_test


def train_fl_client(
    client_name:        str,
    global_model_path:  Optional[str] = None,
    partitions_dir:     str   = "datasets/partitions",
    save_dir:           str   = "checkpoints/fl",
    save:               bool  = False,
    epochs:             Optional[int]   = None,
    lr:                 Optional[float] = None,
    batch_size:         Optional[int]   = None,
    patience:           Optional[int]   = None,
    mu:                 float = 0.0,
) -> Dict:
    """
    Entraine le FLIDSModel (architecture unique) sur les donnees d'un client.

    Parametres
    ----------
    client_name       : nom du client (ex: 'power')
    global_model_path : chemin vers le modele global recu du serveur (.pt)
                        Si None, initialise aleatoirement (round 0).
    partitions_dir    : dossier racine des partitions
    save_dir          : ou sauvegarder le checkpoint local post-entrainement
    save              : si True, sauvegarde le checkpoint
    epochs/lr/batch_size/patience : surcharge les configs par defaut
    mu                : coefficient proximal FedProx (0 = FedAvg standard)
                        Ajoute (μ/2)‖w − w_global‖² a la loss de tache.
    """

    # ── Configs par defaut par client ────────────────────────────────────────
    c_epochs   = epochs     or EPOCHS_CONFIG.get(client_name, 10)
    c_lr       = lr         or LR_CONFIG.get(client_name, 1e-3)
    c_batch    = batch_size or BATCH_CONFIG.get(client_name, 64)
    c_patience = patience   or PATIENCE_CONFIG.get(client_name, 5)

    # ── Chargement des donnees ────────────────────────────────────────────────
    X_all, y_all, X_test, y_test = load_client_data(client_name, partitions_dir)

    # A4 — Split validation interne (10%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, stratify=y_all
    )

    n_normal = int((y_train == 0).sum())
    n_attack = int((y_train == 1).sum())
    pos_w    = n_normal / max(n_attack, 1)

    logger.info(
        "[%s] train=%d | val=%d | test=%d | attaque=%.1f%% | pos_w=%.3f | "
        "epochs=%d | lr=%s | batch=%d | patience=%d | arch=%s | μ=%.4f",
        client_name, len(X_train), len(X_val), len(X_test),
        100 * y_all.mean(), pos_w,
        c_epochs, c_lr, c_batch, c_patience, "(64→32→16)", mu,
    )

    # ── Modele : charge depuis le serveur ou initialise alea ─────────────────
    model = build_fl_model().to(DEVICE)
    if global_model_path and os.path.exists(global_model_path):
        ckpt = torch.load(global_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("[%s] Modele global charge : %s", client_name, global_model_path)
    else:
        logger.info("[%s] Initialisation aleatoire (round 0)", client_name)

    # ── Frozen copy of global weights for FedProx proximal term ──────────────
    global_weights: Optional[List[torch.Tensor]] = None
    if mu > 0.0 and global_model_path and os.path.exists(global_model_path):
        global_weights = [p.data.clone().to(DEVICE) for p in model.parameters()]
        logger.info("[%s] FedProx — poids globaux figes, μ=%.4f", client_name, mu)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=c_batch, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=c_batch, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=c_batch, shuffle=False,
    )

    pos_weight_tensor = torch.tensor([pos_w], dtype=torch.float32).to(DEVICE)
    criterion         = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer         = torch.optim.Adam(model.parameters(), lr=c_lr)

    # ── Entrainement avec Early Stopping (A4) + FedProx ──────────────────────
    best_val_loss = float("inf")
    patience_ctr  = 0
    best_state    = None

    for epoch in range(c_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            task_loss = criterion(model(xb).squeeze(1), yb)

            # FedProx: proximal term  (μ/2)‖w − w_global‖²
            if global_weights is not None:
                prox_loss = (mu / 2.0) * sum(
                    ((p - g) ** 2).sum()
                    for p, g in zip(model.parameters(), global_weights)
                )
                loss = task_loss + prox_loss
            else:
                loss = task_loss

            loss.backward()
            optimizer.step()
            train_loss += task_loss.item() * xb.size(0)   # log task loss only
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb).squeeze(1), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        log_every = max(1, c_epochs // 5)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            logger.info(
                "[%s] epoch %02d/%d  train=%.4f  val=%.4f  μ=%.4f",
                client_name, epoch + 1, c_epochs, train_loss, val_loss, mu,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= c_patience:
                logger.info(
                    "[%s] Early stopping @ epoch %d (best_val=%.4f)",
                    client_name, epoch + 1, best_val_loss,
                )
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── Calibration du seuil (A5) ─────────────────────────────────────────────
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            probs = torch.sigmoid(model(xb.to(DEVICE)).squeeze(1)).cpu().numpy()
            val_probs.extend(probs)
            val_true.extend(yb.numpy().astype(int))

    val_probs = np.array(val_probs)
    val_true  = np.array(val_true)
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.10, 0.91, 0.05):
        f1 = f1_score(val_true, (val_probs > thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    logger.info("[%s] Seuil optimal : %.2f  (F1_val=%.4f)", client_name, best_thr, best_f1)

    # ── Evaluation sur le test set ────────────────────────────────────────────
    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = torch.sigmoid(model(xb.to(DEVICE)).squeeze(1)).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs > best_thr).astype(int))
            all_true.extend(yb.numpy().astype(int))

    metrics = {
        "client":          client_name,
        "fedprox_mu":      mu,
        "n_train":         len(X_all),
        "n_test":          len(X_test),
        "pos_weight":      pos_w,
        "best_threshold":  best_thr,
        "accuracy":        float(accuracy_score(all_true, all_preds)),
        "precision":       float(precision_score(all_true, all_preds, zero_division=0)),
        "recall":          float(recall_score(all_true, all_preds, zero_division=0)),
        "f1_score":        float(f1_score(all_true, all_preds, zero_division=0)),
        "roc_auc":         (float(roc_auc_score(all_true, all_probs))
                            if len(set(all_true)) > 1 else None),
        "confusion_matrix": confusion_matrix(all_true, all_preds).tolist(),
        "architecture":    {"input": FL_INPUT_DIM, "hidden": [64, 32, 16], "dropout": 0.2},
    }

    logger.info(
        "=== [%s] FL-Client Results === "
        "Acc=%.4f | Prec=%.4f | Rec=%.4f | F1=%.4f | AUC=%s | μ=%.4f",
        client_name,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"],   metrics["f1_score"],
        f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "N/A",
        mu,
    )
    logger.info("Conf [[TN,FP],[FN,TP]]:\n%s", np.array(metrics["confusion_matrix"]))
    print(classification_report(all_true, all_preds,
                                target_names=["normal", "attaque"], zero_division=0))

    # ── Sauvegarde checkpoint client (a envoyer au serveur) ───────────────────
    if save:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"client_{client_name}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "client_name":      client_name,
            "n_train":          len(X_all),
            "best_threshold":   best_thr,
            "fedprox_mu":       mu,
            "metrics":          metrics,
            "fl_arch": {
                "input_dim":   FL_INPUT_DIM,
                "hidden_dims": (64, 32, 16),
                "dropout":     0.2,
            },
        }, ckpt_path)
        logger.info("[%s] Checkpoint FL sauvegarde : %s", client_name, ckpt_path)
        metrics["checkpoint_path"] = ckpt_path

    metrics["_model"] = model
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Entrainement FL local (architecture unique figee) avec FedProx"
    )
    parser.add_argument("--client",       choices=ALL_CLIENTS)
    parser.add_argument("--all",          action="store_true")
    parser.add_argument("--global-model", default=None,
                        help="Chemin vers le modele global recu du serveur FL")
    parser.add_argument("--partitions",   default="datasets/partitions")
    parser.add_argument("--save-dir",     default="checkpoints/fl")
    parser.add_argument("--save",         action="store_true")
    parser.add_argument("--save-metrics", action="store_true")
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--batch-size",   type=int,   default=None)
    parser.add_argument(
        "--mu", type=float, default=0.0,
        help="FedProx proximal coefficient μ (0 = FedAvg standard). "
             "Recommended range: 0.001 – 0.1 for Non-IID FL.",
    )
    args = parser.parse_args()

    if not args.client and not args.all:
        parser.error("Preciser --client <nom> ou --all")

    if args.mu > 0.0 and (not args.global_model or not os.path.exists(args.global_model)):
        logger.warning(
            "FedProx μ=%.4f specified but no global model found at '%s'. "
            "Proximal term will be DISABLED for this round (round 0 init).",
            args.mu, args.global_model,
        )

    clients     = ALL_CLIENTS if args.all else [args.client]
    all_results = {}

    for client in clients:
        m = train_fl_client(
            client_name       = client,
            global_model_path = args.global_model,
            partitions_dir    = args.partitions,
            save_dir          = args.save_dir,
            save              = args.save,
            epochs            = args.epochs,
            lr                = args.lr,
            batch_size        = args.batch_size,
            mu                = args.mu,
        )
        all_results[client] = {k: v for k, v in m.items() if k != "_model"}
        print("-" * 70)

    # ── Tableau recap ────────────────────────────────────────────────────────
    if len(clients) > 1:
        logger.info("=" * 70)
        logger.info("RECAP FL LOCAL — Architecture unique 19→64→32→16→1  (μ=%.4f)", args.mu)
        logger.info("=" * 70)
        logger.info("%-15s %8s %8s %8s %8s %6s", "Client", "Acc", "Prec", "Recall", "F1", "Seuil")
        logger.info("-" * 70)
        for c, m in all_results.items():
            logger.info(
                "%-15s %8.4f %8.4f %8.4f %8.4f %6.2f",
                c, m["accuracy"], m["precision"],
                m["recall"], m["f1_score"], m["best_threshold"],
            )

    if args.save_metrics:
        os.makedirs("results/metrics", exist_ok=True)
        out = "results/metrics/fl_client_metrics.json"
        with open(out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info("Metriques FL sauvegardees : %s", out)


if __name__ == "__main__":
    main()
