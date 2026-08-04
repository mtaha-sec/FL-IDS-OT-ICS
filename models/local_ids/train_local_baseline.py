"""
train_local_baseline.py
========================
Teste et valide qu'un IDS "local" (entraine sur les donnees d'UN SEUL client,
SANS federation) fonctionne correctement, AVANT de lancer la simulation FL
complete (6 clients + serveur).

Pourquoi ce script est utile :
  - Sanity check rapide : si le modele n'apprend rien en local (accuracy ~50%,
    loss qui ne baisse pas), inutile de lancer le FL -- le probleme vient du
    pipeline de donnees ou du modele, pas de l'agregation.
  - Baseline de reference : sert a comparer, plus tard, "local-only" vs
    "federe" (FedAvg/FedProx) vs "centralise" (tous les clients fusionnes) --
    argument cle a presenter dans le rapport de PFA.
  - Donne des metriques completes (precision, recall, F1, matrice de
    confusion, ROC-AUC), pas seulement l'accuracy affichee par Flower.

Usage:
    python -m models.local_ids.train_local_baseline --client-name power
    python -m models.local_ids.train_local_baseline --client-name power --epochs 15 --save-metrics
    python -m models.local_ids.train_local_baseline --all   # teste les 6 clients d'affilee
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report,
)

from models.local_ids.model import build_model
from preprocessing.feature_engineering import get_model_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALL_CLIENTS = ["power", "utilities", "sap", "pap", "beneficiation", "granulation"]


def load_client_train_test(client_name: str, partitions_dir: str = "datasets/partitions"):
    client_dir = os.path.join(partitions_dir, client_name)
    train_df = pd.read_csv(os.path.join(client_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(client_dir, "test.csv"))

    feature_cols = get_model_feature_columns()
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Colonnes attendues absentes de train.csv pour '{client_name}': {missing}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    return X_train, y_train, X_test, y_test, feature_cols


def train_and_evaluate(client_name: str, model_type: str = "mlp",
                        epochs: int = 10, batch_size: int = 64, lr: float = 1e-3) -> dict:
    X_train, y_train, X_test, y_test, feature_cols = load_client_train_test(client_name)
    input_dim = len(feature_cols)

    logger.info("[%s] %d features, train=%d (attaque=%.1f%%), test=%d (attaque=%.1f%%), model_type=%s",
                client_name, input_dim, len(X_train), 100 * y_train.mean(),
                len(X_test), 100 * y_test.mean(), model_type)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=batch_size, shuffle=False,
    )

    model = build_model(model_type, input_dim=input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Entrainement local (baseline, sans federation) ---
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            logger.info("[%s] epoch %d/%d - train_loss=%.4f", client_name, epoch + 1, epochs, avg_loss)

    # --- Evaluation complete (pas seulement l'accuracy) ---
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(yb.numpy().astype(int))

    metrics = {
        "client": client_name,
        "model_type": model_type,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "attack_ratio_train": float(y_train.mean()),
        "attack_ratio_test": float(y_test.mean()),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else None,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }

    logger.info("=== Resultats [%s / %s] ===", client_name, model_type)
    logger.info("Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f | ROC-AUC=%s",
                metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"],
                f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A")
    logger.info("Matrice de confusion [[TN,FP],[FN,TP]] :\n%s", np.array(metrics["confusion_matrix"]))
    print(classification_report(all_labels, all_preds, target_names=["normal", "attaque"], zero_division=0))

    return metrics


def interpret(metrics: dict) -> None:
    """Alerte lisible si le modele local semble ne pas fonctionner correctement."""
    acc = metrics["accuracy"]
    f1 = metrics["f1_score"]
    tag = f"{metrics['client']}/{metrics['model_type']}"
    if acc < 0.6:
        logger.warning("[%s] ACCURACY TRES FAIBLE (%.2f) -> probleme probable de features/labels, "
                        "verifier le pipeline de preprocessing.", tag, acc)
    elif f1 < 0.5:
        logger.warning("[%s] F1-SCORE FAIBLE (%.2f) malgre une accuracy correcte -> possible "
                        "desequilibre de classes non gere (le modele predit surtout la classe "
                        "majoritaire).", tag, f1)
    else:
        logger.info("[%s] Le modele local apprend correctement (accuracy=%.2f, f1=%.2f).",
                    tag, acc, f1)


def main():
    parser = argparse.ArgumentParser(description="Test du modele IDS local (baseline non-federee)")
    parser.add_argument("--client-name", choices=ALL_CLIENTS, help="Un seul client a tester")
    parser.add_argument("--all", action="store_true", help="Tester les 6 clients d'affilee")
    parser.add_argument("--model-type", choices=["mlp", "logreg", "both"], default="mlp",
                         help="'both' entraine et compare les deux modeles sur chaque client")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save-metrics", action="store_true",
                         help="Sauvegarder les metriques dans results/metrics/")
    args = parser.parse_args()

    if not args.client_name and not args.all:
        parser.error("Preciser --client-name <nom> ou --all")

    clients_to_test = ALL_CLIENTS if args.all else [args.client_name]
    model_types = ["mlp", "logreg"] if args.model_type == "both" else [args.model_type]

    all_results = {}  # cle = "client/model_type"

    for client_name in clients_to_test:
        for model_type in model_types:
            metrics = train_and_evaluate(client_name, model_type=model_type, epochs=args.epochs)
            interpret(metrics)
            all_results[f"{client_name}/{model_type}"] = metrics
            print("-" * 70)

    # Tableau comparatif final si plusieurs modeles testes (--model-type both)
    if len(model_types) > 1:
        logger.info("=" * 70)
        logger.info("COMPARATIF MLP vs LOGREG (F1-score par client)")
        logger.info("=" * 70)
        for client_name in clients_to_test:
            f1_mlp = all_results[f"{client_name}/mlp"]["f1_score"]
            f1_logreg = all_results[f"{client_name}/logreg"]["f1_score"]
            gagnant = "mlp" if f1_mlp > f1_logreg else ("logreg" if f1_logreg > f1_mlp else "egalite")
            logger.info("%-15s | mlp F1=%.4f | logreg F1=%.4f | gagnant=%s",
                        client_name, f1_mlp, f1_logreg, gagnant)

    if args.save_metrics:
        os.makedirs("results/metrics", exist_ok=True)
        out_path = "results/metrics/local_baseline_metrics.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Metriques sauvegardees : %s", out_path)


if __name__ == "__main__":
    main()