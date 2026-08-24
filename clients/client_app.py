"""
client_app.py
==============
Client Flower GENERIQUE, instancie une fois par site industriel :
    beneficiation, sap, pap, power, utilities, granulation

Chaque client charge UNIQUEMENT ses propres partitions locales
(datasets/partitions/<client_name>/{train,test}.csv) -- jamais les donnees des
autres clients, conformement au principe du federated learning.

Usage (a lancer separement, une fois par client, avec le nom en argument) :
    python clients/client_app.py --client-name power   --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name sap      --server-address 127.0.0.1:8080
    ... (x6, ou orchestre via docker-compose / simulation Flower)
"""

import argparse
import logging
import os

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models.local_ids.model import build_model, get_input_dim

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_client_data(client_name: str, partitions_dir: str = "datasets/partitions"):
    """Charge train.csv / test.csv du client et les convertit en tenseurs."""
    client_dir = os.path.join(partitions_dir, client_name)
    train_df = pd.read_csv(os.path.join(client_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(client_dir, "test.csv"))

    # Colonnes a exclure du vecteur d'entree (IP, labels, metadonnees)
    from preprocessing.feature_engineering import get_model_feature_columns

    feature_cols = get_model_feature_columns()
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Colonnes attendues absentes de train.csv pour '{client_name}': {missing}")

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    logger.info("Client '%s' : %d features, train=%d, test=%d",
                client_name, len(feature_cols), len(X_train), len(X_test))

    return X_train, y_train, X_test, y_test, feature_cols


class IDSFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_name: str, model_type: str = "mlp",
                 batch_size: int = 64, local_epochs: int = 3, lr: float = 1e-3):
        self.client_name = client_name
        self.model_type = model_type
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        X_train, y_train, X_test, y_test, feature_cols = load_client_data(client_name)
        self.input_dim = len(feature_cols)

        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=batch_size, shuffle=True,
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=batch_size, shuffle=False,
        )

        # IMPORTANT (contrainte FedAvg/FedProx) : model_type doit etre IDENTIQUE
        # sur les 6 clients ET sur le serveur (meme architecture partout, sinon
        # l'agregation des poids echoue ou produit un modele incoherent).
        self.model = build_model(model_type, input_dim=self.input_dim).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # --- Interface Flower ---------------------------------------------------
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(xb).squeeze(1)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            logger.info("[%s] epoch %d/%d - loss=%.4f",
                        self.client_name, epoch + 1, self.local_epochs,
                        epoch_loss / len(self.train_loader.dataset))

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total = 0.0, 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = self.model(xb).squeeze(1)
                loss = self.criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                total += yb.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        
        # Calcul des métriques avec sklearn (zero_division=0 pour éviter les warnings si le modèle prédit tout dans une classe)
        accuracy = sum([p == l for p, l in zip(all_preds, all_labels)]) / total if total > 0 else 0.0
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        logger.info("[%s] evaluation locale - loss=%.4f, accuracy=%.4f",
                    self.client_name, avg_loss, accuracy)
                    
        return avg_loss, total, {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }


def main():
    parser = argparse.ArgumentParser(description="Client Flower - site industriel")
    parser.add_argument("--client-name", required=True,
                         choices=["beneficiation", "sap", "pap", "power", "utilities", "granulation"])
    parser.add_argument("--model-type", choices=["mlp", "logreg"], default="mlp",
                         help="Doit etre IDENTIQUE sur les 6 clients et le serveur")
    parser.add_argument("--server-address", default="127.0.0.1:8085")
    parser.add_argument("--local-epochs", type=int, default=3)
    args = parser.parse_args()

    client = IDSFlowerClient(client_name=args.client_name, model_type=args.model_type,
                              local_epochs=args.local_epochs)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()