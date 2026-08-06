"""
models/autoencoder/model.py
============================
Autoencoder pour detection d'anomalies reseau (zero-day inclus).

Principe FL :
  - Entraine uniquement sur le trafic NORMAL de chaque client local.
  - Apprend a reconstruire le profil "habituel" du reseau de ce client.
  - A l'inference, une erreur de reconstruction elevee (MSE > seuil)
    signale une anomalie : attaque connue OU attaque inedite (zero-day).

Architecture :
  Encoder : input_dim -> 64 -> 32 -> latent_dim (16 par defaut)
  Decoder : latent_dim -> 32 -> 64 -> input_dim
  Activation : ReLU partout sauf la sortie (lineaire, car features scalees)
  Regularisation : BatchNorm + Dropout dans l'encodeur uniquement

Compatibilite FL :
  - get_flat_params() / set_flat_params() : serialisation des poids pour FedAvg
  - count_parameters() : monitoring
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IDSAutoencoder(nn.Module):
    """
    Autoencoder symetrique pour detection d'anomalies reseau.

    Parametres
    ----------
    input_dim   : dimension d'entree (= nombre de features, ex: 19)
    latent_dim  : taille de l'espace latent (goulot d'etranglement)
    dropout     : taux de dropout dans l'encodeur (regularisation)
    """

    def __init__(self, input_dim: int = 19, latent_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim

        # ── Encodeur ─────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, latent_dim),
            nn.ReLU(),            # espace latent positif (regularisation legere)
        )

        # ── Decodeur ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, input_dim),
            # Pas d'activation en sortie : les features sont scalees (~N(0,1)),
            # le decodeur doit pouvoir reconstruire des valeurs negatives.
        )

        logger.debug(
            "IDSAutoencoder initialise : input_dim=%d  latent_dim=%d  dropout=%.2f",
            input_dim, latent_dim, dropout,
        )

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode puis decode x. Retourne la reconstruction."""
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne uniquement la representation latente (pour visualisation/clustering)."""
        return self.encoder(x)

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcule l'erreur de reconstruction MSE par echantillon.
        Retourne un tenseur 1D de taille (batch,).
        Une erreur elevee => flux anormal => anomalie.
        """
        self.eval()
        x_hat = self.forward(x)
        # MSE par ligne (mean sur les features, pas sur le batch)
        return ((x - x_hat) ** 2).mean(dim=1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Prediction binaire : 1 = anomalie (attaque ou zero-day), 0 = normal.
        threshold : calcule au prealable sur le jeu de validation normal.
        """
        errors = self.reconstruction_error(x)
        return (errors >= threshold).long()

    # ── Serialisation FL ──────────────────────────────────────────────────────
    def get_flat_params(self) -> torch.Tensor:
        """Aplatit tous les parametres en un vecteur 1D (pour FedAvg)."""
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Reconstruit les parametres depuis un vecteur 1D (reception FedAvg)."""
        offset = 0
        for p in self.parameters():
            size = p.data.numel()
            p.data.copy_(flat[offset: offset + size].view(p.data.shape))
            offset += size

    def count_parameters(self) -> int:
        """Nombre total de parametres apprenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    ae = IDSAutoencoder(input_dim=19, latent_dim=16)
    print(ae)
    print(f"Parametres : {ae.count_parameters():,}")
    x = torch.randn(8, 19)
    err = ae.reconstruction_error(x)
    print(f"Erreurs de reconstruction (batch=8) : {err}")
