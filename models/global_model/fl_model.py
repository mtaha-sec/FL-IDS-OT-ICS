"""
models/global_model/fl_model.py
================================
Architecture UNIQUE et FIGEE utilisee pour le Federated Learning.

Contrainte fondamentale de FedAvg :
  Tous les clients ET le serveur doivent partager EXACTEMENT la meme
  architecture (memes couches, memes dimensions). L'agregation FedAvg
  consiste a calculer une moyenne ponderee des poids :

      w_global = sum(n_k / N * w_k)   pour k in clients

  Si client_1 a une couche Linear(64, 32) et client_2 a Linear(128, 64),
  la moyenne est impossible (les tenseurs n'ont pas la meme forme).

Architecture retenue : 19 -> 80 -> 40 -> 20 -> 1
  - 19  : features d'entree (19 colonnes standardisees post-preprocessing)
  - 64  : premiere couche cachee (representation intermediaire)
  - 32  : deuxieme couche cachee (compression)
  - 16  : troisieme couche cachee (espace latent compact)
  - 1   : sortie logit binaire (normal=0 / attaque=1)

Pourquoi ce compromis ?
  - Pas trop profond -> evite le sur-apprentissage sur les petits clients
    (power : 22 283 flux, utilities : 11 815 flux)
  - Suffisamment expressif -> capture les patterns complexes sur les
    grands clients (granulation : 493 793 flux)
  - BatchNorm1d apres chaque couche -> stabilise l'entrainement distribue
    (chaque client a une distribution de donnees differente / Non-IID)
  - Dropout=0.2 -> regularisation moderee, equilibre entre tous les clients

Parametres totaux : 19*64 + 64 + 64*32 + 32 + 32*16 + 16 + 16*1 + 1
                  = 1216 + 64 + 2048 + 32 + 512 + 16 + 16 + 1 = 3 905 params
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  Constantes de l'architecture FL — NE PAS MODIFIER sans adapter TOUS les
#  checkpoints existants et le serveur d'agregation.
# ─────────────────────────────────────────────────────────────────────────────
FL_INPUT_DIM   : int   = 19
FL_HIDDEN_DIMS : tuple = (80, 40, 20)
FL_DROPOUT     : float = 0.2


class FLIDSModel(nn.Module):
    """
    Modele MLP unique partage entre tous les clients FL et le serveur.

    Architecture : BatchNorm -> Linear -> BatchNorm -> ReLU -> Dropout
    (le BatchNorm avant l'activation accelere la convergence sur donnees
    tabulaires heterogenes).

    Methodes FL-ready :
      get_flat_params()      -> torch.Tensor  (concatenation de tous les poids)
      set_flat_params(flat)  -> None          (injection depuis le serveur)
      count_parameters()     -> int
    """

    def __init__(self,
                 input_dim:   int   = FL_INPUT_DIM,
                 hidden_dims: tuple = FL_HIDDEN_DIMS,
                 dropout:     float = FL_DROPOUT):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))   # logit binaire

        self.net = nn.Sequential(*layers)

        # Initialisation Xavier pour une convergence rapide
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne un logit brut — utiliser BCEWithLogitsLoss a l'entrainement."""
        return self.net(x)

    # ── Methodes FL ──────────────────────────────────────────────────────────

    def get_flat_params(self) -> torch.Tensor:
        """
        Retourne tous les parametres entrainables concatenes en un seul
        vecteur 1D. Utilise par le serveur pour recevoir et stocker les
        deltas de chaque client.
        """
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """
        Injecte un vecteur de parametres 1D dans le modele.
        Utilise par chaque client pour recevoir le modele global du serveur.
        """
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset: offset + numel].view_as(p))
            offset += numel

    def get_state_dict_copy(self) -> dict:
        """Retourne une copie profonde du state_dict (pour l'agregation)."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"FLIDSModel(input={FL_INPUT_DIM}, hidden={FL_HIDDEN_DIMS}, "
            f"dropout={FL_DROPOUT}, params={self.count_parameters()})"
        )


def build_fl_model() -> FLIDSModel:
    """
    Factory unique : construit le modele FL avec l'architecture figee.
    A utiliser partout dans le pipeline FL (clients ET serveur) pour
    garantir la coherence des dimensions.
    """
    return FLIDSModel(
        input_dim   = FL_INPUT_DIM,
        hidden_dims = FL_HIDDEN_DIMS,
        dropout     = FL_DROPOUT,
    )


if __name__ == "__main__":
    model = build_fl_model()
    print(model)
    print(f"\nNombre total de parametres : {model.count_parameters()}")
    # Test forward pass
    x = torch.randn(8, FL_INPUT_DIM)
    out = model(x)
    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}  (logits bruts)")
    # Test FL methods
    flat = model.get_flat_params()
    print(f"Flat params  : {flat.shape[0]} valeurs")
