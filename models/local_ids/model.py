"""
model.py
=========
Architecture du modele d'IDS local, PARTAGEE et IDENTIQUE entre tous les
clients (contrainte obligatoire du federated learning horizontal, cf.
discussion sur le schema commun : FedAvg moyenne les poids, donc l'architecture
et la dimension d'entree doivent etre strictement identiques partout).

MLP simple, adapte a des donnees tabulaires reseau (pas besoin de CNN/RNN ici).
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture unique FL — NE PAS MODIFIER ces constantes sans regenerer
#  tous les checkpoints clients et le modele global du serveur.
# ─────────────────────────────────────────────────────────────────────────────
FL_INPUT_DIM:   int   = 19
FL_HIDDEN_DIMS: tuple = (80, 40, 20)
FL_DROPOUT:     float = 0.2


class IDSMLP(nn.Module):
    """
    MLP pour classification binaire (normal=0 / attaque=1).

    Architecture unique et figee pour le Federated Learning :
        19 -> 80 -> 40 -> 20 -> 1

    Pourquoi cette architecture ?
      - Suffisamment profonde pour capturer des patterns d'attaque
        complexes sur les grands clients (granulation, beneficiation).
      - Suffisamment legere pour ne pas sur-apprendre sur les petits
        clients (power: 22k flux, utilities: 11k flux).
      - BatchNorm1d apres chaque couche : stabilise l'entrainement
        distribue (les clients ont des distributions Non-IID differentes).
      - Xavier init : convergence rapide des le round 1 du FL.

    Contrainte FL (FedAvg) :
      TOUS les clients et le serveur utilisent cette meme classe avec
      les memes dimensions. FedAvg moyenne les poids layer par layer ;
      si les shapes different, la moyenne est impossible.
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
        layers.append(nn.Linear(prev, 1))   # logit binaire brut
        self.net = nn.Sequential(*layers)

        # Initialisation Xavier pour une convergence rapide des le round 0
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne un logit brut — utiliser BCEWithLogitsLoss a l'entrainement."""
        return self.net(x)

    # ── Methodes FL-ready ─────────────────────────────────────────────────────

    def get_flat_params(self) -> torch.Tensor:
        """Concatene tous les parametres entrainables en un vecteur 1D.
        Utilise par le serveur FL pour collecter les deltas de chaque client."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Injecte un vecteur 1D dans le modele (modele global -> client).
        Utilise au debut de chaque round FL pour distribuer le modele global."""
        offset = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(flat[offset: offset + n].view_as(p))
            offset += n

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IDSLogisticRegression(nn.Module):
    """
    Regression logistique pour classification binaire -- baseline volontairement
    simple, servant de point de comparaison au MLP.

    Une seule couche lineaire, AUCUNE non-linearite intermediaire : le modele
    ne peut apprendre qu'une frontiere de decision lineaire dans l'espace des
    19 features (logit = w.x + b). Sert a repondre a la question "les
    attaques sont-elles separables du trafic normal par une simple
    combinaison lineaire des features, ou la non-linearite du MLP est-elle
    necessaire ?"

    Meme interface que IDSMLP (constructeur avec input_dim, forward retourne
    un logit brut) pour rester interchangeable partout ou IDSMLP est utilise
    (train_local_baseline.py, clients/client_app.py).
    """

    def __init__(self, input_dim: int, **kwargs):
        # **kwargs absorbe hidden_dims/dropout si jamais passes par erreur
        # depuis un appel generique -- ignores ici, non pertinents pour un
        # modele lineaire.
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)  # logit brut, meme convention que IDSMLP


MODEL_REGISTRY = {
    "mlp": IDSMLP,
    "logreg": IDSLogisticRegression,
}


def build_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory centralisee : construit le modele demande par son nom, plutot que
    d'importer IDSMLP/IDSLogisticRegression directement partout. Permet de
    changer de modele avec un simple --model-type en ligne de commande, sans
    toucher au code de train_local_baseline.py / client_app.py.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"model_type inconnu : '{model_type}'. Options : {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](input_dim=input_dim, **kwargs)


def get_input_dim() -> int:
    """
    Calcule dynamiquement la dimension d'entree a partir des colonnes de
    features definies dans le pipeline de preprocessing, pour eviter toute
    incoherence manuelle entre le modele et les donnees.

    get_model_feature_columns() retourne DEJA la liste finale post-encodage
    (proto_tcp/proto_udp/proto_icmp/proto_other inclus, attack_type exclu
    car ce n'est pas une feature d'entree -- cf. feature_engineering.py).
    """
    from preprocessing.feature_engineering import get_model_feature_columns
    return len(get_model_feature_columns())


if __name__ == "__main__":
    dim = get_input_dim()
    model = IDSMLP(input_dim=dim)
    print(f"Dimension d'entree calculee : {dim}")
    print(model)