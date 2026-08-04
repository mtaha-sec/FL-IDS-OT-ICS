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


class IDSMLP(nn.Module):
    """
    MLP pour classification binaire (label: normal=0 / attaque=1).

    input_dim : nombre de features en entree (doit correspondre exactement a
                len(get_model_feature_columns()) -- voir
                preprocessing/feature_engineering.py::get_model_feature_columns,
                ou simplement appeler models.local_ids.model.get_input_dim().
    """

    def __init__(self, input_dim: int, hidden_dims=(64, 32, 16), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # sortie binaire (logit)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits bruts ; utiliser BCEWithLogitsLoss a l'entrainement


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