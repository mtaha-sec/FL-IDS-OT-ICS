"""
partitioning.py
================
Repartit les 3 datasets harmonises (ICS-Flow, TON_IoT, X-IIoTID) entre 6
"clients industriels" simulant des sections d'un site industriel (ex. site de
transformation de phosphate) :

    beneficiation, sap, pap, power, utilities, granulation

Strategie de repartition (Option A du plan initial : partitionnement naturel,
non-IID) :
  - ICS-Flow  (trafic OT/reseau industriel)   -> power, utilities
  - TON_IoT   (trafic IoT / reseau IT classique) -> sap, pap
  - X-IIoTID  (trafic IIoT industriel)         -> beneficiation, granulation

Chaque dataset assigne a 2 clients est ensuite scinde en 2 parts non-IID par
regroupement d'adresses IP source (simulateur de "sous-reseau" par site), ce
qui garantit une heterogeneite realiste entre les 2 clients d'un meme dataset,
sans jamais utiliser l'IP comme feature d'entree du modele (cf. Remarque 1).
"""

import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLIENT_DATASET_MAP = {
    "power": "ics_flow",
    "utilities": "ics_flow",
    "sap": "ton_iot",
    "pap": "ton_iot",
    "beneficiation": "x_iiotid",
    "granulation": "x_iiotid",
}

DATASET_CLIENTS = {
    "ics_flow": ["power", "utilities"],
    "ton_iot": ["sap", "pap"],
    "x_iiotid": ["beneficiation", "granulation"],
}


def split_by_src_ip(df: pd.DataFrame, n_splits: int, seed: int = 42) -> list:
    """
    Scinde un DataFrame en n_splits parts non-IID en regroupant les IP sources
    uniques (chaque IP va entierement dans une seule part -> simule des
    sous-reseaux/hotes differents par client, plus realiste qu'un split
    purement aleatoire ligne par ligne).
    """
    rng = np.random.default_rng(seed)
    unique_ips = df["src_ip"].dropna().unique()
    rng.shuffle(unique_ips)

    ip_groups = np.array_split(unique_ips, n_splits)
    splits = []
    for group in ip_groups:
        part = df[df["src_ip"].isin(group)].copy()
        splits.append(part)

    logger.info("Split par src_ip en %d parts : tailles = %s",
                n_splits, [len(s) for s in splits])
    return splits


def build_client_partitions(processed_datasets: dict, output_dir: str, seed: int = 42) -> dict:
    """
    processed_datasets : dict {"ics_flow": df, "ton_iot": df, "x_iiotid": df}
                          (DataFrames deja passes par cleaning + feature_engineering
                          + normalization, PRETS pour l'entrainement)
    output_dir          : dossier racine (ex. "datasets/partitions")

    Retourne un dict {client_name: DataFrame} et ecrit un CSV par client dans
    <output_dir>/<client_name>/data.csv
    """
    client_data = {}

    for dataset_name, clients in DATASET_CLIENTS.items():
        df = processed_datasets[dataset_name]
        parts = split_by_src_ip(df, n_splits=len(clients), seed=seed)
        for client_name, part in zip(clients, parts):
            client_data[client_name] = part

    for client_name, df in client_data.items():
        client_dir = os.path.join(output_dir, client_name)
        os.makedirs(client_dir, exist_ok=True)
        out_path = os.path.join(client_dir, "data.csv")
        df.to_csv(out_path, index=False)
        logger.info("Client '%s' : %d lignes -> %s", client_name, len(df), out_path)

    return client_data


def train_test_split_per_client(client_data: dict, test_size: float = 0.2, seed: int = 42) -> dict:
    """
    Split train/test INDEPENDANT par client (chaque client garde son propre
    jeu de test local, utile pour evaluer la generalisation locale du modele
    federe agrege, cf. plan d'evaluation).
    """
    from sklearn.model_selection import train_test_split

    result = {}
    for client_name, df in client_data.items():
        # Stratification impossible si trop peu de lignes ou une classe avec
        # moins de 2 membres (cas frequent sur de tres petits clients/tests) :
        # on retombe alors sur un split non stratifie plutot que de planter.
        class_counts = df["label"].value_counts()
        can_stratify = df["label"].nunique() > 1 and class_counts.min() >= 2 and len(df) >= 4
        stratify_col = df["label"] if can_stratify else None

        if len(df) < 2:
            # Trop peu de donnees pour un split : tout va en train, test vide.
            result[client_name] = {"train": df, "test": df.iloc[0:0]}
            logger.warning("Client '%s' : trop peu de lignes (%d) pour un split train/test",
                           client_name, len(df))
            continue

        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=stratify_col
        )
        result[client_name] = {"train": train_df, "test": test_df}
        logger.info("Client '%s' : train=%d, test=%d", client_name, len(train_df), len(test_df))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Mapping client -> dataset :")
    for client, dataset in CLIENT_DATASET_MAP.items():
        print(f"  {client:15s} <- {dataset}")
