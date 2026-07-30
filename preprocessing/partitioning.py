"""
partitioning.py
================
Repartit les 3 datasets harmonises (ICS-Flow, TON_IoT, X-IIoTID) entre 6
clients industriels simulant des sections d'un site industriel.

Partitionnement non-IID :
  - ICS-Flow  -> power, utilities
  - TON_IoT   -> sap, pap
  - X-IIoTID  -> beneficiation, granulation

Les clients sont construits par regroupement d'IP source pour simuler des
sous-reseaux differents. Les IP sont utilisees uniquement pour le partitionnement
et sont supprimees avant l'entrainement du modele.
"""

import logging
import os
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DATASET_CLIENTS = {
    "ics_flow": ["power", "utilities"],
    "ton_iot": ["sap", "pap"],
    "x_iiotid": ["beneficiation", "granulation"],
}


def split_by_src_ip(df: pd.DataFrame, n_splits: int, seed: int = 42) -> list:
    """
    Partitionnement non-IID par IP source.
    Les IP restent uniquement un mecanisme de separation des clients.
    """

    if "src_ip" not in df.columns:
        raise ValueError("src_ip obligatoire pour le partitionnement FL")

    rng = np.random.default_rng(seed)

    unique_ips = df["src_ip"].dropna().unique()
    rng.shuffle(unique_ips)

    ip_groups = np.array_split(unique_ips, n_splits)

    splits = []
    for group in ip_groups:
        part = df[df["src_ip"].isin(group)].copy()
        splits.append(part)

    logger.info(
        "Split par src_ip en %d parts : tailles=%s",
        n_splits,
        [len(s) for s in splits]
    )

    return splits


def remove_identifier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les features interdites au modele.
    Les IP servent uniquement au partitionnement FL.
    """
    return df.drop(
        columns=["src_ip", "dst_ip"],
        errors="ignore"
    )


def build_client_partitions(
        processed_datasets: dict,
        output_dir: str,
        seed: int = 42
) -> dict:
    """
    processed_datasets :
        DataFrames apres cleaning + feature_engineering
        (avant normalisation)

    Retourne :
        {client_name: dataframe}
    """

    client_data = {}

    for dataset_name, clients in DATASET_CLIENTS.items():

        df = processed_datasets[dataset_name]

        parts = split_by_src_ip(
            df,
            n_splits=len(clients),
            seed=seed
        )

        for client_name, part in zip(clients, parts):

            # Suppression des identifiants avant entrainement
            client_data[client_name] = remove_identifier_features(part)


    for client_name, df in client_data.items():

        client_dir = os.path.join(output_dir, client_name)
        os.makedirs(client_dir, exist_ok=True)

        out_path = os.path.join(client_dir, "data.csv")

        df.to_csv(
            out_path,
            index=False
        )

        logger.info(
            "Client '%s' : %d lignes -> %s",
            client_name,
            len(df),
            out_path
        )

    return client_data



def train_test_split_per_client(
        client_data: dict,
        test_size: float = 0.2,
        seed: int = 42
) -> dict:
    """
    Split train/test independant par client FL.
    """

    from sklearn.model_selection import train_test_split

    result = {}

    for client_name, df in client_data.items():

        class_counts = df["label"].value_counts()

        can_stratify = (
            df["label"].nunique() > 1
            and class_counts.min() >= 2
            and len(df) >= 4
        )

        stratify_col = df["label"] if can_stratify else None


        if len(df) < 2:

            result[client_name] = {
                "train": df,
                "test": df.iloc[0:0]
            }

            logger.warning(
                "Client '%s' : trop peu de lignes (%d)",
                client_name,
                len(df)
            )

            continue


        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_col
        )


        result[client_name] = {
            "train": train_df,
            "test": test_df
        }


        logger.info(
            "Client '%s' : train=%d test=%d",
            client_name,
            len(train_df),
            len(test_df)
        )


    return result



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("Mapping dataset -> clients :")

    for dataset, clients in DATASET_CLIENTS.items():
        print(f"{dataset}: {clients}")
