"""
run_pipeline.py
================
Orchestre le pipeline complet de preparation des donnees :

  1. Chargement + harmonisation (cleaning.py)
  2. Feature engineering (feature_engineering.py)
  3. Encodage + normalisation (normalization.py)
  4. Partitionnement FL non-IID (partitioning.py)
  5. Split train/test par client
  6. Sauvegarde des datasets prepares
"""

import argparse
import logging
import os

import pandas as pd

from .cleaning import load_ics_flow, load_ton_iot, load_x_iiotid
from .feature_engineering import engineer_features, get_model_feature_columns
from .normalization import (
    one_hot_encode_proto,
    encode_attack_type,
    fit_scaler,
    save_scaler,
    apply_scaler,
    NUMERIC_COLUMNS_TO_SCALE,
)
from .partitioning import build_client_partitions, train_test_split_per_client


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)



def remove_identifier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les identifiants reseau non utilises par le modele.
    src_ip/dst_ip servent uniquement au partitionnement FL.
    """
    return df.drop(
        columns=["src_ip", "dst_ip"],
        errors="ignore"
    )



def run(
        ics_flow_path: str,
        ton_iot_path: str,
        x_iiotid_path: str,
        processed_dir: str = "datasets/processed",
        partitions_dir: str = "datasets/partitions",
        scaler_path: str = "datasets/processed/scaler.joblib"
) -> dict:


    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(partitions_dir, exist_ok=True)



    # ============================================================
    # 1. Chargement + harmonisation
    # ============================================================

    logger.info("=== Etape 1/5 : chargement et harmonisation ===")

    df_ics = load_ics_flow(ics_flow_path)
    df_ton = load_ton_iot(ton_iot_path)
    df_xii = load_x_iiotid(x_iiotid_path)



    # ============================================================
    # 2. Feature engineering
    # ============================================================

    logger.info("=== Etape 2/5 : feature engineering ===")

    df_ics = engineer_features(df_ics)
    df_ton = engineer_features(df_ton)
    df_xii = engineer_features(df_xii)



    # ============================================================
    # 3. Encodage + normalisation
    # ============================================================

    logger.info("=== Etape 3/5 : encodage categoriel + scaling ===")


    df_ics = encode_attack_type(
        one_hot_encode_proto(df_ics)
    )

    df_ton = encode_attack_type(
        one_hot_encode_proto(df_ton)
    )

    df_xii = encode_attack_type(
        one_hot_encode_proto(df_xii)
    )


    # Scaler commun pour garantir un espace de features identique
    # entre les clients FL

    combined = pd.concat(
        [df_ics, df_ton, df_xii],
        ignore_index=True,
        sort=False
    )


    scaler = fit_scaler(
        combined,
        NUMERIC_COLUMNS_TO_SCALE
    )

    save_scaler(
        scaler,
        scaler_path
    )


    df_ics = apply_scaler(
        df_ics,
        NUMERIC_COLUMNS_TO_SCALE,
        scaler
    )

    df_ton = apply_scaler(
        df_ton,
        NUMERIC_COLUMNS_TO_SCALE,
        scaler
    )

    df_xii = apply_scaler(
        df_xii,
        NUMERIC_COLUMNS_TO_SCALE,
        scaler
    )



    # ============================================================
    # Sauvegarde datasets processed
    # ============================================================

    logger.info("=== Sauvegarde datasets preprocesses ===")


    remove_identifier_features(df_ics).to_csv(
        os.path.join(processed_dir, "ics_flow_processed.csv"),
        index=False
    )


    remove_identifier_features(df_ton).to_csv(
        os.path.join(processed_dir, "ton_iot_processed.csv"),
        index=False
    )


    remove_identifier_features(df_xii).to_csv(
        os.path.join(processed_dir, "x_iiotid_processed.csv"),
        index=False
    )



    # ============================================================
    # 4. Partitionnement FL
    # ============================================================

    logger.info("=== Etape 4/5 : partitionnement en clients FL ===")


    processed_datasets = {
        "ics_flow": df_ics,
        "ton_iot": df_ton,
        "x_iiotid": df_xii
    }


    client_data = build_client_partitions(
        processed_datasets,
        output_dir=partitions_dir
    )



    # ============================================================
    # 5. Split train/test local
    # ============================================================

    logger.info("=== Etape 5/5 : split train/test par client ===")


    client_splits = train_test_split_per_client(
        client_data
    )


    for client_name, splits in client_splits.items():

        client_dir = os.path.join(
            partitions_dir,
            client_name
        )

        splits["train"].to_csv(
            os.path.join(client_dir, "train.csv"),
            index=False
        )

        splits["test"].to_csv(
            os.path.join(client_dir, "test.csv"),
            index=False
        )



    logger.info(
        "Pipeline termine. Features finales du modele : %s",
        get_model_feature_columns()
    )


    return client_splits




def main():

    parser = argparse.ArgumentParser(
        description="Pipeline de preparation FL-IDS OT/ICS"
    )

    parser.add_argument(
        "--ics-flow",
        required=True,
        help="Chemin CSV brut ICS-Flow"
    )

    parser.add_argument(
        "--ton-iot",
        required=True,
        help="Chemin CSV brut TON_IoT"
    )

    parser.add_argument(
        "--x-iiotid",
        required=True,
        help="Chemin CSV brut X-IIoTID"
    )


    args = parser.parse_args()


    run(
        args.ics_flow,
        args.ton_iot,
        args.x_iiotid
    )



if __name__ == "__main__":
    main()
