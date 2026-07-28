"""
run_pipeline.py
================
Orchestre le pipeline complet de preparation des donnees, de bout en bout :

  1. Chargement + harmonisation (cleaning.py)          -> schema canonique
  2. Feature engineering (feature_engineering.py)       -> features derivees
  3. Normalisation (normalization.py)                   -> encodage + scaling
  4. Partitionnement en 6 clients (partitioning.py)     -> non-IID par site
  5. Split train/test par client
  6. Sauvegarde dans datasets/processed/ et datasets/partitions/

Usage:
    python -m preprocessing.run_pipeline \
        --ics-flow datasets/raw/iciot/ics_flow.csv \
        --ton-iot  datasets/raw/tonio/ton_iot.csv \
        --x-iiotid datasets/raw/xiiotid/x_iiotid.csv
"""

import argparse
import logging
import os

import pandas as pd

from preprocessing.cleaning import load_ics_flow, load_ton_iot, load_x_iiotid
from preprocessing.feature_engineering import engineer_features, get_model_feature_columns
from preprocessing.normalization import (
    one_hot_encode_proto,
    encode_attack_type,
    fit_scaler,
    save_scaler,
    apply_scaler,
    NUMERIC_COLUMNS_TO_SCALE,
)
from preprocessing.partitioning import build_client_partitions, train_test_split_per_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(ics_flow_path: str, ton_iot_path: str, x_iiotid_path: str,
        processed_dir: str = "datasets/processed",
        partitions_dir: str = "datasets/partitions",
        scaler_path: str = "datasets/processed/scaler.joblib") -> dict:

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(partitions_dir, exist_ok=True)

    # 1. Chargement + harmonisation
    logger.info("=== Etape 1/5 : chargement et harmonisation ===")
    df_ics = load_ics_flow(ics_flow_path)
    df_ton = load_ton_iot(ton_iot_path)
    df_xii = load_x_iiotid(x_iiotid_path)

    # 2. Feature engineering
    logger.info("=== Etape 2/5 : feature engineering ===")
    df_ics = engineer_features(df_ics)
    df_ton = engineer_features(df_ton)
    df_xii = engineer_features(df_xii)

    # 3. Encodage categoriel (proto, attack_type) — vocabulaire fixe, identique
    #    pour les 3 sources (voir normalization.py)
    logger.info("=== Etape 3/5 : encodage categoriel + scaling ===")
    for name, df in [("ics_flow", df_ics), ("ton_iot", df_ton), ("x_iiotid", df_xii)]:
        pass  # placeholder pour clarte de logs

    df_ics = encode_attack_type(one_hot_encode_proto(df_ics))
    df_ton = encode_attack_type(one_hot_encode_proto(df_ton))
    df_xii = encode_attack_type(one_hot_encode_proto(df_xii))

    # Fit du scaler UNE SEULE FOIS sur l'ensemble concatene (etape centralisee
    # de preparation, en amont de la simulation FL -- ne constitue pas une
    # fuite d'information vers les clients au moment de l'entrainement FL
    # lui-meme, seulement une normalisation commune des unites de mesure).
    combined = pd.concat([df_ics, df_ton, df_xii], ignore_index=True, sort=False)
    scaler = fit_scaler(combined, NUMERIC_COLUMNS_TO_SCALE)
    save_scaler(scaler, scaler_path)

    df_ics = apply_scaler(df_ics, NUMERIC_COLUMNS_TO_SCALE, scaler)
    df_ton = apply_scaler(df_ton, NUMERIC_COLUMNS_TO_SCALE, scaler)
    df_xii = apply_scaler(df_xii, NUMERIC_COLUMNS_TO_SCALE, scaler)

    # Sauvegarde des versions "processed" completes (avant partitionnement)
    df_ics.to_csv(os.path.join(processed_dir, "ics_flow_processed.csv"), index=False)
    df_ton.to_csv(os.path.join(processed_dir, "ton_iot_processed.csv"), index=False)
    df_xii.to_csv(os.path.join(processed_dir, "x_iiotid_processed.csv"), index=False)

    # 4. Partitionnement en 6 clients industriels
    logger.info("=== Etape 4/5 : partitionnement en 6 clients ===")
    processed_datasets = {"ics_flow": df_ics, "ton_iot": df_ton, "x_iiotid": df_xii}
    client_data = build_client_partitions(processed_datasets, output_dir=partitions_dir)

    # 5. Split train/test par client
    logger.info("=== Etape 5/5 : split train/test par client ===")
    client_splits = train_test_split_per_client(client_data)

    for client_name, splits in client_splits.items():
        client_dir = os.path.join(partitions_dir, client_name)
        splits["train"].to_csv(os.path.join(client_dir, "train.csv"), index=False)
        splits["test"].to_csv(os.path.join(client_dir, "test.csv"), index=False)

    logger.info("Pipeline termine. Features finales du modele : %s", get_model_feature_columns())
    return client_splits


def main():
    parser = argparse.ArgumentParser(description="Pipeline de preparation FL-IDS OT/ICS")
    parser.add_argument("--ics-flow", required=True, help="Chemin CSV brut ICS-Flow")
    parser.add_argument("--ton-iot", required=True, help="Chemin CSV brut TON_IoT")
    parser.add_argument("--x-iiotid", required=True, help="Chemin CSV brut X-IIoTID")
    args = parser.parse_args()

    run(args.ics_flow, args.ton_iot, args.x_iiotid)


if __name__ == "__main__":
    main()
