"""
feature_engineering.py
=======================
Derive les features "ingenierees" (Table 2 du document de reference) a partir
du DataFrame harmonise produit par preprocessing/cleaning.py.

Groupes derives :
  A - Volume de flux        : src_ip_bytes, dst_ip_bytes
  B - Drapeaux TCP          : syn_flag, ack_flag, fin_flag, rst_flag, synack_flag
  C - Debit temporel        : byte_rate, pkt_rate
  D - Ratios d'asymetrie    : bytes_ratio, pkts_ratio
  E - Protocole & etat      : with_payload

Chaque source (ics_flow / ton_iot / x_iiotid) a une regle de derivation
differente pour le groupe B (cf. Table 2) ; les groupes C, D, E sont calcules
de facon identique pour toutes les sources car ils ne dependent que des
features canoniques directes (deja harmonisees par cleaning.py).

IMPORTANT (Remarque 1 du document) : src_ip et dst_ip NE SONT PAS des features
d'entree du modele. Elles sont conservees dans le DataFrame pour tracabilite
mais doivent etre exclues explicitement avant l'entrainement (cf. get_model_
feature_columns ci-dessous).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-9  # evite les divisions par zero

# TON_IoT : mapping conn_state -> flag TCP (cf. Table 2, groupe B)
TON_CONN_STATE_FLAG_MAP = {
    "S0": "syn_flag",
    "SF": "ack_flag",
    "SH": "fin_flag",
    "SHR": "fin_flag",
    "RSTR": "rst_flag",
    "RSTO": "rst_flag",
    "S1": "synack_flag",
}

FLAG_COLUMNS = ["syn_flag", "ack_flag", "fin_flag", "rst_flag", "synack_flag"]


def _derive_flags_ics_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Groupe B pour ICS-Flow : a partir des taux sSynRate/sAckRate/sFinRate/sRstRate."""
    df["syn_flag"] = (df["_raw_sSynRate"] > 0).astype(int)
    df["ack_flag"] = (df["_raw_sAckRate"] > 0).astype(int)
    df["fin_flag"] = (df["_raw_sFinRate"] > 0).astype(int)
    df["rst_flag"] = (df["_raw_sRstRate"] > 0).astype(int)
    # synack_flag n'est pas derivable directement dans ICS-Flow -> 0 (absence documentee)
    df["synack_flag"] = 0
    return df


def _derive_flags_ton_iot(df: pd.DataFrame) -> pd.DataFrame:
    """Groupe B pour TON_IoT : a partir de conn_state (mapping Table 2)."""
    for flag in FLAG_COLUMNS:
        df[flag] = 0
    conn_state = df["_raw_conn_state"].astype(str).str.strip()
    for state, flag in TON_CONN_STATE_FLAG_MAP.items():
        mask = conn_state == state
        df.loc[mask, flag] = 1
    return df


def _derive_flags_x_iiotid(df: pd.DataFrame) -> pd.DataFrame:
    """Groupe B pour X-IIoTID : a partir des booleens deja presents."""
    df["syn_flag"] = df["_raw_is_syn_only"]
    df["ack_flag"] = df["_raw_is_pure_ack"]
    # 'FIN or RST' est une colonne unique combinee dans X-IIoTID : on l'affecte
    # a la fois a fin_flag et rst_flag (impossible de distinguer les deux avec
    # cette seule colonne source -> limite documentee du dataset original).
    df["fin_flag"] = df["_raw_fin_or_rst"]
    df["rst_flag"] = df["_raw_fin_or_rst"]
    df["synack_flag"] = df["_raw_Is_SYN_ACK"]
    return df


def _derive_ip_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Groupe A : src_ip_bytes / dst_ip_bytes (absents dans ICS-Flow -> 0 + flag manquant)."""
    source = df["source_dataset"].iloc[0] if len(df) else None

    if source == "ton_iot":
        df["src_ip_bytes"] = df["_raw_src_ip_bytes"]
        df["dst_ip_bytes"] = df["_raw_dst_ip_bytes"]
        df["ip_bytes_available"] = 1
    elif source == "x_iiotid":
        df["src_ip_bytes"] = df["_raw_Scr_ip_bytes"]
        df["dst_ip_bytes"] = df["_raw_Des_ip_bytes"]
        df["ip_bytes_available"] = 1
    else:  # ics_flow : non disponible
        df["src_ip_bytes"] = 0.0
        df["dst_ip_bytes"] = 0.0
        df["ip_bytes_available"] = 0  # feature d'indicateur de disponibilite

    return df


def _derive_rates_and_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Groupes C, D, E : identiques pour toutes les sources (dependent uniquement
    des colonnes canoniques directes deja harmonisees)."""
    total_bytes = df["src_bytes"] + df["dst_bytes"]
    total_pkts = df["src_pkts"] + df["dst_pkts"]
    duration_safe = df["duration"].replace(0, np.nan)

    # Groupe C - debit temporel
    df["byte_rate"] = (total_bytes / duration_safe).fillna(0)
    df["pkt_rate"] = (total_pkts / duration_safe).fillna(0)

    # Groupe D - ratios d'asymetrie
    df["bytes_ratio"] = df["src_bytes"] / (total_bytes + EPS)
    df["pkts_ratio"] = df["src_pkts"] / (total_pkts + EPS)

    # Groupe E - protocole & etat
    df["with_payload"] = (df["src_bytes"] > 0).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Point d'entree principal : applique toutes les derivations (Groupes A-E)
    sur un DataFrame harmonise (sortie de cleaning.load_*).

    Le DataFrame doit contenir 'source_dataset' et les colonnes '_raw_*'
    specifiques a la source (cf. cleaning.py).
    """
    if df.empty:
        return df

    source = df["source_dataset"].iloc[0]
    df = df.copy()

    # Groupe A
    df = _derive_ip_bytes(df)

    # Groupe B (regle differente par source)
    if source == "ics_flow":
        df = _derive_flags_ics_flow(df)
    elif source == "ton_iot":
        df = _derive_flags_ton_iot(df)
    elif source == "x_iiotid":
        df = _derive_flags_x_iiotid(df)
    else:
        raise ValueError(f"source_dataset inconnue : {source}")

    # Groupes C, D, E
    df = _derive_rates_and_ratios(df)

    # Nettoyage : on retire les colonnes techniques '_raw_*' (deja exploitees)
    raw_cols = [c for c in df.columns if c.startswith("_raw_")]
    df = df.drop(columns=raw_cols)

    logger.info("Feature engineering applique sur source=%s (%d lignes, %d colonnes)",
                source, len(df), df.shape[1])
    return df


def get_model_feature_columns() -> list:
    """
    Retourne la liste finale des colonnes exploitables par le modele ML/DL
    (exclut src_ip, dst_ip, source_dataset, et les labels).
    proto et attack_type restent categorielles -> a encoder via normalization.py.
    """
    direct = ["proto", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts"]
    group_a = ["src_ip_bytes", "dst_ip_bytes", "ip_bytes_available"]
    group_b = FLAG_COLUMNS
    group_c = ["byte_rate", "pkt_rate"]
    group_d = ["bytes_ratio", "pkts_ratio"]
    group_e = ["with_payload"]
    return direct + group_a + group_b + group_c + group_d + group_e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Colonnes finales du modele :", get_model_feature_columns())
