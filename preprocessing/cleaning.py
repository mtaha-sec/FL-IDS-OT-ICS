"""
cleaning.py
============
Harmonise les 3 datasets sources (ICS-Flow, TON_IoT, X-IIoTID) vers un schema
canonique unique (snake_case, convention Zeek / TON_IoT), tel que defini dans
la Table 1 du document de reference.

Chaque fonction load_* :
  1. Lit le CSV brut
  2. Renomme les colonnes vers les noms canoniques
  3. Normalise les valeurs categorielles (proto, etc.)
  4. Ajoute une colonne 'source_dataset' (utile pour le partitionnement FL)
  5. Retourne un DataFrame au schema canonique (colonnes non-communes conservees
     avec prefixe '_raw_' pour ne rien perdre, mais ignorees par defaut en aval)

Usage:
    from preprocessing.cleaning import load_ics_flow, load_ton_iot, load_x_iiotid
    df_ics = load_ics_flow("datasets/raw/iciot/ics_flow.csv")
    df_ton = load_ton_iot("datasets/raw/tonio/ton_iot.csv")
    df_xii = load_x_iiotid("datasets/raw/xiiotid/x_iiotid.csv")
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema canonique (Table 1 + Table 2 du document de reference)
# ---------------------------------------------------------------------------
CANONICAL_DIRECT_FEATURES = [
    "src_ip", "dst_ip", "proto", "duration",
    "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
]
CANONICAL_LABELS = ["label", "attack_type"]

# Mapping proto -> valeur canonique unique (evite la divergence d'encodage
# entre clients : 'tcp' / 'TCP' / 'IPV4-TCP' -> 'tcp')
PROTO_NORMALIZATION = {
    "tcp": "tcp", "TCP": "tcp", "ipv4-tcp": "tcp", "IPV4-TCP": "tcp",
    "udp": "udp", "UDP": "udp", "ipv4-udp": "udp", "IPV4-UDP": "udp",
    "icmp": "icmp", "ICMP": "icmp", "ipv4-icmp": "icmp", "IPV4-ICMP": "icmp",
}


def _normalize_proto(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .map(lambda x: PROTO_NORMALIZATION.get(x, x.lower()))
    )


def _safe_numeric(series) -> pd.Series:
    """Convertit en numerique, remplace les non-parsables ('-', '', etc.) par 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ---------------------------------------------------------------------------
# ICS-Flow
# ---------------------------------------------------------------------------
def load_ics_flow(path: str) -> pd.DataFrame:
    """
    Charge et harmonise ICS-Flow.

    Notes:
      - src_ip / dst_ip proviennent de sAddress / rAddress (fallback sIPs/rIPs).
      - src_bytes / dst_bytes utilisent sPayloadSum / rPayloadSum (PAS sBytesSum,
        qui inclut les en-tetes IP/TCP -> non comparable aux autres sources,
        cf. remarque du document de reference).
      - label binaire : IT_B_Label (fallback NST_B_Label si IT_B_Label absent).
      - attack_type   : IT_M_Label (fallback NST_M_Label).
    """
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=df.index)

    out["src_ip"] = df["sAddress"] if "sAddress" in df.columns else df["sIPs"]
    out["dst_ip"] = df["rAddress"] if "rAddress" in df.columns else df["rIPs"]
    out["proto"] = _normalize_proto(df["protocol"])
    out["duration"] = _safe_numeric(df["duration"])

    # bytes applicatifs (payload), pas les bytes totaux (sBytesSum inclut headers)
    out["src_bytes"] = _safe_numeric(df["sPayloadSum"])
    out["dst_bytes"] = _safe_numeric(df["rPayloadSum"])

    out["src_pkts"] = _safe_numeric(df["sPackets"])
    out["dst_pkts"] = _safe_numeric(df["rPackets"])

    # Labels
    label_col = "IT_B_Label" if "IT_B_Label" in df.columns else "NST_B_Label"
    attack_col = "IT_M_Label" if "IT_M_Label" in df.columns else "NST_M_Label"
    out["label"] = _binarize_label(df[label_col])
    out["attack_type"] = df[attack_col].astype(str).str.strip()

    # Colonnes brutes utiles pour feature_engineering.py (groupe B, C, D, E)
    out["_raw_sBytesSum"] = _safe_numeric(df["sBytesSum"]) if "sBytesSum" in df.columns else 0
    out["_raw_rBytesSum"] = _safe_numeric(df["rBytesSum"]) if "rBytesSum" in df.columns else 0
    out["_raw_sPayloadSum"] = out["src_bytes"]
    out["_raw_rPayloadSum"] = out["dst_bytes"]
    out["_raw_sPackets"] = out["src_pkts"]
    out["_raw_rPackets"] = out["dst_pkts"]
    out["_raw_sSynRate"] = _safe_numeric(df["sSynRate"]) if "sSynRate" in df.columns else 0
    out["_raw_sAckRate"] = _safe_numeric(df["sAckRate"]) if "sAckRate" in df.columns else 0
    out["_raw_sFinRate"] = _safe_numeric(df["sFinRate"]) if "sFinRate" in df.columns else 0
    out["_raw_sRstRate"] = _safe_numeric(df["sRstRate"]) if "sRstRate" in df.columns else 0
    out["_raw_sLoad"] = _safe_numeric(df["sLoad"]) if "sLoad" in df.columns else 0

    out["source_dataset"] = "ics_flow"
    logger.info("ICS-Flow charge et harmonise : %d lignes", len(out))
    return out


# ---------------------------------------------------------------------------
# TON_IoT
# ---------------------------------------------------------------------------
def load_ton_iot(path: str) -> pd.DataFrame:
    """
    Charge et harmonise TON_IoT.

    Notes:
      - Schema deja tres proche du canonique (convention Zeek d'origine).
      - conn_state sert a deriver les flags TCP (groupe B) dans feature_engineering.py,
        on le conserve donc en colonne brute '_raw_conn_state'.
    """
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=df.index)

    out["src_ip"] = df["src_ip"]
    out["dst_ip"] = df["dst_ip"]
    out["proto"] = _normalize_proto(df["proto"])
    out["duration"] = _safe_numeric(df["duration"])
    out["src_bytes"] = _safe_numeric(df["src_bytes"])
    out["dst_bytes"] = _safe_numeric(df["dst_bytes"])
    out["src_pkts"] = _safe_numeric(df["src_pkts"])
    out["dst_pkts"] = _safe_numeric(df["dst_pkts"])

    out["label"] = _binarize_label(df["label"])
    out["attack_type"] = df["type"].astype(str).str.strip()

    # Colonnes brutes utiles pour feature_engineering.py
    out["_raw_conn_state"] = df["conn_state"].astype(str).str.strip() if "conn_state" in df.columns else ""
    out["_raw_src_ip_bytes"] = _safe_numeric(df["src_ip_bytes"]) if "src_ip_bytes" in df.columns else 0
    out["_raw_dst_ip_bytes"] = _safe_numeric(df["dst_ip_bytes"]) if "dst_ip_bytes" in df.columns else 0

    out["source_dataset"] = "ton_iot"
    logger.info("TON_IoT charge et harmonise : %d lignes", len(out))
    return out


# ---------------------------------------------------------------------------
# X-IIoTID
# ---------------------------------------------------------------------------
def load_x_iiotid(path: str) -> pd.DataFrame:
    """
    Charge et harmonise X-IIoTID.

    Notes:
      - label binaire : class3 (Normal/Attack).
      - attack_type   : class1 (categorie d'attaque, ex: Reconnaissance, DoS...).
      - is_syn_only / Is_SYN_ACK / is_pure_ack / 'FIN or RST' servent a deriver
        les flags TCP dans feature_engineering.py -> conserves en '_raw_'.
    """
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame(index=df.index)

    out["src_ip"] = df["Scr_IP"]
    out["dst_ip"] = df["Des_IP"]
    out["proto"] = _normalize_proto(df["Protocol"])
    out["duration"] = _safe_numeric(df["Duration"])
    out["src_bytes"] = _safe_numeric(df["Scr_bytes"])
    out["dst_bytes"] = _safe_numeric(df["Des_bytes"])
    out["src_pkts"] = _safe_numeric(df["Scr_pkts"])
    out["dst_pkts"] = _safe_numeric(df["Des_pkts"])

    out["label"] = _binarize_label(df["class3"])
    out["attack_type"] = df["class1"].astype(str).str.strip()

    # Colonnes brutes utiles pour feature_engineering.py
    out["_raw_Scr_ip_bytes"] = _safe_numeric(df["Scr_ip_bytes"]) if "Scr_ip_bytes" in df.columns else 0
    out["_raw_Des_ip_bytes"] = _safe_numeric(df["Des_ip_bytes"]) if "Des_ip_bytes" in df.columns else 0
    out["_raw_is_syn_only"] = _to_bool(df["is_syn_only"]) if "is_syn_only" in df.columns else 0
    out["_raw_Is_SYN_ACK"] = _to_bool(df["Is_SYN_ACK"]) if "Is_SYN_ACK" in df.columns else 0
    out["_raw_is_pure_ack"] = _to_bool(df["is_pure_ack"]) if "is_pure_ack" in df.columns else 0
    out["_raw_fin_or_rst"] = _to_bool(df["FIN or RST"]) if "FIN or RST" in df.columns else 0
    out["_raw_is_with_payload"] = _to_bool(df["is_with_payload"]) if "is_with_payload" in df.columns else 0

    out["source_dataset"] = "x_iiotid"
    logger.info("X-IIoTID charge et harmonise : %d lignes", len(out))
    return out


# ---------------------------------------------------------------------------
# Helpers labels
# ---------------------------------------------------------------------------
def _binarize_label(series: pd.Series) -> pd.Series:
    """
    Convertit un label heterogene (0/1, 'Normal'/'Attack', 'normal'/'anomaly', ...)
    en label binaire canonique : 0 = normal, 1 = attaque.
    """
    s = series.astype(str).str.strip().str.lower()
    normal_tokens = {"0", "normal", "benign", "none"}
    return s.map(lambda x: 0 if x in normal_tokens else 1).astype(int)


def _to_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    return s.map(lambda x: 1 if x in {"true", "1", "yes"} else 0).astype(int)


# ---------------------------------------------------------------------------
# Point d'entree (test rapide en local)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Module de nettoyage pret. Importer load_ics_flow / load_ton_iot / load_x_iiotid.")
