"""
normalization.py
=================
Encodage categoriel et normalisation numerique COHERENTS entre les 3 sources.

Principe cle du FL : chaque client doit encoder ses donnees de la MEME facon,
sans avoir vu les donnees des autres clients. On fixe donc ici, a l'avance et
en dur, les vocabulaires (proto, attack_type) et on fit un unique scaler
(offline, avant partitionnement) qui sera ensuite reutilise (jamais refit) par
chaque client.

Contenu :
  - ONE-HOT proto (vocabulaire fixe : tcp / udp / icmp / other)
  - Harmonisation attack_type pour analyse et reporting uniquement
  - StandardScaler fit une seule fois sur l'ensemble concatene (etape de
    preparation centralisee, avant simulation FL) puis sauvegarde/rechargee
"""

import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Proto : vocabulaire fixe (evite des colonnes one-hot differentes par client)
# ---------------------------------------------------------------------------
PROTO_CATEGORIES = ["tcp", "udp", "icmp"]  # tout le reste -> "other"


def one_hot_encode_proto(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["proto"] = df["proto"].where(df["proto"].isin(PROTO_CATEGORIES), "other")
    for cat in PROTO_CATEGORIES + ["other"]:
        df[f"proto_{cat}"] = (df["proto"] == cat).astype(int)
    return df.drop(columns=["proto"])


# ---------------------------------------------------------------------------
# attack_type : mapping vers un vocabulaire canonique reduit et commun
# (les noms exacts different fortement entre ICS-Flow / TON_IoT / X-IIoTID ;
# on regroupe par mots-cles plutot que par egalite stricte)
# ---------------------------------------------------------------------------
CANONICAL_ATTACK_CATEGORIES = [
    "normal",
    "dos_ddos",
    "reconnaissance_scanning",
    "mitm",
    "injection",
    "backdoor",
    "password_bruteforce",
    "ransomware",
    "xss",
    "tampering",
    "other_attack",
]

_ATTACK_KEYWORDS = [
    ("normal", "normal"), ("benign", "normal"),
    ("ddos", "dos_ddos"), ("dos", "dos_ddos"),
    ("recon", "reconnaissance_scanning"), ("scan", "reconnaissance_scanning"),
    ("mitm", "mitm"), ("man-in-the-middle", "mitm"),
    ("injection", "injection"), ("sql", "injection"),
    ("backdoor", "backdoor"),
    ("password", "password_bruteforce"), ("brute", "password_bruteforce"),
    ("ransomware", "ransomware"),
    ("xss", "xss"), ("cross-site", "xss"),
    ("tamper", "tampering"), ("spoof", "tampering"),
]


def canonicalize_attack_type(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()

    def _map(value: str) -> str:
        for keyword, canonical in _ATTACK_KEYWORDS:
            if keyword in value:
                return canonical
        return "other_attack"

    return s.map(_map)


def encode_attack_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise attack_type uniquement pour analyse/reporting.
    Ne crée pas de features ML pour éviter le data leakage.
    """
    df = df.copy()
    df["attack_type"] = canonicalize_attack_type(df["attack_type"])
    return df


# ---------------------------------------------------------------------------
# Scaling numerique (fit une seule fois, offline, sur les donnees concatenees)
# ---------------------------------------------------------------------------
def fit_scaler(df: pd.DataFrame, numeric_columns: list) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[numeric_columns].values)
    logger.info("Scaler fit sur %d colonnes numeriques, %d lignes", len(numeric_columns), len(df))
    return scaler


def save_scaler(scaler: StandardScaler, path: str) -> None:
    joblib.dump(scaler, path)
    logger.info("Scaler sauvegarde : %s", path)


def load_scaler(path: str) -> StandardScaler:
    return joblib.load(path)


def apply_scaler(df: pd.DataFrame, numeric_columns: list, scaler: StandardScaler) -> pd.DataFrame:
    """
    Applique un scaler DEJA FIT (jamais refit par un client individuel — c'est
    la regle a respecter en FL pour garantir un espace de features identique
    entre tous les clients).
    """
    df = df.copy()
    df[numeric_columns] = scaler.transform(df[numeric_columns].values)
    return df


NUMERIC_COLUMNS_TO_SCALE = [
    "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "byte_rate", "pkt_rate",
    "bytes_ratio", "pkts_ratio",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Categories proto :", PROTO_CATEGORIES + ["other"])
    print("Categories attack_type :", CANONICAL_ATTACK_CATEGORIES)
