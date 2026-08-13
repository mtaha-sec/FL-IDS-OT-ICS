"""
FL-IDS-OT-ICS Monitoring Module

Provides:
- centralized logging for server and clients
- round-level metrics
- client-level metrics
- CSV persistence

Directory structure:

monitoring/
├── logs/
│   ├── server.log
│   └── client_<name>.log
└── metrics/
    ├── rounds.csv
    └── clients.csv
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "monitoring" / "logs"
METRICS_DIR = PROJECT_ROOT / "monitoring" / "metrics"

LOG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Logging
# ============================================================

def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger that writes both to the console and to a file.

    Example:
        logger = setup_logger("server", "server.log")
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if setup_logger is called twice
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_path = LOG_DIR / log_file
    file_handler = logging.FileHandler(
        file_path,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_server_logger() -> logging.Logger:
    """Return the global server logger."""
    return setup_logger("FL-SERVER", "server.log")


def get_client_logger(client_name: str) -> logging.Logger:
    """Return a logger dedicated to one FL client."""
    return setup_logger(
        f"FL-CLIENT-{client_name}",
        f"client_{client_name}.log",
    )


# ============================================================
# CSV helper
# ============================================================

def _append_csv(
    filename: str,
    fieldnames: list,
    row: Dict,
) -> None:
    """
    Append a row to a CSV file.

    Creates the file and header automatically if necessary.
    """

    path = METRICS_DIR / filename

    file_exists = path.exists() and path.stat().st_size > 0

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Round metrics
# ============================================================

def record_round_metrics(
    round_number: int,
    loss: Optional[float] = None,
    accuracy: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    f1: Optional[float] = None,
    duration_seconds: Optional[float] = None,
    num_clients: Optional[int] = None,
) -> None:
    """
    Record global federated-learning round metrics.
    """

    fieldnames = [
        "timestamp",
        "round",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "duration_seconds",
        "num_clients",
    ]

    _append_csv(
        "rounds.csv",
        fieldnames,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "round": round_number,
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "duration_seconds": duration_seconds,
            "num_clients": num_clients,
        },
    )


# ============================================================
# Client metrics
# ============================================================

def record_client_metrics(
    client_name: str,
    round_number: int,
    phase: str,
    loss: Optional[float] = None,
    accuracy: Optional[float] = None,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
    f1: Optional[float] = None,
    samples: Optional[int] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    """
    Record metrics for one FL client.

    phase:
        - train
        - evaluate
    """

    fieldnames = [
        "timestamp",
        "client",
        "round",
        "phase",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "samples",
        "duration_seconds",
    ]

    _append_csv(
        "clients.csv",
        fieldnames,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "client": client_name,
            "round": round_number,
            "phase": phase,
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "samples": samples,
            "duration_seconds": duration_seconds,
        },
    )