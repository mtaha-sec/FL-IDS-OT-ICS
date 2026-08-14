"""
FL-IDS-OT-ICS Monitoring Module
================================

Centralized monitoring for the federated learning system.

Provides
--------
- Centralized logging for the FL server
- Dedicated logging for each FL client
- Round-level metrics
- Client-level metrics
- Generic client metrics such as FedProx mu
- CSV persistence

Directory structure
-------------------

monitoring/
├── __init__.py
├── monitor.py
├── logs/
│   ├── server.log
│   ├── client_power.log
│   ├── client_sap.log
│   ├── client_pap.log
│   ├── client_utilities.log
│   ├── client_beneficiation.log
│   └── client_granulation.log
│
└── metrics/
    ├── rounds.csv
    ├── clients.csv
    └── client_metrics.csv

Usage
-----

Server:

    from monitoring.monitor import get_server_logger

    logger = get_server_logger()
    logger.info("Server started")

Client:

    from monitoring.monitor import get_client_logger

    logger = get_client_logger("power")
    logger.info("Client started")

Round metrics:

    from monitoring.monitor import record_round_metrics

    record_round_metrics(
        round_number=1,
        phase="evaluate",
        loss=0.15,
        accuracy=0.95,
        num_clients=6,
    )

Client metrics:

    from monitoring.monitor import record_client_metrics

    record_client_metrics(
        client_name="power",
        round_number=1,
        phase="train",
        loss=0.12,
        samples=5000,
    )

Generic metrics:

    from monitoring.monitor import record_metric

    record_metric(
        client_name="power",
        round_number=1,
        phase="train",
        metric="fedprox_mu",
        value=0.01,
    )
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


# ============================================================
# Paths
# ============================================================

# Project root:
# monitoring/monitor.py
#       ↑
#       parent = monitoring/
#       parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "monitoring" / "logs"
METRICS_DIR = PROJECT_ROOT / "monitoring" / "metrics"

# Create directories automatically
LOG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Timestamp
# ============================================================

def _timestamp() -> str:
    """
    Return the current timestamp in ISO format.

    Example:
        2026-08-14T19:30:25
    """
    return datetime.now().isoformat(timespec="seconds")


# ============================================================
# Logging
# ============================================================

def setup_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger.

    The logger writes simultaneously to:
        1. console
        2. monitoring/logs/<log_file>

    Parameters
    ----------
    name:
        Unique logger name.

    log_file:
        File name inside monitoring/logs/.

    level:
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger.
    """

    logger = logging.getLogger(name)

    logger.setLevel(level)

    # Prevent messages from being propagated to the root logger.
    logger.propagate = False

    # Avoid duplicate handlers if this function is called
    # multiple times for the same logger.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --------------------------------------------------------
    # Console handler
    # --------------------------------------------------------

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # --------------------------------------------------------
    # File handler
    # --------------------------------------------------------

    file_path = LOG_DIR / log_file

    file_handler = logging.FileHandler(
        file_path,
        mode="a",
        encoding="utf-8",
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # --------------------------------------------------------
    # Register handlers
    # --------------------------------------------------------

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_server_logger() -> logging.Logger:
    """
    Return the centralized FL server logger.

    Log file:
        monitoring/logs/server.log
    """

    return setup_logger(
        name="FL-SERVER",
        log_file="server.log",
    )


def get_client_logger(client_name: str) -> logging.Logger:
    """
    Return a logger dedicated to one FL client.

    Example:

        get_client_logger("power")

    creates:

        monitoring/logs/client_power.log
    """

    return setup_logger(
        name=f"FL-CLIENT-{client_name}",
        log_file=f"client_{client_name}.log",
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

    If the CSV file does not exist or is empty,
    the header is automatically created.
    """

    path = METRICS_DIR / filename

    file_exists = (
        path.exists()
        and path.stat().st_size > 0
    )

    try:
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

    except OSError as exc:
        logging.getLogger("FL-MONITORING").error(
            "Failed to write metrics to %s: %s",
            path,
            exc,
        )


# ============================================================
# Round metrics
# ============================================================

def record_round_metrics(
    round_number: int,
    phase: str,
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

    Parameters
    ----------
    round_number:
        FL round number.

    phase:
        Either "train" or "evaluate".

    loss:
        Aggregated loss.

    accuracy:
        Aggregated accuracy.

    precision:
        Aggregated precision.

    recall:
        Aggregated recall.

    f1:
        Aggregated F1 score.

    duration_seconds:
        Round duration.

    num_clients:
        Number of participating clients.

    Output
    ------
    monitoring/metrics/rounds.csv
    """

    fieldnames = [
        "timestamp",
        "round",
        "phase",
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
            "timestamp": _timestamp(),
            "round": round_number,
            "phase": phase,
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

    Parameters
    ----------
    client_name:
        Industrial site name.

    round_number:
        FL round.

    phase:
        "train" or "evaluate".

    loss:
        Local loss.

    accuracy:
        Local accuracy.

    precision:
        Local precision.

    recall:
        Local recall.

    f1:
        Local F1.

    samples:
        Number of samples processed.

    duration_seconds:
        Duration of the operation.

    Output
    ------
    monitoring/metrics/clients.csv
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
            "timestamp": _timestamp(),
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


# ============================================================
# Generic client metrics
# ============================================================

def record_metric(
    client_name: str,
    round_number: int,
    phase: str,
    metric: str,
    value: float,
) -> None:
    """
    Record a generic client metric.

    This function is useful for metrics that do not directly
    correspond to loss/accuracy/etc.

    Examples
    --------
    FedProx coefficient:

        record_metric(
            client_name="power",
            round_number=1,
            phase="train",
            metric="fedprox_mu",
            value=0.01,
        )

    Learning rate:

        record_metric(
            client_name="power",
            round_number=1,
            phase="train",
            metric="learning_rate",
            value=0.001,
        )

    Output
    ------
    monitoring/metrics/client_metrics.csv
    """

    fieldnames = [
        "timestamp",
        "client",
        "round",
        "phase",
        "metric",
        "value",
    ]

    _append_csv(
        "client_metrics.csv",
        fieldnames,
        {
            "timestamp": _timestamp(),
            "client": client_name,
            "round": round_number,
            "phase": phase,
            "metric": metric,
            "value": value,
        },
    )


# ============================================================
# Monitoring initialization
# ============================================================

def initialize_monitoring() -> None:
    """
    Initialize the monitoring directories.

    This function is optional because the directories are already
    created when this module is imported.

    It can nevertheless be called explicitly from server.py
    or client_app.py.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Module test
# ============================================================

if __name__ == "__main__":

    print("FL-IDS-OT-ICS monitoring module test")

    initialize_monitoring()

    # Test server logger
    server_logger = get_server_logger()
    server_logger.info("Monitoring module test: server logger OK")

    # Test client logger
    client_logger = get_client_logger("power")
    client_logger.info("Monitoring module test: client logger OK")

    # Test round metrics
    record_round_metrics(
        round_number=1,
        phase="evaluate",
        loss=0.15,
        accuracy=0.95,
        precision=0.94,
        recall=0.96,
        f1=0.95,
        duration_seconds=12.5,
        num_clients=6,
    )

    # Test client metrics
    record_client_metrics(
        client_name="power",
        round_number=1,
        phase="train",
        loss=0.20,
        samples=5000,
        duration_seconds=4.2,
    )

    # Test generic metric
    record_metric(
        client_name="power",
        round_number=1,
        phase="train",
        metric="fedprox_mu",
        value=0.01,
    )

    print("Monitoring test completed.")
    print(f"Logs directory:    {LOG_DIR}")
    print(f"Metrics directory: {METRICS_DIR}")