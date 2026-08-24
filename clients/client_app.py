"""
client_app.py
=============

Flower client for FL-IDS-OT-ICS with:

  • Homomorphic Encryption (TenSEAL CKKS)
      - get_parameters() encrypts trainable model weights before sending them
        to the server.
      - The server NEVER sees plaintext client parameters.
      - set_parameters() decrypts the aggregated global model received from
        the server using the client's secret key.

  • FedProx (Li et al., 2020)
      - The local training loop adds a proximal term:

            loss = task_loss + (mu / 2) * ||w_local - w_global||²

        where mu is broadcast by the server each round.

  • Monitoring
      - Dedicated client logs
      - Local training loss
      - Evaluation loss
      - Evaluation accuracy
      - FedProx coefficient
      - Number of samples
      - Training duration
      - Evaluation duration

  • TLS
      - Client connects to the Flower server using the CA certificate.

Parameter encoding (agreed with HEFedProxStrategy on the server)
-----------------------------------------------------------------

    parameters[0]:
        trainable parameters

        dtype=uint8
            -> encrypted CKKS ciphertext after round 0

        dtype=float32
            -> plaintext flat array for round 0 initialization

    parameters[1 .. M]:
        plaintext BatchNorm buffers

Usage
-----

    python -m clients.client_app ^
        --client-name power ^
        --server-address 127.0.0.1:8080 ^
        --he-context security/keys/he_context_full.seal

    python -m clients.client_app --client-name sap --server-address 127.0.0.1:8080

    python -m clients.client_app --client-name pap --server-address 127.0.0.1:8080

    python -m clients.client_app --client-name utilities --server-address 127.0.0.1:8080

    python -m clients.client_app --client-name beneficiation --server-address 127.0.0.1:8080

    python -m clients.client_app --client-name granulation --server-address 127.0.0.1:8080
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models.local_ids.model import build_model

from security.he_crypto import (
    decrypt_parameters,
    encrypt_parameters,
    get_param_shapes,
    get_buffer_shapes,
    load_context,
)

# Monitoring module
from monitoring.monitor import (
    get_client_logger,
    record_client_metrics,
    record_metric,
    initialize_monitoring,
)


# ============================================================================
# Global configuration
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DEFAULT_HE_CONTEXT = os.path.join(
    "security",
    "keys",
    "he_context_full.seal",
)

DEFAULT_CA_CERTIFICATE = os.path.join(
    "security",
    "certificates",
    "ca.crt",
)


# ============================================================================
# Base Python logger
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# Data loading
# ============================================================================

def load_client_data(
    client_name: str,
    partitions_dir: str = "datasets/partitions",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
]:
    """
    Load train.csv and test.csv for one industrial site.

    Parameters
    ----------
    client_name:
        Industrial site name.

    partitions_dir:
        Directory containing the client partitions.

    Returns
    -------
    X_train:
        Training features.

    y_train:
        Training labels.

    X_test:
        Test features.

    y_test:
        Test labels.

    feature_cols:
        Model feature names.
    """

    client_dir = os.path.join(
        partitions_dir,
        client_name,
    )

    train_path = os.path.join(
        client_dir,
        "train.csv",
    )

    test_path = os.path.join(
        client_dir,
        "test.csv",
    )

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    from preprocessing.feature_engineering import (
        get_model_feature_columns
    )

    feature_cols = get_model_feature_columns()

    missing_train = [
        column
        for column in feature_cols
        if column not in train_df.columns
    ]

    missing_test = [
        column
        for column in feature_cols
        if column not in test_df.columns
    ]

    if missing_train:
        raise ValueError(
            f"Expected columns missing from train.csv "
            f"for '{client_name}': {missing_train}"
        )

    if missing_test:
        raise ValueError(
            f"Expected columns missing from test.csv "
            f"for '{client_name}': {missing_test}"
        )

    if "label" not in train_df.columns:
        raise ValueError(
            f"'label' column missing from train.csv "
            f"for '{client_name}'"
        )

    if "label" not in test_df.columns:
        raise ValueError(
            f"'label' column missing from test.csv "
            f"for '{client_name}'"
        )

    X_train = train_df[
        feature_cols
    ].values.astype(np.float32)

    y_train = train_df[
        "label"
    ].values.astype(np.float32)

    X_test = test_df[
        feature_cols
    ].values.astype(np.float32)

    y_test = test_df[
        "label"
    ].values.astype(np.float32)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
    )


# ============================================================================
# Flower client
# ============================================================================

class IDSFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for one industrial IDS site.

    Features
    --------
    • TenSEAL CKKS homomorphic encryption
    • FedProx local optimization
    • Monitoring
    • TLS connection to Flower server
    """

    def __init__(
        self,
        client_name: str,
        he_context_path: str = DEFAULT_HE_CONTEXT,
        model_type: str = "mlp",
        batch_size: int = 64,
        local_epochs: int = 3,
        lr: float = 1e-3,
    ) -> None:

        self.client_name = client_name
        self.model_type = model_type
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # --------------------------------------------------------------------
        # Monitoring initialization
        # --------------------------------------------------------------------

        initialize_monitoring()

        self.monitor_logger = get_client_logger(
            client_name
        )

        self.monitor_logger.info(
            "=" * 70
        )

        self.monitor_logger.info(
            "Initializing FL client"
        )

        self.monitor_logger.info(
            "Client name : %s",
            client_name,
        )

        self.monitor_logger.info(
            "Model       : %s",
            model_type,
        )

        self.monitor_logger.info(
            "Device      : %s",
            DEVICE,
        )

        # --------------------------------------------------------------------
        # HE context
        # --------------------------------------------------------------------

        if not os.path.exists(he_context_path):
            raise FileNotFoundError(
                f"HE context not found: {he_context_path}\n"
                f"Run 'python -m security.generate_he_keys' first."
            )

        self.he_ctx = load_context(
            he_context_path
        )

        if not self.he_ctx.is_private():
            raise ValueError(
                "Client must use the FULL HE context "
                "(he_context_full.seal) containing the secret key. "
                "The public-only context is intended for the server."
            )

        self.monitor_logger.info(
            "HE context loaded successfully "
            "(secret key present)"
        )

        # --------------------------------------------------------------------
        # Parameter and buffer shapes
        # --------------------------------------------------------------------

        self.param_shapes = get_param_shapes()

        self.buffer_shapes = get_buffer_shapes()

        self.monitor_logger.info(
            "Parameter shapes loaded: %d",
            len(self.param_shapes),
        )

        self.monitor_logger.info(
            "Buffer shapes loaded: %d",
            len(self.buffer_shapes),
        )

        # --------------------------------------------------------------------
        # Local dataset
        # --------------------------------------------------------------------

        (
            X_train,
            y_train,
            X_test,
            y_test,
            feature_cols,
        ) = load_client_data(
            client_name
        )

        self.input_dim = len(
            feature_cols
        )

        self.monitor_logger.info(
            "Dataset loaded"
        )

        self.monitor_logger.info(
            "Features     : %d",
            self.input_dim,
        )

        self.monitor_logger.info(
            "Train samples: %d",
            len(X_train),
        )

        self.monitor_logger.info(
            "Test samples : %d",
            len(X_test),
        )

        # --------------------------------------------------------------------
        # DataLoaders
        # --------------------------------------------------------------------

        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train),
                torch.tensor(y_train),
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_test),
                torch.tensor(y_test),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        self.n_train = len(
            X_train
        )

        # --------------------------------------------------------------------
        # Model
        # --------------------------------------------------------------------

        self.model = build_model(
            model_type,
            input_dim=self.input_dim,
        ).to(DEVICE)

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        parameter_count = sum(
            p.numel()
            for p in self.model.parameters()
        )

        self.monitor_logger.info(
            "Model initialized"
        )

        self.monitor_logger.info(
            "Trainable parameters: %d",
            parameter_count,
        )

        self.monitor_logger.info(
            "Batch size          : %d",
            batch_size,
        )

        self.monitor_logger.info(
            "Local epochs        : %d",
            local_epochs,
        )

        self.monitor_logger.info(
            "Learning rate       : %.6f",
            lr,
        )

        self.monitor_logger.info(
            "=" * 70
        )

    # ========================================================================
    # Parameter encoding / decoding
    # ========================================================================

    def get_parameters(
        self,
        config: Dict,
    ) -> List[np.ndarray]:
        """
        Encrypt and return local model parameters.

        Encoding
        --------

        parameters[0]
            encrypted trainable parameters

        parameters[1:]
            plaintext model buffers
        """

        self.monitor_logger.info(
            "Encrypting local model parameters..."
        )

        # --------------------------------------------------------------------
        # Trainable parameters
        # --------------------------------------------------------------------

        trainable = [
            p.data.cpu().numpy()
            for p in self.model.parameters()
        ]

        encrypted_parameters = encrypt_parameters(
            trainable,
            self.he_ctx,
        )

        # --------------------------------------------------------------------
        # BatchNorm buffers
        # --------------------------------------------------------------------

        buffers = [
            buffer.cpu().numpy()
            for buffer in self.model.buffers()
        ]

        self.monitor_logger.info(
            "Model parameters encrypted"
        )

        self.monitor_logger.info(
            "Ciphertext size: %d bytes",
            encrypted_parameters.nbytes,
        )

        return [
            encrypted_parameters
        ] + buffers

    # ------------------------------------------------------------------------

    def set_parameters(
        self,
        parameters: List[np.ndarray],
    ) -> None:
        """
        Load global model parameters.

        Two cases are supported:

        float32
            Round-0 plaintext initialization.

        uint8
            Encrypted global model requiring decryption
            with the client's HE secret key.
        """

        encrypted_or_flat = parameters[0]

        buffer_arrays = parameters[1:]

        # --------------------------------------------------------------------
        # Encrypted parameters
        # --------------------------------------------------------------------

        if encrypted_or_flat.dtype == np.uint8:

            self.monitor_logger.info(
                "Encrypted global model received"
            )

            self.monitor_logger.info(
                "Decrypting global model..."
            )

            trainable_list = decrypt_parameters(
                enc_uint8=encrypted_or_flat,
                ctx=self.he_ctx,
                shapes=self.param_shapes,
            )

            self.monitor_logger.info(
                "Global model decrypted successfully"
            )

        # --------------------------------------------------------------------
        # Round-0 plaintext initialization
        # --------------------------------------------------------------------

        else:

            self.monitor_logger.info(
                "Plaintext global model initialization received"
            )

            flat = encrypted_or_flat.astype(
                np.float32
            )

            trainable_list = []

            offset = 0

            for shape in self.param_shapes:

                size = int(
                    np.prod(shape)
                )

                parameter = (
                    flat[
                        offset:
                        offset + size
                    ]
                    .reshape(shape)
                )

                trainable_list.append(
                    parameter
                )

                offset += size

        # --------------------------------------------------------------------
        # Reconstruct model state
        # --------------------------------------------------------------------

        new_state: Dict[
            str,
            torch.Tensor
        ] = {}

        trainable_iterator = iter(
            trainable_list
        )

        for name, _ in self.model.named_parameters():

            new_state[name] = torch.tensor(
                next(trainable_iterator)
            )

        buffer_iterator = iter(
            buffer_arrays
        )

        for name, existing_buffer in (
            self.model.named_buffers()
        ):

            array = next(
                buffer_iterator,
                None,
            )

            if array is not None:
                new_state[name] = torch.tensor(
                    array
                )
            else:
                new_state[name] = existing_buffer

        self.model.load_state_dict(
            new_state,
            strict=True,
        )

    # ========================================================================
    # Local training
    # ========================================================================

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[
        List[np.ndarray],
        int,
        Dict,
    ]:
        """
        Perform local training using FedProx.
        """

        # --------------------------------------------------------------------
        # FL configuration
        # --------------------------------------------------------------------

        mu = float(
            config.get(
                "mu",
                0.0,
            )
        )

        fl_round = int(
            config.get(
                "round",
                0,
            )
        )

        # --------------------------------------------------------------------
        # Training timer
        # --------------------------------------------------------------------

        training_start = time.perf_counter()

        # --------------------------------------------------------------------
        # Load global model
        # --------------------------------------------------------------------

        self.set_parameters(
            parameters
        )

        # --------------------------------------------------------------------
        # Frozen global parameters for FedProx
        # --------------------------------------------------------------------

        global_params: Optional[
            List[torch.Tensor]
        ] = None

        if mu > 0.0:

            global_params = [
                p.data.clone().to(DEVICE)
                for p in self.model.parameters()
            ]

        self.monitor_logger.info(
            "Round %d | local training started | "
            "epochs=%d | FedProx mu=%.6f",
            fl_round,
            self.local_epochs,
            mu,
        )

        # --------------------------------------------------------------------
        # Training mode
        # --------------------------------------------------------------------

        self.model.train()

        avg_task_loss = 0.0

        # --------------------------------------------------------------------
        # Local epochs
        # --------------------------------------------------------------------

        for epoch in range(
            self.local_epochs
        ):

            epoch_task_loss = 0.0

            for xb, yb in self.train_loader:

                xb = xb.to(DEVICE)

                yb = yb.to(DEVICE)

                # ------------------------------------------------------------
                # Reset gradients
                # ------------------------------------------------------------

                self.optimizer.zero_grad()

                # ------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------

                logits = self.model(
                    xb
                ).squeeze(1)

                # ------------------------------------------------------------
                # Classification loss
                # ------------------------------------------------------------

                task_loss = self.criterion(
                    logits,
                    yb,
                )

                # ------------------------------------------------------------
                # FedProx proximal term
                # ------------------------------------------------------------

                if global_params is not None:

                    prox_loss = (
                        mu / 2.0
                    ) * sum(
                        (
                            (p - g) ** 2
                        ).sum()
                        for p, g in zip(
                            self.model.parameters(),
                            global_params,
                        )
                    )

                    loss = (
                        task_loss
                        + prox_loss
                    )

                else:

                    loss = task_loss

                # ------------------------------------------------------------
                # Backpropagation
                # ------------------------------------------------------------

                loss.backward()

                self.optimizer.step()

                # ------------------------------------------------------------
                # Track task loss only
                # ------------------------------------------------------------

                epoch_task_loss += (
                    task_loss.item()
                    * xb.size(0)
                )

            # ----------------------------------------------------------------
            # Epoch average loss
            # ----------------------------------------------------------------

            if self.n_train > 0:

                avg_task_loss = (
                    epoch_task_loss
                    / self.n_train
                )

            else:

                avg_task_loss = 0.0

            # ----------------------------------------------------------------
            # Logging
            # ----------------------------------------------------------------

            self.monitor_logger.info(
                "Round %d | epoch %d/%d | "
                "task_loss=%.6f | FedProx mu=%.6f",
                fl_round,
                epoch + 1,
                self.local_epochs,
                avg_task_loss,
                mu,
            )

            # ----------------------------------------------------------------
            # Monitoring: training loss
            # ----------------------------------------------------------------

            record_client_metrics(
                client_name=self.client_name,
                round_number=fl_round,
                phase="train",
                loss=avg_task_loss,
            )

            # ----------------------------------------------------------------
            # Monitoring: FedProx
            # ----------------------------------------------------------------

            record_metric(
                client_name=self.client_name,
                round_number=fl_round,
                phase="train",
                metric="fedprox_mu",
                value=mu,
            )

        # --------------------------------------------------------------------
        # Training duration
        # --------------------------------------------------------------------

        training_time = (
            time.perf_counter()
            - training_start
        )

        self.monitor_logger.info(
            "Round %d | training completed | "
            "final_loss=%.6f | duration=%.3f s | samples=%d",
            fl_round,
            avg_task_loss,
            training_time,
            self.n_train,
        )

        # --------------------------------------------------------------------
        # Monitoring: training duration
        # --------------------------------------------------------------------

        record_client_metrics(
            client_name=self.client_name,
            round_number=fl_round,
            phase="train",
            loss=avg_task_loss,
            samples=self.n_train,
            duration_seconds=training_time,
        )

        # --------------------------------------------------------------------
        # Monitoring: samples
        # --------------------------------------------------------------------

        record_metric(
            client_name=self.client_name,
            round_number=fl_round,
            phase="train",
            metric="samples",
            value=float(self.n_train),
        )

        # --------------------------------------------------------------------
        # Monitoring: duration
        # --------------------------------------------------------------------

        record_metric(
            client_name=self.client_name,
            round_number=fl_round,
            phase="train",
            metric="duration_seconds",
            value=training_time,
        )

        # --------------------------------------------------------------------
        # Encrypt local model
        # --------------------------------------------------------------------

        self.monitor_logger.info(
            "Round %d | encrypting local model...",
            fl_round,
        )

        encrypted_parameters = self.get_parameters(
            config={}
        )

        self.monitor_logger.info(
            "Round %d | encrypted model ready "
            "for transmission",
            fl_round,
        )

        # --------------------------------------------------------------------
        # Return
        # --------------------------------------------------------------------

        return (
            encrypted_parameters,
            self.n_train,
            {
                "train_loss": avg_task_loss,
            },
        )

    # ========================================================================
    # Evaluation
    # ========================================================================

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[
        float,
        int,
        Dict,
    ]:
        """
        Evaluate the global model on the client's local test set.
        """

        # --------------------------------------------------------------------
        # FL round
        # --------------------------------------------------------------------

        fl_round = int(
            config.get(
                "round",
                0,
            )
        )

        # --------------------------------------------------------------------
        # Evaluation timer
        # --------------------------------------------------------------------

        evaluation_start = (
            time.perf_counter()
        )

        # --------------------------------------------------------------------
        # Load global model
        # --------------------------------------------------------------------

        self.set_parameters(
            parameters
        )

        self.model.eval()

        # --------------------------------------------------------------------
        # Evaluation variables
        # --------------------------------------------------------------------

        total_loss = 0.0

        correct = 0

        total = 0

        # --------------------------------------------------------------------
        # Inference
        # --------------------------------------------------------------------

        with torch.no_grad():

            for xb, yb in self.test_loader:

                xb = xb.to(DEVICE)

                yb = yb.to(DEVICE)

                # ------------------------------------------------------------
                # Forward pass
                # ------------------------------------------------------------

                logits = self.model(
                    xb
                ).squeeze(1)

                # ------------------------------------------------------------
                # Loss
                # ------------------------------------------------------------

                loss = self.criterion(
                    logits,
                    yb,
                )

                total_loss += (
                    loss.item()
                    * xb.size(0)
                )

                # ------------------------------------------------------------
                # Binary prediction
                # ------------------------------------------------------------

                predictions = (
                    torch.sigmoid(logits)
                    > 0.5
                ).float()

                correct += (
                    predictions == yb
                ).sum().item()

                total += yb.size(0)

        # --------------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------------

        avg_loss = (
            total_loss / total
            if total > 0
            else 0.0
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        # --------------------------------------------------------------------
        # Evaluation duration
        # --------------------------------------------------------------------

        evaluation_time = (
            time.perf_counter()
            - evaluation_start
        )

        # --------------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------------

        self.monitor_logger.info(
            "Round %d | evaluation completed | "
            "loss=%.6f | accuracy=%.6f | "
            "duration=%.3f s | samples=%d",
            fl_round,
            avg_loss,
            accuracy,
            evaluation_time,
            total,
        )

        # --------------------------------------------------------------------
        # Monitoring: evaluation metrics
        # --------------------------------------------------------------------

        record_client_metrics(
            client_name=self.client_name,
            round_number=fl_round,
            phase="evaluate",
            loss=avg_loss,
            accuracy=accuracy,
            samples=total,
            duration_seconds=evaluation_time,
        )

        # --------------------------------------------------------------------
        # Generic metrics
        # --------------------------------------------------------------------

        record_metric(
            client_name=self.client_name,
            round_number=fl_round,
            phase="evaluate",
            metric="accuracy",
            value=accuracy,
        )

        record_metric(
            client_name=self.client_name,
            round_number=fl_round,
            phase="evaluate",
            metric="loss",
            value=avg_loss,
        )

        record_metric(
            client_name=self.client_name,
            round_number=fl_round,
            phase="evaluate",
            metric="samples",
            value=float(total),
        )

        record_metric(
            client_name=self.client_name,
            round_number=fl_round,
            phase="evaluate",
            metric="duration_seconds",
            value=evaluation_time,
        )

        # --------------------------------------------------------------------
        # Return evaluation result
        # --------------------------------------------------------------------

        return (
            avg_loss,
            total,
            {
                "accuracy": accuracy,
            },
        )


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "FL-IDS-OT-ICS — Flower client "
            "(HE + FedProx + Monitoring + TLS)"
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    # ------------------------------------------------------------------------
    # Client name
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--client-name",
        required=True,
        choices=[
            "beneficiation",
            "sap",
            "pap",
            "power",
            "utilities",
            "granulation",
        ],
        help=(
            "Industrial site identifier. "
            "Must match the partition directory name."
        ),
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--model-type",
        choices=[
            "mlp",
            "logreg",
        ],
        default="mlp",
        help=(
            "Model architecture. "
            "Must be identical across all clients and server."
        ),
    )

    # ------------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--server-address",
        default="127.0.0.1:8080",
        help=(
            "gRPC address of the Flower aggregation server."
        ),
    )

    # ------------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=3,
        help=(
            "Number of local training epochs "
            "per FL round."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Local mini-batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Local Adam learning rate.",
    )

    # ------------------------------------------------------------------------
    # HE context
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--he-context",
        default=DEFAULT_HE_CONTEXT,
        help=(
            "Path to the FULL TenSEAL HE context "
            "containing the client's secret key."
        ),
    )

    # ------------------------------------------------------------------------
    # TLS CA certificate
    # ------------------------------------------------------------------------

    parser.add_argument(
        "--ca-cert",
        default=DEFAULT_CA_CERTIFICATE,
        help=(
            "Path to the CA certificate used "
            "to verify the Flower TLS server."
        ),
    )

    # ------------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------------

    args = parser.parse_args()

    # ------------------------------------------------------------------------
    # Monitoring initialization
    # ------------------------------------------------------------------------

    initialize_monitoring()

    client_logger = get_client_logger(
        args.client_name
    )

    # ------------------------------------------------------------------------
    # Startup information
    # ------------------------------------------------------------------------

    client_logger.info(
        "=" * 70
    )

    client_logger.info(
        "FL client starting"
    )

    client_logger.info(
        "Client        : %s",
        args.client_name,
    )

    client_logger.info(
        "Server        : %s",
        args.server_address,
    )

    client_logger.info(
        "Model         : %s",
        args.model_type,
    )

    client_logger.info(
        "Local epochs  : %d",
        args.local_epochs,
    )

    client_logger.info(
        "Batch size    : %d",
        args.batch_size,
    )

    client_logger.info(
        "Learning rate : %.6f",
        args.lr,
    )

    client_logger.info(
        "Device        : %s",
        DEVICE,
    )

    client_logger.info(
        "HE context    : %s",
        args.he_context,
    )

    client_logger.info(
        "CA certificate: %s",
        args.ca_cert,
    )

    # ------------------------------------------------------------------------
    # Verify CA certificate
    # ------------------------------------------------------------------------

    if not os.path.exists(
        args.ca_cert
    ):

        client_logger.error(
            "CA certificate not found: %s",
            args.ca_cert,
        )

        raise FileNotFoundError(
            f"CA certificate not found: {args.ca_cert}"
        )

    client_logger.info(
        "TLS CA certificate found"
    )

    # ------------------------------------------------------------------------
    # Create client
    # ------------------------------------------------------------------------

    client = IDSFlowerClient(
        client_name=args.client_name,
        he_context_path=args.he_context,
        model_type=args.model_type,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
    )

    # ------------------------------------------------------------------------
    # Start Flower client
    # ------------------------------------------------------------------------

    client_logger.info(
        "Connecting to Flower server using TLS..."
    )

    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client,
        root_certificates=Path(
            args.ca_cert
        ).read_bytes(),
    )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    main()