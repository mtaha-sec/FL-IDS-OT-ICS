"""
client_app.py
==============
Flower client for FL-IDS-OT-ICS with:

  • Homomorphic Encryption (TenSEAL CKKS)
      - get_parameters() encrypts trainable model weights before sending them
        to the server.  The server NEVER sees plaintext client parameters.
      - set_parameters() decrypts the aggregated global model received from
        the server using the client's secret key.

  • FedProx (Li et al., 2020)
      - The local training loop adds a proximal term:
            loss = task_loss + (μ/2) · ‖w_local − w_global‖²
        where μ is broadcast by the server each round.
        This limits client drift on Non-IID industrial network data.

Parameter encoding (agreed with HEFedProxStrategy on the server)
-----------------------------------------------------------------
  parameters[0]      : trainable params (dtype=uint8 ciphertext after round 0,
                        dtype=float32 flat array for round 0 init)
  parameters[1 .. M] : plaintext BN buffers (running_mean/var/count)

Usage (one terminal per industrial site)
-----------------------------------------
    python clients/client_app.py \\
        --client-name power \\
        --server-address 127.0.0.1:8080 \\
        --he-context security/keys/he_context_full.seal

    python clients/client_app.py --client-name sap      --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name pap      --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name utilities --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name beneficiation --server-address 127.0.0.1:8080
    python clients/client_app.py --client-name granulation --server-address 127.0.0.1:8080
"""

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.local_ids.model import build_model, get_input_dim
from security.he_crypto import (
    decrypt_parameters,
    encrypt_parameters,
    get_param_shapes,
    get_buffer_shapes,
    load_context,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default HE context path (full context with secret key — for clients only)
DEFAULT_HE_CONTEXT = os.path.join("security", "keys", "he_context_full.seal")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_client_data(
    client_name: str,
    partitions_dir: str = "datasets/partitions",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load train.csv / test.csv and convert to float32 tensors."""
    client_dir = os.path.join(partitions_dir, client_name)
    train_df   = pd.read_csv(os.path.join(client_dir, "train.csv"))
    test_df    = pd.read_csv(os.path.join(client_dir, "test.csv"))

    from preprocessing.feature_engineering import get_model_feature_columns
    feature_cols = get_model_feature_columns()

    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(
            f"Expected columns missing from train.csv for '{client_name}': {missing}"
        )

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.float32)

    logger.info(
        "[%s] Data loaded — features=%d  train=%d  test=%d",
        client_name, len(feature_cols), len(X_train), len(X_test),
    )
    return X_train, y_train, X_test, y_test, feature_cols


# ── Flower Client ──────────────────────────────────────────────────────────────

class IDSFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for a single industrial IDS site.

    HE-awareness
    ------------
    • get_parameters() encrypts trainable weights with the CKKS public key.
      The server only ever receives a ciphertext — never the raw weights.
    • set_parameters() detects whether incoming params are:
        - dtype=float32 flat array → round 0 plaintext init from server
        - dtype=uint8             → encrypted aggregated global model (rounds 1+)
      and decrypts accordingly using the full context (secret key).

    FedProx-awareness
    -----------------
    • fit() receives μ from the server config dict.
    • It saves a frozen copy of the global weights at the start of local
      training and adds (μ/2)‖w − w_global‖² to the task loss each step.
    """

    def __init__(
        self,
        client_name:    str,
        he_context_path: str = DEFAULT_HE_CONTEXT,
        model_type:     str  = "mlp",
        batch_size:     int  = 64,
        local_epochs:   int  = 3,
        lr:             float = 1e-3,
    ) -> None:
        self.client_name  = client_name
        self.model_type   = model_type
        self.local_epochs = local_epochs
        self.batch_size   = batch_size

        # ── Load HE context (full, with secret key) ───────────────────────────
        if not os.path.exists(he_context_path):
            raise FileNotFoundError(
                f"HE context not found: {he_context_path}\n"
                f"Run 'python -m security.generate_he_keys' first."
            )
        self.he_ctx = load_context(he_context_path)
        if not self.he_ctx.is_private():
            raise ValueError(
                "Client must use the FULL HE context (he_context_full.seal), "
                "not the public-only context.  The public context is for the server."
            )
        logger.info("[%s] HE context loaded (secret key present ✓)", client_name)

        # Cache parameter / buffer shapes for fast reconstruction
        self.param_shapes  = get_param_shapes()
        self.buffer_shapes = get_buffer_shapes()

        # ── Load local data ───────────────────────────────────────────────────
        X_train, y_train, X_test, y_test, feature_cols = load_client_data(client_name)
        self.input_dim = len(feature_cols)

        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=batch_size, shuffle=True, drop_last=False,
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=batch_size, shuffle=False,
        )
        self.n_train = len(X_train)

        # ── Model — same fixed architecture on all 6 clients + server ─────────
        self.model     = build_model(model_type, input_dim=self.input_dim).to(DEVICE)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        logger.info(
            "[%s] Client ready — model=%s  params=%d  train=%d  device=%s",
            client_name, model_type,
            sum(p.numel() for p in self.model.parameters()),
            self.n_train, DEVICE,
        )

    # ── Parameter Encoding / Decoding ────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Return the local model parameters for transmission to the server.

        Encoding
        --------
        [0] : encrypted trainable params  (dtype=uint8 CKKS ciphertext)
        [1…]: plaintext BN buffers         (running_mean/var/count)

        The server will aggregate [0] homomorphically without decrypting.
        """
        # Encrypt all trainable parameters
        trainable = [p.data.cpu().numpy() for p in self.model.parameters()]
        enc_uint8 = encrypt_parameters(trainable, self.he_ctx)

        # BN buffers sent in plaintext
        buffers = [b.cpu().numpy() for b in self.model.buffers()]

        logger.debug(
            "[%s] get_parameters — ciphertext: %d bytes  buffers: %d",
            self.client_name, enc_uint8.nbytes, len(buffers),
        )
        return [enc_uint8] + buffers

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load incoming global model parameters into the local model.

        Handles two cases automatically:
        ┌───────────────┬──────────────────────────────────────────────────────┐
        │ params[0] dtype│ Meaning                                             │
        ├───────────────┼──────────────────────────────────────────────────────┤
        │ float32       │ Round-0 plaintext init sent by the server            │
        │ uint8         │ Encrypted aggregated global model (rounds 1+)        │
        └───────────────┴──────────────────────────────────────────────────────┘
        The client decrypts uint8 ciphertexts with its secret key.
        """
        enc_or_flat   = parameters[0]
        buffer_arrays = parameters[1:]

        # ── Reconstruct trainable parameters ──────────────────────────────────
        if enc_or_flat.dtype == np.uint8:
            # Encrypted ciphertext → decrypt with secret key
            trainable_list = decrypt_parameters(
                enc_uint8=enc_or_flat,
                ctx=self.he_ctx,
                shapes=self.param_shapes,
            )
            logger.debug("[%s] set_parameters — decrypted global model.", self.client_name)
        else:
            # Plaintext flat float32 array (round 0 server init)
            flat = enc_or_flat.astype(np.float32)
            trainable_list, offset = [], 0
            for shape in self.param_shapes:
                size = int(np.prod(shape))
                trainable_list.append(flat[offset : offset + size].reshape(shape))
                offset += size
            logger.debug("[%s] set_parameters — plaintext init loaded.", self.client_name)

        # ── Build new state dict ───────────────────────────────────────────────
        new_state: Dict[str, torch.Tensor] = {}

        # Trainable params (weights + biases, including BN affine)
        trainable_iter = iter(trainable_list)
        for name, _ in self.model.named_parameters():
            new_state[name] = torch.tensor(next(trainable_iter))

        # BN running buffers
        buf_iter = iter(buffer_arrays)
        for name, existing_buf in self.model.named_buffers():
            arr = next(buf_iter, None)
            new_state[name] = torch.tensor(arr) if arr is not None else existing_buf

        self.model.load_state_dict(new_state, strict=True)

    # ── Flower Callbacks ──────────────────────────────────────────────────────

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Local training with FedProx proximal regularisation.

        Receives
        --------
        parameters : global model parameters from the server
        config     : {"mu": float, "round": int}

        FedProx loss
        ------------
            L(w) = BCE(w; D_k) + (μ/2) · ‖w − w_global‖²

        where w_global is the FROZEN parameter vector received at the start of
        this round.  The proximal term prevents excessive client drift on the
        Non-IID industrial network data.
        """
        mu         = float(config.get("mu", 0.0))
        fl_round   = int(config.get("round", 0))

        self.set_parameters(parameters)

        # Frozen copy of global weights for the proximal term (FedProx)
        global_params: Optional[List[torch.Tensor]] = None
        if mu > 0.0:
            global_params = [p.data.clone().to(DEVICE) for p in self.model.parameters()]

        logger.info(
            "[%s] Round %d — local training  epochs=%d  μ=%.4f",
            self.client_name, fl_round, self.local_epochs, mu,
        )

        self.model.train()
        for epoch in range(self.local_epochs):
            epoch_task_loss = 0.0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()

                logits    = self.model(xb).squeeze(1)
                task_loss = self.criterion(logits, yb)

                # FedProx proximal term
                if global_params is not None:
                    prox_loss = (mu / 2.0) * sum(
                        ((p - g) ** 2).sum()
                        for p, g in zip(self.model.parameters(), global_params)
                    )
                    loss = task_loss + prox_loss
                else:
                    loss = task_loss

                loss.backward()
                self.optimizer.step()
                epoch_task_loss += task_loss.item() * xb.size(0)

            avg_task_loss = epoch_task_loss / self.n_train
            logger.info(
                "[%s] epoch %d/%d — task_loss=%.4f  μ=%.4f",
                self.client_name, epoch + 1, self.local_epochs, avg_task_loss, mu,
            )

        return self.get_parameters(config={}), self.n_train, {"train_loss": avg_task_loss}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """
        Local evaluation on the client's held-out test set.

        Decrypts the received global model, runs inference, and returns
        loss + accuracy (+ optional precision/recall/f1).
        """
        self.set_parameters(parameters)
        self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb  = xb.to(DEVICE), yb.to(DEVICE)
                logits   = self.model(xb).squeeze(1)
                loss     = self.criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                preds    = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == yb).sum().item()
                total   += yb.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct   / total if total > 0 else 0.0
        logger.info(
            "[%s] evaluate — loss=%.5f  accuracy=%.4f",
            self.client_name, avg_loss, accuracy,
        )
        return avg_loss, total, {"accuracy": accuracy}


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FL-IDS-OT-ICS — Flower client (HE + FedProx)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--client-name", required=True,
        choices=["beneficiation", "sap", "pap", "power", "utilities", "granulation"],
        help="Industrial site identifier (must match partition directory name).",
    )
    parser.add_argument(
        "--model-type", choices=["mlp", "logreg"], default="mlp",
        help="Model architecture. Must be identical across all 6 clients and the server.",
    )
    parser.add_argument(
        "--server-address", default="127.0.0.1:8080",
        help="gRPC address of the Flower aggregation server.",
    )
    parser.add_argument(
        "--local-epochs", type=int, default=3,
        help="Local training epochs per FL round.",
    )
    parser.add_argument(
        "--he-context",
        default=DEFAULT_HE_CONTEXT,
        help="Path to the FULL HE context file (with secret key). "
             "Distribute securely from security/generate_he_keys.py output.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size for local training.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Local Adam learning rate.",
    )
    args = parser.parse_args()

    logger.info(
        "Starting FL client '%s' → server %s",
        args.client_name, args.server_address,
    )

    client = IDSFlowerClient(
        client_name      = args.client_name,
        he_context_path  = args.he_context,
        model_type       = args.model_type,
        batch_size       = args.batch_size,
        local_epochs     = args.local_epochs,
        lr               = args.lr,
    )
    fl.client.start_numpy_client(
        server_address = args.server_address,
        client         = client,
    )


if __name__ == "__main__":
    main()