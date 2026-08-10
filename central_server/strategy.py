"""
central_server/strategy.py
===========================
HEFedProxStrategy — Flower aggregation strategy combining:

  • Homomorphic Encryption (TenSEAL CKKS)
      The server aggregates model parameters IN THE ENCRYPTED DOMAIN.
      It holds only the PUBLIC key and never decrypts individual client params.
      Only clients (secret-key holders) can decrypt the aggregated global model.

  • FedProx (Li et al., 2020)
      Each client adds a proximal term  (μ/2)‖w_k − w^t‖²  to its local
      objective during training.  This limits client drift on Non-IID data.
      μ is broadcast to clients via Flower's config dict every round.

Parameter encoding (transported through Flower's numpy parameter system)
------------------------------------------------------------------------
  parameters[0]      : encrypted trainable params   (dtype=uint8, ciphertext)
  parameters[1 .. M] : plaintext BN buffers          (running_mean/var/count)

The separation lets us protect the sensitive learned weights/biases with HE
while keeping BatchNorm statistics (which reveal only data-distribution moments)
in plaintext to avoid CKKS slot overflow.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from security.he_crypto import aggregate_encrypted_params, load_context

logger = logging.getLogger(__name__)


class HEFedProxStrategy(fl.server.strategy.Strategy):
    """
    Federated Learning strategy with server-side Homomorphic Encryption
    aggregation and FedProx proximal regularisation on the client side.

    Parameters
    ----------
    public_context_path : str
        Path to the server's PUBLIC HE context (.seal file, no secret key).
    mu : float
        FedProx proximal coefficient (0.0 = pure FedAvg).  Default 0.01.
    min_fit_clients : int
        Minimum clients required before a fit round starts.
    min_evaluate_clients : int
        Minimum clients for an evaluation round.
    min_available_clients : int
        Minimum connected clients before *any* round starts.
    initial_parameters : Optional[Parameters]
        Plaintext initial global model parameters for round 0.
        If None, clients initialise randomly.
    save_dir : str
        Directory to save global model checkpoints after every round.
    fraction_fit : float
        Fraction of available clients sampled per fit round (1.0 = all).
    fraction_evaluate : float
        Fraction of available clients sampled per evaluation round.
    """

    def __init__(
        self,
        public_context_path:   str,
        mu:                    float = 0.01,
        min_fit_clients:       int   = 2,
        min_evaluate_clients:  int   = 2,
        min_available_clients: int   = 2,
        initial_parameters:    Optional[Parameters] = None,
        save_dir:              str   = "checkpoints/fl",
        fraction_fit:          float = 1.0,
        fraction_evaluate:     float = 1.0,
    ) -> None:
        super().__init__()
        self.public_context_path   = public_context_path
        self.mu                    = mu
        self.min_fit_clients       = min_fit_clients
        self.min_evaluate_clients  = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters    = initial_parameters
        self.save_dir              = save_dir
        self.fraction_fit          = fraction_fit
        self.fraction_evaluate     = fraction_evaluate

        # Load the public HE context — server never has the secret key
        self.pub_ctx = load_context(public_context_path)
        if self.pub_ctx.is_private():
            raise ValueError(
                "The server loaded a FULL HE context (contains secret key). "
                "Pass the PUBLIC context file (he_context_public.seal) to the server. "
                "The secret key must NOT reside on the server."
            )

        os.makedirs(save_dir, exist_ok=True)
        logger.info(
            "HEFedProxStrategy ready — μ=%.4f  min_clients=%d  "
            "HE public context: %s",
            mu, min_fit_clients, public_context_path,
        )

    # ── Flower Strategy Interface ─────────────────────────────────────────────

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Optional[Parameters]:
        """Return plaintext initial parameters for round 0, if provided."""
        if self.initial_parameters is not None:
            logger.info("Round 0 — sending plaintext initial model to clients.")
            return self.initial_parameters
        logger.info("Round 0 — no initial parameters; clients will init randomly.")
        return None

    # ─────────────────────────────────────────────────────────────────────────
    def configure_fit(
        self,
        server_round: int,
        parameters:   Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Sample clients and send them:
          • Current global parameters  (encrypted ciphertext + plaintext BN buffers)
          • Config dict: FedProx μ and the current round number
        """
        config = {
            "mu":    float(self.mu),
            "round": server_round,
        }
        fit_ins = FitIns(parameters=parameters, config=config)

        n_available = client_manager.num_available()
        sample_size = max(
            self.min_fit_clients,
            int(n_available * self.fraction_fit),
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
        )
        logger.info(
            "[Round %d] configure_fit — %d/%d clients selected, μ=%.4f",
            server_round, len(clients), n_available, self.mu,
        )
        return [(c, fit_ins) for c in clients]

    # ─────────────────────────────────────────────────────────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results:  List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client model updates **in the encrypted domain**.

        Steps
        -----
        1. Extract encrypted trainable params  (ciphertext, dtype=uint8)
        2. Compute per-client weights  n_k / N
        3. Homomorphic aggregation:  Enc(w_global) = Σ_k (n_k/N) · Enc(w_k)
           → server never calls decrypt!
        4. Weighted average of plaintext BN buffers
        5. Pack aggregated (still encrypted) result → send to clients
        6. Clients decrypt w_global with their secret key
        """
        if not results:
            logger.warning("[Round %d] No fit results received — skipping.", server_round)
            return None, {}

        if failures:
            logger.warning(
                "[Round %d] %d client(s) failed during fit — proceeding with %d.",
                server_round, len(failures), len(results),
            )

        # ── Per-client sample counts and aggregation weights ──────────────────
        n_examples = [fit_res.num_examples for _, fit_res in results]
        N          = sum(n_examples)
        # Standard linear weighting: w_k = n_k / N
        # (This allows the large, balanced 'granulation' dataset to serve as an anchor)
        weights = [n / N for n in n_examples]

        client_ids = [cp.cid for cp, _ in results]
        logger.info(
            "[Round %d] Aggregating %d clients — total_samples=%d",
            server_round, len(results), N,
        )
        logger.info(
            "[Round %d] Clients: %s | weights: %s",
            server_round,
            client_ids,
            [f"{w:.4f}" for w in weights],
        )

        # ── Extract parameter arrays from each client ─────────────────────────
        # parameters[0]    = encrypted trainable params (dtype=uint8 ciphertext)
        # parameters[1..M] = plaintext BN buffers
        all_ndarrays     = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        enc_params_list  = [ndarrays[0] for ndarrays in all_ndarrays]
        buffer_lists     = [ndarrays[1:] for ndarrays in all_ndarrays]

        # ── Step 3 : Homomorphic aggregation (no decryption on server) ────────
        logger.info(
            "[Round %d] Running HE aggregation — server holds public key only, "
            "no client weights are ever decrypted here.",
            server_round,
        )
        agg_enc = aggregate_encrypted_params(
            enc_list=enc_params_list,
            weights=weights,
            pub_ctx=self.pub_ctx,
        )
        logger.info(
            "[Round %d] HE aggregation done — ciphertext size: %d bytes.",
            server_round, agg_enc.nbytes,
        )

        # ── Step 4 : Weighted average of plaintext BN buffers ─────────────────
        agg_buffers: List[np.ndarray] = []
        if buffer_lists and buffer_lists[0]:
            n_buffers = len(buffer_lists[0])
            for i in range(n_buffers):
                client_bufs = [bl[i] for bl in buffer_lists]
                orig_dtype  = client_bufs[0].dtype
                agg_buf = sum(
                    w * b.astype(np.float64)
                    for w, b in zip(weights, client_bufs)
                ).astype(orig_dtype)
                agg_buffers.append(agg_buf)
            logger.info(
                "[Round %d] %d BN buffer(s) averaged (plaintext).",
                server_round, n_buffers,
            )

        # ── Step 5 : Pack and return aggregated parameters ────────────────────
        agg_ndarrays = [agg_enc] + agg_buffers
        agg_params   = ndarrays_to_parameters(agg_ndarrays)

        # ── Aggregate scalar metrics ──────────────────────────────────────────
        metrics_agg: Dict[str, Scalar] = {
            "round":       server_round,
            "num_clients": len(results),
            "total_samples": N,
        }
        client_metrics = [fit_res.metrics for _, fit_res in results]
        for key in ("train_loss",):
            vals = [m.get(key) for m in client_metrics]
            if all(v is not None and isinstance(v, (int, float)) for v in vals):
                metrics_agg[key] = float(sum(w * v for w, v in zip(weights, vals)))  # type: ignore

        logger.info("[Round %d] aggregate_fit complete — metrics: %s", server_round, metrics_agg)
        return agg_params, metrics_agg

    # ─────────────────────────────────────────────────────────────────────────
    def configure_evaluate(
        self,
        server_round:   int,
        parameters:     Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Send global model to a sample of clients for local evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        n_available = client_manager.num_available()
        sample_size = max(
            self.min_evaluate_clients,
            int(n_available * self.fraction_evaluate),
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients,
        )
        evaluate_ins = EvaluateIns(parameters=parameters, config={"round": server_round})
        logger.info(
            "[Round %d] configure_evaluate — %d clients selected.",
            server_round, len(clients),
        )
        return [(c, evaluate_ins) for c in clients]

    # ─────────────────────────────────────────────────────────────────────────
    def aggregate_evaluate(
        self,
        server_round: int,
        results:  List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Weighted average of client evaluation losses and accuracy metrics."""
        if not results:
            return None, {}

        n_examples = [eval_res.num_examples for _, eval_res in results]
        N          = sum(n_examples)
        weights = [n / N for n in n_examples]

        avg_loss = float(sum(w * r.loss for w, (_, r) in zip(weights, results)))

        metrics_agg: Dict[str, Scalar] = {"round": server_round}
        for key in ("accuracy", "precision", "recall", "f1"):
            vals = [r.metrics.get(key) for _, r in results]
            if all(v is not None and isinstance(v, (int, float)) for v in vals):
                metrics_agg[key] = float(sum(w * v for w, v in zip(weights, vals)))  # type: ignore

        logger.info(
            "[Round %d] Eval aggregated — loss=%.5f  accuracy=%.4f",
            server_round, avg_loss,
            float(metrics_agg.get("accuracy", float("nan"))),
        )
        return avg_loss, metrics_agg

    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        server_round: int,
        parameters:   Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Optional server-side evaluation.
        Not used — relies on distributed client-side evaluation instead.
        The server cannot evaluate anyway since it holds only encrypted weights.
        """
        return None
