"""
central_server/strategy.py
===========================

HEFedProxStrategy — Flower aggregation strategy combining:

  • Homomorphic Encryption (TenSEAL CKKS)
      The server aggregates model parameters IN THE ENCRYPTED DOMAIN.
      It holds only the PUBLIC key and never decrypts individual client params.

  • FedProx (Li et al., 2020)
      Each client adds a proximal term:
          (μ/2) ||w_k − w^t||²
      during local training.

  • Monitoring
      Records FL round metrics through monitoring.monitor.record_metric().

Parameter encoding
------------------
  parameters[0]      : encrypted trainable params (dtype=uint8)
  parameters[1 .. M] : plaintext BN buffers
"""

import logging
import os
import time
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

from security.he_crypto import (
    aggregate_encrypted_params,
    load_context,
)

from monitoring.monitor import record_metric


logger = logging.getLogger(__name__)


class HEFedProxStrategy(fl.server.strategy.Strategy):
    """
    Federated Learning strategy with:

      - Homomorphic Encryption aggregation
      - FedProx client-side regularisation
      - Server-side monitoring
    """

    def __init__(
        self,
        public_context_path: str,
        mu: float = 0.01,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        save_dir: str = "checkpoints/fl",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
    ) -> None:

        super().__init__()

        self.public_context_path = public_context_path
        self.mu = mu

        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        self.initial_parameters = initial_parameters
        self.save_dir = save_dir

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate

        # ---------------------------------------------------------------
        # Load PUBLIC HE context
        # ---------------------------------------------------------------

        self.pub_ctx = load_context(public_context_path)

        if self.pub_ctx.is_private():
            raise ValueError(
                "The server loaded a FULL HE context containing the "
                "secret key. The server must use the PUBLIC context only."
            )

        # ---------------------------------------------------------------
        # Checkpoint directory
        # ---------------------------------------------------------------

        os.makedirs(save_dir, exist_ok=True)

        # ---------------------------------------------------------------
        # Monitoring
        # ---------------------------------------------------------------

        self.round_start_times: Dict[int, float] = {}

        logger.info(
            "HEFedProxStrategy ready — μ=%.4f  min_clients=%d "
            "HE public context=%s",
            mu,
            min_fit_clients,
            public_context_path,
        )

    # ===================================================================
    # INITIAL PARAMETERS
    # ===================================================================

    def initialize_parameters(
        self,
        client_manager: ClientManager,
    ) -> Optional[Parameters]:

        """
        Return plaintext initial parameters for round 0.
        """

        if self.initial_parameters is not None:

            logger.info(
                "Round 0 — sending plaintext initial model to clients."
            )

            return self.initial_parameters

        logger.info(
            "Round 0 — no initial parameters; "
            "clients will initialise randomly."
        )

        return None

    # ===================================================================
    # CONFIGURE FIT
    # ===================================================================

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:

        """
        Select clients and send:

          - current global parameters
          - FedProx μ
          - current round number
        """

        # ---------------------------------------------------------------
        # Start round timer
        # ---------------------------------------------------------------

        self.round_start_times[server_round] = time.perf_counter()

        # ---------------------------------------------------------------
        # Configuration sent to clients
        # ---------------------------------------------------------------

        config = {
            "mu": float(self.mu),
            "round": server_round,
        }

        fit_ins = FitIns(
            parameters=parameters,
            config=config,
        )

        # ---------------------------------------------------------------
        # Client selection
        # ---------------------------------------------------------------

        n_available = client_manager.num_available()

        sample_size = max(
            self.min_fit_clients,
            int(n_available * self.fraction_fit),
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
        )

        # ---------------------------------------------------------------
        # Logs
        # ---------------------------------------------------------------

        logger.info(
            "[Round %d] configure_fit — %d/%d clients selected, μ=%.4f",
            server_round,
            len(clients),
            n_available,
            self.mu,
        )

        # ---------------------------------------------------------------
        # Monitoring
        # ---------------------------------------------------------------

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="round",
            metric="fedprox_mu",
            value=float(self.mu),
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="round",
            metric="clients_selected",
            value=len(clients),
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="round",
            metric="clients_available",
            value=n_available,
        )

        return [
            (client, fit_ins)
            for client in clients
        ]

    # ===================================================================
    # AGGREGATE FIT
    # ===================================================================

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[
            Union[
                Tuple[ClientProxy, FitRes],
                BaseException,
            ]
        ],
    ) -> Tuple[
        Optional[Parameters],
        Dict[str, Scalar],
    ]:

        """
        Aggregate encrypted client model parameters.

        The server NEVER decrypts the client parameters.
        """

        # ---------------------------------------------------------------
        # No results
        # ---------------------------------------------------------------

        if not results:

            logger.warning(
                "[Round %d] No fit results received — skipping.",
                server_round,
            )

            return None, {}

        # ---------------------------------------------------------------
        # Failures
        # ---------------------------------------------------------------

        if failures:

            logger.warning(
                "[Round %d] %d client(s) failed during fit — "
                "proceeding with %d successful clients.",
                server_round,
                len(failures),
                len(results),
            )

        # ---------------------------------------------------------------
        # Sample counts
        # ---------------------------------------------------------------

        n_examples = [
            fit_res.num_examples
            for _, fit_res in results
        ]

        N = sum(n_examples)

        if N <= 0:

            logger.error(
                "[Round %d] Invalid total number of samples: %d",
                server_round,
                N,
            )

            return None, {}

        # ---------------------------------------------------------------
        # FedAvg weighting
        # ---------------------------------------------------------------

        weights = [
            n / N
            for n in n_examples
        ]

        client_ids = [
            client_proxy.cid
            for client_proxy, _ in results
        ]

        logger.info(
            "[Round %d] Aggregating %d clients — total_samples=%d",
            server_round,
            len(results),
            N,
        )

        logger.info(
            "[Round %d] Clients: %s | weights: %s",
            server_round,
            client_ids,
            [
                f"{weight:.4f}"
                for weight in weights
            ],
        )

        # ---------------------------------------------------------------
        # Monitoring — aggregation information
        # ---------------------------------------------------------------

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_fit",
            metric="clients",
            value=len(results),
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_fit",
            metric="samples",
            value=N,
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_fit",
            metric="failed_clients",
            value=len(failures),
        )

        # ---------------------------------------------------------------
        # Convert Flower parameters
        # ---------------------------------------------------------------

        all_ndarrays = [
            parameters_to_ndarrays(
                fit_res.parameters
            )
            for _, fit_res in results
        ]

        # parameters[0] = encrypted model
        enc_params_list = [
            ndarrays[0]
            for ndarrays in all_ndarrays
        ]

        # parameters[1:] = BN buffers
        buffer_lists = [
            ndarrays[1:]
            for ndarrays in all_ndarrays
        ]

        # ---------------------------------------------------------------
        # Homomorphic aggregation
        # ---------------------------------------------------------------

        logger.info(
            "[Round %d] Running HE aggregation — "
            "server holds public key only.",
            server_round,
        )

        aggregation_start = time.perf_counter()

        agg_enc = aggregate_encrypted_params(
            enc_list=enc_params_list,
            weights=weights,
            pub_ctx=self.pub_ctx,
        )

        aggregation_duration = (
            time.perf_counter()
            - aggregation_start
        )

        logger.info(
            "[Round %d] HE aggregation done — "
            "ciphertext size=%d bytes — duration=%.3fs",
            server_round,
            agg_enc.nbytes,
            aggregation_duration,
        )

        # ---------------------------------------------------------------
        # Monitoring — HE aggregation
        # ---------------------------------------------------------------

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_fit",
            metric="he_aggregation_duration_seconds",
            value=aggregation_duration,
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_fit",
            metric="ciphertext_bytes",
            value=agg_enc.nbytes,
        )

        # ---------------------------------------------------------------
        # Aggregate plaintext BN buffers
        # ---------------------------------------------------------------

        agg_buffers: List[np.ndarray] = []

        if buffer_lists and buffer_lists[0]:

            n_buffers = len(
                buffer_lists[0]
            )

            for i in range(n_buffers):

                client_bufs = [
                    buffer_list[i]
                    for buffer_list in buffer_lists
                ]

                original_dtype = (
                    client_bufs[0].dtype
                )

                agg_buf = sum(
                    weight * buffer.astype(np.float64)
                    for weight, buffer
                    in zip(
                        weights,
                        client_bufs,
                    )
                ).astype(original_dtype)

                agg_buffers.append(
                    agg_buf
                )

            logger.info(
                "[Round %d] %d BN buffer(s) "
                "averaged in plaintext.",
                server_round,
                n_buffers,
            )

        # ---------------------------------------------------------------
        # Pack aggregated parameters
        # ---------------------------------------------------------------

        agg_ndarrays = [
            agg_enc
        ] + agg_buffers

        agg_params = ndarrays_to_parameters(
            agg_ndarrays
        )

        # ---------------------------------------------------------------
        # Aggregate training metrics
        # ---------------------------------------------------------------

        metrics_agg: Dict[str, Scalar] = {
            "round": server_round,
            "num_clients": len(results),
            "total_samples": N,
        }

        client_metrics = [
            fit_res.metrics
            for _, fit_res in results
        ]

        # Weighted training loss
        train_loss_values = [
            metrics.get("train_loss")
            for metrics in client_metrics
        ]

        if all(
            value is not None
            and isinstance(value, (int, float))
            for value in train_loss_values
        ):

            weighted_train_loss = float(
                sum(
                    weight * float(value)
                    for weight, value
                    in zip(
                        weights,
                        train_loss_values,
                    )
                )
            )

            metrics_agg["train_loss"] = (
                weighted_train_loss
            )

            record_metric(
                client_name="server",
                round_number=server_round,
                phase="aggregate_fit",
                metric="train_loss",
                value=weighted_train_loss,
            )

        # ---------------------------------------------------------------
        # Round duration
        # ---------------------------------------------------------------

        if server_round in self.round_start_times:

            round_duration = (
                time.perf_counter()
                - self.round_start_times[
                    server_round
                ]
            )

            record_metric(
                client_name="server",
                round_number=server_round,
                phase="aggregate_fit",
                metric="round_duration_seconds",
                value=round_duration,
            )

        # ---------------------------------------------------------------
        # Final log
        # ---------------------------------------------------------------

        logger.info(
            "[Round %d] aggregate_fit complete — metrics=%s",
            server_round,
            metrics_agg,
        )

        return (
            agg_params,
            metrics_agg,
        )

    # ===================================================================
    # CONFIGURE EVALUATE
    # ===================================================================

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[
        Tuple[ClientProxy, EvaluateIns]
    ]:

        """
        Send the encrypted global model to clients
        for local evaluation.
        """

        if self.fraction_evaluate == 0.0:
            return []

        n_available = (
            client_manager.num_available()
        )

        sample_size = max(
            self.min_evaluate_clients,
            int(
                n_available
                * self.fraction_evaluate
            ),
        )

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients,
        )

        evaluate_ins = EvaluateIns(
            parameters=parameters,
            config={
                "round": server_round
            },
        )

        logger.info(
            "[Round %d] configure_evaluate — "
            "%d clients selected.",
            server_round,
            len(clients),
        )

        # ---------------------------------------------------------------
        # Monitoring
        # ---------------------------------------------------------------

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="evaluate",
            metric="clients_selected",
            value=len(clients),
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="evaluate",
            metric="clients_available",
            value=n_available,
        )

        return [
            (client, evaluate_ins)
            for client in clients
        ]

    # ===================================================================
    # AGGREGATE EVALUATE
    # ===================================================================

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[
            Tuple[ClientProxy, EvaluateRes]
        ],
        failures: List[
            Union[
                Tuple[ClientProxy, EvaluateRes],
                BaseException,
            ]
        ],
    ) -> Tuple[
        Optional[float],
        Dict[str, Scalar],
    ]:

        """
        Aggregate client evaluation metrics.
        """

        if not results:

            logger.warning(
                "[Round %d] No evaluation results.",
                server_round,
            )

            return None, {}

        # ---------------------------------------------------------------
        # Sample counts
        # ---------------------------------------------------------------

        n_examples = [
            evaluate_res.num_examples
            for _, evaluate_res in results
        ]

        N = sum(n_examples)

        if N <= 0:

            logger.error(
                "[Round %d] Invalid evaluation sample count.",
                server_round,
            )

            return None, {}

        weights = [
            n / N
            for n in n_examples
        ]

        # ---------------------------------------------------------------
        # Weighted loss
        # ---------------------------------------------------------------

        avg_loss = float(
            sum(
                weight * evaluate_res.loss
                for weight, (_, evaluate_res)
                in zip(
                    weights,
                    results,
                )
            )
        )

        # ---------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------

        metrics_agg: Dict[str, Scalar] = {
            "round": server_round,
            "num_clients": len(results),
            "total_samples": N,
        }

        metric_names = (
            "accuracy",
            "precision",
            "recall",
            "f1",
        )

        for key in metric_names:

            values = [
                evaluate_res.metrics.get(key)
                for _, evaluate_res in results
            ]

            if all(
                value is not None
                and isinstance(value, (int, float))
                for value in values
            ):

                weighted_value = float(
                    sum(
                        weight * float(value)
                        for weight, value
                        in zip(
                            weights,
                            values,
                        )
                    )
                )

                metrics_agg[key] = (
                    weighted_value
                )

        # ---------------------------------------------------------------
        # Logs
        # ---------------------------------------------------------------

        logger.info(
            "[Round %d] Eval aggregated — "
            "loss=%.5f | accuracy=%.4f",
            server_round,
            avg_loss,
            float(
                metrics_agg.get(
                    "accuracy",
                    float("nan"),
                )
            ),
        )

        # ---------------------------------------------------------------
        # Monitoring — evaluation
        # ---------------------------------------------------------------

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_evaluate",
            metric="loss",
            value=avg_loss,
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_evaluate",
            metric="clients",
            value=len(results),
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_evaluate",
            metric="samples",
            value=N,
        )

        record_metric(
            client_name="server",
            round_number=server_round,
            phase="aggregate_evaluate",
            metric="failed_clients",
            value=len(failures),
        )

        for key in metric_names:

            if key in metrics_agg:

                record_metric(
                    client_name="server",
                    round_number=server_round,
                    phase="aggregate_evaluate",
                    metric=key,
                    value=float(
                        metrics_agg[key]
                    ),
                )

        # ---------------------------------------------------------------
        # Cleanup round timer
        # ---------------------------------------------------------------

        self.round_start_times.pop(
            server_round,
            None,
        )

        return (
            avg_loss,
            metrics_agg,
        )

    # ===================================================================
    # SERVER-SIDE EVALUATION
    # ===================================================================

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[
        Tuple[
            float,
            Dict[str, Scalar],
        ]
    ]:

        """
        Server-side evaluation is disabled.

        The server holds only the public HE context and therefore
        cannot decrypt the encrypted global model.

        Evaluation is performed by the clients.
        """

        return None