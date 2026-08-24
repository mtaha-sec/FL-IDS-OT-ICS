import argparse
import logging
import flwr as fl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CustomFedProx(fl.server.strategy.FedProx):
    def aggregate_evaluate(self, server_round, results, failures):
        # Appel de la méthode parente pour calculer la loss globale
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if not results:
            return loss, metrics
            
        # Agrégation personnalisée de toutes nos métriques (pondération par le nombre d'exemples de test)
        total_examples = sum([res.num_examples for _, res in results])
        aggregated_metrics = {}
        
        if total_examples > 0:
            for metric_name in ["accuracy", "precision", "recall", "f1"]:
                weighted_sum = sum([
                    res.num_examples * res.metrics.get(metric_name, 0.0) 
                    for _, res in results
                ])
                aggregated_metrics[metric_name] = weighted_sum / total_examples
                
        # Impression formatée pour que FLRunnerWorker la parse facilement
        logger.info(
            f"[GLOBAL_METRICS] round={server_round} "
            f"loss={loss:.4f} "
            f"accuracy={aggregated_metrics.get('accuracy', 0.0):.4f} "
            f"precision={aggregated_metrics.get('precision', 0.0):.4f} "
            f"recall={aggregated_metrics.get('recall', 0.0):.4f} "
            f"f1={aggregated_metrics.get('f1', 0.0):.4f}"
        )
        
        return loss, aggregated_metrics

def main():
    parser = argparse.ArgumentParser(description="Serveur Central Flower (FedProx)")
    parser.add_argument("--num-rounds", type=int, default=10, help="Nombre de rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Nombre minimum de clients pour commencer")
    parser.add_argument("--mu", type=float, default=0.1, help="Paramètre FedProx µ")
    args = parser.parse_args()

    # Création de la stratégie FedProx modifiée
    strategy = CustomFedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        proximal_mu=args.mu,
    )

    logger.info(f"Démarrage du serveur Flower avec FedProx (Rounds={args.num_rounds}, MinClients={args.min_clients}, µ={args.mu})")

    fl.server.start_server(
        server_address="127.0.0.1:8085",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
