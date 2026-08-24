import os
import sys
import subprocess
import threading
import re
import time
from PySide6.QtCore import QThread, Signal

class FLRunnerWorker(QThread):
    """
    Worker responsable de lancer le véritable processus Flower :
    - Lance le serveur central via subprocess
    - Lance les clients sélectionnés via subprocess
    - Capture et parse leurs logs pour mettre à jour l'interface
    """

    log_message = Signal(str)
    metrics_update = Signal(dict)
    progress_update = Signal(int, int)
    error_occurred = Signal(str)
    
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params
        self.processes = []
        self._is_running = True
        self._current_round = 0
        self._client_losses_buffer = []

    def run(self):
        self.log_message.emit("============================================================")
        self.log_message.emit("🚀 [FL RUNNER] Démarrage de l'apprentissage fédéré (Subprocess)")
        self.log_message.emit(f"   ► Rounds   : {self.params.get('num_rounds')}")
        self.log_message.emit(f"   ► Clients  : {', '.join(self.params.get('selected_clients', []))}")
        self.log_message.emit(f"   ► Stratégie: FedProx (µ={self.params.get('mu')})")
        self.log_message.emit("============================================================")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        cwd = os.getcwd()

        # 1. Démarrage du serveur central
        num_rounds = self.params.get("num_rounds", 10)
        selected_clients = self.params.get("selected_clients", [])
        
        if not selected_clients:
            self.log_message.emit("❌ Aucun client sélectionné. Arrêt.")
            return

        server_cmd = [
            sys.executable, "-m", "central_server.server",
            "--num-rounds", str(num_rounds),
            "--min-clients", str(len(selected_clients)),
            "--mu", str(self.params.get("mu", 0.1))
        ]
        
        self.log_message.emit("\n🌐 Lancement du serveur central...")
        server_proc = subprocess.Popen(
            server_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1, env=env, cwd=cwd
        )
        self.processes.append(server_proc)
        threading.Thread(target=self._monitor_output, args=(server_proc, "SERVEUR"), daemon=True).start()
        
        # Laisser un peu de temps au serveur pour démarrer
        time.sleep(2)
        
        if not self._is_running:
            self.stop()
            return

        # 2. Démarrage des clients
        self.log_message.emit("\n💻 Lancement des clients locaux...")
        for client_name in selected_clients:
            if not self._is_running:
                break
                
            client_cmd = [
                sys.executable, "-m", "clients.client_app",
                "--client-name", client_name,
                "--local-epochs", str(self.params.get("local_epochs", 3))
            ]
            self.log_message.emit(f"   ► Démarrage de '{client_name}'...")
            client_proc = subprocess.Popen(
                client_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                text=True, bufsize=1, env=env, cwd=cwd
            )
            self.processes.append(client_proc)
            threading.Thread(target=self._monitor_output, args=(client_proc, f"CLIENT-{client_name.upper()}"), daemon=True).start()
            
            # Petit délai pour éviter de saturer le CPU d'un coup
            time.sleep(1)

        # 3. Attente de la fin des processus
        for p in self.processes:
            try:
                p.wait()
            except Exception:
                pass
                
        self.log_message.emit("\n✅ [FL RUNNER] Processus terminés.")

    def _monitor_output(self, proc, prefix):
        for line in proc.stdout:
            if not self._is_running:
                break
            line = line.strip()
            if line:
                self.log_message.emit(f"[{prefix}] {line}")
                self._parse_metrics(prefix, line)

    def _parse_metrics(self, prefix, line):
        """
        Analyse les lignes de log pour extraire les métriques.
        C'est très basique et dépend du format exact des logs.
        """
        metrics = {}
        
        # Exemple de format attendu du serveur : 
        # "fit progress: (1, 0.5432, {'accuracy': 0.82})"
        # ou autre selon Flower. On va faire au mieux.
        
        # Parsing Global Loss & Metrics (depuis le serveur Flower)
        if prefix == "SERVEUR":
            # Round progress & Metrics
            # Format attendu : [GLOBAL_METRICS] round=1 loss=0.4500 accuracy=0.8800 precision=0.8700 recall=0.8900 f1=0.8800
            match_global = re.search(r"\[GLOBAL_METRICS\] round=(\d+) loss=([0-9\.]+) accuracy=([0-9\.]+) precision=([0-9\.]+) recall=([0-9\.]+) f1=([0-9\.]+)", line)
            if match_global:
                self._current_round = int(match_global.group(1))
                metrics["round"] = self._current_round
                metrics["global_loss"] = float(match_global.group(2))
                metrics["global_accuracy"] = float(match_global.group(3))
                metrics["global_precision"] = float(match_global.group(4))
                metrics["global_recall"] = float(match_global.group(5))
                metrics["global_f1"] = float(match_global.group(6))
                
                self.progress_update.emit(self._current_round, self.params.get("num_rounds", 10))
                # Clear buffer for the next round
                self._client_losses_buffer.clear()
                
        # Parsing Local Loss (depuis les clients)
        elif prefix.startswith("CLIENT"):
            # Exemple: "[sap] epoch 3/3 - loss=0.4567"
            match_loss = re.search(r"loss=([0-9\.]+)", line)
            if match_loss:
                loss_val = float(match_loss.group(1))
                self._client_losses_buffer.append(loss_val)
                metrics["round"] = max(1, self._current_round)
                metrics["client_losses"] = list(self._client_losses_buffer)

        if metrics:
            self.metrics_update.emit(metrics)

    def stop(self):
        """Arrêt d'urgence de tous les sous-processus."""
        self._is_running = False
        self.log_message.emit("🛑 ARRÊT D'URGENCE DEMANDÉ. Terminaison des processus...")
        for p in self.processes:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception as e:
                self.log_message.emit(f"Erreur lors de l'arrêt d'un processus : {e}")
        self.processes.clear()
