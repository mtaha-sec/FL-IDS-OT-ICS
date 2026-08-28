🛡️ FL-IDS-OT-ICS
Privacy-Preserving Federated Intrusion Detection System for Industrial OT/ICS Environments

A privacy-preserving Federated Learning framework for collaborative intrusion detection across distributed Industrial OT/ICS environments, combining FedProx, Homomorphic Encryption (CKKS/TenSEAL), and secure TLS communication.

📌 Overview

Industrial Operational Technology (OT) and Industrial Control Systems (ICS) generate security-sensitive network traffic across multiple sites and infrastructures.

Traditional centralized Intrusion Detection Systems require organizations to share raw network data with a central entity, which can create significant privacy, confidentiality, and data-governance risks.

This project proposes a Federated Learning-based Intrusion Detection System (FL-IDS) where each industrial site trains a local IDS model on its own data without sharing the raw dataset.

The architecture consists of:

🖥️ 1 Global Server responsible for federated aggregation
🏭 6 distributed industrial clients/sites
🤖 Local IDS models trained independently at each client
🔄 FedProx for federated optimization under heterogeneous/non-IID data
🔐 Homomorphic Encryption (HE) using CKKS/TenSEAL to protect model parameters during aggregation
🔒 TLS to secure communication between clients and the global server
📊 Multiple industrial cybersecurity datasets
Main objective

Enable collaborative IDS model training without requiring clients to share their raw industrial network traffic or expose their local model parameters to the aggregation server in plaintext.

Dataset Repository:
Google Drive:
        ├── ICS-Flow.csv
        ├── TON_IoT.csv
        ├── ML-EdgeIIoT.csv
        └── X-IIoTID.csv
    

    Dataset download:
https://drive.google.com/drive/folders/1wtewxDU6VNqBmPSFRZfspqvBfTv8hvZF?usp=sharing
