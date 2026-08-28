#  FL-IDS-OT-ICS
## Privacy-Preserving Federated Intrusion Detection System for Industrial OT/ICS Environments

A privacy-preserving **Federated Learning-based Intrusion Detection System (FL-IDS)** designed for collaborative cybersecurity across distributed **Industrial OT/ICS environments**.

The framework combines **FedProx**, **Homomorphic Encryption (CKKS/TenSEAL)**, and **secure TLS communication** to enable collaborative model training while protecting sensitive industrial data and model parameters.

---

##  Overview

Industrial **Operational Technology (OT)** and **Industrial Control Systems (ICS)** generate highly sensitive network traffic across multiple sites, plants, and infrastructures.

Traditional centralized Intrusion Detection Systems (IDS) require organizations to send their raw network traffic to a central entity. This approach can introduce significant:

-  Privacy risks
-  Industrial confidentiality concerns
-  Data-governance issues
- Security and compliance risks

This project addresses these challenges through a **Federated Learning-based IDS**, where each industrial site trains its own local IDS model without sharing its raw network traffic with the central server.

Instead of exchanging raw datasets, clients collaboratively contribute to the training of a **global IDS model** through federated optimization.

---

## System Architecture

The proposed architecture consists of:

- **1 Global Server** responsible for federated model aggregation
- **6 Distributed Industrial Clients/Sites**
- **Local IDS models** independently trained at each client
- **FedProx** for federated optimization under heterogeneous and non-IID data distributions
-  **Homomorphic Encryption (HE)** using **CKKS/TenSEAL** to protect model parameters during aggregation
-  **TLS** to secure communication between clients and the global server
- Multiple industrial cybersecurity datasets
  
  ![FL-IDS-OT-ICS Architecture](architecture/architecture.jpeg)
###  Main Objective

> **Enable collaborative IDS model training without requiring clients to share their raw industrial network traffic or expose their local model parameters to the aggregation server in plaintext.**

##  How It Works

The FL-IDS-OT-ICS framework follows an end-to-end privacy-preserving federated learning workflow. Each industrial client keeps its local dataset, performs local IDS training, encrypts its model update, and securely transmits the encrypted update to the Global Server. The server aggregates the contributions and sends the updated global model back to all clients for the next federated round.

### End-to-End Federated Learning Workflow

```text
                         🏭 DISTRIBUTED INDUSTRIAL OT/ICS ENVIRONMENT

 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │     PAP      │   │  UTILITIES   │   │ GRANULATION  │   │ BENEFICIATION│   │     POWER    │   │      SAP     │
 │   Client 1 │   │   Client 2 │   │       Client 3 │      Client 4 │     │     Client 5 │   │      Client 6 │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                    │                    │                    │                    │                    │
        │ 1️-Local Data     │                    │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼                    ▼                    ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ Local Dataset│   │ Local Dataset│   │ Local Dataset│   │ Local Dataset│   │ Local Dataset│   │ Local Dataset│
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                    │                    │                    │                    │                    │
        │ 2️-Preprocessing  │                    │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼                    ▼                    ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │Preprocessing │   │Preprocessing │   │Preprocessing │   │Preprocessing │   │Preprocessing │   │Preprocessing │
 │ & Features   │   │ & Features   │   │ & Features   │   │ & Features   │   │ & Features   │   │ & Features   │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                    │                    │                    │                    │                    │
        │ 3️-Local Training │                    │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼                    ▼                    ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ Local IDS    │   │ Local IDS    │   │ Local IDS    │   │ Local IDS    │   │ Local IDS    │   │ Local IDS    │
 │ Model Train. │   │ Model Train. │   │ Model Train. │   │ Model Train. │   │ Model Train. │   │ Model Train. │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                    │                    │                    │                    │                    │
        │ 4️-Local Model    │                    │                    │                    │                    │
        │    Update          │                    │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼                    ▼                    ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  Local Model │   │  Local Model │   │  Local Model │   │  Local Model │   │  Local Model │   │  Local Model │
 │    Update    │   │    Update    │   │    Update    │   │    Update    │   │    Update    │   │    Update    │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                    │                    │                    │                    │                    │
        │ 5️-CKKS Encryption│                    │                    │                    │                    │
        ▼                    ▼                    ▼                    ▼                    ▼                    ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  Encrypted │   │  Encrypted │   │   Encrypted │   │      Encrypted │   │    Encrypted │   │    Encrypted │
 │ Model Update │   │ Model Update │   │ Model Update │   │ Model Update │   │ Model Update │   │ Model Update │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                    │                    │                    │                    │                    │
        └────────────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┘
                                                     │
                                                     │
                                          6️-SECURE TLS TRANSMISSION
                                                     │
                                                     ▼
                              ┌────────────────────────────────────────────┐
                              │               GLOBAL SERVER              │
                              │                                            │
                              │  Receive Encrypted Client Updates          │
                              │                    │                       │
                              │                    ▼                       │
                              │        Privacy-Preserving Aggregation    │
                              │                    │                       │
                              │                    ▼                       │
                              │                 FedProx                    │
                              │          Global Model Update               │
                              └────────────────────┬───────────────────────┘
                                                   │
                                                   │
                                     7️-UPDATED GLOBAL MODEL
                                                   │
                                                   │  TLS
                                                   ▼
        ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
        │                         SECURE GLOBAL MODEL DISTRIBUTION                                    │
        └─────────────────────────────────────────────────────────────────────────────────────────────┘
          │                    │                    │                    │                    │                    │
          ▼                    ▼                    ▼                    ▼                    ▼                    ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │  Updated   │   │   Updated   │   │    Updated   │   │    Updated   │   │    Updated   │   │     Updated   │
 │ Global Model │   │ Global Model │   │ Global Model │   │ Global Model │   │ Global Model │   │ Global Model │
 │    → PAP     │   │ → UTILITIES  │   │ → GRANULATION│   │→ BENEFICIATION│  │   → POWER    │   │    → SAP     │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                    │                    │                    │                    │                    │
        │           8️-Continue Local Training using the Updated Global Model
        ▼                    ▼                    ▼                    ▼                    ▼                    ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │    PAP       │   │  UTILITIES   │   │ GRANULATION  │   │ BENEFICIATION│   │     POWER    │   │      SAP     │
 │   Round 2    │   │   Round 2    │   │   Round 2    │   │   Round 2    │   │   Round 2    │   │   Round 2    │
 └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
        │                    │                    │                    │                    │                    │
        └────────────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┘
                                                     │
                                                     │
                                  REPEAT FEDERATED TRAINING ROUNDS
                                                     │
                                                     └──────────────► 1️-Next Round(...until the 10 round)

Dataset Repository:
Google Drive:
        ├── ICS-Flow.csv
        ├── TON_IoT.csv
        ├── ML-EdgeIIoT.csv
        └── X-IIoTID.csv
    

    Dataset download:
https://drive.google.com/drive/folders/1wtewxDU6VNqBmPSFRZfspqvBfTv8hvZF?usp=sharing
