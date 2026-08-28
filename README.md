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
  ![FL-IDS-OT-ICS Architecture](architecture/architecture.jpg)
###  Main Objective

> **Enable collaborative IDS model training without requiring clients to share their raw industrial network traffic or expose their local model parameters to the aggregation server in plaintext.**

Dataset Repository:
Google Drive:
        ├── ICS-Flow.csv
        ├── TON_IoT.csv
        ├── ML-EdgeIIoT.csv
        └── X-IIoTID.csv
    

    Dataset download:
https://drive.google.com/drive/folders/1wtewxDU6VNqBmPSFRZfspqvBfTv8hvZF?usp=sharing
