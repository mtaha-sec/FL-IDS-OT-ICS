# Root directory = dossier actuel

# Fichiers racine
New-Item README.md -ItemType File -Force
New-Item LICENSE -ItemType File -Force
New-Item .gitignore -ItemType File -Force
New-Item requirements.txt -ItemType File -Force
New-Item docker-compose.yml -ItemType File -Force

# Documentation
mkdir docs
mkdir docs\report_assets

# Datasets
mkdir datasets
mkdir datasets\raw
mkdir datasets\processed
mkdir datasets\partitions

# Preprocessing
mkdir preprocessing
New-Item preprocessing\cleaning.py -ItemType File -Force
New-Item preprocessing\feature_engineering.py -ItemType File -Force
New-Item preprocessing\normalization.py -ItemType File -Force
New-Item preprocessing\partitioning.py -ItemType File -Force

# Models
mkdir models
mkdir models\local_ids
mkdir models\global_model

# Industrial Clients
mkdir clients
mkdir clients\beneficiation
mkdir clients\sap
mkdir clients\pap
mkdir clients\power
mkdir clients\utilities
mkdir clients\granulation

# Aggregators
mkdir aggregators
mkdir aggregators\aggregator_1
mkdir aggregators\aggregator_2

# Central Server
mkdir central_server
New-Item central_server\server.py -ItemType File -Force
New-Item central_server\strategy.py -ItemType File -Force

# Security
mkdir security
mkdir security\certificates
mkdir security\tls
mkdir security\authentication

# Docker
mkdir docker
mkdir docker\clients
mkdir docker\aggregators
mkdir docker\server

# Monitoring
mkdir monitoring
mkdir monitoring\metrics
mkdir monitoring\logs

# Results
mkdir results
mkdir results\figures
mkdir results\metrics
mkdir results\reports


Write-Host "FL-IDS OT/ICS structure created successfully!"