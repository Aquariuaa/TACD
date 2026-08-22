# TACD: Topology-Aware Attention Causal Discovery for Network Anomaly Localization

This repository contains the official implementation of **TACD** (Topology-Aware Attention Causal Discovery), a real-time causal discovery method for topological event sequences in telecommunication networks.

# Data Preparation

## Synthetic Data
Synthetic datasets can be generated using the [gCastle API](https://www.gcastle.com). 
Default parameters: 20 event types, 40 devices, baseline intensity mu in [3e-5, 5e-5], excitation intensity alpha in [0.02, 0.03], generation period T = 2 days.

## Real-World Data
Three metropolitan telecommunication network alarm datasets from Huawei PCIC 2021 are supported:

| Dataset | Event Types | Devices | Alarm Logs |
|---------|-------------|---------|------------|
| 18V-55N-Wireless | 18 | 55 | 34,839 |
| 24V-439N-Microwave | 24 | 439 | 64,599 |
| 25V-474N-Microwave | 25 | 474 | 48,573 |

# Dependencies
The code requires the following dependencies:
- Python >= 3.9
- PyTorch >= 2.6.0
- NumPy >= 1.26.0
- Pandas >= 2.2.0
- scikit-learn >= 1.6.0
- gcastle >= 1.0.4

# Quick Start
```bash
python main.py --dataset <dataset_name>
python main.py --dataset 24V_439N_Microwave
```

## Acknowledgements

This work is built upon prior research including S^2GCSL, THP, and CausalNET. We thank the Huawei PCIC 2021 competition for providing the real-world datasets.
