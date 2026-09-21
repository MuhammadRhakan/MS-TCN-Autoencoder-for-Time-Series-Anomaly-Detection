# Multi-Scale Temporal Convolutional Network Autoencoder

Autoencoder architecture for time-series anomaly detection with a multi-scale encoder block and multiple TCN blocks on the decoder layer.

## Executive Summary
This project implements an end-to-end unsupervised anomaly detection pipeline for multivariate time-series data in Cyber-Physical Systems. Trained exclusively on normal operational conditions, the autoencoder detects abnormal behaviours by measuring reconstruction error deviations against defined thresholds.

## Key Highlights
* **Multi-Scale Feature Extraction:** Captures short-term spikes alongside long-range temporal dependencies across industrial sensors.
* **Temporal Convolutional Network (TCN):** Employs causal dilated convolutions to expand receptive fields without recurring memory bottlenecks typical of recurrent neural networks.
* **Project Pipeline:** Modular codebase separating preprocessing, windowing, reconstruction-based inference, and metric evaluation.

### Performance Results
| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Precision** | `0.99` | Minimized false positives during steady-state |
| **Recall** | `0.65` | High capture rate on stealthy bias attacks |
| **F1-Score** | `0.79` | Overall harmonic mean balance |
| **ROC-AUC** | `0.83` | Threshold-independent anomaly scoring |

## 🛠️ Architecture Design
![architecture design](/assets/architecture.png)

## Project Structure
```
Multi-Scale-TCN-Autoencoder/
├── assets/                        # Resources
├── model/
│   └── mstcn.py                   # Model architecture
├── notebooks/
│   └── data_analysis.ipynb        # Exploratory data analysis
├── src/
│   ├── config.py                  # Predefined hyperparameters and model configurations
│   ├── detect.py                  # Anomaly detection from reconstructed errors
│   ├── evaluate.py                # Evaluation metrics
│   ├── preprocessing.py           # Data preprocessing steps
│   ├── sliding_window.py          # Sliding window for neural network input
│   └── train.py                   # Model training
├── .gitignore
├── main.py                        # Code execution from training to evaluation
└── README.md
```

## Dataset
The SWaT dataset used for this model could not be disclosed due to confidentiality. Therefore, the data path must be defined according to the user’s local environment. However, the dataset can be directly accessed upon request via:
```
https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/
```

The raw data consists of two xlsx files:
|File|Description|
|:----|:----|
|SWaT_Dataset_Normal_v1| 4 days normal operation for model training |
|SWaT_Dataset_Attack_v0| 7 days attacked operations for validation and testing |

## Execution Sample
Train neural network
```
python main.py --mode train --data ./datasets/
```
Evaluate trained network
```
python main.py --mode eval --weights ./model/checkpoints/best_weights.pt
```

## 📚 References
* J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, “A Dataset to Support Research in the Design of Secure Water Treatment Systems.”
* X. Ma, F. Wu, J. Yue, P. Feng, X. Peng, and J. Chu, “MSE-TCN: Multi-scale temporal convolutional network with channel attention for open-set gas classification,” Microchemical Journal, vol. 207, Dec. 2024.