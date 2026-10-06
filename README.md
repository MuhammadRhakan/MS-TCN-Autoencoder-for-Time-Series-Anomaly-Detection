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
├── assets/                        # Image resources
├── model/
│   ├── mstcn.py                   # Model architecture
│   └── trainer.py                 # Wrapper file for modeling pipeline
├── notebooks/
│   └── data_analysis.ipynb        # Exploratory data analysis
├── src/
│   ├── callbacks.py               # Neural network callback configurations
│   ├── preprocessing.py           # Data preprocessing steps
│   ├── run_data_preparation.py    # Full data preparation execution
│   └── sliding_window.py          # Sliding window for neural network input
├── .gitignore
├── README.md                   
├── config.yaml                    
└── main.py                        # Code execution from training to evaluation
```

## Dataset
The SWaT dataset used for this model could not be disclosed due to confidentiality. Therefore, the data path must be defined according to the user’s local environment. However, the dataset can be directly accessed upon request via:
```
https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/
```

The raw data consists of two xlsx files:
|File|Description|
|:----|:----|
|SWaT_Dataset_Normal_v1| 4 days normal operation for model training and validation |
|SWaT_Dataset_Attack_v0| 7 days attacked operations for testing |

## Execution Sample
Run data preparation
```
python -m src.run_data_preparation --config "config.yaml"
```

Execute main script.py
```
python main.py
```

## 📚 References
* J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, “A Dataset to Support Research in the Design of Secure Water Treatment Systems.”
* X. Ma, F. Wu, J. Yue, P. Feng, X. Peng, and J. Chu, “MSE-TCN: Multi-scale temporal convolutional network with channel attention for open-set gas classification,” Microchemical Journal, vol. 207, Dec. 2024.
