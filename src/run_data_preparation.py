import numpy as np
import pandas as pd
import argparse
import yaml

from src.callbacks import *
from src.preprocessing import clean_data, fill_missing_timestamps, filter_operating_state
from src.sliding_windows import sliding_windows
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


parser = argparse.ArgumentParser(description="Parse data preparation parameters.")
parser.add_argument("--config", type=Path, required=True, help="Path to the configuration file.")
args = parser.parse_args()

def main():
    # ------------------------------
    # 1. Load raw dataset
    # -------------------------------
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    normal = pd.read_csv(
        Path(config['dataset']['raw_dir']) / config['dataset']['normal_file'],
        sep=';')  
      
    attack = pd.read_csv(
        Path(config['dataset']['raw_dir']) / config['dataset']['attack_file'],
        sep=';')

    normal.columns = normal.columns.str.strip()
    attack.columns = attack.columns.str.strip()

    timestamp = 'Timestamp'
    target = 'Normal/Attack'
    actuators = [a for a in normal.columns if (normal[a].nunique() <=3) and (a not in [timestamp, target])]
    sensors = [s for s in normal.columns if (s not in actuators) and (s not in [timestamp, target])]
    dataset_features = normal.drop(columns={timestamp, target}).columns

    # ----------------------------
    # 2. Clean and filter dataset
    # ----------------------------
    # Normal Dataset
    normal[dataset_features] = normal.loc[:, dataset_features].astype(str).replace(",", ".", regex=True)
    normal[dataset_features] = normal[dataset_features].apply(pd.to_numeric, errors='coerce')

    normal_clean = clean_data(normal)
    normal_fill = fill_missing_timestamps(normal_clean, actuators, sensors, target)
    normal_filtered = filter_operating_state(normal_fill, target, hours=4)

    # Attack Dataset
    attack[dataset_features] = attack.loc[:, dataset_features].astype(str).replace(",", ".", regex=True)
    attack[dataset_features] = attack[dataset_features].apply(pd.to_numeric, errors='coerce')

    attack_clean = clean_data(attack)
    attack_fill = fill_missing_timestamps(attack_clean, actuators, sensors, target)
    
    # ----------------
    # 3. Split dataset 
    # ----------------
    X_normal = normal_filtered[dataset_features].values
    y_normal = normal_filtered[target].values

    X_attack = attack_fill[dataset_features].values
    y_attack = attack_fill[target].values

    train_size = int(len(X_normal) * 0.8)

    X_train, y_train = X_normal[:train_size], y_normal[:train_size]
    X_val, y_val = X_normal[train_size:], y_normal[train_size:]
    X_test, y_test = X_attack, y_attack

    # --------------------------
    # 4. Normalization and PCA
    # --------------------------
    scaler = MinMaxScaler()

    X_train_transformed = scaler.fit_transform(X_train)
    X_val_transformed = scaler.transform(X_val)
    X_test_transformed = scaler.transform(X_test)

    # --------------------------
    # 5. Create sliding windows
    # --------------------------
    X_train_windows = sliding_windows(X_train, config['window_shape'], config['train_stride'])
    X_val_windows = sliding_windows(X_val, config['window_shape'], config['test_stride'])
    X_test_windows = sliding_windows(X_test, config['window_shape'], config['test_stride'])

    y_train_windows = sliding_windows(y_train, config['window_shape'], config['train_stride'])
    y_val_windows = sliding_windows(y_val, config['window_shape'], config['test_stride'])
    y_test_windows = sliding_windows(y_test, config['window_shape'], config['test_stride'])

    print(f"Training dataset shape: {X_train_windows.shape}")
    print(f"Validation dataset shape: {X_val_windows.shape}")
    print(f"Testing dataset shape: {X_test_windows.shape}")
    print(f"Validation labels: {y_val_windows.shape}")
    print(f"Ground Truth shape: {y_test_windows.shape}")

    #----------------------------
    # Save all datasets
    #----------------------------
    np.savez_compressed(
        Path(config['output']['processed_dir']) / "processed_datasets.npz",
        X_train=X_train_transformed,
        X_val=X_val_transformed,
        X_test=X_test_transformed,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test)
    print("Preprocessed datasets saved successfully.")


    np.savez_compressed(
        Path(config['output']['tensor_dir']) / "tensors.npz",
        train_ds=X_train_windows,
        val_ds=X_val_windows,
        test_ds=X_test_windows,
        train_labels=y_train_windows,
        val_labels=y_val_windows,
        test_labels=y_test_windows)
    print("Tensor datasets are saved successfully.")


if __name__ == "__main__":
    main()