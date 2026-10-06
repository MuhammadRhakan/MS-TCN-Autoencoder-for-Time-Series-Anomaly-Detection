import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model.mstcn import AutoEncoder
from src.callbacks import configuration_settings
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    auc)


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class MSTCN_AutoEncoder:
    def __init__(
        self,
        features,
        latent_dim,
        train_ds,
        val_ds,
        test_ds,
    ):    
        self.batch_size = config['batch_size']
        self.features = features
        self.latent_dim = latent_dim

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.train_tensor = (
            tf.data.Dataset.from_tensor_slices((train_ds, train_ds))
            .shuffle(buffer_size=len(train_ds))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE))
        
        self.val_tensor = (
            tf.data.Dataset.from_tensor_slices((val_ds, val_ds))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE))
        
        self.test_tensor = (
            tf.data.Dataset.from_tensor_slices((test_ds, test_ds))
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE))

        self.mstcn_autoencoder = AutoEncoder(
            features=self.features,
            latent_dim=self.latent_dim)

    def train(self, epochs, learning_rate, callbacks):
        self.mstcn_autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse')

        self.mstcn_autoencoder.fit(
            x=self.train_tensor,
            validation_data=self.val_tensor,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1)

    def calculate_anomaly_scores(self, data):
        reconstruction = self.mstcn_autoencoder.predict(data)
        return np.square(data - reconstruction)

    def determine_thresholds(self, reference_anomaly_scores, percentile):
        return np.percentile(reference_anomaly_scores, percentile, axis=(0, 1))

    def detect_anomalies(self, baseline='train', percentile=95):
        if baseline == 'train':
            reference_anomaly_scores = self.calculate_anomaly_scores(self.train_ds)
        elif baseline == 'val':
            reference_anomaly_scores = self.calculate_anomaly_scores(self.val_ds)

        thresholds = self.determine_thresholds(reference_anomaly_scores, percentile)
        anomaly_scores = self.calculate_anomaly_scores(self.test_ds)

        return (anomaly_scores > thresholds).astype(int)
        
    def evaluate_detection_results(self, ground_truth, detection_results, minimum_abnormal_features):
        # Threshold-dependent evaluation metrics
        full_timesteps = ground_truth.flatten()
        anomaly_timesteps = (np.sum(detection_results, axis=2) >= minimum_abnormal_features).flatten().astype(int)

        precision = precision_score(full_timesteps, anomaly_timesteps, zero_division=0)
        recall = recall_score(full_timesteps, anomaly_timesteps, zero_division=0)
        f1 = f1_score(full_timesteps, anomaly_timesteps, zero_division=0)

        print("Threshold-dependent evaluation metrics:")
        print("Precision : ", precision)
        print("Recall    : ", recall)
        print("F1-score  : ", f1)

        # Threshold-independent evaluation metrics
        anomaly_scores = self.calculate_anomaly_scores(self.test_tensor)
        mean_anomaly_scores1D = np.mean(anomaly_scores, axis=2).flatten()

        precision, recall, threshold = precision_recall_curve(full_timesteps, mean_anomaly_scores1D)
        average_precision = average_precision_score(full_timesteps, mean_anomaly_scores1D)

        fpr, tpr, threshold = roc_curve(full_timesteps, mean_anomaly_scores1D)
        area_under_curve = auc(fpr, tpr)

        print()
        print("Threshold-independent evaluation metrics:")
        print("Precision scores : ", precision)
        print("Recall scores    : ", recall)
        print("Thresholds       : ", threshold)
        print("Avg_prec         : ", average_precision)
        print("Threshold from roc curve : ", threshold)
        print("Area Under Curve         : ", area_under_curve)

        # Visualize Precision-Recall Curve
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, color='red', label='ROC Curve')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--', label='Random Guess Baseline')
        plt.legend()
        plt.tight_layout()
        plt.show()





