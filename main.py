import yaml
import argparse
import numpy as np
import tensorflow as tf
from callbacks import configuration_settings
from model.trainer import MSTCN_AutoEncoder


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file) 

dataset = config['output']['tensor_directory'] + "/tensors.npz"

def main():
    model = MSTCN_AutoEncoder(
        features=dataset['train_ds'].shape[-1],
        latent_dim=config['latent_dim'],
        train_ds=dataset["train_ds"],
        val_ds=dataset["val_ds"],
        test_ds=dataset["test_ds"])

    model.train(
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        callbacks=configuration_settings())

    detection_results = model.detect_anomalies(
        baseline='train',
        percentile=config['percentile'])

    model.evaluate(
        ground_truth=dataset["test_labels"],
        detection_results=detection_results,
        minimum_abnormal_features=config['minimum_abnormal_features'])


if __name__ == "__main__":
    main()