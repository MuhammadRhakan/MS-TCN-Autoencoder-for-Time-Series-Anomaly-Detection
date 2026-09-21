import numpy as np
from src.train import train_model
from src.detect import detect_anomalies
from src.evaluate import evaluate_detection_results
import argparse


parser = argparse.ArgumentParser(description="Parse anomaly detection parameters.")
parser.add_argument("--mode", type=str, choices=["demo", "train", "eval"], required=True)
parser.add_argument("--data", type=str, help="The data used for training the neural network")
parser.add_argument("--weights", type=str, help="Store weights for evaluation the trained model")
parser.add_argument("--eval_points", action=argparse.BooleanOptionalAction, help="Whether evaluating point-wise or window-wise", default=False, required=True)
parser.add_argument("--static_type", type=str, choices=["percentile", "stddev"],  help="Thresholding strategy: 'percentile' or 'stddev'", required=False)
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument("--n_percentile", type=float, default=None, help="Percentile cutoff when --static_type is 'percentile' (e.g., 95)")
parser.add_argument("--k", type=float, default=None, help="Multiplier for standard deviation when --static_type is 'stddev' (e.g., 3.0)")
args = parser.parse_args()

# Set the hyperparameters based on the user-selected method
"""
If static_type is percentile (n-th%), then parameter-n is required, while k is set to None. Vice versa.
"""
if args.static_type == "percentile":
    if args.k is not None:
        parser.error("--k is not valid with percentile")

    n_percentile = args.n_percentile if args.n_percentile is not None else 95.0
    parameter_k = None

elif args.static_type == "stddev":
    if args.n_percentile is not None:
        parser.error("--n_percentile is not valid with stddev")

    parameter_k = args.k if args.k is not None else 3.0
    n_percentile = None


def main():
    if args.mode == "demo":
        pass

    elif args.mode == "train":
        #-------------------
        # Load the datasets
        #-------------------
        with np.load(args.data) as data:
            train_ds = data["train_ds"]
            val_ds = data["val_ds"]
            test_ds = data["test_ds"]
            test_labels = data["test_labels"]

        #-----------------
        # Train the model
        #-----------------
        model, history = train_model(train_ds, val_ds)

        #--------------------------------------
        # Make predictions and calculate errors
        #--------------------------------------
        predictions = model.predict(test_ds)
        errors = np.mean(np.square(test_ds - predictions), axis=2)

    elif args.mode == "eval":
        #--------------------------
        # Perform anomaly detection
        #--------------------------
        # Get threshold from training error distribution  
        if args.threshold is None:
            train_predictions = model.predict(train_ds)
            train_errors = np.mean(np.square(train_ds - train_predictions), axis=2)
            reference_errors = train_errors
        else:
            reference_errors = None

        result = detect_anomalies(
            errors,
            test_labels,
            eval_points=args.eval_points,
            threshold=args.threshold,
            reference_errors=reference_errors,
            static_type=args.static_type,
            n_percentile=n_percentile,
            parameter_k=parameter_k)

        #------------------
        # Evaluate Result
        #------------------
        evaluate_detection_results(test_labels, result)

    else:
        raise ValueError("Select mode either 'mode', 'train', 'eval'")

if __name__ == "__main__":
    main()