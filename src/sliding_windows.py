import pandas as pd
import numpy as np

# Create sliding windows function
def sliding_windows(X, window_shape, stride):
    n_windows =  len(X) - window_shape + 1
    start = np.arange(0, n_windows, stride)
    windows = np.array([X[j:j+ window_shape] for j in start])
    return windows
