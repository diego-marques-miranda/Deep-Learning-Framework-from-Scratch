import numpy as np

def train_test_split(X, y, test_size):
    """Splits arrays or matrices into random train and validation/test subsets using index shuffling."""
    indexes = np.arange(X.shape[0])
    np.random.shuffle(indexes)
    split_idx = int(X.shape[0] * (1 - test_size))
    
    train_idx, val_idx = indexes[:split_idx], indexes[split_idx:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]