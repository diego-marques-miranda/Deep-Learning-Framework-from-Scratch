import numpy as np

def train_test_split(X, y, test_size, rng=np.random.default_rng(), stratify=None):
    """Split arrays into random train and validation/test subsets.

    When provided, ``stratify`` contains one class label per sample and keeps
    the class proportions similar in both subsets.
    """
    indexes = np.arange(X.shape[0])
    split_idx = int(X.shape[0] * (1 - test_size))

    if stratify is None:
        rng.shuffle(indexes)
        train_idx, test_idx = indexes[:split_idx], indexes[split_idx:]
    else:
        stratify = np.asarray(stratify)
        if stratify.shape[0] != X.shape[0]:
            raise ValueError("stratify must have the same number of samples as X")

        classes, class_indexes = np.unique(stratify, return_inverse=True)
        class_counts = np.bincount(class_indexes)
        test_count = X.shape[0] - split_idx

        # Distribute the test samples across classes using the largest remainders.
        raw_test_counts = class_counts * test_count / X.shape[0]
        test_counts = np.floor(raw_test_counts).astype(int)
        remainder = test_count - test_counts.sum()
        if remainder > 0:
            order = np.argsort(raw_test_counts - test_counts)[::-1]
            test_counts[order[:remainder]] += 1

        train_parts, test_parts = [], []
        for class_index in range(classes.size):
            # Shuffle each class separately before assigning its samples.
            class_indices = indexes[class_indexes == class_index]
            rng.shuffle(class_indices)
            class_test_count = test_counts[class_index]
            test_parts.append(class_indices[:class_test_count])
            train_parts.append(class_indices[class_test_count:])

        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]