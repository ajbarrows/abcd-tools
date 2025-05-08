
import numpy as np


def haufe_transform(model_weights, X, batch_size=1000, flip_sign=False):
    """
    Memory-efficient Haufe transformation with corrected sign.
    Simply negate the result of the original implementation.

    Parameters:
    -----------
    model_weights : array-like, shape (n_features,)
        Weight vector from a linear model
    X : array-like, shape (n_samples, n_features)
        Feature matrix used to train the model
    batch_size : int, default=1000
        Size of batches to process at once

    Returns:
    --------
    activation_patterns : array, shape (n_features,)
        Transformed weights with correct sign
    """
    X = np.asarray(X)
    model_weights = np.asarray(model_weights).flatten()
    n_samples, n_features = X.shape

    # Compute means
    means = np.mean(X, axis=0)

    # Compute using original implementation
    activation_patterns = np.zeros(n_features)

    for i in range(0, n_samples, batch_size):
        batch = X[i:min(i+batch_size, n_samples)]
        centered_batch = batch - means
        activation_patterns += np.dot(centered_batch.T,
                np.dot(centered_batch, model_weights)
                )

    # Normalize by n_samples - 1
    activation_patterns /= (n_samples - 1)

    if flip_sign:
        # Flip the sign of the activation patterns
        activation_patterns = -activation_patterns

    return activation_patterns
