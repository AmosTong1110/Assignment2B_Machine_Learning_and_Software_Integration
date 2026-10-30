"""Calculate regression errors in the same units as the target."""
import numpy as np

def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError('Actual and predicted must be nonempty equal-length vectors.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Metrics require finite values.')
    errors = predicted - actual
    return {'mae': float(np.mean(np.abs(errors))), 'rmse': float(np.sqrt(np.mean(errors ** 2)))}
