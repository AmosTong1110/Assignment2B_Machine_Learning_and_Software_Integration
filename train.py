"""Train one model on chronological observations."""
import json
from pathlib import Path
import numpy as np
from data_processing import load_site_series
from models import _artifact_path, _torch, _network
from evaluation import regression_metrics

def train_and_evaluate(csv_path, site_id, model_type, model_dir, window=8,
                       epochs=20, train_fraction=0.8, hidden_size=16,
                       learning_rate=0.01, ridge_alpha=0.01):
    """Fit one site's model and save it. Returns held-out one-step MAE/RMSE.

    Fit means target time belongs strictly to training period. Each held-out
    test target uses the preceding observed values, so the test is a rolling
    next-step test, not a multi-step forecast without later observations.
    Do not tune hyperparameters against the test results repeatedly.
    """
    site_id, model_type = str(site_id).strip(), str(model_type).lower().strip()
    path = _artifact_path(model_dir, model_type, site_id)
    if not isinstance(window, int) or window < 1:
        raise ValueError('window must be a positive integer.')
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError('epochs must be a positive integer.')
    if not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError('hidden_size must be a positive integer.')
    if not 0 < train_fraction < 1 or learning_rate <= 0 or ridge_alpha < 0:
        raise ValueError('Invalid train fraction, learning rate or ridge alpha.')
    stamps, values = load_site_series(csv_path, site_id)
    values = np.asarray(values, dtype=np.float64)
    split = int(len(values) * train_fraction)
    if split <= window or len(values) - split < 1:
        raise ValueError('Not enough records for the selected window and train/test split.')
    train_min = float(np.min(values[:split]))
    train_range = float(np.max(values[:split]) - train_min) or 1.0
    norm = (values - train_min) / train_range
    x_train = np.asarray([norm[i-window:i] for i in range(window, split)])
    y_train = norm[window:split]
    x_test = np.asarray([norm[i-window:i] for i in range(split, len(values))])
    y_test = values[split:]
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {'format_version': 1, 'site_id': site_id, 'model_type': model_type,
                'window': window, 'train_min': train_min, 'train_range': train_range,
                'flow_unit': 'measurement counts per 15-minute interval',
                'description': 'One-step rolling next-interval prediction'}
    if model_type == 'linear':
        # Unpenalised intercept; pseudoinverse handles singular matrices.
        design = np.column_stack([np.ones(len(x_train)), x_train])
        penalty = np.eye(window + 1) * ridge_alpha
        penalty[0, 0] = 0
        weights = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_train
        pred_norm = np.column_stack([np.ones(len(x_test)), x_test]) @ weights
        artifact['weights'] = weights.tolist()
        artifact['ridge_alpha'] = ridge_alpha
    else:
        torch = _torch()
        torch.manual_seed(42)
        net = _network(torch, model_type, hidden_size)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        x = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y_train, dtype=torch.float32)
        net.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(net(x), y)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            pred_norm = net(torch.tensor(x_test, dtype=torch.float32).unsqueeze(-1)).numpy()
        # JSON stores model parameters as simple numeric arrays; no pickle loads.
        artifact['state_dict'] = {k: v.detach().cpu().numpy().tolist() for k,v in net.state_dict().items()}
        artifact['hidden_size'] = hidden_size
        artifact['epochs'] = epochs
        artifact['learning_rate'] = learning_rate
    predictions = np.maximum(0.0, np.asarray(pred_norm) * train_range + train_min)
    if not np.isfinite(predictions).all():
        raise ValueError('Training produced non-finite predictions; check the data and settings.')
    metrics = regression_metrics(y_test, predictions)
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    return {'site_id': site_id, 'model_type': model_type,
            'train_count': int(len(x_train)), 'test_count': int(len(x_test)),
            'train_start': stamps[0], 'train_end': stamps[split-1],
            'test_start': stamps[split], 'test_end': stamps[-1],
            'window': window, 'horizon_minutes': 15,
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'unit': artifact['flow_unit'], 'artifact_path': str(path)}


