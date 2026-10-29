"""Per-site next-interval forecasts for a demonstration route-guidance prototype.

CSV columns: site_id,timestamp,flow; timestamps ISO 8601, exactly 15 minutes
apart per site; flow is a non-negative count per 15-minute interval. This is
an adapter format, not a claim about the supplied university dataset's schema.

Three model choices: `linear` is ridge regression (candidate requiring tutor
approval); `lstm` and `gru` need PyTorch. Models are independent per site.
Training uses an 80/20 chronological split by default and scaler parameters
fit on the training period only. Test is one-step rolling evaluation: prior
observations, including earlier test observations, are inputs; future values
are never used. Metrics are in vehicles per 15-minute interval.

This prototype is not a validated scientific experiment. Do not load model
files from unknown sources. Inspect the dataset and choose settings with the
team; reserve independent data for any model selection and final assessment.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np

INTERVAL = timedelta(minutes=15)
MODEL_TYPES = ('linear', 'lstm', 'gru')


def load_site_series(csv_path, site_id):
    """Return (sorted timestamps, float flows) for one site, rejecting gaps.

    Rejects duplicates, missing/invalid values, negative counts and irregular
    timestamps rather than silently imputing or creating incorrect windows.
    Timestamps for each site must use a consistent timezone convention.
    """
    site_id = str(site_id).strip()
    if not site_id:
        raise ValueError('Site ID must not be empty.')
    records = []
    with Path(csv_path).open('r', newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        required = {'site_id', 'timestamp', 'flow'}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError('CSV requires columns site_id, timestamp, flow.')
        for line, row in enumerate(reader, 2):
            if (row.get('site_id') or '').strip() != site_id:
                continue
            try:
                stamp = datetime.fromisoformat((row.get('timestamp') or '').strip().replace('Z', '+00:00'))
                value = float(row.get('flow') or '')
                if not math.isfinite(value) or value < 0:
                    raise ValueError('flow must be finite and non-negative')
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid timestamp or flow on CSV line {line}: {exc}') from exc
            records.append((stamp, value))
    if not records:
        raise ValueError(f'No records found for site {site_id!r}.')
    try:
        records.sort(key=lambda r: r[0])
        for previous, current in zip(records, records[1:]):
            if current[0] - previous[0] != INTERVAL:
                raise ValueError('Site timestamps must be unique and exactly 15 minutes apart; clean the dataset first.')
    except TypeError as exc:
        raise ValueError('Site timestamps must use a consistent timezone convention.') from exc
    return [r[0].isoformat() for r in records], [r[1] for r in records]


def _artifact_path(model_dir, model_type, site_id):
    if model_type not in MODEL_TYPES:
        raise ValueError(f'Model must be one of {MODEL_TYPES}.')
    digest = hashlib.sha256(str(site_id).encode()).hexdigest()[:16]
    return Path(model_dir) / f'{model_type}_{digest}.json'


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError('LSTM and GRU require PyTorch. Install it in the project environment before training or prediction.') from exc


def _network(torch, model_type, hidden_size):
    class Recurrent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            recurrent = torch.nn.LSTM if model_type == 'lstm' else torch.nn.GRU
            self.rnn = recurrent(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.head = torch.nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)
    return Recurrent()


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
                'flow_unit': 'vehicles per 15-minute interval',
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
    errors = predictions - y_test
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    return {'site_id': site_id, 'model_type': model_type,
            'train_count': int(len(x_train)), 'test_count': int(len(x_test)),
            'train_start': stamps[0], 'train_end': stamps[split-1],
            'test_start': stamps[split], 'test_end': stamps[-1],
            'window': window, 'horizon_minutes': 15,
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'unit': artifact['flow_unit'], 'artifact_path': str(path)}


def predict_next(model_dir, model_type, site_id, history):
    """Forecast the next 15-minute count using the latest observed values."""
    site_id, model_type = str(site_id).strip(), str(model_type).lower().strip()
    path = _artifact_path(model_dir, model_type, site_id)
    if not path.is_file():
        raise FileNotFoundError(f'No trained {model_type} model for site {site_id}. Train it first.')
    artifact = json.loads(path.read_text(encoding='utf-8'))
    if artifact['site_id'] != site_id or artifact['model_type'] != model_type:
        raise ValueError('Saved model does not match the requested site and model.')
    window = int(artifact['window'])
    history = np.asarray(history, dtype=float)
    if history.ndim != 1 or len(history) < window:
        raise ValueError(f'At least {window} observed 15-minute values are required.')
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError('History must contain finite, non-negative values.')
    x = (history[-window:] - artifact['train_min']) / artifact['train_range']
    if model_type == 'linear':
        pred_norm = float(np.r_[1.0, x] @ np.asarray(artifact['weights']))
    else:
        torch = _torch()
        net = _network(torch, model_type, int(artifact['hidden_size']))
        state = {k: torch.tensor(v, dtype=torch.float32) for k,v in artifact['state_dict'].items()}
        net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            pred_norm = float(net(torch.tensor(x, dtype=torch.float32).reshape(1, window, 1)).item())
    return max(0.0, pred_norm * artifact['train_range'] + artifact['train_min'])
