"""Model implementations and saved predictions."""
import hashlib, json
from pathlib import Path
import numpy as np

MODEL_TYPES = ('linear', 'lstm', 'gru')

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


