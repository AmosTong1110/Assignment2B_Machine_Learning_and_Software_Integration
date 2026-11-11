"""
train_lstm.py  -  train + evaluate the LSTM for every site
==========================================================
Run from the project root:      python Source/Models/train_lstm.py
Quick test on two sites:        python Source/Models/train_lstm.py --sites 2000 3002 --epochs 5

Settings (epochs, units, ...) come from config.yaml.

Outputs
    saved_models/lstm/<site>.keras + <site>_scaler.json   one trained model per site
    Results/lstm_metrics.csv       MAE / RMSE per site (shared format, see evaluate.py)
    Results/lstm_predictions.csv   predictions vs truth (for plots in the report)
    Results/lstm_history.csv       training / validation loss per epoch (learning curves)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root, so 'Source.Models...' imports work

import Source.Models.deep_common as dc
import Source.Models.lstm_model as lstm

if __name__ == "__main__":
    dc.run_cli(lstm.MODEL_NAME, lstm.MODEL_KEY, lstm.LAYER)
