"""
train_gru.py  -  train + evaluate the GRU for every site
========================================================
Run from the project root:      python -m src.Models.train_gru
Quick test on two sites:        python -m src.Models.train_gru --sites 2000 3002 --epochs 5

Settings (epochs, units, ...) come from config.yaml.

Outputs
    saved_models/gru/<site>.keras + <site>_scaler.json    one trained model per site
    results/gru_metrics.csv        MAE / RMSE per site (shared format, see evaluate.py)
    results/gru_predictions.csv    predictions vs truth (for plots in the report)
    results/gru_history.csv        training / validation loss per epoch (learning curves)
"""
import src.Models.deep_common as dc
import src.Models.gru_model as gru

if __name__ == "__main__":
    dc.run_cli(gru.MODEL_NAME, gru.MODEL_KEY, gru.LAYER)