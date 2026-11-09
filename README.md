# TBRGS - Traffic-Based Route Guidance System
COS30019 Introduction to AI - Assignment 2, Part B (Machine Learning and Software Integration)

Predict traffic flow at SCATS intersections in Boroondara (Melbourne) with machine learning,
then use the predictions to find the best routes between two intersections.

---

## 1. Project status

| Task | What it is | Status |
|---|---|---|
| 1 | Data processing | Done (`Data/data_processing.py`) |
| 2 | ML models (Linear Regression, LSTM, GRU) | Linear done. LSTM and GRU code written, **not trained yet** |
| 3 | Travel time for each edge | Not started |
| 4 | Integrate Part A (top-k routes) + GUI | Not started (`main.py` is empty) |

Update this table as the team makes progress.

---

## 2. Folder structure

```
Assignment2B/
├── README.md
├── requirements.txt          libraries to install
├── config.yaml               default settings (epochs, lags, paths ...)
├── main.py                   entry point / GUI (Task 4, not written yet)
├── Data/
│   ├── Raw_Data/             original VicRoads files (do not edit)
│   ├── Processed_Data/       cleaned data made by data_processing.py
│   ├── data_processing.py    Task 1
│   └── edges.csv             road links between SCATS sites (DRAFT, see section 6)
├── Source/
│   ├── Models/               Task 2
│   │   ├── evaluate.py            one shared "referee" for all models
│   │   ├── linear_regression_model.py
│   │   ├── lstm_model.py / gru_model.py / deep_common.py
│   │   ├── train_linear.py / train_lstm.py / train_all.py
│   │   └── plot_results.py        figures for the report
│   └── Routing/
│       └── build_edges.py    makes Data/edges.csv
├── Results/                  metrics, predictions, plots
├── saved_models/             trained models (created when you train)
├── tests/                    26 automatic tests
├── Assignment PDF/           the assignment brief
└── teacher_give_info/        files given by the teacher
```

---

## 3. Install

- Use Python 3.10 - 3.12.
- Make a virtual environment (recommended), then install:

```bash
pip install -r requirements.txt
```

- `tensorflow` is large. It is only needed for LSTM and GRU.

---

## 4. How to run

Run every command from the **project root** (the folder that contains this README).

### Task 1 - clean the data (already done, outputs are in `Data/Processed_Data/`)
```bash
python Data/data_processing.py --xls "Data/Raw_Data/Scats Data October 2006.xls" --out Data/Processed_Data
```

### Task 2 - train and compare models
```bash
python Source/Models/train_all.py                                   # everything
python Source/Models/train_all.py --sites 2000 3002 --epochs 5      # quick test (2 sites)
python Source/Models/train_all.py --only gru                        # one model only
python Source/Models/train_all.py --skip-training                   # only compare + plots
```
- Settings come from `config.yaml`. Change a value there instead of editing the code.
- Results go to `Results/` (CSV files and `Results/plots/`).

### Build the draft road links
```bash
python Source/Routing/build_edges.py
```

### Run the tests
```bash
python -m pytest -v
```

---

## 5. How the models are compared (fair rules)

All models use the same rules, set in `Source/Models/evaluate.py`:

- **One model per SCATS site.**
- **Test set** = the last 7 days of October 2006 (672 rows of 15 minutes). Models never see it in training.
- **Metrics** = MAE and RMSE, in vehicles per 15 minutes (lower is better).
- **Only real measurements are scored.** Repaired (imputed) values are skipped.
- **Low-quality sites** (3685, 4264) are left out.
- **Baseline** = "next value = last value". Every model should beat it.
- Inputs: the last 12 values (3 hours), time of day (sin/cos) and a weekend flag.

Results so far (38 sites):

| Model | Mean MAE | Mean RMSE |
|---|---|---|
| Persistence baseline | 33.93 | 47.62 |
| Linear Regression | 29.59 | 40.08 |
| LSTM | not trained yet | |
| GRU | not trained yet | |

---

## 6. About `Data/edges.csv`

- The VicRoads data has the intersections but **not** the roads between them.
- `build_edges.py` makes a **draft** with a simple rule:
  - Each detector name tells us the two roads at a site (`WARRIGAL_RD N of TOORAK_RD`).
  - Sites on the same road are sorted along the road, and neighbours are linked.
- Columns: `from_site, to_site, road, distance_km` (every link appears in both directions).
- Current draft: 94 directed edges, 36 sites in one connected network.
- **Not linked yet (add by hand):**

| Site | Closest sites (km) |
|---|---|
| 2200 | 3126 (1.25), 4063 (1.60) |
| 2846 | 4043 (1.64), 4273 (2.02) |
| 4812 | 4262 (0.83), 4263 (1.06) |
| 4821 | 4262 (1.12), 3001 (1.24) |

- **Please check the draft on a real map (Google Maps or OpenStreetMap) before using it.**
  - The assignment says the given latitude and longitude are not exact.
  - Some real roads may be missing or wrong. Edit the CSV by hand.
  - Tell the marker in the report that the edges are team-made.

---

## 7. Tests (26 automatic tests, TC-01 to TC-26)

| Group | Test cases | What they check |
|---|---|---|
| `test_data_processing.py` | TC-01 to TC-05 | No missing or negative flow, full month per site, 15-minute steps, coordinates, quality report |
| `test_evaluate.py` | TC-06 to TC-12 | MAE and RMSE maths, 7-day test set, rejects scaled predictions, skips imputed rows, Linear beats baseline |
| `test_models.py` | TC-13 to TC-20 | No future data in features, time features, scaler, `predict_hourly` = 4 x `predict`, config loading |
| `test_edges.py` | TC-21 to TC-26 | Distance, road-name parsing, chaining, long links dropped, `edges.csv` is consistent, route 2000 to 3002 exists |

- Tests need only numpy, pandas, scikit-learn, pyyaml, joblib, xlrd and pytest.
- TC-18 and TC-19 import `deep_common.py`. If TensorFlow is not installed they still pass, because TensorFlow is only imported inside the training functions.
- Put the test results (the pass/fail output) in the report's Testing section.

---

## 8. Known issues (fix these first)

- **Import path bug.** `data_processing.py` is in `Data/`, but `train_linear.py` and `deep_common.py` look for it in `src/` or the project root. Training scripts crash with `No module named 'data_processing'`.
  - Easy fix: move `data_processing.py` to `Source/` and change the imports.
  - The tests already add `Data/` to the path, so they are not affected.
- **Old folder names.** `train_linear.py` and `linear_regression_model.py` still use `data/processed`, `results` and `src`. Change them to `Data/Processed_Data` and `Results`.
- **`predict()` needs history.** It needs the 12 intervals right before the requested time. The data only covers October 2006, so decide how the route finder chooses a departure time.
- **LSTM and GRU have not been trained yet.**
- **Third algorithm.** Ask the tutor to approve Linear Regression as the third model.

---

## 9. Team

| Name | Student ID | Contribution |
|---|---|---|
| | | |
| | | |
| | | |
