# COS30019 traffic guidance educational prototype

This is an educational starting point, not a completed assignment submission. The traffic data and the road graph are fictional. No Boroondara dataset, Part A program, or course traffic-flow-to-travel-time conversion was available when it was created.

## What it does
- Reads a simple long CSV format, trains per-site ridge regression, LSTM, or GRU, and evaluates a chronological held-out period.
- Provides a Tkinter GUI and a command-line demonstration, with an example configuration file.
- Finds up to five loopless routes using fixed non-negative edge weights. The demonstration edge-time conversion uses free-flow speed and an example delay, **ignores traffic predictions**, and is not the course formula.
- Includes 15 automated tests using synthetic data. This does not replace testing the actual assignment system.

Third model: the linear ridge regression is only a proposed model; tutor approval is required. LSTM/GRU require PyTorch.

## Setup
Use Python 3.10 or newer with tkinter support. Open a terminal in the extracted folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` and use `py` in place of `python3` if needed. PyTorch installation can depend on your operating system; consult the official PyTorch installation instructions if installation fails.

## Run
The fictional data files are included. Recreate them with `python generate_demo.py` if needed.

```sh
python cli.py
python cli.py --model lstm --epochs 20
python cli.py --model gru --epochs 20
python app.py
python -m unittest discover -s tests -p 'test_*.py' -v
python -m unittest -v tests.test_01_load_regular_site_data
```

The command-line demonstration trains one model per site and displays synthetic evaluation and demo routes. The GUI lets you edit paths/settings, train individual or all sites, see a prediction chart, and save defaults. To display example routes in the GUI, select the checkbox explicitly marked as a fictional demonstration and train models for all sites first. Its route times ignore predictions and are not assignment-compliant. If tkinter is unavailable, the GUI requires a Python installation with Tk support and a working graphical desktop.

## Input formats
Traffic CSV columns: `site_id,timestamp,flow`, where timestamps are ISO format at exactly 15-minute intervals per site and flow is the nonnegative count for the interval. Values with gaps, duplicate times or invalid entries are rejected rather than silently repaired. This is a prototype format, not a verified format for the real dataset.

Road CSV columns: `origin,destination,distance_km`. Each row is directed; add a reverse row if travel in the opposite direction is allowed. Site identifiers must match traffic data. A real road network and distances must be sourced or supplied by your course.

## Required work before assignment use
1. Obtain the provided dataset and inspect its real schema. Create and verify an adapter for it and document the handling of missing or irregular records.
2. Obtain the actual Boroondara graph and connect the group's Part A search implementation. The included generic route finder is not the group's Part A code.
3. Obtain and implement the course's traffic-flow-to-travel-time conversion, including its hourly aggregation and intersection delay rules. Remove the free-flow demo method for assignment results. Presently the GUI's `4 * next_interval_forecast` is only a demonstration assumption and is ignored by its demo conversion.
4. Obtain tutor approval for the third model. Train all three with consistent splits; use a separate validation period for tuning and reserve the test period for final evaluation.
5. Verify graphical operation, model training, evaluation and routes against real data. Record real results, any missing features and known bugs in the report. Research papers do not establish correctness of this code.

## Validation performed for this starting point
- PASS: 13 automated synthetic tests, including the ridge model, scaler boundary, and generic routing.
- NOT RUN: 2 PyTorch tests (PyTorch not installed in the checked environment); LSTM/GRU training.
- NOT RUN: actual Boroondara data, course conversion, Part A integration, and full GUI interaction. GUI launch failed in the checked environment due to Tcl/Tk display setup; the GUI requires further verification.
- The command-line fictional example ran successfully. Its numbers are not assignment evidence.

## Files
- `forecast.py` data validation, per-site training, saving and prediction.
- `routing.py` generic route finder and explicit conversion interface.
- `app.py` graphical interface, example chart and fictional conversion.
- `cli.py` terminal demonstration.
- `generate_demo.py` creates fictional input CSV files.
- `config.json` GUI defaults.
- `tests/` contains 15 separate automated test files, a shared support fixture and a package initializer.
- `requirements.txt` Python dependencies.
- `demo_data/` fictional traffic and directed edges.
