# TBRGS - Traffic-Based Route Guidance System (Boroondara)

COS30019 Introduction to AI, Assignment 2 Part B.

The system predicts traffic flow at the Boroondara SCATS sites (October 2006 data) with
machine-learning models, turns the predicted flow into a travel time for every road link,
and uses the Part A tree-based searches to find up to **5 routes** between two SCATS sites.

```
SCATS data -> clean (Task 1) -> LSTM / GRU / Linear models (Task 2)
          -> flow -> speed -> edge travel time (Task 3) -> Part A search + top-5 routes (Task 4) -> GUI
```

## 1. Setup

Needs **Python 3.14** (the saved models only load with the pinned versions).

```bash
bash setup_env.sh          # creates .venv with the exact versions; do NOT use plain pip install
source .venv/bin/activate
```

No TensorFlow? The GUI still works with **Linear regression** and **Measured flow**; GRU and
LSTM are hidden automatically.

## 2. Run

### GUI (easiest)

```bash
python main.py
```

1. Choose **From** and **To** (SCATS sites), the **Date** and **Leave at** time (1-31 Oct 2006).
2. Press **Find routes** (or Enter). The first search takes about 10 s while the models load.
3. Click a route in the table: it is drawn bold on the map and listed step by step on the right.

Settings: traffic model (GRU / LSTM / Linear / Measured), search method, number of routes (1-5).
Advanced: where the flow is taken (origin or destination site), flow per road, and colouring
the roads by predicted speed (green fast, red slow).

**Train models...** (bottom left) opens a window that trains Linear / LSTM / GRU and shows the
output live. *Quick demo* (3 sites, 2 epochs, about 1 minute) writes to a temporary folder and
does not change the saved models. *Full training* (20+ minutes) replaces them, so restart the app afterwards.

Departures before about 03:00 on 1 Oct 2006 are rejected: the models need the previous 3 hours.

### Command line

```bash
python main.py --cli -o 2000 -d 3002 -t "2006-10-26 08:00"                 # defaults: GRU, A*, 5 routes
python main.py --cli -o 2000 -d 3002 -t "2006-10-26 17:30" --model linear --method BFS -k 3
```

Methods: `AS` (A*), `BFS`, `DFS`, `GBFS`, `CUS1` (IDDFS), `CUS2` (IDA*). Models: `gru`, `lstm`, `linear`, `actual`.

### Pictures for the report

```bash
python -m src.GUI.report_figures -o 2000 -d 3002 -t "2006-10-26 08:00" --model gru
```
Saves the route map, the speed-coloured map and the route table to `results/plots/`.

## 3. Settings (config.yaml)

All defaults live in `config.yaml`. The ones that matter for routing (`travel_time:`):

| Key | Meaning |
|---|---|
| `flow_site` | `origin` (follows the conversion PDF) or `destination` (follows the brief): whose flow sets the speed of edge A->B |
| `flow_mode` | `per_detector` (site total / number of detectors) or `total`. `total` puts most roads at capacity speed, so `per_detector` is the default |
| `default_model` | `linear`, `lstm`, `gru` or `actual` |
| `speed_limit_kmh`, `capacity_flow_vph`, `capacity_speed_kmh`, `intersection_delay_s` | the flow-to-travel-time constants |

## 4. Train / rebuild (only if needed; trained models are included)

```bash
python -m src.Models.data_processing        # raw .xls -> data/Processed_Data
python -m src.Models.train_all              # linear + LSTM + GRU, comparison and plots (slow: ~20 min)
python -m src.Routing.graph_builder         # rebuild the road network (edges.csv + network map)
python -m src.Routing.edge_costs --time "2006-10-26 08:00" --model gru     # travel time of every edge
```

## 5. Tests

```bash
python -m unittest discover tests -v        # 44 tests, about a second, no TensorFlow needed
```

## 6. Project layout

| Path | Content |
|---|---|
| `main.py` | entry point: GUI, or `--cli` |
| `src/Models/` | Tasks 1-2: data processing, linear / LSTM / GRU, evaluation, plots |
| `src/Routing/` | Task 3-4: network (`graph_builder`), flow -> time (`travel_time`, `edge_costs`), `route_finder` |
| `src/Search/` | Part A searches (DFS, BFS, GBFS, A*, IDDFS, IDA*) and `top_k.py` (Yen's algorithm for k routes) |
| `src/GUI/` | the Tkinter GUI and the report-figure script |
| `tests/` | unit tests (travel time, graph, edge costs, top-k, route finder) |
| `data/`, `results/`, `saved_models/` | data, metrics and plots, trained models |

## 7. Known limits

- Edge length is the straight-line distance between two sites, not the road length.
- SCATS coordinates are slightly off the real intersections; 6 links between roads that change
  name were added by hand (`EXTRA_EDGES` in `src/Routing/graph_builder.py`).
- Sites 3685 and 4264 have no trained model (low data quality) and use the historical average flow.
- Only October 2006 can be queried.
- Only A* is guaranteed to return the true 5 fastest routes; the other methods return valid,
  different routes that may be slower.
