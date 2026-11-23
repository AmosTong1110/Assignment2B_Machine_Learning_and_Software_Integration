# TBRGS test cases

**50 test cases:** 45 automated unit tests and 5 command-line system tests.
**Result: all 50 pass** (run on 23 Nov 2026).

- Automated tests: `python -m unittest discover tests -v` (about 1 second, no TensorFlow needed)
- System tests: run by hand on the command line (`python main.py --cli ...`)
- The ID is written as a comment next to each test in the code, e.g. `# DP-04`.

| Group | IDs | File | What is covered |
|---|---|---|---|
| Data processing (Task 1) | DP-01 – DP-07 | `test_data_processing.py` | cleaning rules, validation, saved dataset |
| ML helpers and models (Task 2) | ML-01 – ML-05 | `test_data_processing.py` | scaling, features, no look-ahead, predictions |
| Evaluation | EV-01 – EV-03 | `test_data_processing.py` | metrics, test split, input checks |
| Travel time (Task 3) | TT-01 – TT-05 | `test_travel_time.py` | flow → speed → time (conversion PDF) |
| Road network (Task 3) | GB-01 – GB-04 | `test_travel_time.py` | building edges between SCATS sites |
| Edge costs (Tasks 2+3) | EC-01 – EC-04 | `test_travel_time.py` | flow per link, model fallback, settings |
| Top-k routes (Task 4) | TK-01 – TK-03 | `test_route_finder.py` | Yen's algorithm on all Part A searches |
| Route finder (Task 4) | RF-01 – RF-04 | `test_route_finder.py` | end-to-end routing on the real map |
| OSM map (Research) | MAP-01 – MAP-10 | `test_web_map.py` | coordinate correction, map page |
| System (end-to-end) | ST-01 – ST-05 | command line | whole program, valid and invalid input |

Some IDs cover a test method with several checks. The tables list each method once.

---

## 1. Data processing (Task 1)

The cleaning rules are tested on small hand-made tables, so the exact expected output is known.

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| DP-01 | Find long runs of zero flow (outage detector) | `[1,1,0,1,1,1,1,0,1]`, minimum run 3 | Only the run of four is marked | Pass |
| DP-02 | Drop a dead detector | Site with one normal detector and one reading 0 all month | Dead detector removed and reported; normal one kept | Pass |
| DP-03 | Add detectors into one site flow; blank incomplete days | Detector A = 30, detector B = 20, B missing on day 1 | Day 1 = missing (NaN); other days = 50; 1 incomplete day counted | Pass |
| DP-04 | Flag outages and spikes but keep real rush hours | 3 weeks with a 07–09 peak, plus a 2.5 h zero run and one 900 veh value | Zero run and spike set to NaN (10 + 1 values); rush-hour peaks untouched | Pass |
| DP-05 | Repair gaps | 30 min gap between 30 and 60; 5 h gap | Short gap filled by a straight line (40, 50); long gap filled with the same-weekday profile (40); all 22 repairs flagged `imputed=1` | Pass |
| DP-06 | Quality checklist catches broken data | Real cleaned data; then a copy with one NaN; then a duplicate row | Real data passes; both broken copies raise an error | Pass |
| DP-07 | Saved clean dataset is complete | `scats_clean_15min.csv`, `site_metadata.csv` | 40 sites × 2,976 rows (31 days × 96); site "0970" keeps its leading zero | Pass |

## 2. ML helpers and models (Task 2)

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| ML-01 | Scaler maps to 0–1 and back, and survives saving to JSON | `[10, 55, 100]` | `[0, 0.5, 1]`; reloaded scaler gives back the original values | Pass |
| ML-02 | Time features | Fri 23:45, Sat 00:00, Fri 12:00 | 23:45 is next to 00:00; noon is opposite midnight; weekend flag only on Saturday | Pass |
| ML-03 | Training windows never use future data | 20 values in reverse order, 12 → 4 lags | Sorted by time first; every input is earlier than its target | Pass |
| ML-04 | Linear model prediction | Site 2000 at 01 Oct 01:00; then at 26 Oct 08:00 | First is rejected (needs 3 h of history); second is ≥ 0 and within 30 % of the measured flow | Pass |
| ML-05 | Every model beats the naive baseline ("next = last value") | `results/model_comparison.csv` | GRU, LSTM and Linear MAE all lower than persistence | Pass |

## 3. Evaluation

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| EV-01 | Metrics give known values | true `[10,20,30,40]`, predicted `[12,18,30,44]` | MAE = 2, RMSE = √6, NRMSE = √6 / 25 | Pass |
| EV-02 | Test set = last 7 days, the same for every model | Site 2000 | 672 rows, 25 Oct 00:00 – 31 Oct 23:45 | Pass |
| EV-03 | Common mistakes are caught | One prediction missing; predictions still scaled 0–1; perfect predictions | First two raise an error; perfect gives MAE 0 and scores only real (non-imputed) rows | Pass |

## 4. Travel time (Task 3)

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| TT-01 | Key points of the flow–speed curve | Flow 0, 351, 1500, 3000, −1 veh/h | 60 km/h up to 351; 32 km/h at 1500; above capacity clamped to 32; negative flow rejected; A = −1.46484375 | Pass |
| TT-02 | Curve shape | Flow 400 → 1400 | Speed always falls; matches the PDF formula both ways; congested branch < 32 km/h | Pass |
| TT-03 | Edge travel time | 1 km at flow 0 and at 1500; 0 km | 90 s; 142.5 s; 30 s; more traffic is never faster | Pass |
| TT-04 | Settings from config.yaml are used | Speed limit 50, delay 0 | 1 km = 72 s | Pass |
| TT-05 | Distance between coordinates | 0.01° of latitude | ≈ 1.112 km; same point = 0 | Pass |

## 5. Road network (Task 3)

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| GB-01 | Read road names from detector text | "WARRIGAL_RD N of TOORAK_RD", a typo alias, nonsense text | Both roads found; alias corrected; nonsense rejected | Pass |
| GB-02 | Join neighbouring sites on the same road | Sites A, B, C in a line on X_RD | A–B and B–C both ways, but not A–C | Pass |
| GB-03 | Ignore far-apart sites that share a road name | Two sites 11 km apart on "HIGH_ST" | No edge; join reported as dropped | Pass |
| GB-04 | The real network is connected | `edges.csv` | Every site can reach every other site; all distances > 0 | Pass |

## 6. Edge costs (Tasks 2 + 3)

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| EC-01 | Flow per link (origin site, per detector) | Site A 100 veh/15 min, 2 detectors | Link flow 200 veh/h; travel time 90 s | Pass |
| EC-02 | Destination site, total flow | Site B 300 veh/15 min | Link flow 1200 veh/h; speed from the curve | Pass |
| EC-03 | Model prediction, plus fallback for a site with no model | Fake GRU = 1500 veh/h at A; B has no model | A uses GRU (142.5 s); B uses the historical average | Pass |
| EC-04 | Bad settings are rejected | Unknown flow_site, flow_mode, model | Each raises an error; normal costs are ≥ 30 s | Pass |

## 7. Top-k routes, Yen's algorithm (Task 4)

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| TK-01 | Textbook example | Yen's example graph (C → H), k = 3 and k = 50 | Costs 5, 7, 8; asking for 50 returns every route, no duplicates, no loops | Pass |
| TK-02 | No route exists; invalid k | H → C (edges point the wrong way); k = 0 | Empty list; k = 0 rejected | Pass |
| TK-03 | All six Part A methods | DFS, BFS, GBFS, A\*, IDDFS, IDA\*, k = 4 | 4 valid routes each, sorted by cost, costs add up | Pass |

## 8. Route finder (Task 4)

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| RF-01 | The brief's example | 2000 → 3002, every method, k = 5 and k = 2 | Up to 5 routes from 2000 to 3002 with every method; k limits the count | Pass |
| RF-02 | A\* returns the true 5 fastest routes | 3 trips, compared with checking every possible path | Same 5 costs; heuristic never overestimates | Pass |
| RF-03 | Route time = driving time + 30 s per intersection | 2000 → 3002 at 600 and 1400 veh/h | Times add up; heavier traffic gives a slower route | Pass |
| RF-04 | Invalid input is rejected | k = 6, same origin and destination, site 9999, method "DIJKSTRA", site "970" | All rejected with a clear message; "970" accepted as "0970" | Pass |

## 9. OpenStreetMap map (Research)

No internet is needed: the OSM roads are small hand-made examples.

| ID | Scenario | Input | Expected result | Result |
|---|---|---|---|---|
| MAP-01 | SCATS and OSM names match | "WARRIGAL_RD" vs "Warrigal Road", "SEYMOUR_GV" vs "Seymour Grove", … | Same key | Pass |
| MAP-02 | Different roads stay different | "HIGH_ST" vs "High Street Road" | Different keys | Pass |
| MAP-03 | Split a site label into roads | "TOORAK_RD / WARRIGAL_RD" | {TOORAKRD, WARRIGALRD} | Pass |
| MAP-04 | Snap a site to the shared intersection node | Two crossing roads; site 90 m away | Moved exactly onto the crossing | Pass |
| MAP-05 | Only one road found | One of the site's two roads nearby | Moved onto the nearest point of that road | Pass |
| MAP-06 | Nothing found | No matching road | SCATS position kept | Pass |
| MAP-07 | Divided road (two crossing nodes) | Two carriageways 22 m apart | Centre of the two crossings | Pass |
| MAP-08 | Map data for a search | 2000 → 3002, k = 3 | 3 routes; every edge in the network; each route's edges join up | Pass |
| MAP-09 | Time slider | Extra prediction at 07:00 | Slider steps 07:00 and 08:00 | Pass |
| MAP-10 | HTML file is written | Write to a temporary folder | No placeholder left; default background from OpenStreetMap Germany plus a menu to switch background; no extra plugin; data readable | Pass |

## 10. System tests (whole program, command line)

| ID | Scenario | Command | Expected result | Actual result | Result |
|---|---|---|---|---|---|
| ST-01 | Brief example | `-o 2000 -d 3002 -t "2006-10-26 08:00" --model linear` | 5 routes, fastest first | 5 routes: 14.6, 14.6, 15.4, 15.5, 15.5 min (A\*, 486 nodes) | Pass |
| ST-02 | Departure without 3 h of history | `-t "2006-10-01 01:00"` | Clear error, no crash | "need the 12 intervals right before 2006-10-01 01:00" | Pass |
| ST-03 | Unknown destination | `-d 1234` | Clear error listing valid sites | "Destination 1234 is not a SCATS site on the map. Known sites: …" | Pass |
| ST-04 | Too many routes | `-k 7` | Rejected (brief: up to 5) | "number of routes must be between 1 and 5, got 7" | Pass |
| ST-05 | Predicted traffic changes the route | Same trip at 03:00 and 08:00 | Different route 1 | 03:00 via Toorak Rd, Burke Rd, Barkers Rd (12.3 min); 08:00 via Warrigal Rd, Riversdale Rd, Burke Rd, Barkers Rd (14.6 min) | Pass |

---

## ML model evaluation (from `results/model_comparison.csv`)

Testing the models themselves: the last 7 days of October (672 slots per site, 38 sites), scored
only on real (non-imputed) measurements. Flow is in vehicles per 15 minutes.

| Model | MAE | RMSE | NRMSE | Best on (sites) | Training time (s) |
|---|---|---|---|---|---|
| GRU | 24.25 | 33.10 | 0.096 | 19 | 1,178 |
| LSTM | 24.34 | 33.03 | 0.096 | 19 | 1,757 |
| Linear regression | 29.58 | 40.07 | 0.117 | 0 | 0.4 |
| Persistence (baseline) | 33.93 | 47.62 | 0.138 | – | 0 |

All three models beat the baseline (checked automatically by ML-05). GRU and LSTM are about
18 % more accurate than linear regression (MAE), and GRU trains about 33 % faster than LSTM.
