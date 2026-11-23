# Fixes for Assign_2B_Astral_Illusion (paste into the Word file)

Each fix says **where** it goes, **what to delete**, and the **new text** to paste in.
Page numbers refer to the current PDF.

---

## Fix 1: Instructions, section 1 (page 3): Python version

**Delete:** "Python 3.14 is recommended operating system that is compatible with Python 3.11–3.14."

**Paste:**

> The program needs **Python 3.14**: the saved models only load with the exact library versions
> pinned in `requirements.txt`.

---

## Fix 2: Instructions, step A.3 (page 3): search time

**Delete:** "Subsequent searches take less than 0.5 seconds."

**Paste:**

> Later searches take about one second. Changing only the search method or the number of routes
> is instant.

---

## Fix 3: Instructions: add the traffic map (after step A.4, page 4)

**Paste as a new step 5:**

> 5. **Traffic map on OpenStreetMap (Research).** After a search, press **Open traffic map
>    (OpenStreetMap)**. The routes and predicted traffic open in the web browser on a real street
>    map, with a slider for every hour of the day. The first time, run
>    `python -m src.Routing.osm_geometry` (needs internet, about 2 minutes) to place the SCATS sites
>    on the real intersections.

---

## Fix 4: Instructions, section 4 (page 4): test count and new commands

**Delete:** "Run the 20 automated unit test cases verifying Haversine calculations, flow-to-speed
parabolic conversions, intersection delay additions, graph connectivity, A* admissibility, and
Yen's top-k loop prevention:"

**Paste:**

> Software unit test suite: 45 automated tests (about 1 second, no TensorFlow needed). They cover
> data cleaning, the model helpers and evaluation, flow-to-travel-time conversion, the road
> network, Yen's top-k routes, the route finder and the OpenStreetMap map:

**At the end of section 4, add:**

> ♦ Report figures for the four experiments: `python -m src.Routing.experiment_figures`
>
> ♦ Traffic map pictures for the Research section: `python -m src.GUI.map_screenshots`

---

## Fix 5: Features, Bugs and Missing Requirements (pages 6–7): replace the whole section

**Delete everything from the first heading "Features, Bugs and Missing Requirements" down to the
end of the paragraph after Table 2** ("…does not change when predicted traffic changes."). This
includes:

- the first small table (it says "fictional CSV format", "PyTorch was unavailable" and
  "Traffic-based travel time: Not implemented", none of which is true);
- the second heading with the same name;
- the 38-row file list (rows 34–38 are empty and number 8 is missing);
- Table 2 and both paragraphs.

**Paste:**

> ### Features, Bugs and Missing Requirements
>
> **Features implemented.** All required features are implemented and tested.
>
> | No. | Feature | Main file(s) | Status |
> |---|---|---|---|
> | 1 | Data processing: clean the October 2006 SCATS data (dead detectors, outages, spikes, gaps) and save 15-minute flows for 40 sites | `data_processing.py` | Done: 8 quality checks pass, 4.6 % of values repaired and flagged |
> | 2 | Linear Regression model (third ML method) | `linear_regression_model.py` | Done ⟦add: approved by tutor on …⟧ |
> | 3 | LSTM model | `lstm_model.py`, `deep_common.py` | Done: 38 sites |
> | 4 | GRU model | `gru_model.py`, `deep_common.py` | Done: 38 sites |
> | 5 | Evaluation and comparison of the models against a naive baseline | `evaluate.py`, `train_all.py`, `plot_results.py` | Done |
> | 6 | Road network of Boroondara: 40 sites, 108 one-way links | `graph_builder.py` | Done |
> | 7 | Travel time from predicted flow (conversion PDF, 60 km/h limit, 30 s per intersection) | `travel_time.py`, `edge_costs.py` | Done |
> | 8 | Six Part A search methods (A\*, BFS, DFS, GBFS, IDDFS, IDA\*) | `src/Search/` | Done |
> | 9 | Up to 5 routes from origin to destination (Yen's algorithm) | `top_k.py`, `route_finder.py` | Done |
> | 10 | Desktop GUI: trip input, settings, route table, map, step-by-step legs | `app.py` | Done |
> | 11 | Model training window in the GUI | `train_dialog.py` | Done |
> | 12 | Configuration file with all defaults | `config.yaml` | Done |
> | 13 | Command-line version | `main.py --cli` | Done |
> | 14 | Experiments and figures for the report | `experiments.py`, `experiment_figures.py` | Done |
> | 15 | Research: traffic map on OpenStreetMap with corrected site positions | `osm_geometry.py`, `web_map.py` | Done |
> | 16 | Automated tests (45) | `tests/` | Done: all pass |
>
> *Table 1: Implemented features*
>
> **Missing requirements.** None.
>
> **Known bugs and limitations**
>
> | Limitation | Effect |
> |---|---|
> | Road length is the straight-line distance between two sites | Travel times are slightly too short. The order of the routes changes little (see Research) |
> | Sites 3685 and 4264 have no trained model (too much missing data) | They use the historical average flow for that weekday and time |
> | Only 1–31 October 2006 can be queried; departures before 03:00 on 1 October are rejected | The models need the 3 hours before the departure time |
> | Only A\* is guaranteed to return the true 5 fastest routes | Other methods return valid routes that may be slower (see Insights) |
> | The first search takes about 10 seconds | TensorFlow and the models load once; later searches take about 1 second |
>
> *Table 2: Known bugs and limitations*

---

## Fix 6: Testing (pages 8–9): add the new test cases

Keep TC-01 to TC-20. **Add these rows after TC-20** (same columns):

> | ID | Scenario (input) | Expected result | Result |
> |---|---|---|---|
> | TC-21 | Find long runs of zeros in `[1,1,0,1,1,1,1,0,1]` (minimum 3) | Only the run of four is marked | Pass |
> | TC-22 | A detector that reads 0 all month | Dead detector removed and reported; the normal one kept | Pass |
> | TC-23 | Two detectors (30 + 20) at one site, one missing for a day | Site flow = 50; the incomplete day is left blank for repair | Pass |
> | TC-24 | 2.5 h of zeros and one value of 900 in 3 weeks of normal traffic | Both flagged as bad; normal rush-hour peaks kept | Pass |
> | TC-25 | A 30-minute gap and a 5-hour gap | Short gap filled by a straight line; long gap by the usual value for that weekday and time; all repairs flagged | Pass |
> | TC-26 | Quality check on real data, then on data with a blank value or a duplicate row | Real data passes; both broken copies are rejected | Pass |
> | TC-27 | Saved clean dataset | 40 sites × 2,976 rows; site "0970" keeps its leading zero | Pass |
> | TC-28 | Scale `[10, 55, 100]` to 0–1 and back, after saving the scaler | `[0, 0.5, 1]` and the original values back | Pass |
> | TC-29 | Time features for Fri 23:45, Sat 00:00, Fri 12:00 | 23:45 is close to 00:00; weekend flag only on Saturday | Pass |
> | TC-30 | Build training samples from unsorted data | Every input is earlier than its target (no look into the future) | Pass |
> | TC-31 | Linear prediction at 1 Oct 01:00, then at 26 Oct 08:00 | First rejected (no 3 h history); second within 30 % of the real flow | Pass |
> | TC-32 | Compare every model with the naive baseline | GRU, LSTM and Linear all have lower MAE | Pass |
> | TC-33 | MAE / RMSE / NRMSE on known numbers | 2, √6, √6 / 25 | Pass |
> | TC-34 | Test set for site 2000 | Last 7 days: 672 rows, 25–31 Oct | Pass |
> | TC-35 | Predictions with one missing, still scaled 0–1, or perfect | First two rejected; perfect gives MAE 0 on real (not repaired) rows | Pass |
> | TC-36 | "WARRIGAL_RD" vs "Warrigal Road" and other spellings | Recognised as the same road | Pass |
> | TC-37 | "HIGH_ST" vs "High Street Road" | Recognised as different roads | Pass |
> | TC-38 | Site label "TOORAK_RD / WARRIGAL_RD" | Split into its two roads | Pass |
> | TC-39 | Site 90 m from where its two roads cross | Moved exactly onto the crossing | Pass |
> | TC-40 | Only one of the site's roads nearby | Moved onto that road | Pass |
> | TC-41 | None of the site's roads nearby | Position not changed | Pass |
> | TC-42 | Divided road (two crossing points 22 m apart) | Moved to the middle of the two | Pass |
> | TC-43 | Map data for 2000 → 3002, 3 routes | 3 routes; every road included; each route's roads join up | Pass |
> | TC-44 | Extra prediction at 07:00 for the time slider | Slider shows 07:00 and 08:00 | Pass |
> | TC-45 | Write the map web page | File complete, uses OpenStreetMap, data readable | Pass |
> | TC-46 | Command line: 2000 → 3002 at 08:00 | 5 routes, fastest first (14.6 min) | Pass |
> | TC-47 | Command line: departure 1 Oct 01:00 | Clear error message, no crash | Pass |
> | TC-48 | Command line: destination 1234 | Error listing the valid sites | Pass |
> | TC-49 | Command line: 7 routes asked | Rejected (maximum 5) | Pass |
> | TC-50 | Same trip at 03:00 and 08:00 | Different fastest route (via Toorak Rd at 03:00, via Warrigal Rd at 08:00) | Pass |
>
> **Add one sentence under the table:** All 50 test cases pass. TC-01 to TC-45 run automatically
> with `python -m unittest discover tests -v`; TC-46 to TC-50 were run by hand on the command line.

---

## Fix 7: Insights, first paragraphs (page 9): remove the repeat

**Delete the second paragraph**, the one starting "To avoid data leakage, we only fitted our data
scalers on the training data…". The first paragraph already says the same thing.

---

## Fix 8: Insights, Figure 5 (page 12): use the clearer chart

**Replace the picture** with `results/experiments/fig2_time_of_day.png`.

**New caption:**

> *Figure 5: (a) Fastest-route time for each trip compared with 03:00; a hollow circle means a
> different route was chosen. (b) Average predicted flow and speed over the whole network
> (Thursday 26 October 2006, GRU).*

Also delete the stray "(" just above the old caption.

---

## Fix 9: Research (page 13)

Paste the section from `report/research_section.md` under the empty "Research" heading.

---

## Fix 10: Conclusion, "Summary of Machine Learning Suitability" (page 13)

**Delete:** "GRU emerged as the best overall choice. It delivered the highest accuracy while
training roughly 9% faster than LSTM (640 seconds vs. 706 seconds) due to its simpler internal
gating architecture."

**Paste:**

> **GRU emerged as the best overall choice.** It was as accurate as LSTM (MAE 24.25 vs 24.34,
> same NRMSE 0.096) but trained about 33 % faster (1,178 s vs 1,757 s for all 38 sites),
> because its simpler design has fewer parameters to learn.

The old numbers (640 s vs 706 s, 9 %) disagree with the Insights section and with
`results/model_comparison.csv`.

---

## Fix 11: Conclusion, "End-to-End System Integration", point 1 (page 14)

**Delete:** "The data cleaning pipeline ensured 100% complete, outlier-free data across all 40
SCATS sites"

**Paste:**

> The data cleaning pipeline produced a complete dataset for all 40 SCATS sites: faulty values
> were removed and the gaps repaired (4.6 % of all values, each one flagged).

---

## Fix 12: Conclusion, "Potential Improvements" (page 14)

Point 3 is cut off ("bad weathe"). **Finish it**, and **add a 4th point** that links to the
Research section:

> 3. … or bad weather.
> 4. **Real road distances:** using the real road length from OpenStreetMap (Research section)
>    instead of the straight-line distance would make travel times more accurate.

---

## Fix 13: Page count (whole report)

The brief allows **8–10 pages**, not counting the cover page and the table of contents. The
report is now about 12 pages before Research is added. Fix 5 saves about 1.5 pages. To save
more:

- Shorten Instructions (pages 3–5) to about 1 page. The README already has the full details.
- Make the Testing table smaller (font 9 pt, narrower columns), or ask your tutor whether it can
  go in an appendix.
- Show Table 5 (model choice, pages 11–12) as Figure `fig4_model_choice.png` instead. A picture
  is shorter than the 8-row table.

---

## Fix 14: Table of Contents

Update the TOC after the changes above (in Word: right-click the TOC → Update Field → Update
entire table). The heading "Research if Applicable" can just be "Research".
