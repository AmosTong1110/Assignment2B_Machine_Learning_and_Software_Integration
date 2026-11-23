# Experiments: figures and discussion (draft for the report)

All experiments use the test week (Thu 26 Oct 2006) and four trips across the network:
2000→3002 (the brief's example), 4821→2827, 0970→4051 and 3001→2200.
Tables: `results/experiments/*.csv` (`python -m src.Routing.experiments`).
Figures: `python -m src.Routing.experiment_figures`.

---

## Figure 1: Part A search methods (`fig1_search_methods.png`)

*Figure 1. Search effort (a, log scale) and route quality (b) of the six Part A methods when
finding 5 routes with Yen's algorithm (GRU predictions, 08:00). In (b), no bar means the method
returned the same 5 routes as A\*.*

- **A\* is the only method guaranteed to find the true 5 fastest routes.** It needed 486–7,188
  search nodes, depending on the trip.
- **BFS, GBFS, IDDFS and IDA\* found the same fastest route as A\* on every trip.** Their
  2nd–5th routes were sometimes slightly slower (up to +0.64 min on average for GBFS on
  4821→2827). Route 1 matches because the fastest route here is also one with few
  intersections, so even methods that ignore cost find it.
- **DFS gave the worst routes.** The average of its 5 routes was 1.9–9.8 min slower than A\*'s.
  On 0970→4051 its *best* route was 3.5 min slower (21.2 vs 17.7 min). DFS takes the first path
  it reaches, not the cheapest one.
- **The iterative-deepening methods (IDDFS, IDA\*) cost the most.** On the longer trips they
  created 37,000–63,000 nodes, 8–10× more than A\*, because each new depth limit repeats the
  earlier work.
- **GBFS created the fewest nodes on short trips (301 and 413) but was the slowest method on
  4821→2827 (1,028 ms).** A greedy search fits Yen's spur searches poorly: inside each spur
  search it often follows the heuristic into a dead end.

**Take-away:** A\* is the right default for TBRGS. It is the only optimal method, and in our
data it was also among the cheapest.

## Figure 2: Departure time (`fig2_time_of_day.png`)

*Figure 2. (a) Fastest-route time for each trip relative to 03:00; hollow markers show that a
different route was chosen. (b) Mean predicted site flow (bars) and mean link speed (line) over
the whole network.*

- **The predicted traffic follows a clear daily pattern.** Mean site flow rises from 25 veh/h at
  03:00 to 883 veh/h at 08:00, and mean link speed falls from 60 to 52 km/h. The morning peak
  is heavier than the evening peak (806 veh/h, 53.4 km/h at 17:00).
- **Trip times grow by 7–18 % in the morning peak.** The brief's example 2000→3002 grows most,
  from 12.3 min at 03:00 to 14.5 min at 08:00.
- **The recommended route changes with the time of day.** For 2000→3002 the system chooses
  Toorak Rd → Burke Rd → Barkers Rd off-peak, but Warrigal Rd → Riversdale Rd → Burke Rd at
  07:00, 08:00 and 17:00, when Toorak Rd is predicted to be busy. The other three trips keep
  their route all day because their alternatives are much longer.

**Take-away:** the ML predictions change the routes the system recommends, not only the time
estimates. That is the point of a *traffic-based* route guidance system.

## Figure 3: Flow per link (`fig3_flow_mode.png`)

*Figure 3. Effect of the `flow_mode` setting: (a) number of the 108 directed links at the
capacity speed of 32 km/h; (b) mean fastest-route time over the four trips.*

- **SCATS flow is the total of all detectors at an intersection, but the conversion formula
  expects the flow on one road.** Using the total (`total`) puts 102 of 108 links at capacity
  at 08:00, and still 87 at midday. Trip times rise by about 44 % (19.6 vs 13.6 min).
- **With every link capped at 32 km/h, the traffic information disappears.** Routes are then
  chosen almost only by distance and number of intersections. At 12:00, for example, 2000→3002
  switches to a 6-road detour.
- **Dividing by the number of detectors (`per_detector`) keeps links below capacity**
  (2 links at 08:00, 0 otherwise), so speeds still differ between roads.

**Take-away:** this justifies `per_detector` as the default in `config.yaml`. It is a
simplification (detectors are not split exactly by approach), but it keeps the flow-to-speed
curve in its useful range.

## Figure 4: Choice of ML model (`fig4_model_choice.png`)

*Figure 4. Routing with GRU, LSTM and Linear predictions compared with routing on the measured
flow: (a) error of the predicted fastest-route time; (b) time lost in real (measured) traffic by
following the predicted route; (c) mean absolute error of the link travel times.*

- **All three models recommend the same route as the measured flow in 7 of 8 cases.** In the
  eighth (2000→3002 at 17:00), all three choose a route that is only 4.8 s slower in real
  traffic.
- **Predicted trip times are within 16 s of the measured-flow times (at most 2.2 %).** The
  errors are mostly negative: the models slightly *underestimate* peak traffic, most of all
  for 4821→2827 and 3001→2200 at 08:00.
- **On link travel time the deep models beat linear regression** (MAE 1.10 s for LSTM,
  1.15 s for GRU, 1.50 s for Linear). This agrees with the model comparison in the Insights
  section.
- **The difference between models matters little for routing.** A link's travel time is
  mostly distance at the speed limit plus the fixed 30 s intersection delay. Speed only varies
  between 32 and 60 km/h, so a flow error of tens of vehicles per hour moves a link's time by
  about a second.

**Take-away:** better flow prediction (GRU/LSTM) gives more accurate time *estimates*, but with
this travel-time model the *route choice* is robust even to the simple linear model. A
travel-time model that reacts more strongly to congestion (for example the congested branch of
the curve, or real signal delays) would make model accuracy matter more.
