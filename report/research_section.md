# Research: showing our predictions on a real map

*Draft for the report. Pictures are in `results/maps/screenshots/`; the offset of every site is
in `results/maps/screenshots/site_offsets.csv`.*

---

## Research

### What we did

The brief suggested building something like **Google Maps Traffic** using free map data from
**OpenStreetMap**. We built a web map that shows:

- every road coloured by our **predicted** speed (green = fast, red = slow);
- the 5 recommended routes drawn on the real streets;
- a slider to see the predicted traffic at any hour of the day.

The map opens from the GUI with the button **Open traffic map (OpenStreetMap)**.

### Problem 1: the SCATS positions are in the wrong place

The brief warns that the SCATS latitude and longitude do not match the real intersections.
We checked this for all 40 sites.

**How we fixed it.** Each site's name tells us which roads meet there (for example, site 2000 =
Warrigal Rd and Toorak Rd). We downloaded the real roads near each site from OpenStreetMap and
found the point where those roads actually cross. The site was moved to that point.

**What we found.** 38 of the 40 sites were matched to their real intersection. They were on
average **204 m** away from it, and 33 of the 38 were between 180 and 230 m away. The direction
was also the same almost everywhere: on average the real intersection is **169 m north and
113 m east** of the SCATS position. So the error is a **constant shift** in the data (every point
sits about 200 m south-west of where it should be), not random mistakes. Figure R2 shows this
for site 2000.

*Table R1. How far the SCATS positions were from the real intersections (40 sites).*

| Result | Number of sites | Distance moved |
|---|---|---|
| Moved to the real intersection | 38 | average 204 m (most between 180 and 230 m) |
| Moved onto the nearest correct road only | 2 | 121 m and 170 m |
| Not moved (no match found) | 0 | – |

Two sites could only be moved onto one of their roads:

- **Site 2825 (Burke Rd / Eastern Fwy):** the freeway goes under a bridge, so the two roads
  never touch on the map.
- **Site 2200:** SCATS calls the road "Maroondah Hwy", but today it is named "Whitehorse Road"
  on the map.

*Figure R1. Sites before and after the correction. Black dots: the corrected positions on the
real intersections. Purple circles and dashed lines: the original SCATS positions and how far
each one moved.* → `positions_after.png` (optionally next to `positions_before.png`)

*Figure R2. Close-up of site 2000. The green dot is the real Warrigal Rd / Toorak Rd
intersection. The original SCATS position (purple) is about 205 m to the south-west, near
Chester Street.* → `positions_zoom_2000.png`

### Problem 2: straight lines are not real roads

Our routing measures each road as a straight line between two sites. To draw the real
streets, we asked a free route planner (OSRM) for the actual driving path between each pair of
sites.

**What we found.** 100 of the 108 road links now follow the real streets. The other 8 are drawn
as straight lines because the route planner took a detour (for example at the Eastern Freeway
ramps). The real roads are on average about **5 % longer** than the straight lines
(133.1 km vs 127.3 km in total), so our travel times are slightly too short. Because almost all
roads are affected in a similar way, the order of the recommended routes hardly changes.

### What the map shows

*Figure R3. Predicted traffic (GRU) for a trip from 2000 to 3002 on Thursday 26 October 2006:
(a) leaving at 03:00, (b) leaving at 08:00.* → `traffic_0300.png`, `traffic_0800.png`

The map makes our results easy to see:

- **At 03:00** every road is green (60 km/h). Route 1 goes along Toorak Rd and takes 12.3 min.
- **At 08:00** Toorak Rd and Warrigal Rd south of the start turn red. The system now sends the
  driver north along Warrigal Rd and Riversdale Rd instead (14.5 min).

This is the same result as Experiment 2 in *Insights*, now shown on a real map.

### Limitations

- The free OpenStreetMap servers are slow, so the correction is run once and saved. After that,
  the map works without them.
- The map shows today's roads, which can differ from 2006.
- To keep all numbers in this report the same, the route search still uses the original SCATS
  positions and straight-line distances. The corrected positions are only used for the map.

### Testing

Ten automated tests (TC-36 to TC-45 in *Testing*) check this feature. For example, they check
that "WARRIGAL_RD" matches "Warrigal Road", and that a site moves to the correct crossing point.
