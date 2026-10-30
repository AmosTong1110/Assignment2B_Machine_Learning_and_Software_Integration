# Map trace draft

This is an explicitly incomplete, approximate interpretation of IBM data_Bor.pdf, not a validated network. The user suggested assuming roads are usable in both directions; each candidate segment is therefore present in both directions. That does not verify actual traffic rules or historical direction restrictions. No routes from 2000 to 3002 can be generated using this draft alone.

The original PDF was rendered to 3200 pixels in width. The scale bar marked 0–4 km spans approximately 577 pixels in the render, or 144.25 pixels/km. Node positions were visually estimated at pink marker centres. Distances below are Euclidean distances in the map projection, approximating the corresponding near-straight road segment on this printed map. They are not measured road lengths, and diagram distortion and imprecise marker/scale endpoints mean these figures should not be called exact. No path-length tracing, road geometries, turns or connectivity at crossings have been verified. No confidence intervals are claimed. Manually confirm each connection against map, supporting road data, and tutor before using.

| Sites | Render marker centres (x,y pixels) | Approx km | Visual basis |
|---|---|---|---|
| 2000–3682 | (2139,1540), (2172,1309) | 1.62 | The map appears to show a continuous segment along Warrigal Road. |
| 3002–4263 | (1284,953), (1265,1079) | 0.88 | The map appears to show a continuous segment along the same north–south road. |
| 3180–4057 | (2003,652), (1982,793) | 0.99 | The map appears to show a continuous segment along the same north–south road. |
| 4321–4032 | (1571,725), (1720,749) | 1.05 | The map appears to show a continuous road segment. Labels and nearby markers warrant independent review. |

Controlled-intersection status is left blank because the image alone does not establish the required metadata. Source 2006 workbook covers only 40 sites and includes multiple measurement locations per site; the map includes many additional SCATS sites and does not supply compatible predicted hourly flows for all of them. This limited extraction is not a complete road network, should not be mixed with synthetic edges, and is unsuitable to generate purported assignment routes until graph and flow coverage are established.

Additional provisional segments on the same visible east–west road: 2000–4050 is approximately 0.19 km from marker centres (2139,1540) and (2112,1537); 4050–4049 is approximately 0.81 km from (2112,1537) to (1997,1519). The latter has closely spaced labels and should be independently checked. Neither 4050 nor 4049 appears among the 40 traffic workbook sites; a complete time-dependent route cannot be calculated without a defensible approach to missing flows.
