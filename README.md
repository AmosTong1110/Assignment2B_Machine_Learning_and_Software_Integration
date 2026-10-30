# Assignment 2B Traffic Prediction and Route Guidance

Work in progress. The supplied October 2006 Boroondara workbook is the primary dataset. The October 2026 traffic files previously added are separate and their suitability requires tutor approval. The 2006 data is by measurement location, not automatically an intersection total.

## Prepare data

Install requirements with `python -m pip install -r requirements.txt`. The .xls reader needs xlrd; openpyxl is also provided for a converted .xlsx. Example using the supplied workbook:

```sh
python prepare_2006.py --input 'data/raw/2006/Scats Data October 2006.xls' --site 2000 --location 'WARRIGAL_RD N of TOORAK_RD' --hf-id 6246 --output data/prepared/new_2006_series.csv --longest-contiguous
python cli.py --data data/prepared/site_2000_hf6246_2006.csv --site 2000 --model linear
python -m unittest discover -s tests -p 'test_*.py' -v
python app.py
```

The provided prepared example for site 2000 and HF ID 6246 is ONE measurement location, not the whole intersection. Its 28 recorded days have three continuous segments; the explicitly chosen longest segment keeps 1,152 of 2,688 observations and discards 1,536. This is only a demonstration, not an endorsed approach: inspect missing days and avoid biased data loss. Report the chosen method. Input series are identified by SCATS number, exact location and HF internal ID, since multiple series can share a location. The V00 header represents 00:00; Date cells are normalised to midnight. Leading zero site IDs are preserved.

Raw files in `data/raw/` are ignored by Git. The GUI expects a prepared CSV. PyTorch is needed for LSTM and GRU; the proposed linear ridge model needs tutor approval. Keep test data held out from model selection.

## Travel times and routing

`travel_time.py` implements the *printed* equation in the course PDF: q = -1.4648375 v^2 + 93.75 v, choosing the under-capacity higher-speed root and capping at 60 km/h. Time is 60*distance_km/speed plus 0.5 minutes only if the intersection is confirmed controlled. Flows above the quadratic maximum are rejected; the course does not specify a higher-flow policy. The graph in the PDF displays a slightly different coefficient; verify with the tutor.

The assignment brief says flow at the destination site; the conversion PDF says the starting site. The user chose the brief's destination convention pending tutor confirmation. Existing route_search.py applies destination flow. It requires explicit controlled_intersection values in the edge CSV for course conversion.

The supplied Boroondara PDF is a static map, not a verified road-edge table. The site listing and traffic location CSV do not supply verified SCATS-to-SCATS connections, distances or driving direction. Do not infer routes from proximity. GUI routing remains disabled until a verified graph, controlled-site metadata, site-level aggregation and hourly flow mapping are available. The original Part A code is not integrated.

## Files and validation

`app.py` GUI; `data_processing.py` 2026 detector conversion; `prepare_2006.py` 2006 workbook converter; `models.py` models; `train.py` training; `evaluation.py` errors; `graph.py` and `route_search.py` graph search; `travel_time.py` course conversion; `forecast.py` and `routing.py` compatibility wrappers; `tests/` test cases.

Tests use both small fictional cases and checked source data for an example; they do not verify the full Boroondara application. LSTM/GRU need PyTorch; GUI interaction needs a working desktop. No real-data model comparison or complete assignment route has been validated.
