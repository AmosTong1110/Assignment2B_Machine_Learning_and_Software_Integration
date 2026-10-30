"""Convert SCATS detector-day CSVs and validate a single-detector time series.

A detector is NOT automatically a whole-intersection traffic total. The
course must specify which detectors can be combined. Negative values are
unverified missing/invalid indicators and are never treated as zero.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

INTERVAL = timedelta(minutes=15)


def load_site_series(csv_path, site_id):
    """Load a prepared long CSV. Reject missing, duplicate or irregular data."""
    site_id = str(site_id).strip()
    if not site_id:
        raise ValueError('Site ID must not be empty.')
    records = []
    with Path(csv_path).open('r', newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'site_id','timestamp','flow'}.issubset(reader.fieldnames or []):
            raise ValueError('CSV requires columns site_id, timestamp, flow. Run the preparation command first.')
        for line, row in enumerate(reader, 2):
            if (row.get('site_id') or '').strip() != site_id:
                continue
            try:
                stamp = datetime.fromisoformat((row.get('timestamp') or '').strip().replace('Z','+00:00'))
                value = float(row.get('flow') or '')
                if not math.isfinite(value) or value < 0:
                    raise ValueError('flow must be finite and non-negative')
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid timestamp or flow on CSV line {line}: {exc}') from exc
            records.append((stamp,value))
    if not records:
        raise ValueError(f'No records found for site {site_id!r}.')
    try:
        records.sort(key=lambda r:r[0])
        for a,b in zip(records,records[1:]):
            if b[0]-a[0] != INTERVAL:
                raise ValueError('Site timestamps must be unique and exactly 15 minutes apart; clean the dataset first.')
    except TypeError as exc:
        raise ValueError('Site timestamps must use a consistent timezone convention.') from exc
    return [r[0].isoformat() for r in records], [r[1] for r in records]


def prepare_detector(input_dir, site, detector, output, longest_contiguous=False):
    """Convert one site AND detector; do not aggregate detector counts.

    Negative/blank values are excluded and counted. Gaps cause a clear error
    unless longest_contiguous is selected, which discards other valid periods
    and reports how many observations were retained. No imputation occurs.
    """
    site,detector=str(site).strip(),str(detector).strip()
    if not site or not detector:
        raise ValueError('Both site and detector are required.')
    files=sorted(Path(input_dir).glob('VSDATA_*.csv'))
    if not files: raise ValueError('No VSDATA_*.csv files found in input folder.')
    output=Path(output)
    if output.exists(): raise FileExistsError(f'Output exists: {output}; choose a new file name.')
    records=[]; bad=0; matched=0; observed=set()
    required={'NB_SCATS_SITE','QT_INTERVAL_COUNT','NB_DETECTOR'} | {f'V{i:02d}' for i in range(96)}
    for file in files:
        with file.open(newline='',encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'Missing required fields in {file.name}')
            for row in reader:
                if (row.get('NB_SCATS_SITE') or '').strip()!=site or (row.get('NB_DETECTOR') or '').strip()!=detector:
                    continue
                matched+=1
                try: day=datetime.fromisoformat(row['QT_INTERVAL_COUNT'].strip())
                except ValueError as exc: raise ValueError(f'Invalid date in {file.name}') from exc
                if day.time().isoformat()!='00:00:00': raise ValueError('Expected a date without a time component.')
                if day.date() in observed: raise ValueError(f'Duplicate site/detector/day: {day.date()}')
                observed.add(day.date())
                for i in range(96):
                    raw=(row.get(f'V{i:02d}') or '').strip()
                    try: value=float(raw)
                    except ValueError: bad+=1;continue
                    if not math.isfinite(value) or value<0: bad+=1;continue
                    records.append((day+INTERVAL*i,value))
    if not matched or not records: raise ValueError(f'No valid records for site {site}, detector {detector}.')
    records.sort()
    chunks=[]; current=[]
    for record in records:
        if current and record[0]-current[-1][0]!=INTERVAL:
            chunks.append(current);current=[]
        current.append(record)
    if current: chunks.append(current)
    if len(chunks)>1 and not longest_contiguous:
        raise ValueError(f'{len(chunks)} contiguous periods and {bad} invalid/missing values found. Inspect data or use --longest-contiguous to explicitly discard other periods.')
    selected=max(chunks,key=len) if longest_contiguous else chunks[0]
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['site_id','timestamp','flow'])
        for stamp,value in selected:w.writerow([site,stamp.isoformat(),value])
    return {'site':site,'detector':detector,'source_files':len(files),'matching_day_rows':matched,'invalid_or_missing_values':bad,'valid_observations':len(records),'contiguous_periods':len(chunks),'retained_observations':len(selected),'valid_observations_discarded':len(records)-len(selected),'start':selected[0][0].isoformat(),'end':selected[-1][0].isoformat(),'output':str(output),'unit':'detector counts per 15-minute interval'}


def main():
    p=argparse.ArgumentParser(description='Prepare SCATS data for one chosen site and detector; no detector aggregation.')
    p.add_argument('--input-dir',required=True);p.add_argument('--site',required=True);p.add_argument('--detector',required=True);p.add_argument('--output',required=True)
    p.add_argument('--longest-contiguous',action='store_true',help='Explicitly keep only the longest continuous valid segment if gaps exist')
    a=p.parse_args()
    print(json.dumps(prepare_detector(a.input_dir,a.site,a.detector,a.output,a.longest_contiguous),indent=2))
if __name__=='__main__':main()
