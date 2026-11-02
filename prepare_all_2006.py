"""Export each 2006 site/location/HF series as separate continuous CSVs.

No aggregation, imputation, or dropping of observations is performed.
The output directory must not already exist. See manifest.csv for provenance.
"""
import argparse
import csv
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

from prepare_2006 import iter_rows


def prepare_all(source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    series = defaultdict(list)
    seen_days = set()
    for row in iter_rows(source):
        site = str(row['SCATS Number']).strip()
        location = str(row['Location']).strip()
        raw_hf = row['HF VicRoads Internal']
        hf = str(int(raw_hf)) if isinstance(raw_hf, (int, float)) else str(raw_hf).strip()
        day = row['Date']
        if not site or not location or not hf or not isinstance(day, (date, datetime)):
            raise ValueError('Invalid site, location, HF ID or date in workbook')
        day = day.date() if isinstance(day, datetime) else day
        key = (site, location, hf)
        if (key, day) in seen_days:
            raise ValueError(f'Duplicate day for {key}: {day}')
        seen_days.add((key, day))
        for i in range(96):
            value = float(row[f'V{i:02d}'])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid count for {key} on {day}, interval {i}')
            series[key].append((datetime.combine(day, time.min) + timedelta(minutes=15*i), value))
    if not series:
        raise ValueError('Workbook contains no measurement series')
    all_segments = []
    for key, records in sorted(series.items()):
        chunks = []
        for stamp, value in sorted(records):
            if not chunks or stamp - chunks[-1][-1][0] != timedelta(minutes=15):
                chunks.append([])
            chunks[-1].append((stamp, value))
        for segment in chunks:
            all_segments.append((key, segment, len(chunks)))
    output.mkdir(parents=True, exist_ok=False)
    manifest_rows = []
    for (site, location, hf), records, segment_count in all_segments:
        start, end = records[0][0], records[-1][0]
        name = f'site_{site}_hf{hf}_{start:%Y%m%d_%H%M}.csv'
        with (output / name).open('x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['site_id', 'timestamp', 'flow'])
            writer.writerows((site, stamp.isoformat(), value) for stamp, value in records)
        manifest_rows.append([site, location, hf, name, start.isoformat(), end.isoformat(), len(records), segment_count])
    with (output / 'manifest.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['site_id', 'location', 'hf_id', 'csv_file', 'start', 'end', 'observations', 'segments_in_series'])
        writer.writerows(manifest_rows)
    return len(series), len(manifest_rows), sum(len(rows) for rows in series.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    series, files, observations = prepare_all(args.input, args.output_dir)
    print(f'Exported {series} measurement series into {files} CSV files; {observations} observations. See manifest.csv.')


if __name__ == '__main__':
    main()
