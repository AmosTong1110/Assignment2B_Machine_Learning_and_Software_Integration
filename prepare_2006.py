"""Prepare one explicitly selected SCATS 2006 measurement series.

Rows are site/location/day, not a site total. HF internal ID distinguishes
multiple series with the same site and location. Missing days are never filled.
"""
import argparse, csv, json, math
from datetime import datetime, date, timedelta
from pathlib import Path


def iter_rows(path):
    path=Path(path)
    if path.suffix.lower()=='.xls':
        try: import xlrd
        except ImportError as exc: raise RuntimeError('Reading .xls requires xlrd. Install requirements.txt first.') from exc
        book=xlrd.open_workbook(str(path),on_demand=True); sheet=book.sheet_by_name('Data')
        header=sheet.row_values(1)
        try:
            for i in range(2,sheet.nrows):
                vals=sheet.row_values(i)
                d=sheet.cell(i,header.index('Date'))
                if d.ctype==xlrd.XL_CELL_DATE: vals[header.index('Date')]=xlrd.xldate_as_datetime(d.value,book.datemode)
                yield dict(zip(header,vals))
        finally: book.release_resources()
    elif path.suffix.lower()=='.xlsx':
        try: import openpyxl
        except ImportError as exc: raise RuntimeError('Reading .xlsx requires openpyxl. Install requirements.txt first.') from exc
        book=openpyxl.load_workbook(path,read_only=True,data_only=True)
        try:
            rows=book['Data'].iter_rows(values_only=True);next(rows);header=next(rows)
            for vals in rows:yield dict(zip(header,vals))
        finally:book.close()
    else: raise ValueError('Expected an .xls or .xlsx workbook.')


def prepare_2006(path,site,location,hf_id,output,longest_contiguous=False):
    site,location,hf_id=str(site).strip(),str(location).strip(),str(hf_id).strip()
    if not all((site,location,hf_id)):raise ValueError('site, location and HF ID are all required.')
    out=Path(output)
    if out.exists():raise FileExistsError('Output already exists; choose a new filename.')
    records=[]; seen=set(); days=0
    for row in iter_rows(path):
        raw_site=row.get('SCATS Number'); raw_hf=row.get('HF VicRoads Internal')
        # Keep leading zero site identifiers as provided by workbook.
        row_site=str(raw_site).strip(); row_hf=str(int(raw_hf)) if isinstance(raw_hf,(int,float)) and raw_hf is not None else str(raw_hf).strip()
        if row_site!=site or str(row.get('Location') or '').strip()!=location or row_hf!=hf_id:continue
        day=row.get('Date')
        if not isinstance(day,(datetime,date)):raise ValueError('Invalid date for selected record.')
        day=datetime.combine(day.date() if isinstance(day,datetime) else day,datetime.min.time())
        if day in seen:raise ValueError('Duplicate selected site/location/HF ID/date; investigate source.')
        seen.add(day);days+=1
        for i in range(96):
            try:value=float(row[f'V{i:02d}'])
            except (TypeError,ValueError,KeyError) as exc:raise ValueError(f'Invalid flow at {day.date()} V{i:02d}.') from exc
            if not math.isfinite(value) or value<0:raise ValueError('Missing or negative count; investigate source.')
            records.append((day+timedelta(minutes=15*i),value))
    if not records:raise ValueError('No records matched site, exact location and HF ID.')
    records.sort();chunks=[];current=[]
    for record in records:
        if current and record[0]-current[-1][0]!=timedelta(minutes=15):chunks.append(current);current=[]
        current.append(record)
    if current:chunks.append(current)
    if len(chunks)>1 and not longest_contiguous:raise ValueError(f'{len(chunks)} separate continuous segments; inspect missing days or explicitly use --longest-contiguous.')
    chosen=max(chunks,key=len) if longest_contiguous else chunks[0]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['site_id','timestamp','flow'])
        for stamp,value in chosen:w.writerow([site,stamp.isoformat(),value])
    return {'site':site,'location':location,'hf_internal_id':hf_id,'days':days,'valid_observations':len(records),'segments':len(chunks),'retained_observations':len(chosen),'discarded_observations':len(records)-len(chosen),'start':chosen[0][0].isoformat(),'end':chosen[-1][0].isoformat(),'output':str(out)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--site',required=True);p.add_argument('--location',required=True);p.add_argument('--hf-id',required=True);p.add_argument('--output',required=True);p.add_argument('--longest-contiguous',action='store_true');a=p.parse_args()
    print(json.dumps(prepare_2006(a.input,a.site,a.location,a.hf_id,a.output,a.longest_contiguous),indent=2))
if __name__=='__main__':main()
