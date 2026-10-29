"""Generate entirely fictional input data for checking the prototype."""
import csv, math
from datetime import datetime, timedelta
from pathlib import Path
root = Path(__file__).resolve().parent / 'demo_data'
root.mkdir(exist_ok=True)
with (root/'traffic.csv').open('w', newline='') as f:
    w=csv.writer(f); w.writerow(['site_id','timestamp','flow'])
    for site in ['1000','2000','3000','4000','5000']:
        for i in range(240):
            date=datetime(2026,1,1)+timedelta(minutes=15*i)
            flow=round(70+10*int(site[0])+24*math.sin(2*math.pi*i/96)+8*math.sin(2*math.pi*i/20),2)
            w.writerow([site,date.isoformat(),flow])
with (root/'edges.csv').open('w', newline='') as f:
    w=csv.writer(f);w.writerow(['origin','destination','distance_km'])
    w.writerows([['1000','2000',1.2],['2000','5000',1.6],['1000','3000',1],['3000','5000',2.1],['1000','4000',1.7],['4000','5000',1.8],['2000','3000',0.6],['3000','4000',0.8]])
print('Created synthetic traffic.csv and directed edges.csv in demo_data')
