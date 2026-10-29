"""Command-line demo that does not require a graphical display."""
import argparse, csv
from pathlib import Path
from forecast import load_site_series, train_and_evaluate, predict_next
from routing import load_edges, find_routes
def site_ids(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        return sorted({row['site_id'].strip() for row in csv.DictReader(stream) if row.get('site_id','').strip()})

def demo_edge_minutes(edge, hourly_flow):
    # Fictional free-flow-only demonstration; ignores traffic.
    return edge.distance_km + 0.5

BASE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='Fictional-data demonstration, not the course conversion')
    parser.add_argument('--model',choices=['linear','lstm','gru'],default='linear')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--window',type=int,default=8)
    args=parser.parse_args()
    data=BASE/'demo_data'/'traffic.csv'; edges=load_edges(BASE/'demo_data'/'edges.csv')
    directory=BASE/'models'; hourly={}
    print('DEMO ONLY: synthetic traffic and travel times; not assignment results.')
    print('The demo conversion ignores predicted flow and is not the course formula.')
    for site in site_ids(data):
        r=train_and_evaluate(data,site,args.model,directory,window=args.window,epochs=args.epochs)
        _,history=load_site_series(data,site)
        predicted=predict_next(directory,args.model,site,history)
        hourly[site]=4*predicted  # Demonstration assumption, not the verified course method.
        print(f"{site} | MAE {r['mae']:.2f} | RMSE {r['rmse']:.2f} | next interval {predicted:.2f} vehicles")
    routes=find_routes(edges,'1000','5000',hourly,demo_edge_minutes)
    for n,route in enumerate(routes,1):
        print(f"{n}. {' -> '.join(route.sites)} | {route.minutes:.2f} minutes (demo only)")
if __name__=='__main__': main()
