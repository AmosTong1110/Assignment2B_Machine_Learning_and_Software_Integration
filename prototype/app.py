"""Educational GUI prototype. Real course travel-time conversion is NOT supplied."""
import csv, json, threading, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from forecast import load_site_series, train_and_evaluate, predict_next
from routing import load_edges, find_routes

BASE=Path(__file__).resolve().parent
CONFIG=BASE/'config.json'

def path_for(text):
    p=Path(text).expanduser()
    return p if p.is_absolute() else BASE/p

def site_ids(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        if not r.fieldnames or 'site_id' not in r.fieldnames:
            raise ValueError('Traffic CSV must include site_id')
        return sorted({row['site_id'].strip() for row in r if row.get('site_id','').strip()})

def demo_edge_minutes(edge, hourly_flow):
    """NOT course formula: free-flow 60 km/h plus 0.5 minute per edge.
    It ignores hourly_flow, so routes are not traffic-sensitive.
    """
    return edge.distance_km + 0.5

class App:
    def __init__(self, root):
        self.root=root
        root.title('Traffic prediction and route guidance - educational prototype')
        root.geometry('1000x760')
        cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
        self.vars={key:tk.StringVar(value=str(cfg.get(key,default))) for key,default in {
            'traffic_csv':'demo_data/traffic.csv','edges_csv':'demo_data/edges.csv',
            'model_dir':'models','model':'linear','origin':'1000','destination':'5000',
            'window':8,'epochs':20,'train_fraction':0.8,'k':5}.items()}
        self.demo=tk.BooleanVar(value=bool(cfg.get('use_demo_conversion',False)))
        title=ttk.Label(root,text='Educational prototype - fictional data by default',font=('Arial',15,'bold'))
        title.pack(anchor='w',padx=12,pady=(10,2))
        ttk.Label(root,text='The demo conversion ignores traffic and is not the course formula. Third model needs tutor approval.').pack(anchor='w',padx=12)
        form=ttk.Frame(root,padding=10);form.pack(fill='x')
        fields=[('Traffic CSV','traffic_csv'),('Road edges CSV','edges_csv'),('Saved models','model_dir'),('Model','model'),('Origin SCATS','origin'),('Destination SCATS','destination'),('Window (intervals)','window'),('Training epochs','epochs'),('Training fraction','train_fraction'),('Maximum routes','k')]
        for i,(label,key) in enumerate(fields):
            row,col=divmod(i,2);c=col*2
            ttk.Label(form,text=label).grid(row=row,column=c,sticky='w',padx=4,pady=3)
            widget=ttk.Combobox(form,textvariable=self.vars[key],values=('linear','lstm','gru'),state='readonly',width=34) if key=='model' else ttk.Entry(form,textvariable=self.vars[key],width=37)
            widget.grid(row=row,column=c+1,sticky='ew',padx=4,pady=3)
        form.columnconfigure(1,weight=1);form.columnconfigure(3,weight=1)
        ttk.Checkbutton(root,text='Use fictional free-flow demo conversion (not suitable for submission)',variable=self.demo).pack(anchor='w',padx=15)
        buttons=ttk.Frame(root,padding=(10,4));buttons.pack(fill='x')
        for text,command in [('Train selected site',self.train_one),('Train all sites',self.train_all),('Predict selected site',self.predict_one),('Show example routes',self.route),('Save defaults',self.save)]:
            ttk.Button(buttons,text=text,command=command).pack(side='left',padx=3)
        self.status=ttk.Label(root,text='Ready');self.status.pack(anchor='w',padx=12)
        self.canvas=tk.Canvas(root,height=180,bg='white',highlightthickness=1,highlightbackground='#cccccc');self.canvas.pack(fill='x',padx=12,pady=5)
        self.canvas.create_text(10,10,anchor='nw',text='Prediction chart will appear here')
        self.output=tk.Text(root,height=12,wrap='word');self.output.pack(fill='both',expand=True,padx=12,pady=(0,10))
        self.log('Fictional data and demonstration graph. No real SCATS or verified course conversion is included.')
    def log(self,msg):
        self.output.insert('end',str(msg)+'\n');self.output.see('end')
    def params(self):
        v={k:val.get().strip() for k,val in self.vars.items()}
        v['window']=int(v['window']);v['epochs']=int(v['epochs']);v['k']=int(v['k']);v['train_fraction']=float(v['train_fraction'])
        return v
    def run_background(self,fn):
        try: p=self.params()
        except Exception as e: messagebox.showerror('Invalid settings',str(e));return
        self.status.config(text='Working...')
        def work():
            try:
                result=fn(p)
                self.root.after(0,lambda:self.done(result))
            except Exception as e:
                msg=str(e)
                self.root.after(0,lambda:self.failed(msg))
        threading.Thread(target=work,daemon=True).start()
    def done(self,text): self.status.config(text='Ready');self.log(text)
    def failed(self,text): self.status.config(text='Error');self.log('ERROR: '+text);messagebox.showerror('Prototype error',text)
    def train(self,p,ids):
        output=[]
        for site in ids:
            r=train_and_evaluate(path_for(p['traffic_csv']),site,p['model'],path_for(p['model_dir']),window=p['window'],epochs=p['epochs'],train_fraction=p['train_fraction'])
            output.append(f"{site} {p['model']}: MAE={r['mae']:.2f}, RMSE={r['rmse']:.2f} {r['unit']}; held-out {r['test_start']} to {r['test_end']}")
        return '\n'.join(output)
    def train_one(self): self.run_background(lambda p:self.train(p,[p['origin']]))
    def train_all(self): self.run_background(lambda p:self.train(p,site_ids(path_for(p['traffic_csv']))))
    def predict_one(self):
        try:
            p=self.params();_,values=load_site_series(path_for(p['traffic_csv']),p['origin'])
            pred=predict_next(path_for(p['model_dir']),p['model'],p['origin'],values)
            self.log(f"Site {p['origin']}: predicted next 15-minute volume = {pred:.2f} vehicles. This is not a measured observation.")
            self.draw(values[-24:],pred)
        except Exception as e: self.failed(str(e))
    def draw(self,values,pred):
        self.canvas.delete('all');self.root.update_idletasks()
        width=max(self.canvas.winfo_width(),200);height=180; series=values+[pred]; vmax=max(series) or 1
        pts=[(25+i*(width-50)/max(1,len(series)-1),height-25-v/vmax*(height-55)) for i,v in enumerate(series)]
        self.canvas.create_text(10,8,anchor='nw',text='Last observed values (blue) and next predicted value (red)')
        for a,b in zip(pts[:-2],pts[1:-1]):self.canvas.create_line(*a,*b,fill='#2763a4',width=2)
        self.canvas.create_line(*pts[-2],*pts[-1],fill='#c73939',dash=(4,3),width=2)
        self.canvas.create_oval(pts[-1][0]-4,pts[-1][1]-4,pts[-1][0]+4,pts[-1][1]+4,fill='#c73939')
    def route(self):
        if not self.demo.get():
            messagebox.showwarning('Course conversion missing','The course travel-time conversion is not included. Enable the clearly labelled fictional demo only for demonstration, or implement the verified course formula.');return
        def run(p):
            edges=load_edges(path_for(p['edges_csv'])); ids={e.destination for e in edges}
            # Demonstration assumption only: scale the next 15-minute forecast by 4.
            # This is NOT a verified course-prescribed aggregation, and the demo
            # conversion ignores it; real work requires the official method.
            hourly={}
            for site in ids:
                _,values=load_site_series(path_for(p['traffic_csv']),site)
                hourly[site]=4*predict_next(path_for(p['model_dir']),p['model'],site,values)
            routes=find_routes(edges,p['origin'],p['destination'],hourly,demo_edge_minutes,k=p['k'])
            lines=['DEMONSTRATION ONLY: free-flow conversion ignores traffic. Not assignment-compliant travel times.']
            lines.extend(f"{i}. {' -> '.join(route.sites)}: {route.minutes:.2f} min" for i,route in enumerate(routes,1))
            if not routes: lines.append('No route found.')
            return '\n'.join(lines)
        self.run_background(run)
    def save(self):
        try:
            p=self.params();p['use_demo_conversion']=self.demo.get();CONFIG.write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8');self.log('Defaults saved to config.json')
        except Exception as e:self.failed(str(e))

if __name__=='__main__':
    root=tk.Tk();App(root);root.mainloop()
