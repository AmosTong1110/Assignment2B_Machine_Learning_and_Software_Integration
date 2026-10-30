"""Traffic prediction desktop dashboard.

Routes remain unavailable until a verified graph and site-level flows exist.
"""
import csv
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from forecast import load_site_series, train_and_evaluate, predict_next

BASE = Path(__file__).resolve().parent
CONFIG = BASE / 'config.json'
BG = '#f3f6f6'
CARD = '#ffffff'
INK = '#173137'
MUTED = '#61767a'
ACCENT = '#116d69'
BORDER = '#dce7e5'


def path_for(text):
    path = Path(text).expanduser()
    return path if path.is_absolute() else BASE / path


def site_ids(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or 'site_id' not in reader.fieldnames:
            raise ValueError('Traffic CSV must include site_id')
        return sorted({row['site_id'].strip() for row in reader if row.get('site_id', '').strip()})


class App:
    def __init__(self, root):
        self.root = root
        self.events = queue.Queue()
        self.busy = False
        self.settings_window = None
        self.chart_values = None
        root.title('Traffic route guidance')
        root.geometry('1120x790')
        root.minsize(980, 720)
        root.configure(bg=BG)
        cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
        defaults = {'traffic_csv': 'data/prepared/site_2000_hf6246_2006.csv',
                    'edges_csv': 'data/boroondara_edges.csv', 'model_dir': 'saved_models',
                    'model': 'linear', 'origin': '2000', 'destination': '3002',
                    'window': 8, 'epochs': 20, 'train_fraction': 0.8, 'k': 5}
        self.vars = {key: tk.StringVar(value=str(cfg.get(key, value))) for key, value in defaults.items()}
        self.prediction = tk.StringVar(value='—')
        self.prediction_detail = tk.StringVar(value='Train a model, then select Predict traffic.')
        self.status = tk.StringVar(value='Ready')
        self._styles()
        self._build()
        self.root.after(100, self._poll_events)

    def _styles(self):
        style = ttk.Style(self.root)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
        style.configure('Primary.TButton', background=ACCENT, foreground='white', padding=(16, 10), borderwidth=0, font=('Arial', 11, 'bold'))
        style.map('Primary.TButton', background=[('disabled', '#a6b8b5'), ('active', '#0d5854')])
        style.configure('Secondary.TButton', background='#edf4f2', foreground=INK, padding=(14, 10), borderwidth=0, font=('Arial', 11, 'bold'))
        style.map('Secondary.TButton', background=[('disabled', '#e4ecea'), ('active', '#dcebe7')], foreground=[('disabled', '#90a29f')])
        style.configure('Dashboard.TEntry', padding=8, fieldbackground='white', foreground=INK, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
        style.configure('Dashboard.TCombobox', padding=7, fieldbackground='white', foreground=INK, bordercolor=BORDER)

    def _label(self, parent, text, size=11, color=INK, bold=False, bg=CARD, **kwargs):
        return tk.Label(parent, text=text, bg=bg, fg=color, font=('Arial', size, 'bold' if bold else 'normal'), anchor='w', justify='left', **kwargs)

    def _card(self, parent):
        return tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1, bd=0)

    def _field(self, parent, title, key, row, combo=False):
        self._label(parent, title, size=10, color=MUTED, bold=True).grid(row=row, column=0, sticky='w', pady=(12, 4))
        if combo:
            widget = ttk.Combobox(parent, textvariable=self.vars[key], values=('linear', 'lstm', 'gru'), state='readonly', style='Dashboard.TCombobox')
        else:
            widget = ttk.Entry(parent, textvariable=self.vars[key], style='Dashboard.TEntry')
        widget.grid(row=row + 1, column=0, sticky='ew', ipady=2)
        return widget

    def _build(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', padx=28, pady=(20, 14))
        self._label(top, 'TRAFFIC LAB  /  BOROONDARA', 10, ACCENT, True, BG).pack(anchor='w')
        self._label(top, 'Traffic prediction', 25, INK, True, BG).pack(anchor='w', pady=(5, 1))
        self._label(top, 'Explore a traffic measurement and prepare route guidance', 11, MUTED, bg=BG).pack(anchor='w')
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill='both', expand=True, padx=28, pady=(0, 16))
        content.grid_columnconfigure(0, minsize=290, weight=0)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        left = self._card(content); left.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        right = tk.Frame(content, bg=BG); right.grid(row=0, column=1, sticky='nsew')
        left_body = tk.Frame(left, bg=CARD, padx=20, pady=18); left_body.pack(fill='both', expand=True)
        left_body.grid_columnconfigure(0, weight=1)
        self._label(left_body, 'Prediction setup', 16, INK, True).grid(row=0, column=0, sticky='w')
        self._label(left_body, 'Choose a site and a model', 11, MUTED).grid(row=1, column=0, sticky='w', pady=(4, 0))
        self._field(left_body, 'ORIGIN SCATS SITE', 'origin', 2)
        self._field(left_body, 'DESTINATION SCATS SITE', 'destination', 4)
        self._field(left_body, 'PREDICTION MODEL', 'model', 6, combo=True)
        self._label(left_body, 'Linear is a proposed third model;\ntutor approval is required.', 10, MUTED).grid(row=8, column=0, sticky='w', pady=(8, 12))
        self.train_button = ttk.Button(left_body, text='1   Train selected site', style='Primary.TButton', command=self.train_one)
        self.train_button.grid(row=9, column=0, sticky='ew', pady=(0, 8))
        self.predict_button = ttk.Button(left_body, text='2   Predict traffic', style='Secondary.TButton', command=self.predict_one)
        self.predict_button.grid(row=10, column=0, sticky='ew')
        self._label(left_body, 'Train uses the prepared data. Predict uses\nthe latest available observations.', 10, MUTED).grid(row=11, column=0, sticky='w', pady=(12, 0))
        ttk.Button(left_body, text='Advanced settings', style='Secondary.TButton', command=self.open_settings).grid(row=12, column=0, sticky='ew', pady=(16, 0))
        tk.Label(left_body, textvariable=self.status, bg=CARD, fg=ACCENT, font=('Arial', 10, 'bold'), anchor='w').grid(row=13, column=0, sticky='w', pady=(16, 0))
        summary = self._card(right); summary.pack(fill='x', pady=(0, 12))
        summary_body = tk.Frame(summary, bg=CARD, padx=20, pady=15); summary_body.pack(fill='x')
        self._label(summary_body, 'NEXT 15-MINUTE PREDICTION', 10, MUTED, True).pack(anchor='w')
        tk.Label(summary_body, textvariable=self.prediction, bg=CARD, fg=INK, font=('Arial', 25, 'bold'), anchor='w').pack(anchor='w', pady=(3, 0))
        tk.Label(summary_body, textvariable=self.prediction_detail, bg=CARD, fg=MUTED, font=('Arial', 10), anchor='w').pack(anchor='w', pady=(2, 0))
        chart = self._card(right); chart.pack(fill='both', expand=True, pady=(0, 12))
        chart_head = tk.Frame(chart, bg=CARD, padx=20, pady=12); chart_head.pack(fill='x')
        self._label(chart_head, 'Traffic history', 15, INK, True).pack(anchor='w')
        self._label(chart_head, 'Recent observations and the next prediction', 10, MUTED).pack(anchor='w', pady=(2, 0))
        self.canvas = tk.Canvas(chart, height=190, bg=CARD, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=12, pady=(0, 8))
        self.canvas.bind('<Configure>', lambda event: self.draw_chart())
        bottom = self._card(right); bottom.pack(fill='x')
        area = tk.Frame(bottom, bg=CARD, padx=20, pady=14); area.pack(fill='x')
        self._label(area, 'Route guidance', 15, INK, True).pack(anchor='w')
        self._label(area, 'Unavailable until road connections and site-level hourly flows are verified.', 10, MUTED, wraplength=650).pack(anchor='w', pady=(4, 9))
        ttk.Button(area, text='Find routes', style='Secondary.TButton', state='disabled').pack(anchor='w')
        activity = tk.Frame(self.root, bg=BG); activity.pack(fill='x', padx=28, pady=(0, 16))
        self._label(activity, 'ACTIVITY', 10, MUTED, True, BG).pack(anchor='w')
        self.output = tk.Text(activity, height=3, wrap='word', bg=BG, fg=INK, relief='flat', font=('Arial', 10), highlightthickness=0)
        self.output.pack(fill='x', pady=(4, 0))
        self.log('Ready. The prepared example contains one measurement location, not a whole-intersection total.')

    def open_settings(self):
        if self.settings_window and self.settings_window.winfo_exists():
            self.settings_window.lift(); return
        window = tk.Toplevel(self.root); self.settings_window = window
        window.title('Advanced settings'); window.configure(bg=BG); window.geometry('600x590'); window.minsize(520, 500)
        body = tk.Frame(window, bg=CARD, padx=22, pady=18); body.pack(fill='both', expand=True, padx=14, pady=14)
        body.grid_columnconfigure(1, weight=1)
        self._label(body, 'Advanced settings', 18, INK, True).grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 4))
        self._label(body, 'File paths and model training settings', 10, MUTED).grid(row=1, column=0, columnspan=2, sticky='w', pady=(0, 12))
        fields = [('Prepared traffic CSV', 'traffic_csv'), ('Road edges CSV', 'edges_csv'), ('Saved model folder', 'model_dir'), ('Input window', 'window'), ('Training epochs', 'epochs'), ('Training fraction', 'train_fraction'), ('Maximum routes', 'k')]
        for i, (label, key) in enumerate(fields, 2):
            self._label(body, label, 11).grid(row=i, column=0, sticky='w', padx=(0, 12), pady=5)
            ttk.Entry(body, textvariable=self.vars[key], style='Dashboard.TEntry').grid(row=i, column=1, sticky='ew', pady=5)
        actions = tk.Frame(body, bg=CARD); actions.grid(row=9, column=0, columnspan=2, sticky='ew', pady=(14, 0))
        self.train_all_button = ttk.Button(actions, text='Train all sites', style='Secondary.TButton', command=self.train_all)
        self.train_all_button.pack(side='left', padx=(0, 8))
        if self.busy: self.train_all_button.state(['disabled'])
        ttk.Button(actions, text='Save defaults', style='Primary.TButton', command=self.save).pack(side='left')
        self._label(body, 'Changes are saved to config.json only when you select Save defaults.', 10, MUTED, wraplength=500).grid(row=10, column=0, columnspan=2, sticky='w', pady=(12, 0))

    def log(self, message):
        self.output.insert('end', str(message) + '\n'); self.output.see('end')

    def params(self):
        values = {key: value.get().strip() for key, value in self.vars.items()}
        for key in ('window', 'epochs', 'k'): values[key] = int(values[key])
        values['train_fraction'] = float(values['train_fraction'])
        return values

    def _set_busy(self, value):
        self.busy = value
        for button in (self.train_button, self.predict_button):
            button.state(['disabled'] if value else ['!disabled'])
        if hasattr(self, 'train_all_button') and self.train_all_button.winfo_exists():
            self.train_all_button.state(['disabled'] if value else ['!disabled'])
        self.status.set('Working…' if value else 'Ready')

    def _start(self, task):
        if self.busy: return
        try: params = self.params()
        except (TypeError, ValueError) as exc:
            messagebox.showerror('Check your settings', str(exc), parent=self.root); return
        self._set_busy(True)
        def worker():
            try: self.events.put(('success', task(params)))
            except Exception as exc: self.events.put(('error', str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self):
        try:
            while True:
                kind, result = self.events.get_nowait()
                self._set_busy(False)
                if kind == 'error':
                    self.status.set('Check the activity message'); self.log('Error: ' + result)
                    messagebox.showerror('Unable to continue', result, parent=self.root)
                elif result['kind'] == 'prediction':
                    self.prediction.set(f"{result['value']:.1f} counts")
                    self.prediction_detail.set(f"Site {result['site']} · {result['model'].upper()} · one measurement location")
                    self.chart_values = (result['history'][-24:], result['value'])
                    self.draw_chart()
                    self.log(f"Prediction: {result['value']:.2f} measurement counts for the next 15 minutes.")
                else:
                    self.log(result['message'])
        except queue.Empty:
            pass
        if self.root.winfo_exists(): self.root.after(100, self._poll_events)

    def _train(self, params, sites):
        lines = []
        for site in sites:
            result = train_and_evaluate(path_for(params['traffic_csv']), site, params['model'], path_for(params['model_dir']), window=params['window'], epochs=params['epochs'], train_fraction=params['train_fraction'])
            lines.append(f"{site} {params['model'].upper()} | MAE {result['mae']:.2f} | RMSE {result['rmse']:.2f} | test {result['test_start']} to {result['test_end']}")
        return {'kind': 'training', 'message': '\n'.join(lines)}

    def train_one(self): self._start(lambda p: self._train(p, [p['origin']]))
    def train_all(self): self._start(lambda p: self._train(p, site_ids(path_for(p['traffic_csv']))))

    def predict_one(self):
        def task(params):
            _, values = load_site_series(path_for(params['traffic_csv']), params['origin'])
            value = predict_next(path_for(params['model_dir']), params['model'], params['origin'], values)
            return {'kind': 'prediction', 'site': params['origin'], 'model': params['model'], 'value': value, 'history': values}
        self._start(task)

    def draw_chart(self):
        self.canvas.delete('all')
        width = max(self.canvas.winfo_width(), 250); height = max(self.canvas.winfo_height(), 160)
        if self.chart_values is None:
            self.canvas.create_text(width / 2, height / 2, text='Your prediction chart will appear here', fill=MUTED, font=('Arial', 11)); return
        history, predicted = self.chart_values
        series = history + [predicted]; maximum = max(series) or 1
        left, right, top, bottom = 52, width - 18, 18, height - 37
        for fraction in (0, 0.5, 1):
            y = bottom - fraction * (bottom - top)
            self.canvas.create_line(left, y, right, y, fill='#e6eeec')
            self.canvas.create_text(left - 8, y, text=f'{maximum*fraction:.0f}', anchor='e', fill=MUTED, font=('Arial', 9))
        points = [(left + i * (right-left) / max(1, len(series)-1), bottom - value / maximum * (bottom-top)) for i, value in enumerate(series)]
        for a, b in zip(points[:-2], points[1:-1]): self.canvas.create_line(*a, *b, fill=ACCENT, width=2)
        self.canvas.create_line(*points[-2], *points[-1], fill='#cc7859', width=2, dash=(4, 3))
        x, y = points[-1]; self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='#cc7859', outline='')
        self.canvas.create_text(left, height-12, text='Earlier observations', anchor='w', fill=MUTED, font=('Arial', 9))
        self.canvas.create_text(right, height-12, text='Next interval', anchor='e', fill=MUTED, font=('Arial', 9))

    def save(self):
        try:
            values = self.params()
            CONFIG.write_text(json.dumps(values, indent=2) + '\n', encoding='utf-8')
            self.log('Default settings saved to config.json.')
            messagebox.showinfo('Settings saved', 'Your default settings have been saved.', parent=self.settings_window)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Unable to save settings', str(exc), parent=self.settings_window)


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
