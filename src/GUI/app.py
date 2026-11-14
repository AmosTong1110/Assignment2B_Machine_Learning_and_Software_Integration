"""
app.py  -  TBRGS desktop GUI (Tkinter + Matplotlib, no extra installs)
======================================================================
Run from the project root:      python main.py

Left  : trip (origin, destination, date, time) and settings (model, search method, ...).
Right : the routes found (fastest first), a map of the Boroondara network with the selected
        route highlighted, and a leg-by-leg breakdown of the selected route.

Defaults come from config.yaml (travel_time: section); the user can change them here.
The ML prediction runs in a background thread so the window never freezes.  The first search
takes ~10 s (TensorFlow and the models load once); later searches take about a second, and
changing only the search method / number of routes is instant (edge costs are cached).
"""
from __future__ import annotations

import importlib.util
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import src.Models.deep_common as dc
from src.Routing import edge_costs as ec
from src.Routing import route_finder as rf
from src.Routing.graph_builder import load_edges, load_nodes
from src.Search import SEARCH_METHODS

# Choices shown to the user  ->  value passed to the code
MODELS = {"GRU": "gru", "LSTM": "lstm", "Linear regression": "linear", "Measured flow (no prediction)": "actual"}

# Fallback: GRU/LSTM need TensorFlow.  If it is not installed, the GUI still works with
# Linear regression (scikit-learn) and measured flow, instead of failing to start.
HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
if not HAS_TENSORFLOW:
    MODELS = {k: v for k, v in MODELS.items() if v not in ("gru", "lstm")}
METHODS = {("A* search (recommended)" if code == "AS" else name): code
           for code, (_, name) in SEARCH_METHODS.items()}
FLOW_SITES = {"Origin site (PDF)": "origin", "Destination site (brief)": "destination"}
FLOW_MODES = {"Per detector (one approach)": "per_detector", "Intersection total": "total"}

DAYS = pd.date_range("2006-10-01", "2006-10-31", freq="D")
TIMES = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
DEFAULT_TRIP = ("2000", "3002", "2006-10-26", "08:00")    # the example in the brief, Thursday 8 am

ROUTE_COLOUR = "#1d4ed8"


def _key_for(mapping: dict, value: str) -> str:
    """Label in a {label: value} dict for a value (used to show config defaults)."""
    return next((k for k, v in mapping.items() if v == value), next(iter(mapping)))


class RouteApp(ttk.Frame):
    def __init__(self, root: tk.Tk):
        super().__init__(root, padding=10)
        self.root = root
        self.cfg = dc.load_config()
        self.nodes = load_nodes(self.cfg)
        self.edges = load_edges(self.cfg)
        self.labels = rf.site_labels()
        self.site_choices = [f"{s}  {self.labels.get(s, '')}".strip() for s in sorted(self.nodes["scats_number"])]
        self.pos = self.nodes.set_index("scats_number")[["longitude", "latitude"]]

        self.result: rf.RouteResult | None = None
        self.cost_cache: dict[tuple, pd.DataFrame] = {}     # (time, model, flow_site, flow_mode) -> edge table
        self.jobs: queue.Queue = queue.Queue()

        self._build_controls()
        self._build_results()
        self.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.draw_map()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _combo(self, parent, row, label, values, default, width=34):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        var = tk.StringVar(value=default)
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width
                     ).grid(row=row, column=1, sticky="ew", pady=3)
        return var

    def _build_controls(self):
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        s = ec.settings(self.cfg)
        o, d, day, time = DEFAULT_TRIP

        trip = ttk.LabelFrame(left, text="1. Your trip", padding=8)
        trip.grid(row=0, column=0, sticky="ew")
        self.origin = self._combo(trip, 0, "From", self.site_choices, self._choice(o))
        ttk.Button(trip, text="⇅  Swap", command=self.swap).grid(row=1, column=1, sticky="e")
        self.destination = self._combo(trip, 2, "To", self.site_choices, self._choice(d))
        self.day = self._combo(trip, 3, "Date", [f"{x:%a %d %b %Y}" for x in DAYS],
                               f"{pd.Timestamp(day):%a %d %b %Y}")
        self.time = self._combo(trip, 4, "Leave at", TIMES, time)

        opts = ttk.LabelFrame(left, text="2. Settings", padding=8)
        opts.grid(row=1, column=0, sticky="ew", pady=10)
        self.model = self._combo(opts, 0, "Traffic model", list(MODELS), _key_for(MODELS, s["default_model"]))
        self.method = self._combo(opts, 1, "Search method", list(METHODS), _key_for(METHODS, "AS"))
        ttk.Label(opts, text="Routes to show").grid(row=2, column=0, sticky="w", pady=3)
        self.k = tk.IntVar(value=rf.MAX_ROUTES)
        ttk.Spinbox(opts, from_=1, to=rf.MAX_ROUTES, textvariable=self.k, width=5, state="readonly"
                    ).grid(row=2, column=1, sticky="w", pady=3)

        adv = ttk.LabelFrame(left, text="Advanced (travel-time model)", padding=8)
        adv.grid(row=2, column=0, sticky="ew")
        self.flow_site = self._combo(adv, 0, "Flow taken at", list(FLOW_SITES), _key_for(FLOW_SITES, s["flow_site"]), 26)
        self.flow_mode = self._combo(adv, 1, "Flow per road", list(FLOW_MODES), _key_for(FLOW_MODES, s["flow_mode"]), 26)
        self.traffic = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv, text="Colour roads by predicted speed", variable=self.traffic,
                        command=self.draw_map).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.go = ttk.Button(left, text="Find routes", command=self.start_search)
        self.go.grid(row=3, column=0, sticky="ew", pady=(14, 4), ipady=6)
        start_text = "Choose a trip and press Find routes." if HAS_TENSORFLOW else \
            "TensorFlow is not installed, so GRU and LSTM are hidden. Linear regression still works."
        self.status = ttk.Label(left, text=start_text, wraplength=330, foreground="#555")
        self.status.grid(row=4, column=0, sticky="ew")
        self.root.bind("<Return>", lambda _e: self.start_search())

    def _build_results(self):
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self.summary = ttk.Label(right, text="Routes", font=("TkDefaultFont", 13, "bold"))
        self.summary.grid(row=0, column=0, sticky="w")

        cols = {"route": ("Route", 55), "time": ("Travel time", 90), "dist": ("Distance", 80),
                "hops": ("Intersections", 95), "via": ("Via", 520)}
        self.table = ttk.Treeview(right, columns=list(cols), show="headings", height=5, selectmode="browse")
        for c, (title, w) in cols.items():
            self.table.heading(c, text=title)
            self.table.column(c, width=w, stretch=(c == "via"), anchor="w" if c == "via" else "center")
        self.table.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.table.bind("<<TreeviewSelect>>", lambda _e: self.show_selected())

        bottom = ttk.PanedWindow(right, orient="horizontal")
        bottom.grid(row=2, column=0, sticky="nsew")

        map_frame = ttk.Frame(bottom)
        self.fig = Figure(figsize=(6.5, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=map_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        bottom.add(map_frame, weight=3)

        legs_frame = ttk.LabelFrame(bottom, text="Selected route, step by step", padding=4)
        legs_cols = {"leg": ("From → To", 105), "road": ("Road", 130), "km": ("km", 50),
                     "flow": ("Flow veh/h", 75), "speed": ("km/h", 50), "sec": ("Time s", 60)}
        self.legs = ttk.Treeview(legs_frame, columns=list(legs_cols), show="headings")
        for c, (title, w) in legs_cols.items():
            self.legs.heading(c, text=title)
            self.legs.column(c, width=w, anchor="w" if c in ("leg", "road") else "e")
        self.legs.pack(fill="both", expand=True)
        ttk.Label(legs_frame, text="Each step = driving time at the predicted speed + 30 s intersection delay.",
                  foreground="#555", wraplength=440).pack(anchor="w", pady=(4, 0))
        bottom.add(legs_frame, weight=2)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _choice(self, site: str) -> str:
        return next(c for c in self.site_choices if c.split()[0] == site)

    @staticmethod
    def _site(choice: str) -> str:
        return choice.split()[0]

    def swap(self):
        o, d = self.origin.get(), self.destination.get()
        self.origin.set(d)
        self.destination.set(o)

    def set_status(self, text: str, error: bool = False):
        self.status.configure(text=text, foreground="#b91c1c" if error else "#555")

    # ------------------------------------------------------------------
    # Search (background thread)
    # ------------------------------------------------------------------
    def start_search(self):
        if str(self.go["state"]) == "disabled":
            return
        origin, dest = self._site(self.origin.get()), self._site(self.destination.get())
        if origin == dest:
            self.set_status("Origin and destination are the same. Please choose two different sites.", True)
            return
        day = pd.Timestamp(DAYS[[f"{x:%a %d %b %Y}" for x in DAYS].index(self.day.get())])
        when = pd.Timestamp(f"{day:%Y-%m-%d} {self.time.get()}")
        job = {
            "origin": origin, "destination": dest, "when": when,
            "model": MODELS[self.model.get()], "method": METHODS[self.method.get()], "k": int(self.k.get()),
            "flow_site": FLOW_SITES[self.flow_site.get()], "flow_mode": FLOW_MODES[self.flow_mode.get()],
        }
        cache_key = (when, job["model"], job["flow_site"], job["flow_mode"])
        first = not self.cost_cache and job["model"] != "actual"
        self.go.configure(state="disabled")
        self.set_status("Predicting traffic and searching…" +
                        (" (first run loads the models, about 10 seconds)" if first else ""))
        threading.Thread(target=self._worker, args=(job, cache_key), daemon=True).start()
        self.root.after(100, self._poll)

    def _worker(self, job: dict, cache_key: tuple):
        try:
            cfg = rf.config_with(self.cfg, flow_site=job["flow_site"], flow_mode=job["flow_mode"])
            table = self.cost_cache.get(cache_key)
            if table is None:
                table = ec.edge_costs(job["when"], job["model"], cfg)
            result = rf.find_routes(job["origin"], job["destination"], job["when"], job["model"],
                                    job["method"], job["k"], cfg, edge_table=table)
            self.jobs.put(("ok", (cache_key, result)))
        except Exception as e:                                    # shown to the user, never crashes the GUI
            self.jobs.put(("error", e))

    def _poll(self):
        try:
            kind, payload = self.jobs.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll)
            return
        self.go.configure(state="normal")
        if kind == "error":
            msg = str(payload)
            if "intervals right before" in msg or "No measured flow" in msg:
                msg += "\n\nThe data starts on 1 Oct 2006 00:00 and the models need the 3 hours before " \
                       "the departure time. Please choose a later time."
            self.set_status("Could not find routes. " + msg.splitlines()[0], True)
            messagebox.showerror("Could not find routes", msg)
            return
        cache_key, result = payload
        self.cost_cache[cache_key] = result.edge_table
        self.show_result(result)

    # ------------------------------------------------------------------
    # Showing the result
    # ------------------------------------------------------------------
    def show_result(self, result: rf.RouteResult):
        self.result = result
        self.table.delete(*self.table.get_children())
        self.legs.delete(*self.legs.get_children())
        name = SEARCH_METHODS[result.method][1]
        if not result.routes:
            self.summary.configure(text=f"No route from {result.origin} to {result.destination}")
            self.set_status("No route found between these sites.", True)
            self.draw_map()
            return

        self.summary.configure(text=f"{len(result.routes)} route(s): {result.origin} → {result.destination}, "
                                    f"leaving {result.when:%a %d %b %H:%M}")
        for r in result.routes:
            self.table.insert("", "end", iid=str(r.rank), values=(
                r.rank, f"{r.minutes:.1f} min", f"{r.distance_km:.2f} km", len(r.path) - 1, " → ".join(r.roads)))
        fallback = sorted(set(result.edge_table.loc[result.edge_table["source"] == "historical_avg", "flow_site"]))
        note = f" Sites {', '.join(fallback)} have no model and use the historical average." if fallback else ""
        self.set_status(f"Done. {name}, {result.nodes_created} search nodes created.{note}")
        self.table.selection_set("1")                              # triggers show_selected()

    def selected_route(self) -> rf.Route | None:
        sel = self.table.selection()
        if not self.result or not sel:
            return None
        return self.result.routes[int(sel[0]) - 1]

    def show_selected(self):
        route = self.selected_route()
        self.legs.delete(*self.legs.get_children())
        if route is not None:
            for leg in route.legs.itertuples(index=False):
                self.legs.insert("", "end", values=(
                    f"{leg[0]} → {leg[1]}", leg.road, f"{leg.distance_km:.2f}", f"{leg.flow_vph:.0f}",
                    f"{leg.speed_kmh:.1f}", f"{leg.travel_time_s:.0f}"))
            total = route.legs[["distance_km", "travel_time_s"]].sum()
            self.legs.insert("", "end", values=("Total", "", f"{total.distance_km:.2f}", "", "",
                                                f"{total.travel_time_s:.0f}"))
        self.draw_map()

    def draw_map(self):
        ax, pos = self.ax, self.pos
        ax.clear()
        speed = None                                              # {(a, b): km/h, average of both directions}
        if self.traffic.get() and self.result is not None:
            t = self.result.edge_table
            pairs = [tuple(sorted(p)) for p in zip(t["from"], t["to"])]
            speed = pd.Series(t["speed_kmh"].to_numpy()).groupby(pd.Series(pairs)).mean().to_dict()

        # 1. the whole network
        drawn = set()
        for a, b in zip(self.edges["from"], self.edges["to"]):
            key = tuple(sorted((a, b)))
            if key in drawn:
                continue
            drawn.add(key)
            colour, width = "#c8ccd2", 1.5
            if speed is not None and key in speed:
                s = speed[key]
                colour, width = ("#22c55e" if s >= 55 else "#f59e0b" if s >= 45 else "#ef4444"), 2.5
            ax.plot(*zip(pos.loc[a], pos.loc[b]), color=colour, lw=width, zorder=1, solid_capstyle="round")

        # 2. other routes thin and dashed, the selected one bold (blue never clashes with traffic colours)
        selected = self.selected_route()
        if self.result is not None:
            for r in self.result.routes:
                xy = pos.loc[r.path].to_numpy()
                if selected is not None and r.rank == selected.rank:
                    ax.plot(xy[:, 0], xy[:, 1], color=ROUTE_COLOUR, lw=6, alpha=0.9, zorder=3,
                            solid_capstyle="round", solid_joinstyle="round", label=f"Route {r.rank}")
                else:
                    ax.plot(xy[:, 0], xy[:, 1], color=ROUTE_COLOUR, lw=1.5, ls="--", alpha=0.6, zorder=2)

        # 3. sites, origin and destination
        ax.scatter(pos["longitude"], pos["latitude"], s=18, color="#374151", zorder=4)
        for site, (x, y) in pos.iterrows():
            ax.annotate(site, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color="#374151")
        o, d = self._site(self.origin.get()), self._site(self.destination.get())
        if self.result is not None:
            o, d = self.result.origin, self.result.destination
        ax.scatter(*pos.loc[o], s=150, color="#16a34a", edgecolor="white", zorder=5, label=f"Start {o}")
        ax.scatter(*pos.loc[d], s=150, color="#dc2626", edgecolor="white", marker="s", zorder=5, label=f"End {d}")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        if speed is not None:
            ax.set_title("Predicted speed:  green ≥ 55 km/h   amber 45–55   red < 45", fontsize=9)
        ax.set_aspect(1 / np.cos(np.radians(pos["latitude"].mean())))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout()
        self.canvas.draw_idle()


def run() -> None:
    root = tk.Tk()
    root.title("TBRGS - Boroondara Traffic-Based Route Guidance")
    root.geometry("1400x860")
    root.minsize(1100, 700)
    RouteApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
