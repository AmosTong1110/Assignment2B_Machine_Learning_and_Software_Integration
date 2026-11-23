"""
experiment_figures.py  -  report figures from the four experiment tables
=======================================================================
Run from the project root AFTER  python -m src.Routing.experiments :

    python -m src.Routing.experiment_figures

Reads  results/experiments/{1_search_methods,2_time_of_day,3_flow_mode,4_model_choice}.csv
Writes results/experiments/fig1_search_methods.png ... fig4_model_choice.png
No ML and no TensorFlow: it only re-draws the saved tables.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")                            # no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.Models.deep_common as dc

METHOD_ORDER = ["AS", "BFS", "DFS", "GBFS", "CUS1", "CUS2"]
METHOD_LABEL = {"AS": "A*", "BFS": "BFS", "DFS": "DFS", "GBFS": "GBFS", "CUS1": "IDDFS", "CUS2": "IDA*"}
MODEL_LABEL = {"gru": "GRU", "lstm": "LSTM", "linear": "Linear"}
MODEL_COLOUR = {"gru": "#2563eb", "lstm": "#16a34a", "linear": "#f59e0b"}
TRIP_COLOURS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def _grouped_bars(ax, table: pd.DataFrame, colours, log=False, fmt="{:.0f}"):
    """table: rows = groups on the x axis, columns = bars inside a group."""
    n = len(table.columns)
    width = 0.8 / n
    x = np.arange(len(table.index))
    for i, col in enumerate(table.columns):
        bars = ax.bar(x + (i - (n - 1) / 2) * width, table[col], width, label=col, color=colours[i % len(colours)])
        if fmt:
            ax.bar_label(bars, labels=[fmt.format(v) if v else "" for v in table[col]], fontsize=6, padding=1)
    ax.set_xticks(x, table.index)
    if log:
        ax.set_yscale("log")


def fig_search_methods(df: pd.DataFrame, path) -> None:
    """Search effort (nodes created) and route quality (extra minutes vs A*) per method."""
    df = df.assign(m=df["method"].map(METHOD_LABEL))
    order = [METHOD_LABEL[m] for m in METHOD_ORDER]
    nodes = df.pivot(index="m", columns="trip", values="nodes_created").loc[order]
    extra = df.pivot(index="m", columns="trip", values="extra_vs_astar_min").loc[order]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    _grouped_bars(a1, nodes, TRIP_COLOURS, log=True, fmt=None)
    a1.set_ylabel("Search nodes created (log scale)")
    a1.set_title("(a) Search effort for 5 routes (Yen's algorithm)")
    _grouped_bars(a2, extra, TRIP_COLOURS, fmt="{:.2f}")
    a2.set_ylabel("Mean of the 5 routes: minutes slower than A*")
    a2.text(0.98, 0.95, "no bar = same 5 routes as A*", transform=a2.transAxes, ha="right", fontsize=7, color="#6b7280")
    a2.set_title("(b) Route quality compared with A*")
    a1.legend(title="Trip", fontsize=7, title_fontsize=7)
    fig.suptitle("Part A search methods on the Boroondara network (GRU, Thu 26 Oct 2006 08:00)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_time_of_day(df: pd.DataFrame, path) -> None:
    """Fastest-route time per trip through the day, with network speed; hollow marker = route changed."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    for colour, (trip, g) in zip(TRIP_COLOURS, df.groupby("trip", sort=False)):
        base = g["route1_min"].iloc[0]
        a1.plot(g["time"], g["route1_min"] / base * 100 - 100, "-", color=colour, label=trip)
        changed = g[g["route_changed_vs_0300"]]
        a1.scatter(changed["time"], changed["route1_min"] / base * 100 - 100, s=60,
                   facecolor="white", edgecolor=colour, zorder=3)
    a1.set_ylabel("Fastest route: % slower than at 03:00")
    a1.set_xlabel("Departure time")
    a1.set_title("(a) Fastest route time (hollow marker = a different route is chosen)")
    a1.legend(fontsize=7)

    net = df.drop_duplicates("time")
    a2.bar(net["time"], net["network_mean_flow_vph"], color="#cbd5e1", label="Mean site flow")
    a2.set_ylabel("Mean predicted flow at a site (veh/h)")
    a2.set_xlabel("Departure time")
    a2b = a2.twinx()
    a2b.plot(net["time"], net["network_mean_speed_kmh"], "o-", color="#dc2626", label="Mean link speed")
    a2b.set_ylabel("Mean link speed (km/h)", color="#dc2626")
    a2b.spines["right"].set_visible(True)
    a2.set_title("(b) Whole network")
    fig.suptitle("Effect of departure time (GRU predictions, Thu 26 Oct 2006)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_flow_mode(df: pd.DataFrame, path) -> None:
    """per_detector vs total: links at capacity and resulting route times."""
    net = df.drop_duplicates(["time", "flow_mode"]).pivot(index="time", columns="flow_mode",
                                                          values="edges_at_capacity")
    total_edges = int(df["edges_total"].iloc[0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    _grouped_bars(a1, net[["per_detector", "total"]], ["#2563eb", "#dc2626"])
    a1.axhline(total_edges, color="#6b7280", ls="--", lw=1)
    a1.text(-0.4, total_edges + 2, f"all {total_edges} links", fontsize=7, color="#6b7280")
    a1.set_ylim(0, total_edges * 1.15)
    a1.set_ylabel("Links at capacity speed (32 km/h)")
    a1.set_xlabel("Departure time")
    a1.set_title("(a) How many links are 'at capacity'")
    a1.legend(title="flow_mode", fontsize=7, title_fontsize=7)

    ratio = df.pivot_table(index="time", columns="flow_mode", values="route1_min", aggfunc="mean")
    _grouped_bars(a2, ratio[["per_detector", "total"]], ["#2563eb", "#dc2626"], fmt="{:.1f}")
    a2.set_ylabel("Fastest route, mean over 4 trips (min)")
    a2.set_xlabel("Departure time")
    a2.set_title("(b) Predicted trip time")
    fig.suptitle("Flow per link: site total divided by detectors (per_detector) vs intersection total", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_model_choice(df: pd.DataFrame, path) -> None:
    """Prediction error of route time, regret in real traffic, and edge-time MAE per model."""
    ml = df[df["model"] != "actual"].copy()
    actual = df[df["model"] == "actual"].set_index(["time", "trip"])["predicted_route1_min"]
    ml["error_s"] = (ml["predicted_route1_min"] - [actual[(t, r)] for t, r in zip(ml["time"], ml["trip"])]) * 60
    ml["regret_s"] = (ml["chosen_route_in_real_traffic_min"] - ml["best_possible_min"]) * 60
    ml["case"] = ml["time"] + "  " + ml["trip"]
    models = [m for m in MODEL_LABEL if m in set(ml["model"])]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [4, 4, 2]})
    err = ml.pivot(index="case", columns="model", values="error_s")[models].rename(columns=MODEL_LABEL)
    _grouped_bars(a1, err, [MODEL_COLOUR[m] for m in models], fmt=None)
    a1.axhline(0, color="black", lw=0.8)
    a1.set_ylabel("Predicted − measured route time (s)")
    a1.set_title("(a) Error of the predicted trip time")
    a1.tick_params(axis="x", labelsize=7, labelrotation=40)
    plt.setp(a1.get_xticklabels(), ha="right", rotation_mode="anchor")
    a1.legend(fontsize=7)

    reg = ml.pivot(index="case", columns="model", values="regret_s")[models].rename(columns=MODEL_LABEL)
    _grouped_bars(a2, reg, [MODEL_COLOUR[m] for m in models], fmt=None)
    a2.set_ylabel("Time lost in real traffic (s)")
    a2.set_title("(b) Regret: chosen route vs best route (measured flow)")
    a2.tick_params(axis="x", labelsize=7, labelrotation=40)
    plt.setp(a2.get_xticklabels(), ha="right", rotation_mode="anchor")
    a2.set_ylim(0, max(10, reg.to_numpy().max() * 1.5))

    mae = ml.groupby("model")["edge_time_mae_s"].mean()[models]
    bars = a3.bar([MODEL_LABEL[m] for m in models], mae, color=[MODEL_COLOUR[m] for m in models])
    a3.bar_label(bars, fmt="%.2f", fontsize=7)
    a3.set_ylabel("Mean abs. error per link (s)")
    a3.set_title("(c) Link travel-time error")
    fig.suptitle("Does the choice of ML model change the recommended route? (Thu 26 Oct 2006)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    out = dc.get_path(dc.load_config(), "results_dir") / "experiments"
    jobs = [("1_search_methods.csv", fig_search_methods, "fig1_search_methods.png"),
            ("2_time_of_day.csv", fig_time_of_day, "fig2_time_of_day.png"),
            ("3_flow_mode.csv", fig_flow_mode, "fig3_flow_mode.png"),
            ("4_model_choice.csv", fig_model_choice, "fig4_model_choice.png")]
    for csv, draw, png in jobs:
        df = pd.read_csv(out / csv, dtype={"time": str})
        draw(df, out / png)
        print(f"Saved {out / png}")


if __name__ == "__main__":
    main()
