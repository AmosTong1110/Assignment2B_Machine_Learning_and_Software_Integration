"""
train_dialog.py  -  "Train models" window of the GUI
====================================================
Trains the ML models from the GUI by running  python -m src.Models.train_all  in a separate
process (so the window never freezes and TensorFlow stays out of the GUI process).
The output of the training is shown live in the window.

Two modes:
    Quick demo     3 sites, 2 epochs, written to a TEMPORARY folder - about a minute,
                   does not touch the saved models.  Use it to show that training works.
    Full training  every site, up to `epochs` from config.yaml - 20+ minutes with LSTM/GRU,
                   REPLACES the models in saved_models/ and the files in results/.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import yaml

import src.Models.deep_common as dc

QUICK_SITES = ["2000", "3002", "4035"]
QUICK_EPOCHS = 2
MODEL_LABELS = {"linear": "Linear regression", "lstm": "LSTM", "gru": "GRU"}


class TrainDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, has_tensorflow: bool, on_full_training_done=None):
        super().__init__(parent)
        self.title("Train models")
        self.geometry("760x520")
        self.minsize(600, 400)
        self.on_full_training_done = on_full_training_done
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.tmp: tempfile.TemporaryDirectory | None = None
        self.full_run = False

        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Models to train").grid(row=0, column=0, sticky="w")
        self.use = {}
        for i, (key, label) in enumerate(MODEL_LABELS.items()):
            ok = has_tensorflow or key == "linear"
            self.use[key] = tk.BooleanVar(value=ok)
            ttk.Checkbutton(top, text=label + ("" if ok else " (needs TensorFlow)"), variable=self.use[key],
                            state="normal" if ok else "disabled").grid(row=0, column=i + 1, padx=8, sticky="w")

        self.mode = tk.StringVar(value="quick")
        ttk.Radiobutton(top, variable=self.mode, value="quick",
                        text=f"Quick demo: {len(QUICK_SITES)} sites, {QUICK_EPOCHS} epochs, about 1 minute. "
                             "Safe: your saved models are not changed."
                        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Radiobutton(top, variable=self.mode, value="full",
                        text="Full training: all sites, 20+ minutes with LSTM/GRU. "
                             "Replaces the saved models and results."
                        ).grid(row=2, column=0, columnspan=4, sticky="w")

        buttons = ttk.Frame(self, padding=(10, 0))
        buttons.pack(fill="x")
        self.start_btn = ttk.Button(buttons, text="Start training", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.state = ttk.Label(buttons, text="Ready.", foreground="#555")
        self.state.pack(side="left", padx=10)

        self.bar = ttk.Progressbar(self, mode="indeterminate")
        self.bar.pack(fill="x", padx=10, pady=8)
        box = ttk.Frame(self)
        box.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(box, wrap="none", height=14, font=("Menlo", 10), state="disabled")
        scroll = ttk.Scrollbar(box, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.close)

    # ------------------------------------------------------------------
    def _write(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def command(self) -> list[str]:
        chosen = [k for k, v in self.use.items() if v.get()]
        cmd = [sys.executable, "-u", "-m", "src.Models.train_all", "--only", *chosen]
        if self.mode.get() == "quick":
            # temporary config: same settings, but models and results go to a throw-away folder
            self.tmp = tempfile.TemporaryDirectory(prefix="tbrgs_train_")
            base = Path(self.tmp.name)
            cfg = {"paths": {"models_dir": str(base / "saved_models"), "results_dir": str(base / "results")}}
            for sub in ("saved_models", "results"):
                (base / sub).mkdir()
            for m in chosen:                                     # predict() needs the folder to exist
                (base / "saved_models" / m).mkdir()
            conf = base / "config.yaml"
            conf.write_text(yaml.safe_dump({**dc.load_config(), "paths": {**dc.load_config()["paths"],
                                                                           **cfg["paths"]}}))
            cmd += ["--config", str(conf), "--sites", *QUICK_SITES, "--epochs", str(QUICK_EPOCHS)]
        return cmd

    def start(self) -> None:
        if self.proc is not None:
            return
        if not any(v.get() for v in self.use.values()):
            messagebox.showinfo("Train models", "Please tick at least one model.", parent=self)
            return
        self.full_run = self.mode.get() == "full"
        if self.full_run and not messagebox.askyesno(
                "Replace saved models?",
                "Full training takes 20+ minutes and REPLACES the models in saved_models/ and the "
                "files in results/.\n\nContinue?", parent=self):
            return
        cmd = self.command()
        self._write(f"$ {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))}\n\n")
        self.proc = subprocess.Popen(cmd, cwd=dc.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.state.configure(text="Training...")
        self.bar.start(12)
        self.after(100, self._poll)

    def _reader(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self.lines.put(line)
        proc.wait()
        self.lines.put(None)                                     # end marker

    def _poll(self) -> None:
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    return self._finished()
                self._write(line)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._poll)

    def _finished(self) -> None:
        code = self.proc.returncode if self.proc else -1
        self.proc = None
        self.bar.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        if code == 0:
            msg = "Finished. Models trained and compared."
            if self.full_run:
                msg += " Restart the app to use the new models."
                if self.on_full_training_done:
                    self.on_full_training_done()
            else:
                msg += " (Quick demo: your saved models were not changed.)"
            self.state.configure(text=msg, foreground="#15803d")
        else:
            self.state.configure(text=f"Stopped or failed (exit code {code}). See the log.", foreground="#b91c1c")
        if self.tmp is not None:
            self.tmp.cleanup()
            self.tmp = None

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()

    def close(self) -> None:
        if self.proc is not None and not messagebox.askyesno(
                "Training is running", "Stop the training and close this window?", parent=self):
            return
        self.stop()
        self.destroy()
