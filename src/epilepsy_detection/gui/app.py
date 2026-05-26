"""Desktop GUI for seizure detection (EDF → detect when–when seizure occurs)."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline


class EpilepsyGUI:
    """Detect ictal periods in an EDF using a pre-trained model."""

    def __init__(self) -> None:
        self.pipeline = DetectionPipeline()
        self.root = tk.Tk()
        self.root.title("Epilepsy Seizure Detection")
        self.root.geometry("640x520")

        self.edf_path = tk.StringVar()
        self.model_path = tk.StringVar(value="models/seizure_model.joblib")
        self.status = tk.StringVar(value="Load an EDF and pre-trained model, then click Detect Seizures.")

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frame,
            text="Detect ictal (seizure) periods in an EEG recording",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, **pad)

        ttk.Label(frame, text="EDF recording:").grid(row=1, column=0, sticky=tk.W, **pad)
        ttk.Entry(frame, textvariable=self.edf_path, width=52).grid(row=1, column=1, **pad)
        ttk.Button(frame, text="Browse", command=self._browse_edf).grid(row=1, column=2, **pad)

        ttk.Label(frame, text="Pre-trained model:").grid(row=2, column=0, sticky=tk.W, **pad)
        ttk.Entry(frame, textvariable=self.model_path, width=52).grid(row=2, column=1, **pad)
        ttk.Button(frame, text="Browse", command=self._browse_model).grid(row=2, column=2, **pad)

        ttk.Label(
            frame,
            text="Train models in the legacy notebook; place seizure_model.joblib here.",
            foreground="gray",
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=8)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=12)
        ttk.Button(
            btn_frame,
            text="Detect Seizures",
            command=self._run_detect,
        ).pack(side=tk.LEFT, padx=4)

        ttk.Label(frame, text="Results:").grid(row=5, column=0, sticky=tk.NW, **pad)
        self.results_text = scrolledtext.ScrolledText(frame, width=62, height=16, wrap=tk.WORD)
        self.results_text.grid(row=5, column=1, columnspan=2, sticky=tk.NSEW, **pad)

        ttk.Label(frame, textvariable=self.status, wraplength=580).grid(
            row=6, column=0, columnspan=3, sticky=tk.W, **pad
        )

        frame.rowconfigure(5, weight=1)
        frame.columnconfigure(1, weight=1)

    def _browse_edf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf"), ("All", "*.*")])
        if path:
            self.edf_path.set(path)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Model", "*.joblib"), ("All", "*.*")])
        if path:
            self.model_path.set(path)

    def _run_detect(self) -> None:
        if not self.edf_path.get().strip():
            messagebox.showwarning("Missing input", "Select an EDF recording.")
            return
        if not Path(self.model_path.get()).exists():
            messagebox.showwarning(
                "Missing model",
                "Pre-trained model not found.\n\n"
                "Train with notebooks/legacy/Epilepsy.ipynb and save to:\n"
                f"  {self.model_path.get()}",
            )
            return

        def worker() -> None:
            try:
                result = self.pipeline.detect_from_edf(
                    self.edf_path.get(),
                    self.model_path.get(),
                )
                out_csv = Path("reports/detection_result.csv")
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                result.per_epoch.to_csv(out_csv, index=False)

                self.root.after(0, lambda: self._show_results(result.report, str(out_csv)))
            except Exception as exc:
                self.root.after(0, lambda: self._show_error(str(exc)))

        self.status.set("Processing EDF — extracting features and detecting seizures...")
        self.results_text.delete("1.0", tk.END)
        threading.Thread(target=worker, daemon=True).start()

    def _show_results(self, report: str, csv_path: str) -> None:
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, report)
        self.results_text.insert(tk.END, f"\n\nPer-epoch details saved to:\n  {csv_path}")
        self.status.set("Detection complete.")
        messagebox.showinfo("Detection complete", report)

    def _show_error(self, msg: str) -> None:
        self.status.set(f"Error: {msg}")
        messagebox.showerror("Error", msg)

    def run(self) -> None:
        self.root.mainloop()


def run_gui() -> None:
    EpilepsyGUI().run()
