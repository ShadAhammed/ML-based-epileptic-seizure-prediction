"""Optional desktop GUI for epilepsy detection."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline


class EpilepsyGUI:
    """Thin GUI over DetectionPipeline."""

    def __init__(self) -> None:
        self.pipeline = DetectionPipeline()
        self.root = tk.Tk()
        self.root.title("Epilepsy Detection")
        self.root.geometry("520x420")

        self.edf_path = tk.StringVar()
        self.features_path = tk.StringVar()
        self.model_path = tk.StringVar(value="models/seizure_model.joblib")
        self.sz_start = tk.StringVar(value="2382")
        self.sz_end = tk.StringVar(value="2447")
        self.output_path = tk.StringVar(value="data/features.parquet")
        self.status = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="EDF file:").grid(row=0, column=0, sticky=tk.W, **pad)
        ttk.Entry(frame, textvariable=self.edf_path, width=45).grid(row=0, column=1, **pad)
        ttk.Button(frame, text="Browse", command=self._browse_edf).grid(row=0, column=2, **pad)

        ttk.Label(frame, text="Seizure start:").grid(row=1, column=0, sticky=tk.W, **pad)
        ttk.Entry(frame, textvariable=self.sz_start, width=12).grid(row=1, column=1, sticky=tk.W, **pad)
        ttk.Label(frame, text="Seizure end:").grid(row=2, column=0, sticky=tk.W, **pad)
        ttk.Entry(frame, textvariable=self.sz_end, width=12).grid(row=2, column=1, sticky=tk.W, **pad)

        ttk.Label(frame, text="Features output:").grid(row=3, column=0, sticky=tk.W, **pad)
        ttk.Entry(frame, textvariable=self.output_path, width=45).grid(row=3, column=1, **pad)

        ttk.Label(frame, text="Features file:").grid(row=4, column=0, sticky=tk.W, **pad)
        ttk.Entry(frame, textvariable=self.features_path, width=45).grid(row=4, column=1, **pad)
        ttk.Button(frame, text="Browse", command=self._browse_features).grid(row=4, column=2, **pad)

        ttk.Label(frame, text="Model path:").grid(row=5, column=0, sticky=tk.W, **pad)
        ttk.Entry(frame, textvariable=self.model_path, width=45).grid(row=5, column=1, **pad)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=0, columnspan=3, pady=12)
        ttk.Button(btn_frame, text="Extract Features", command=self._run_extract).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_frame, text="Train Model", command=self._run_train).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Predict", command=self._run_predict).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Evaluate", command=self._run_evaluate).pack(side=tk.LEFT, padx=4)

        ttk.Label(frame, textvariable=self.status, wraplength=480).grid(
            row=7, column=0, columnspan=3, sticky=tk.W, **pad
        )

    def _browse_edf(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("EDF files", "*.edf"), ("All", "*.*")])
        if path:
            self.edf_path.set(path)

    def _browse_features(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Data files", "*.parquet *.csv *.xlsx"), ("All", "*.*")]
        )
        if path:
            self.features_path.set(path)

    def _run_async(self, func, success_msg: str) -> None:
        def worker() -> None:
            try:
                func()
                self.root.after(0, lambda: self._set_status(success_msg))
                self.root.after(0, lambda: messagebox.showinfo("Success", success_msg))
            except Exception as exc:
                self.root.after(0, lambda: self._set_status(f"Error: {exc}"))
                self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))

        self._set_status("Running...")
        threading.Thread(target=worker, daemon=True).start()

    def _run_extract(self) -> None:
        def task() -> None:
            self.pipeline.extract_features(
                self.edf_path.get(),
                int(self.sz_start.get()),
                int(self.sz_end.get()),
                self.output_path.get(),
            )
            self.features_path.set(self.output_path.get())

        self._run_async(task, "Feature extraction complete.")

    def _run_train(self) -> None:
        def task() -> None:
            out = Path(self.model_path.get()).parent
            self.pipeline.train(self.features_path.get(), out)

        self._run_async(task, "Training complete.")

    def _run_predict(self) -> None:
        def task() -> None:
            preds = self.pipeline.predict(self.features_path.get(), self.model_path.get())
            out = Path("reports/predictions.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            preds.to_csv(out, index=False)

        self._run_async(task, "Predictions saved to reports/predictions.csv")

    def _run_evaluate(self) -> None:
        def task() -> None:
            report = self.pipeline.evaluate(
                self.features_path.get(),
                self.model_path.get(),
                "reports",
            )
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Evaluation",
                    f"Accuracy: {report['accuracy']:.4f}\n"
                    f"Sensitivity: {report['sensitivity']:.4f}",
                ),
            )

        self._run_async(task, "Evaluation complete.")

    def _set_status(self, msg: str) -> None:
        self.status.set(msg)

    def run(self) -> None:
        self.root.mainloop()


def run_gui() -> None:
    EpilepsyGUI().run()
