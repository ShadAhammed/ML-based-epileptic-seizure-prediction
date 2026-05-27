#!/usr/bin/env python
"""Save seizure_model.joblib from an active Jupyter kernel (notebook still open)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROBE = "print('OK' if all(n in globals() for n in ['scaler','Smodel','f','X','y']) else 'MISSING')"

SAVE_CODE = f"""
import sys
from pathlib import Path
repo = Path(r'{ROOT}')
sys.path.insert(0, str(repo / 'src'))

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier
from epilepsy_detection.notebook_export import save_trained_model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rfe = RFECV(estimator=XGBClassifier(), min_features_to_select=16, cv=3)
rfe.fit(X_train, y_train)

out = save_trained_model(
    scaler=scaler,
    rfe=rfe,
    model=Smodel,
    feature_columns=list(X.columns),
    selected_features=f,
    strategy='smote',
    output_path=str(repo / 'models' / 'seizure_model.joblib'),
)
print('SAVED:', out)
"""


def _run(client, code: str, timeout: float = 30.0) -> tuple[bool, str]:
    msg_id = client.execute(code)
    text_parts: list[str] = []
    try:
        while True:
            msg = client.get_iopub_msg(timeout=timeout)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            t = msg["header"]["msg_type"]
            if t == "stream":
                text_parts.append(msg["content"].get("text", ""))
            elif t == "error":
                return False, "".join(msg["content"].get("traceback", []))
            elif t == "status" and msg["content"].get("execution_state") == "idle":
                break
    except Exception as exc:
        return False, str(exc)
    return True, "".join(text_parts)


def try_kernel(kernel_json: Path) -> bool:
    from jupyter_client.manager import KernelManager

    km = KernelManager(connection_file=str(kernel_json))
    km.load_connection_file()
    if not km.is_alive():
        return False

    client = km.client()
    client.start_channels()
    try:
        ok, out = _run(client, PROBE, timeout=15.0)
        if not ok or "OK" not in out:
            return False
        ok, out = _run(client, SAVE_CODE, timeout=300.0)
        if ok:
            print(out, end="")
        return ok and (ROOT / "models" / "seizure_model.joblib").exists()
    finally:
        client.stop_channels()


def main() -> None:
    runtime = Path.home() / "AppData" / "Roaming" / "jupyter" / "runtime"
    kernels = sorted(runtime.glob("kernel-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not kernels:
        print("No Jupyter kernels found. Open Epilepsy.ipynb and run the last cell instead.")
        sys.exit(1)

    for kf in kernels[:20]:
        print(f"Probing {kf.name}...")
        try:
            if try_kernel(kf):
                print(f"Success: {ROOT / 'models' / 'seizure_model.joblib'}")
                sys.exit(0)
        except Exception as exc:
            print(f"  skip: {exc}")

    print(
        "No kernel with trained variables (scaler, Smodel, f, X, y).\n"
        "Open notebooks/legacy/Epilepsy.ipynb, run through the SMOTE cell, "
        "then run the last cell (Save model) or:\n"
        "  python scripts/save_model.py --features path/to/TestSeizureData3.xlsx"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
