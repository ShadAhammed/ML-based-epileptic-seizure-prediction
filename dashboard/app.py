"""
Epilepsy Seizure Detection Dashboard

Run after clone:
    pip install -e .
    epilepsy dashboard
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline  # noqa: E402

st.set_page_config(
    page_title="Epilepsy Seizure Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 1.75rem; font-weight: 600; color: #1e3a5f; }
    .sub-header { color: #5a6a7a; margin-bottom: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline() -> DetectionPipeline:
    return DetectionPipeline()


def default_model_path() -> Path:
    return ROOT / "models" / "seizure_model.joblib"


def main() -> None:
    st.markdown('<p class="main-header">Epilepsy Seizure Detection</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload an EDF recording to detect ictal (seizure) time windows.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Configuration")
        model_path = st.text_input(
            "Pre-trained model path",
            value=str(default_model_path()),
            help="Train locally with your research notebook; place seizure_model.joblib in models/",
        )
        st.divider()
        st.markdown("**Data policy**")
        st.caption(
            "No clinical EEG is stored in this repo. "
            "Obtain CHB-MIT data via [PhysioNet](https://physionet.org/content/chbmit/1.0.0/). "
            "See docs/DATA.md."
        )
        st.divider()
        st.markdown("**How it works**")
        st.caption("EDF → feature extraction → ML classifier → seizure intervals (from–to).")

    uploaded = st.file_uploader(
        "Upload EDF recording",
        type=["edf"],
        help="Scalp EEG recording in European Data Format (.edf)",
    )

    col_run, col_clear = st.columns([1, 4])
    with col_run:
        run_detect = st.button("Detect Seizures", type="primary", use_container_width=True)

    if not run_detect:
        _show_welcome()
        return

    if uploaded is None:
        st.error("Please upload an EDF file before running detection.")
        return

    model_file = Path(model_path)
    if not model_file.exists():
        st.error(
            f"Model not found: `{model_file}`\n\n"
            "Train a model with your local research notebook and save it as "
            "`models/seizure_model.joblib`."
        )
        return

    with st.spinner("Extracting features and running detection…"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            pipeline = get_pipeline()
            result = pipeline.detect_from_edf(tmp_path, model_file)
            tmp_path.unlink(missing_ok=True)
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            return

    _render_results(result)


def _show_welcome() -> None:
    st.info(
        "Configure the model path in the sidebar, upload an EDF file, then click **Detect Seizures**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Input", "EDF recording")
    c2.metric("Output", "Seizure time windows")
    c3.metric("Epoch resolution", "1 second")


def _render_results(result) -> None:
    st.success("Detection complete")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recording length", f"{result.recording_seconds:,} s")
    m2.metric("Epochs analyzed", f"{result.n_epochs:,}")
    m3.metric("Seizures detected", len(result.seizures))
    m4.metric(
        "Ictal coverage",
        f"{result.per_epoch['predicted'].sum()} epochs",
    )

    st.subheader("Detected seizure periods")
    if result.seizures:
        rows = [
            {
                "#": i,
                "Start (s)": s.start_seconds,
                "End (s)": s.end_seconds,
                "Duration (s)": s.duration_seconds,
                "Start epoch": s.start_epoch,
                "End epoch": s.end_epoch,
            }
            for i, s in enumerate(result.seizures, 1)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No seizure activity detected in this recording.")

    st.subheader("Timeline")
    df = result.per_epoch.copy()
    fig = px.scatter(
        df,
        x="time_seconds",
        y="probability_seizure" if "probability_seizure" in df.columns else "predicted",
        color=df["predicted"].map({0: "Interictal", 1: "Ictal"}),
        labels={"time_seconds": "Time (seconds)", "color": "State"},
        title="Per-epoch seizure prediction",
        color_discrete_map={"Interictal": "#94a3b8", "Ictal": "#dc2626"},
    )
    fig.update_layout(height=380, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Per-epoch predictions (table)"):
        st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        file_name="detection_result.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
