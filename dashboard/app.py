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
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
for p in (str(ROOT / "src"), str(DASHBOARD_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline  # noqa: E402

from charts import (  # noqa: E402
    fig_eeg_with_seizures,
    fig_epoch_map,
    fig_probability_timeline,
    fig_seizure_gantt,
    fig_state_timeline,
    seizure_summary_cards,
)

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
    .seizure-banner {
        background: linear-gradient(90deg, #fef2f2 0%, #fff 100%);
        border-left: 4px solid #dc2626;
        padding: 0.75rem 1rem;
        margin: 0.35rem 0;
        border-radius: 4px;
        font-size: 1.05rem;
    }
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
        '<p class="sub-header">Upload an EDF recording to detect and visualize ictal (seizure) time windows.</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Configuration")
        model_path = st.text_input(
            "Pre-trained model path",
            value=str(default_model_path()),
            help="Train locally with your research notebook; place seizure_model.joblib in models/",
        )
        show_eeg = st.checkbox("Show EEG waveform preview", value=True)
        st.divider()
        st.markdown("**Data policy**")
        st.caption(
            "No clinical EEG is stored in this repo. "
            "Obtain CHB-MIT data via [PhysioNet](https://physionet.org/content/chbmit/1.0.0/). "
            "See docs/DATA.md."
        )

    uploaded = st.file_uploader(
        "Upload EDF recording",
        type=["edf"],
        help="Scalp EEG recording in European Data Format (.edf)",
    )

    if uploaded is not None:
        st.caption(f"Loaded: **{uploaded.name}** ({uploaded.size / 1e6:.2f} MB)")

    run_detect = st.button("Detect Seizures", type="primary")

    if st.session_state.get("detection_result") and not run_detect:
        _render_results(
            st.session_state["detection_result"],
            st.session_state.get("edf_temp_path"),
            show_eeg,
        )
        return

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
            old = st.session_state.pop("edf_temp_path", None)
            if old and Path(old).exists():
                Path(old).unlink(missing_ok=True)

            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)

            pipeline = get_pipeline()
            result = pipeline.detect_from_edf(tmp_path, model_file)

            st.session_state["detection_result"] = result
            st.session_state["edf_temp_path"] = tmp_path
            st.session_state["edf_name"] = uploaded.name
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            return

    _render_results(result, st.session_state.get("edf_temp_path"), show_eeg)


def _show_welcome() -> None:
    st.info(
        "Upload an EDF file and click **Detect Seizures** to see **from–to** predicted seizure windows."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Input", "EDF recording")
    c2.metric("Output", "When → when seizure periods")
    c3.metric("Resolution", "1 second per epoch")


def _render_results(result, edf_path: Path | None, show_eeg: bool) -> None:
    st.success("Detection complete")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recording length", f"{result.recording_seconds:,} s")
    m2.metric("Epochs analyzed", f"{result.n_epochs:,}")
    m3.metric("Seizures detected", len(result.seizures))
    m4.metric("Ictal epochs", f"{int(result.per_epoch['predicted'].sum()):,}")

    st.subheader("Predicted seizure periods (when → when)")
    if result.seizures:
        for line in seizure_summary_cards(result.seizures):
            st.markdown(f'<div class="seizure-banner">{line}</div>', unsafe_allow_html=True)

        rows = [
            {
                "Seizure": f"#{i}",
                "From (s)": s.start_seconds,
                "To (s)": s.end_seconds,
                "Duration (s)": s.duration_seconds,
                "From (epoch)": s.start_epoch,
                "To (epoch)": s.end_epoch,
            }
            for i, s in enumerate(result.seizures, 1)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No seizure activity detected in this recording.")

    st.subheader("Visual timeline")
    tab_gantt, tab_state, tab_prob, tab_epoch, tab_eeg = st.tabs(
        ["When → when (bars)", "State along recording", "Probability", "Epoch map", "EEG + seizures"]
    )

    with tab_gantt:
        st.plotly_chart(
            fig_seizure_gantt(result.seizures, result.recording_seconds),
            use_container_width=True,
        )

    with tab_state:
        st.plotly_chart(
            fig_state_timeline(result.per_epoch, result.seizures),
            use_container_width=True,
        )

    with tab_prob:
        st.plotly_chart(
            fig_probability_timeline(result.per_epoch, result.seizures),
            use_container_width=True,
        )

    with tab_epoch:
        st.plotly_chart(fig_epoch_map(result.per_epoch), use_container_width=True)

    with tab_eeg:
        if show_eeg and edf_path and Path(edf_path).exists():
            fig = fig_eeg_with_seizures(edf_path, result.seizures)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not render EEG preview.")
        else:
            st.caption("Enable **Show EEG waveform preview** in the sidebar to overlay seizures on the signal.")

    with st.expander("Per-epoch predictions (table)"):
        st.dataframe(result.per_epoch, use_container_width=True)

    st.download_button(
        "Download predictions (CSV)",
        result.per_epoch.to_csv(index=False),
        file_name="detection_result.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
