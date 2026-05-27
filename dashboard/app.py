"""
Epilepsy Seizure Detection Dashboard
=====================================

A professional web interface for detecting ictal (seizure) periods in scalp
EEG recordings using a pre-trained XGBoost classifier.

Workflow
--------
1. Load the pre-trained model (trained locally from the research notebook).
2. Upload an EDF recording (.edf).
3. Click **Analyse Recording**.
4. The dashboard displays:
   - A clear SEIZURE DETECTED / NO SEIZURE verdict.
   - Exact from-to time windows for each predicted seizure.
   - Interactive charts: Gantt timeline, probability curve, raw EEG overlay.
   - A downloadable CSV of per-epoch predictions.

Running
-------
After cloning:

    pip install -e .
    epilepsy dashboard

Or on Windows, double-click ``run_dashboard.bat``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup (works whether installed as a package or run from repo root) ──
ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT / "src"), str(DASHBOARD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from epilepsy_detection.pipeline.detection_pipeline import DetectionPipeline  # noqa: E402
from epilepsy_detection.data.edf_loader import EDFLoader  # noqa: E402

from charts import (  # noqa: E402
    fig_eeg_with_seizures,
    fig_epoch_raster,
    fig_probability_timeline,
    fig_seizure_gantt,
    fig_state_timeline,
    seizure_summary_html,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EpilepsyDetector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Typography + dark navy shell */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #0a1628;
        background-image: linear-gradient(180deg, #0a1628 0%, #0d1f36 100%);
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1f36;
        border-right: 1px solid #1e3a5f;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #cbd5e1;
    }
    .main .block-container {
        color: #e2e8f0;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: inherit;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #0f2137;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: #e2e8f0 !important;
    }

    /* App header */
    .app-title {
        font-size: 1.85rem; font-weight: 700;
        color: #f1f5f9; letter-spacing: -0.02em; margin-bottom: 0;
    }
    .app-subtitle {
        font-size: 0.95rem; color: #94a3b8; margin-top: 0.1rem;
    }

    /* Verdict cards */
    .verdict-seizure {
        background: linear-gradient(135deg, #3b1520 0%, #4a1a24 100%);
        border: 2px solid #f87171;
        border-radius: 12px; padding: 1.4rem 1.8rem;
        text-align: center;
    }
    .verdict-seizure h2 { color: #fca5a5; font-size: 1.7rem; margin: 0 0 0.3rem 0; }
    .verdict-seizure p  { color: #fecaca; font-size: 1rem; margin: 0; }

    .verdict-normal {
        background: linear-gradient(135deg, #0f2a1f 0%, #143528 100%);
        border: 2px solid #4ade80;
        border-radius: 12px; padding: 1.4rem 1.8rem;
        text-align: center;
    }
    .verdict-normal h2 { color: #86efac; font-size: 1.7rem; margin: 0 0 0.3rem 0; }
    .verdict-normal p  { color: #bbf7d0; font-size: 1rem; margin: 0; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #0f2137;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
    }
    div[data-testid="stMetric"] > div:first-child { color: #94a3b8; font-size: 0.82rem; }
    div[data-testid="stMetric"] > div:last-child  { color: #f1f5f9; font-size: 1.35rem; }

    /* Seizure window badge */
    .sz-window {
        background: #1a2744;
        border-left: 4px solid #f87171;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #e2e8f0;
    }

    /* Model status badge */
    .model-ok   { color: #4ade80; font-weight: 600; }
    .model-miss { color: #f87171; font-weight: 600; }

    /* Divider */
    hr.light { border: none; border-top: 1px solid #1e3a5f; margin: 1rem 0; }

    /* Step indicator */
    .step-label {
        font-size: 0.78rem; font-weight: 600; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_pipeline() -> DetectionPipeline:
    """Single shared pipeline instance (cached across Streamlit reruns)."""
    return DetectionPipeline()


def _default_model_path() -> Path:
    """Resolve the default model location relative to the repo root."""
    return ROOT / "models" / "seizure_model.joblib"


def _model_status(path: Path) -> tuple[bool, str]:
    """Return (found, message) for a given model path."""
    if path.exists():
        size_kb = path.stat().st_size / 1024
        return True, f"{path.name}  ({size_kb:,.0f} KB)"
    return False, f"Not found: {path}"


def _recording_info(edf_bytes: bytes, filename: str) -> dict | None:
    """Extract EDF metadata without storing the file to disk permanently."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(edf_bytes)
            info = EDFLoader().recording_info(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)
        return info
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[Path, int, bool]:
    """Render the sidebar and return (model_path, channel_idx, show_eeg)."""
    with st.sidebar:
        st.markdown(
            "<div style='text-align:center; padding:0.5rem 0 1rem'>"
            "<span style='font-size:2.2rem'>🧠</span><br>"
            "<span style='font-weight:700; font-size:1.1rem; color:#f1f5f9'>EpilepsyDetector</span><br>"
            "<span style='font-size:0.8rem; color:#94a3b8'>CHB-MIT · XGBoost · v1.0</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # --- Model configuration ---
        st.markdown('<p class="step-label">Step 1 · Model</p>', unsafe_allow_html=True)
        raw_path = st.text_input(
            "Model path (.joblib)",
            value=str(_default_model_path()),
            label_visibility="collapsed",
            help="Path to seizure_model.joblib trained from the research notebook.",
        )
        model_path = Path(raw_path)
        found, status_msg = _model_status(model_path)
        css_class = "model-ok" if found else "model-miss"
        icon = "✔" if found else "✘"
        st.markdown(
            f'<p class="{css_class}" style="font-size:0.82rem">{icon} {status_msg}</p>',
            unsafe_allow_html=True,
        )
        if not found:
            st.warning(
                "The notebook trains the model **in memory only** — it does not "
                "create this file automatically.\n\n"
                "**Fix:** Paste `scripts/save_model_cell.py` in a new cell at the "
                "end of `Epilepsy.ipynb` (after SMOTE training), **or** run:\n\n"
                "`python scripts/save_model.py --features your_features.xlsx`"
            )

        st.divider()

        # --- Visualisation options ---
        st.markdown('<p class="step-label">Options</p>', unsafe_allow_html=True)
        show_eeg = st.checkbox("Show EEG signal overlay", value=True)
        channel_idx = st.number_input(
            "EEG channel to display",
            min_value=0,
            max_value=99,
            value=0,
            help="Zero-based channel index.  Channel 0 is typically EEG FP1-F7.",
        )

        st.divider()

        # --- About panel ---
        with st.expander("About this tool"):
            st.markdown(
                """
                **Epilepsy Seizure Detection** applies a pre-trained
                XGBoost classifier to CHB-MIT-style scalp EEG recordings.

                **How the model works:**
                Each 1-second epoch is characterised by statistical and
                band-energy features per channel (10 features × N channels).
                The model predicts whether each epoch is ictal (seizure)
                or interictal (normal).  Consecutive ictal epochs are
                merged into reported seizure windows.

                **Data privacy:** No EEG data is stored in this repository.
                See `docs/DATA.md` for CHB-MIT access instructions.
                """,
                unsafe_allow_html=False,
            )

    return model_path, int(channel_idx), show_eeg


# ── Welcome screen ────────────────────────────────────────────────────────────

def _render_welcome() -> None:
    """Show usage instructions when no analysis has been run yet."""
    st.markdown(
        """
        <div style="background:#0f2137; border:1px solid #1e3a5f; border-radius:12px;
                    padding:2rem; text-align:center; margin-top:1rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem">📂</div>
            <h3 style="color:#f1f5f9; margin:0 0 0.5rem">Upload an EDF recording</h3>
            <p style="color:#94a3b8; max-width:420px; margin:0 auto">
                Select a <code>.edf</code> file from the CHB-MIT database or any
                compatible EEG system, then click <strong>Analyse Recording</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='light'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.info("**Input:** EDF recording (.edf)")
    c2.info("**Output:** Seizure time windows (seconds)")
    c3.info("**Resolution:** 1 epoch = 1 second")


# ── Results rendering ─────────────────────────────────────────────────────────

def _render_results(result, edf_path: Path | None, channel_idx: int, show_eeg: bool) -> None:
    """Render the full analysis results panel."""

    # ── Verdict ──────────────────────────────────────────────────────────────
    if result.seizure_detected:
        n = len(result.seizures)
        total_s = sum(s.duration_seconds for s in result.seizures)
        st.markdown(
            f"""
            <div class="verdict-seizure">
                <h2>⚠ SEIZURE DETECTED</h2>
                <p>{n} seizure period{'s' if n > 1 else ''} identified &nbsp;·&nbsp;
                   Total ictal duration: <strong>{total_s}s</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="verdict-normal">
                <h2>✓ NO SEIZURE DETECTED</h2>
                <p>The classifier found no ictal activity in this recording.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recording duration", f"{result.recording_seconds:,} s")
    m2.metric("Epochs analysed", f"{result.n_epochs:,}")
    m3.metric("Seizures detected", str(len(result.seizures)))
    ictal_pct = (
        100.0 * int(result.per_epoch["predicted"].sum()) / result.n_epochs
        if result.n_epochs > 0
        else 0.0
    )
    m4.metric("Ictal fraction", f"{ictal_pct:.1f}%")

    st.markdown("<hr class='light'>", unsafe_allow_html=True)

    # ── Seizure window table ──────────────────────────────────────────────────
    st.markdown("#### Seizure windows (from → to)")
    if result.seizures:
        for html in seizure_summary_html(result.seizures):
            st.markdown(f'<div class="sz-window">{html}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        table_rows = [
            {
                "Seizure #": f"#{i}",
                "Start (s)": sz.start_seconds,
                "End (s)": sz.end_seconds,
                "Duration (s)": sz.duration_seconds,
                "Start epoch": sz.start_epoch,
                "End epoch": sz.end_epoch,
            }
            for i, sz in enumerate(result.seizures, 1)
        ]
        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Start (s)": st.column_config.NumberColumn(format="%d s"),
                "End (s)": st.column_config.NumberColumn(format="%d s"),
                "Duration (s)": st.column_config.NumberColumn(format="%d s"),
            },
        )
    else:
        st.info("No seizure windows detected in this recording.")

    st.markdown("<hr class='light'>", unsafe_allow_html=True)

    # ── Visualisation tabs ────────────────────────────────────────────────────
    st.markdown("#### Visualisations")
    tab_gantt, tab_prob, tab_state, tab_eeg, tab_raster = st.tabs(
        ["Seizure windows", "Probability", "State timeline", "EEG signal", "Epoch raster"]
    )

    with tab_gantt:
        st.caption(
            "Each bar shows one detected seizure window against the full recording length."
        )
        st.plotly_chart(
            fig_seizure_gantt(result.seizures, result.recording_seconds),
            use_container_width=True,
        )

    with tab_prob:
        st.caption(
            "Per-second seizure probability from the XGBoost classifier.  "
            "Red shading marks predicted ictal periods."
        )
        st.plotly_chart(
            fig_probability_timeline(result.per_epoch, result.seizures),
            use_container_width=True,
        )

    with tab_state:
        st.caption(
            "Binary ictal / interictal state along the full recording."
        )
        st.plotly_chart(
            fig_state_timeline(result.per_epoch, result.seizures),
            use_container_width=True,
        )

    with tab_eeg:
        if show_eeg and edf_path and edf_path.exists():
            st.caption(
                f"Raw EEG channel {channel_idx} with predicted seizure windows highlighted "
                "in red.  Signal is down-sampled for display."
            )
            fig = fig_eeg_with_seizures(edf_path, result.seizures, channel_idx=channel_idx)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("EEG preview unavailable — check that the EDF file is valid.")
        else:
            st.info(
                'Enable **Show EEG signal overlay** in the sidebar, '
                "then re-run the analysis to see the raw waveform."
            )

    with tab_raster:
        st.caption(
            "Each dot is one epoch (1 second).  "
            "Crimson dots are predicted ictal; grey dots are predicted interictal."
        )
        st.plotly_chart(fig_epoch_raster(result.per_epoch), use_container_width=True)

    # ── Per-epoch detail table (collapsed by default) ─────────────────────────
    with st.expander("Per-epoch prediction detail"):
        st.dataframe(result.per_epoch, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<hr class='light'>", unsafe_allow_html=True)
    col_dl, col_rpt = st.columns([2, 3])
    with col_dl:
        st.download_button(
            label="Download predictions (CSV)",
            data=result.per_epoch.to_csv(index=False),
            file_name="seizure_detection_result.csv",
            mime="text/csv",
        )
    with col_rpt:
        st.download_button(
            label="Download text report",
            data=result.report,
            file_name="seizure_detection_report.txt",
            mime="text/plain",
        )


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    """Main Streamlit entry point."""

    # ── Sidebar ───────────────────────────────────────────────────────────────
    model_path, channel_idx, show_eeg = _render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<p class="app-title">Epilepsy Seizure Detection</p>'
        '<p class="app-subtitle">Upload an EEG recording to detect and visualise ictal periods '
        "using a pre-trained XGBoost classifier.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='light'>", unsafe_allow_html=True)

    # ── Upload panel ──────────────────────────────────────────────────────────
    st.markdown('<p class="step-label">Step 2 · Upload EDF recording</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "EDF file",
        type=["edf"],
        label_visibility="collapsed",
        help="European Data Format scalp EEG recording (.edf)",
    )

    # Show lightweight recording metadata immediately after upload
    if uploaded is not None:
        info = _recording_info(uploaded.getvalue(), uploaded.name)
        if info:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("File", uploaded.name)
            c2.metric("Channels", info["n_channels"])
            c3.metric("Sample rate", f"{info['sample_rate']} Hz")
            c4.metric("Duration", f"{info['duration_seconds']:.1f} s")

    # ── Analyse button ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="step-label">Step 3 · Run analysis</p>', unsafe_allow_html=True)
    run = st.button("Analyse Recording", type="primary", use_container_width=False)

    # ── Show existing result without re-running ───────────────────────────────
    if st.session_state.get("result") and not run:
        st.markdown("<hr class='light'>", unsafe_allow_html=True)
        st.markdown('<p class="step-label">Step 4 · Results</p>', unsafe_allow_html=True)
        edf_cached = st.session_state.get("edf_temp_path")
        _render_results(
            st.session_state["result"],
            Path(edf_cached) if edf_cached else None,
            channel_idx,
            show_eeg,
        )
        return

    if not run:
        _render_welcome()
        return

    # ── Validation ────────────────────────────────────────────────────────────
    if uploaded is None:
        st.error("Please upload an EDF file.")
        return

    if not model_path.exists():
        st.error(
            f"Model not found at `{model_path}`.  "
            "Train the model with the research notebook and place "
            "`seizure_model.joblib` in the `models/` folder."
        )
        return

    # ── Run detection ─────────────────────────────────────────────────────────
    progress = st.progress(0, text="Preparing…")
    try:
        # Clean up any previous temp file
        old = st.session_state.pop("edf_temp_path", None)
        if old:
            Path(old).unlink(missing_ok=True)

        progress.progress(10, text="Writing EDF to disk…")
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        progress.progress(30, text="Extracting features (this may take a moment)…")
        pipeline = _get_pipeline()

        progress.progress(60, text="Running XGBoost classifier…")
        result = pipeline.detect_from_edf(tmp_path, model_path)

        progress.progress(90, text="Generating charts…")
        st.session_state["result"] = result
        st.session_state["edf_temp_path"] = str(tmp_path)

        progress.progress(100, text="Done.")
        progress.empty()

    except Exception as exc:
        progress.empty()
        st.error(f"Analysis failed: {exc}")
        return

    # ── Render results ────────────────────────────────────────────────────────
    st.markdown("<hr class='light'>", unsafe_allow_html=True)
    st.markdown('<p class="step-label">Step 4 · Results</p>', unsafe_allow_html=True)
    _render_results(result, tmp_path, channel_idx, show_eeg)


if __name__ == "__main__":
    main()
