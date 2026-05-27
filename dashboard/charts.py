"""
Plotly chart builders for the epilepsy detection dashboard.

Each function returns a :class:`plotly.graph_objects.Figure` that is rendered
directly by Streamlit with ``st.plotly_chart(fig, use_container_width=True)``.

Design conventions:
    - Ictal (seizure) regions: crimson ``#b91c1c``
    - Interictal (normal) regions: slate blue ``#64748b``
    - Probability curve: deep blue ``#1d4ed8``
    - Background panels: dark navy ``#0f2137``
    - Annotations use the same crimson for consistency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from epilepsy_detection.detection.intervals import DetectedSeizure

# ── Colour palette ──────────────────────────────────────────────────────────
_ICTAL_SOLID = "#b91c1c"
_ICTAL_FILL = "rgba(185, 28, 28, 0.18)"
_ICTAL_FILL_STRONG = "rgba(185, 28, 28, 0.45)"
_INTERICTAL_FILL = "rgba(100, 116, 139, 0.25)"
_PROB_LINE = "#1d4ed8"
_BG = "#0f2137"
_GRID = "#1e3a5f"
_PAPER = "#0a1628"
_TEXT = "#e2e8f0"

# ── Shared layout defaults ───────────────────────────────────────────────────
_BASE_LAYOUT = dict(
    plot_bgcolor=_BG,
    paper_bgcolor=_PAPER,
    font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=_TEXT),
    xaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False),
    margin=dict(t=50, b=40, l=60, r=30),
    hovermode="x unified",
)


# ── Summary cards ────────────────────────────────────────────────────────────

def seizure_summary_html(seizures: list[DetectedSeizure]) -> list[str]:
    """Return an HTML string per seizure for dashboard banner rendering.

    Each string contains bold time labels and a duration badge suitable
    for injection into a styled ``<div>``.
    """
    rows = []
    for i, sz in enumerate(seizures, 1):
        rows.append(
            f"<b>Seizure {i}</b>&nbsp;&nbsp;"
            f"<span style='color:#b91c1c; font-weight:600'>"
            f"{sz.start_seconds}s &rarr; {sz.end_seconds}s"
            f"</span>"
            f"&nbsp;&nbsp;<span style='color:#6b7280; font-size:0.92em'>"
            f"Duration: {sz.duration_seconds}s &nbsp;|&nbsp; "
            f"Epochs {sz.start_epoch}&ndash;{sz.end_epoch}"
            f"</span>"
        )
    return rows


# ── Chart 1: Gantt-style seizure windows ─────────────────────────────────────

def fig_seizure_gantt(
    seizures: list[DetectedSeizure],
    recording_seconds: int,
) -> go.Figure:
    """Horizontal Gantt chart showing each seizure window against the full recording.

    The top bar represents the entire recording duration in light grey.
    Each additional bar is one detected seizure, coloured in crimson with
    the from-to label rendered inside the bar.

    Args:
        seizures: Detected seizure windows.
        recording_seconds: Total recording duration in seconds.

    Returns:
        Plotly :class:`~plotly.graph_objects.Figure`.
    """
    fig = go.Figure()

    # Full recording background bar
    fig.add_trace(
        go.Bar(
            x=[max(recording_seconds, 1)],
            y=["Recording"],
            orientation="h",
            marker=dict(color="#1e3a5f", line=dict(color="#334155", width=1)),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if not seizures:
        fig.update_layout(
            title=dict(text="Seizure windows - none detected", font=dict(size=14)),
            xaxis_title="Time (seconds)",
            height=160,
            **_BASE_LAYOUT,
        )
        return fig

    for i, sz in enumerate(seizures, 1):
        label = f"{sz.start_seconds}s – {sz.end_seconds}s"
        fig.add_trace(
            go.Bar(
                x=[sz.duration_seconds],
                y=[f"Seizure {i}"],
                orientation="h",
                base=sz.start_seconds,
                marker=dict(
                    color=_ICTAL_SOLID,
                    line=dict(color="#7f1d1d", width=1),
                ),
                text=label,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white", size=12, family="Inter, sans-serif"),
                name=f"Seizure {i}",
                hovertemplate=(
                    f"<b>Seizure {i}</b><br>"
                    f"Start: {sz.start_seconds}s<br>"
                    f"End: {sz.end_seconds}s<br>"
                    f"Duration: {sz.duration_seconds}s"
                    "<extra></extra>"
                ),
            )
        )

    row_order = ["Recording"] + [f"Seizure {i}" for i in range(len(seizures), 0, -1)]
    fig.update_layout(
        title=dict(text="Predicted seizure windows", font=dict(size=14)),
        xaxis_title="Time (seconds)",
        barmode="overlay",
        height=max(200, 100 + 52 * (len(seizures) + 1)),
        showlegend=False,
        **_BASE_LAYOUT,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=row_order,
    )
    return fig


# ── Chart 2: Full-recording state timeline ───────────────────────────────────

def fig_state_timeline(
    df: pd.DataFrame,
    seizures: list[DetectedSeizure],
) -> go.Figure:
    """Area chart showing interictal and ictal state along the full recording.

    Args:
        df: Per-epoch prediction DataFrame (must have ``time_seconds`` and
            ``predicted`` columns).
        seizures: Detected seizure windows for red-band annotations.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    # Interictal fill (always present)
    fig.add_trace(
        go.Scatter(
            x=df["time_seconds"],
            y=df["predicted"],
            mode="lines",
            line=dict(color="#64748b", width=0),
            fill="tozeroy",
            fillcolor=_INTERICTAL_FILL,
            name="Interictal",
            hovertemplate="Time: %{x}s | State: interictal<extra></extra>",
        )
    )

    # Ictal fill (only for positive epochs)
    ictal_df = df[df["predicted"] == 1]
    if not ictal_df.empty:
        fig.add_trace(
            go.Scatter(
                x=ictal_df["time_seconds"],
                y=ictal_df["predicted"],
                mode="markers",
                marker=dict(color=_ICTAL_SOLID, size=4, symbol="square"),
                fill="tozeroy",
                fillcolor=_ICTAL_FILL_STRONG,
                name="Ictal",
                hovertemplate="Time: %{x}s | State: <b>ICTAL</b><extra></extra>",
            )
        )

    # Vertical bands and labels for each seizure
    for i, sz in enumerate(seizures, 1):
        fig.add_vrect(
            x0=sz.start_seconds,
            x1=sz.end_seconds,
            fillcolor=_ICTAL_FILL,
            line_width=1.5,
            line_color=_ICTAL_SOLID,
            annotation_text=f"S{i}",
            annotation_position="top left",
            annotation_font=dict(color=_ICTAL_SOLID, size=11),
        )

    fig.update_layout(
        title=dict(text="Ictal / interictal classification along the recording", font=dict(size=14)),
        xaxis_title="Time (seconds)",
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=["Interictal", "Ictal"],
            range=[-0.08, 1.25],
            gridcolor=_GRID,
        ),
        height=280,
        **{k: v for k, v in _BASE_LAYOUT.items() if k != "yaxis"},
    )
    return fig


# ── Chart 3: Seizure probability over time ───────────────────────────────────

def fig_probability_timeline(
    df: pd.DataFrame,
    seizures: list[DetectedSeizure],
) -> go.Figure:
    """Line chart of per-epoch seizure probability with shaded seizure bands.

    Args:
        df: Per-epoch DataFrame.  Uses ``probability_seizure`` when available;
            falls back to the binary ``predicted`` column.
        seizures: Detected seizure windows.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    y_col = "probability_seizure" if "probability_seizure" in df.columns else "predicted"
    y_label = "Seizure probability" if y_col == "probability_seizure" else "Predicted class"

    fig.add_trace(
        go.Scatter(
            x=df["time_seconds"],
            y=df[y_col],
            mode="lines",
            line=dict(color=_PROB_LINE, width=1.8),
            name=y_label,
            hovertemplate="Time: %{x}s<br>Probability: %{y:.3f}<extra></extra>",
        )
    )

    # Decision threshold reference line
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="threshold (0.5)",
        annotation_font=dict(color="#94a3b8", size=10),
    )

    # Shaded seizure windows with time labels
    for i, sz in enumerate(seizures, 1):
        fig.add_vrect(
            x0=sz.start_seconds,
            x1=sz.end_seconds,
            fillcolor=_ICTAL_FILL,
            line_width=0,
            annotation_text=f"{sz.start_seconds}s–{sz.end_seconds}s",
            annotation_position="top center",
            annotation_font=dict(color=_ICTAL_SOLID, size=10),
        )

    fig.update_layout(
        title=dict(text="Seizure probability per second", font=dict(size=14)),
        xaxis_title="Time (seconds)",
        yaxis_title=y_label,
        yaxis=dict(range=[-0.05, 1.1], gridcolor=_GRID),
        height=320,
        **{k: v for k, v in _BASE_LAYOUT.items() if k != "yaxis"},
    )
    return fig


# ── Chart 4: Raw EEG with seizure overlay ────────────────────────────────────

def fig_eeg_with_seizures(
    edf_path: str | Path,
    seizures: list[DetectedSeizure],
    channel_idx: int = 0,
    max_points: int = 10_000,
) -> go.Figure | None:
    """Plot a single EEG channel with predicted seizure windows highlighted.

    The raw signal is down-sampled for display when it exceeds *max_points*
    so the browser renders it responsively without losing the overall shape.

    Args:
        edf_path: Path to the source ``.edf`` file.
        seizures: Detected seizure windows.
        channel_idx: Zero-based channel index to plot (default 0 = first channel).
        max_points: Maximum number of data points rendered on screen.

    Returns:
        Plotly figure, or ``None`` if the file is unavailable.
    """
    try:
        import mne  # imported lazily; not needed in unit tests
    except ImportError:
        return None

    path = Path(edf_path)
    if not path.exists():
        return None

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    channel_idx = min(channel_idx, len(raw.ch_names) - 1)
    ch_name = raw.ch_names[channel_idx]
    data, times = raw.get_data(picks=[channel_idx], return_times=True)
    signal = data[0]

    # Downsample for display performance
    if len(signal) > max_points:
        step = len(signal) // max_points
        signal = signal[::step]
        times = times[::step]

    # Convert amplitude to microvolts for readability
    signal_uv = signal * 1e6

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=signal_uv,
            mode="lines",
            line=dict(color="#1e40af", width=0.9),
            name=ch_name,
            hovertemplate="Time: %{x:.1f}s<br>Amplitude: %{y:.2f} uV<extra></extra>",
        )
    )

    for i, sz in enumerate(seizures, 1):
        fig.add_vrect(
            x0=sz.start_seconds,
            x1=sz.end_seconds,
            fillcolor="rgba(185, 28, 28, 0.22)",
            line_width=2,
            line_color=_ICTAL_SOLID,
            layer="below",
            annotation_text=f"Seizure {i}: {sz.start_seconds}s–{sz.end_seconds}s",
            annotation_position="top left",
            annotation_font=dict(color=_ICTAL_SOLID, size=10, family="Inter, sans-serif"),
        )

    fig.update_layout(
        title=dict(
            text=f"EEG signal - channel <b>{ch_name}</b> with predicted seizure regions",
            font=dict(size=14),
        ),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (uV)",
        height=400,
        **_BASE_LAYOUT,
    )
    return fig


# ── Chart 5: Epoch raster / dot map ──────────────────────────────────────────

def fig_epoch_raster(df: pd.DataFrame) -> go.Figure:
    """Compact single-row raster plot coloured by predicted class.

    Each dot is one epoch (1 second).  Crimson dots are predicted ictal;
    grey dots are predicted interictal.  Useful for a quick overview of
    seizure density and distribution.

    Args:
        df: Per-epoch prediction DataFrame.

    Returns:
        Plotly figure.
    """
    plot_df = df[["time_seconds", "predicted"]].copy()
    plot_df["State"] = plot_df["predicted"].map({0: "Interictal", 1: "Ictal"})

    fig = px.scatter(
        plot_df,
        x="time_seconds",
        y=plot_df["predicted"].map({0: 0, 1: 0}),
        color="State",
        color_discrete_map={"Interictal": "#94a3b8", "Ictal": _ICTAL_SOLID},
        labels={"time_seconds": "Time (seconds)"},
        title="Epoch raster - each dot = 1 second",
    )
    fig.update_traces(marker=dict(size=7, symbol="circle", opacity=0.85))
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(
        height=130,
        margin=dict(t=45, b=30, l=30, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor=_BG,
        paper_bgcolor=_PAPER,
    )
    return fig
