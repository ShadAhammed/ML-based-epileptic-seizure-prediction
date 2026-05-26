"""Plotly and EEG charts for the seizure detection dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from epilepsy_detection.detection.intervals import DetectedSeizure


def seizure_summary_cards(seizures: list[DetectedSeizure]) -> list[str]:
    """HTML-friendly one-line summaries per seizure."""
    return [
        f"**Seizure {i}:** {s.start_seconds}s → {s.end_seconds}s "
        f"({s.duration_seconds}s, epochs {s.start_epoch}–{s.end_epoch})"
        for i, s in enumerate(seizures, 1)
    ]


def fig_seizure_gantt(seizures: list[DetectedSeizure], recording_seconds: int) -> go.Figure:
    """Gantt-style chart: when–when each predicted seizure occurs."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[max(recording_seconds, 1)],
            y=["Full recording"],
            orientation="h",
            marker_color="#e2e8f0",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if not seizures:
        fig.update_layout(
            title="Predicted seizure windows — none detected",
            xaxis_title="Time (seconds)",
            height=200,
        )
        return fig

    for i, sz in enumerate(seizures, 1):
        fig.add_trace(
            go.Bar(
                x=[sz.duration_seconds],
                y=[f"Seizure {i}"],
                orientation="h",
                base=sz.start_seconds,
                marker_color="#dc2626",
                name=f"Seizure {i}",
                text=f"{sz.start_seconds}s – {sz.end_seconds}s",
                textposition="inside",
                hovertemplate=(
                    f"Seizure {i}<br>"
                    f"Start: {sz.start_seconds}s<br>"
                    f"End: {sz.end_seconds}s<br>"
                    f"Duration: {sz.duration_seconds}s<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Predicted seizure windows (when → when)",
        xaxis_title="Time (seconds)",
        barmode="overlay",
        height= max(220, 80 + 56 * max(len(seizures), 1)),
        showlegend=False,
        margin=dict(l=120),
    )
    fig.update_yaxes(categoryorder="array", categoryarray=["Full recording"] + [
        f"Seizure {i}" for i in range(len(seizures), 0, -1)
    ])
    return fig


def fig_state_timeline(df: pd.DataFrame, seizures: list[DetectedSeizure]) -> go.Figure:
    """Full-recording strip: red bands = predicted ictal, gray = interictal."""
    fig = go.Figure()
    times = df["time_seconds"].values
    state = df["predicted"].values

    fig.add_trace(
        go.Scatter(
            x=times,
            y=state,
            mode="lines",
            line=dict(color="#64748b", width=0),
            fill="tozeroy",
            fillcolor="rgba(148, 163, 184, 0.35)",
            name="Interictal (0)",
            hovertemplate="Time: %{x}s<br>State: interictal<extra></extra>",
        )
    )

    ictal = df[df["predicted"] == 1]
    if not ictal.empty:
        fig.add_trace(
            go.Scatter(
                x=ictal["time_seconds"],
                y=ictal["predicted"],
                mode="lines",
                line=dict(color="#dc2626", width=0),
                fill="tozeroy",
                fillcolor="rgba(220, 38, 38, 0.55)",
                name="Ictal (1)",
                hovertemplate="Time: %{x}s<br>State: ictal<extra></extra>",
            )
        )

    for i, sz in enumerate(seizures, 1):
        fig.add_vrect(
            x0=sz.start_seconds,
            x1=sz.end_seconds,
            fillcolor="rgba(220, 38, 38, 0.15)",
            line_width=2,
            line_color="#dc2626",
            annotation_text=f"S{i}",
            annotation_position="top left",
        )

    fig.update_layout(
        title="Recording timeline — predicted seizure regions highlighted",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=["Interictal", "Ictal"],
            range=[-0.1, 1.2],
        ),
        height=280,
        hovermode="x unified",
    )
    return fig


def fig_probability_timeline(df: pd.DataFrame, seizures: list[DetectedSeizure]) -> go.Figure:
    """Seizure probability over time with shaded when–when windows."""
    fig = go.Figure()
    y_col = "probability_seizure" if "probability_seizure" in df.columns else "predicted"
    y_label = "Seizure probability" if y_col == "probability_seizure" else "Predicted class"

    fig.add_trace(
        go.Scatter(
            x=df["time_seconds"],
            y=df[y_col],
            mode="lines",
            line=dict(color="#2563eb", width=1.5),
            name=y_label,
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8", annotation_text="threshold")

    for i, sz in enumerate(seizures, 1):
        fig.add_vrect(
            x0=sz.start_seconds,
            x1=sz.end_seconds,
            fillcolor="rgba(220, 38, 38, 0.25)",
            line_width=0,
            annotation_text=f"{sz.start_seconds}s–{sz.end_seconds}s",
            annotation_position="top center",
        )

    fig.update_layout(
        title="Per-second prediction with seizure intervals",
        xaxis_title="Time (seconds)",
        yaxis_title=y_label,
        height=320,
    )
    return fig


def fig_eeg_with_seizures(edf_path: str | Path, seizures: list[DetectedSeizure], max_points: int = 8000) -> go.Figure | None:
    """Plot first EEG channel with predicted seizure windows shaded."""
    try:
        import mne
    except ImportError:
        return None

    path = Path(edf_path)
    if not path.exists():
        return None

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    data, times = raw.get_data(picks=[0], return_times=True)
    signal = data[0]
    ch_name = raw.ch_names[0]

    if len(signal) > max_points:
        step = len(signal) // max_points
        signal = signal[::step]
        times = times[::step]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=signal,
            mode="lines",
            line=dict(color="#1e40af", width=0.8),
            name=ch_name,
        )
    )
    for i, sz in enumerate(seizures, 1):
        fig.add_vrect(
            x0=sz.start_seconds,
            x1=sz.end_seconds,
            fillcolor="rgba(220, 38, 38, 0.35)",
            line_width=2,
            line_color="#dc2626",
            layer="below",
            annotation_text=f"Seizure {i}: {sz.start_seconds}s–{sz.end_seconds}s",
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"EEG preview ({ch_name}) — predicted seizure periods",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (µV)",
        height=360,
    )
    return fig


def fig_epoch_map(df: pd.DataFrame) -> go.Figure:
    """Heatmap-style view of predicted state across epochs."""
    plot_df = df[["time_seconds", "predicted"]].copy()
    plot_df["state"] = plot_df["predicted"].map({0: "Interictal", 1: "Ictal"})
    fig = px.scatter(
        plot_df,
        x="time_seconds",
        y=[0] * len(plot_df),
        color="state",
        color_discrete_map={"Interictal": "#94a3b8", "Ictal": "#dc2626"},
        labels={"time_seconds": "Time (seconds)"},
        title="Epoch-level classification along the recording",
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_layout(height=120, margin=dict(t=60, b=40))
    return fig
