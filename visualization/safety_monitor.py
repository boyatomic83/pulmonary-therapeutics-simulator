"""
Therapeutic Window & Safety Hazard Monitor
Combines local target inhibition (Emax/Hill) and systemic blood exposure (Db).
Provides:
  - Efficacy Gauge (Sub-Therapeutic <50%, Therapeutic 50-90%, High >90%)
  - Systemic Blood Exposure Gauge with configurable safety threshold
  - Dual-axis Efficacy vs. Systemic Exposure time-series plot
All zones are explicitly model-defined simulation assumptions, not validated clinical limits.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

def create_efficacy_gauge(sim_result: Dict[str, Any]) -> go.Figure:
    """
    Creates an interactive Plotly gauge showing efficacy zones:
      - 0% to 50%: Sub-Therapeutic (Amber/Gray)
      - 50% to 90%: Therapeutic Simulation Window (Emerald Green)
      - 90% to 100%: High-Efficacy Zone (Blue)
    """
    metrics = sim_result["metrics"]
    peak_eff = float(metrics["max_efficacy"])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=peak_eff,
        number=dict(suffix="%", font=dict(size=28, color="#0F172A")),
        delta=dict(
            reference=50.0,
            increasing=dict(color="#10B981"),
            decreasing=dict(color="#F59E0B"),
            position="bottom",
        ),
        title=dict(
            text="<b>Peak Pulmonary Efficacy (E_max)</b><br><span style='font-size:0.8em;color:#64748B'>Model-defined zones</span>",
            font=dict(size=14, color="#1E293B"),
        ),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 90, 100],
                ticktext=["0%", "25%", "50%", "75%", "90%", "100%"],
                tickcolor="#64748B",
            ),
            bar=dict(color="#0F172A", thickness=0.3),
            bgcolor="#F8FAFC",
            borderwidth=1,
            bordercolor="#E2E8F0",
            steps=[
                dict(range=[0, 50], color="rgba(245, 158, 11, 0.25)"),
                dict(range=[50, 90], color="rgba(16, 185, 129, 0.35)"),
                dict(range=[90, 100], color="rgba(59, 130, 246, 0.30)"),
            ],
            threshold=dict(
                line=dict(color="#EF4444", width=3),
                thickness=0.85,
                value=50.0,
            ),
        ),
    ))
    
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=50, b=20),
        paper_bgcolor="#FFFFFF",
        font=dict(family="sans-serif"),
    )
    
    return fig

def create_systemic_exposure_gauge(sim_result: Dict[str, Any], safety_threshold_db: float = 5.0) -> go.Figure:
    """
    Creates an interactive gauge for systemic blood exposure (Db) relative to safety hazard threshold.
    """
    metrics = sim_result["metrics"]
    peak_db = float(metrics["peak_blood_conc"])
    
    max_range = float(max(safety_threshold_db * 1.5, peak_db * 1.25, 6.0))
    is_hazard = bool(peak_db > safety_threshold_db)
    bar_color = "#EF4444" if is_hazard else "#10B981"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=peak_db,
        number=dict(suffix=" mg/L", font=dict(size=28, color="#EF4444" if is_hazard else "#0F172A")),
        title=dict(
            text=f"<b>Peak Blood Exposure (max D_b)</b><br><span style='font-size:0.8em;color:#64748B'>Hazard Limit: {safety_threshold_db:.1f} mg/L</span>",
            font=dict(size=14, color="#1E293B"),
        ),
        gauge=dict(
            axis=dict(range=[0, max_range], tickcolor="#64748B"),
            bar=dict(color=bar_color, thickness=0.3),
            bgcolor="#F8FAFC",
            borderwidth=1,
            bordercolor="#E2E8F0",
            steps=[
                dict(range=[0, safety_threshold_db], color="rgba(16, 185, 129, 0.25)"),
                dict(range=[safety_threshold_db, max_range], color="rgba(239, 68, 68, 0.35)"),
            ],
            threshold=dict(
                line=dict(color="#DC2626", width=3),
                thickness=0.85,
                value=safety_threshold_db,
            ),
        ),
    ))
    
    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=50, b=20),
        paper_bgcolor="#FFFFFF",
        font=dict(family="sans-serif"),
    )
    
    return fig

def plot_efficacy_vs_exposure_timeseries(sim_result: Dict[str, Any], safety_threshold_db: float = 5.0) -> go.Figure:
    """
    Dual-axis time-series visualization displaying local pulmonary anti-inflammatory
    efficacy E(t) alongside systemic blood concentration Db(t).
    """
    times = sim_result["times"]
    efficacy = sim_result["efficacy"]
    db = sim_result["db"]
    
    fig = go.Figure()
    
    # Efficacy Line (Left Axis)
    fig.add_trace(go.Scatter(
        x=times,
        y=efficacy,
        mode="lines",
        name="Local Pulmonary Efficacy E(t)",
        line=dict(color="#059669", width=3),
        hovertemplate="Time: %{x:.2f} h<br>Efficacy: %{y:.1f}%<extra></extra>",
    ))
    
    # Systemic Blood Exposure (Right Axis)
    fig.add_trace(go.Scatter(
        x=times,
        y=db,
        mode="lines",
        name="Systemic Blood Exposure D_b(t)",
        line=dict(color="#DC2626", width=2.5, dash="dot"),
        yaxis="y2",
        hovertemplate="Time: %{x:.2f} h<br>Blood Exposure: %{y:.3f} mg/L<extra></extra>",
    ))
    
    # Shaded Therapeutic Window Band (50% to 90%)
    fig.add_hrect(
        y0=50.0,
        y1=90.0,
        fillcolor="rgba(16, 185, 129, 0.12)",
        line_width=0,
        annotation_text="Therapeutic Simulation Window (50% - 90%)",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="#047857",
    )
    
    # Horizontal reference for 50% threshold
    fig.add_hline(
        y=50.0,
        line=dict(color="#047857", width=1.5, dash="dash"),
    )
    
    # Safety Hazard line on secondary axis
    fig.add_hline(
        y=safety_threshold_db,
        line=dict(color="#991B1B", width=1.5, dash="dashdot"),
        yref="y2",
        annotation_text=f"Safety Hazard Threshold ({safety_threshold_db:.1f} mg/L)",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="#991B1B",
    )
    
    fig.update_layout(
        title=dict(
            text="<b>Therapeutic Window & Safety Hazard Monitor (0–24h)</b>",
            x=0.01,
            font=dict(size=14, color="#0F172A"),
        ),
        xaxis=dict(
            title="Time post-inhalation (hours)",
            showgrid=True,
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title="Local Pulmonary Efficacy (%)",
            range=[0, 105],
            showgrid=True,
            gridcolor="#F1F5F9",
            title_font=dict(color="#059669"),
            tickfont=dict(color="#059669"),
        ),
        yaxis2=dict(
            title="Systemic Blood Exposure D_b (mg/L)",
            overlaying="y",
            side="right",
            showgrid=False,
            title_font=dict(color="#DC2626"),
            tickfont=dict(color="#DC2626"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=50, t=50, b=40),
        height=380,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    
    return fig
