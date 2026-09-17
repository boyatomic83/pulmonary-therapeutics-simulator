"""
Plotly Visualizations for Inhaled Therapeutics Simulator
  1. plot_pk_profiles: 4-compartment PK profile (Nm, Dm, Dt, Db) over 0-24 hours
  2. plot_pd_response: Local anti-inflammatory efficacy with 50% therapeutic threshold line
  3. plot_deposition_breakdown: Regional deposition breakdown bar chart
  4. plot_mmad_curve: Continuous MMAD vs regional deposition curve with current marker
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any

from models.aerosol import generate_deposition_curve

# Clean scientific color palette
COLOR_NM = "#2563EB"    # Blue - Nanoparticle in Mucus
COLOR_DM = "#0D9488"    # Teal - Free Drug in Mucus
COLOR_DT = "#D97706"    # Amber - Target Epithelial Tissue
COLOR_DB = "#DC2626"    # Red - Systemic Blood Exposure
COLOR_EFF = "#16A34A"   # Emerald Green - PD Efficacy
COLOR_REF = "#94A3B8"   # Slate - Reference lines

LAYOUT_DEFAULTS = dict(
    font=dict(family="system-ui, -apple-system, sans-serif", size=12, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    margin=dict(l=55, r=25, t=45, b=45),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#E2E8F0",
        borderwidth=1,
    ),
)

def plot_pk_profiles(sim_result: Dict[str, Any]) -> go.Figure:
    """
    Visualization 1: Four-Compartment PK Profile (0-24 hours).
    Plots trajectories of Nm, Dm, Dt, and Db.
    """
    times = sim_result["times"]
    nm = sim_result["nm"]
    dm = sim_result["dm"]
    dt = sim_result["dt"]
    db = sim_result["db"]
    
    fig = go.Figure()
    
    # Nm: Nanoparticle Reservoir in Mucus
    fig.add_trace(go.Scatter(
        x=times,
        y=nm,
        mode="lines",
        name="Nm: Nanoparticle Payload (Mucus)",
        line=dict(color=COLOR_NM, width=2.8),
        hovertemplate="<b>Nm (Nanoparticles)</b>: %{y:.3f} mg<extra></extra>",
    ))
    
    # Dm: Free Drug in Mucus
    fig.add_trace(go.Scatter(
        x=times,
        y=dm,
        mode="lines",
        name="Dm: Free Drug (Mucus)",
        line=dict(color=COLOR_DM, width=2.5, dash="dot"),
        hovertemplate="<b>Dm (Free in Mucus)</b>: %{y:.3f} mg<extra></extra>",
    ))
    
    # Dt: Target Epithelial Tissue
    fig.add_trace(go.Scatter(
        x=times,
        y=dt,
        mode="lines",
        name="Dt: Target Epithelial Tissue",
        line=dict(color=COLOR_DT, width=3.2),
        hovertemplate="<b>Dt (Target Tissue)</b>: %{y:.3f} mg<extra></extra>",
    ))
    
    # Db: Systemic Blood Exposure
    fig.add_trace(go.Scatter(
        x=times,
        y=db,
        mode="lines",
        name="Db: Systemic Blood Exposure",
        line=dict(color=COLOR_DB, width=2.2, dash="dash"),
        hovertemplate="<b>Db (Systemic Blood)</b>: %{y:.3f} mg<extra></extra>",
    ))
    
    # Peak tissue marker
    t_max = sim_result["metrics"]["t_max_tissue"]
    c_max = sim_result["metrics"]["peak_tissue_conc"]
    fig.add_trace(go.Scatter(
        x=[t_max],
        y=[c_max],
        mode="markers+text",
        name="Peak Tissue (Cmax)",
        marker=dict(color=COLOR_DT, size=9, symbol="diamond-open", line=dict(width=2)),
        text=[f"Peak: {c_max:.2f} mg @ {t_max:.1f}h"],
        textposition="top right",
        showlegend=False,
        hoverinfo="skip",
    ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="<b>Four-Compartment Pulmonary & Systemic PK Profile</b>",
            font=dict(size=14, color="#1E293B"),
            x=0.0,
        ),
        xaxis=dict(
            title="<b>Time (hours)</b>",
            gridcolor="#F1F5F9",
            zeroline=False,
            range=[0, times[-1]],
            tickmode="linear",
            dtick=4,
        ),
        yaxis=dict(
            title="<b>Amount / Concentration (mg)</b>",
            gridcolor="#F1F5F9",
            zeroline=True,
            zerolinecolor="#E2E8F0",
        ),
    )
    
    return fig

def plot_pd_response(sim_result: Dict[str, Any]) -> go.Figure:
    """
    Visualization 2: Pulmonary Pharmacodynamic Response (0-24 hours).
    Plots local anti-inflammatory efficacy (%) and 50% therapeutic threshold line.
    """
    times = sim_result["times"]
    efficacy = sim_result["efficacy"]
    
    fig = go.Figure()
    
    # 50% Therapeutic Threshold Line
    fig.add_trace(go.Scatter(
        x=[0, times[-1]],
        y=[50, 50],
        mode="lines",
        name="50% Therapeutic Threshold",
        line=dict(color="#EF4444", width=1.8, dash="dash"),
        hoverinfo="skip",
    ))
    
    # Shaded region under curve
    fig.add_trace(go.Scatter(
        x=times,
        y=efficacy,
        mode="lines",
        name="Local Anti-Inflammatory Efficacy E(t)",
        line=dict(color=COLOR_EFF, width=3.0),
        fill="tozeroy",
        fillcolor="rgba(22, 163, 74, 0.08)",
        hovertemplate="<b>Efficacy E(Dt)</b>: %{y:.1f}%<br>Time: %{x:.1f} h<extra></extra>",
    ))
    
    max_eff = sim_result["metrics"]["max_efficacy"]
    max_idx = int(np.argmax(efficacy))
    t_peak = float(times[max_idx])
    
    fig.add_trace(go.Scatter(
        x=[t_peak],
        y=[max_eff],
        mode="markers+text",
        name="Max Efficacy",
        marker=dict(color=COLOR_EFF, size=8, symbol="circle"),
        text=[f"Max: {max_eff:.1f}%"],
        textposition="top center",
        showlegend=False,
        hoverinfo="skip",
    ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="<b>Pulmonary Pharmacodynamic Response: Local Anti-Inflammatory Effect</b>",
            font=dict(size=14, color="#1E293B"),
            x=0.0,
        ),
        xaxis=dict(
            title="<b>Time (hours)</b>",
            gridcolor="#F1F5F9",
            zeroline=False,
            range=[0, times[-1]],
            tickmode="linear",
            dtick=4,
        ),
        yaxis=dict(
            title="<b>Efficacy E(Dt) (%)</b>",
            gridcolor="#F1F5F9",
            range=[0, 105],
            zeroline=True,
            zerolinecolor="#E2E8F0",
        ),
    )
    
    return fig

def plot_deposition_breakdown(deposition: Dict[str, float], inhaled_dose: float) -> go.Figure:
    """
    Visualization 3: Regional Deposition Breakdown across Upper, Bronchial, and Alveolar regions.
    """
    regions = ["Upper Airways<br><i>(Oropharynx / Larynx)</i>", "Conducting Bronchial<br><i>(Tracheobronchial)</i>", "Deep Alveolar<br><i>(Pulmonary Acinus)</i>"]
    percentages = [
        deposition["f_upper"] * 100.0,
        deposition["f_bronchial"] * 100.0,
        deposition["f_alveolar"] * 100.0,
    ]
    doses = [
        deposition["upper_dose"],
        deposition["bronchial_dose"],
        deposition["alveolar_dose"],
    ]
    colors = ["#94A3B8", "#0284C7", "#059669"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=regions,
        y=percentages,
        text=[f"<b>{p:.1f}%</b><br>({d:.2f} mg)" for p, d in zip(percentages, doses)],
        textposition="inside",
        marker=dict(color=colors, line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Fraction: %{y:.1f}%<extra></extra>",
    ))
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="<b>Regional Inhaled Particle Deposition Fractions</b>",
            font=dict(size=14, color="#1E293B"),
            x=0.0,
        ),
        yaxis=dict(
            title="<b>Deposition Fraction (%)</b>",
            gridcolor="#F1F5F9",
            range=[0, 100],
        ),
        xaxis=dict(gridcolor="#FFFFFF"),
        showlegend=False,
        height=320,
    )
    
    return fig

def plot_mmad_curve(current_mmad: float) -> go.Figure:
    """
    Continuous Aerosol Deposition vs. MMAD (0.5 to 10.0 um).
    Shows how particle size governs impaction vs. alveolar sedimentation.
    """
    curve_data = generate_deposition_curve(points=80)
    mmad = curve_data["mmad"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=mmad,
        y=curve_data["upper_pct"],
        mode="lines",
        name="Upper Airways (Impaction)",
        line=dict(color="#94A3B8", width=2.2),
        hovertemplate="MMAD: %{x:.2f} um<br>Upper: %{y:.1f}%<extra></extra>",
    ))
    
    fig.add_trace(go.Scatter(
        x=mmad,
        y=curve_data["bronchial_pct"],
        mode="lines",
        name="Conducting Bronchial",
        line=dict(color="#0284C7", width=2.2),
        hovertemplate="MMAD: %{x:.2f} um<br>Bronchial: %{y:.1f}%<extra></extra>",
    ))
    
    fig.add_trace(go.Scatter(
        x=mmad,
        y=curve_data["alveolar_pct"],
        mode="lines",
        name="Deep Alveolar (Sedimentation)",
        line=dict(color="#059669", width=2.8),
        hovertemplate="MMAD: %{x:.2f} um<br>Alveolar: %{y:.1f}%<extra></extra>",
    ))
    
    # Vertical line at current MMAD
    fig.add_vline(
        x=current_mmad,
        line_width=2,
        line_dash="dash",
        line_color="#E11D48",
        annotation_text=f"Selected MMAD: {current_mmad:.1f} μm",
        annotation_position="top left",
        annotation_font=dict(color="#BE123C", size=11),
    )
    
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text="<b>Aerosol Deposition Efficiency vs. Particle Diameter (MMAD)</b>",
            font=dict(size=14, color="#1E293B"),
            x=0.0,
        ),
        xaxis=dict(
            title="<b>Mass Median Aerodynamic Diameter — MMAD (μm)</b>",
            gridcolor="#F1F5F9",
            range=[0.5, 10.0],
            dtick=1.0,
        ),
        yaxis=dict(
            title="<b>Deposition Fraction (%)</b>",
            gridcolor="#F1F5F9",
            range=[0, 100],
        ),
        height=320,
    )
    
    return fig
