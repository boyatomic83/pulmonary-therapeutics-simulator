"""
Inhaled Nanoparticle & Biological Respiratory Therapeutics Simulator
Integrates:
  1. Aerosol Deposition Physics (MMAD 0.5–10.0 um, inertial impaction, sedimentation, airway caliber)
  2. Four-Compartment Pulmonary Mucosal & Tissue PK Model (solve_ivp RK45)
  3. Disease-State Physiological Presets (Healthy Adult, Cystic Fibrosis, Asthma/COPD, Pediatric, Custom)
  4. Local Anti-Inflammatory Pharmacodynamics (Emax Hill equation)
  5. Interactive 3D Airway & Particle Deposition Visualizer (G0–G23 dichotomous branching)
  6. Therapeutic Window & Safety Hazard Monitor (Efficacy gauge, systemic blood exposure hazard)
  7. Simulation Dossier Export (Pandas CSV generation)

Notice: All disease parameters, kinetics, and safety limits are simulation assumptions / illustrative
model parameters, and do NOT represent validated clinical dosing or toxicity recommendations.
"""

import streamlit as st
import numpy as np
import pandas as pd

from models.presets import PRESETS, get_preset
from models.aerosol import calculate_regional_deposition, generate_deposition_curve
from simulation.solver import run_simulation
from visualization.plots import (
    plot_pk_profiles,
    plot_pd_response,
    plot_deposition_breakdown,
    plot_mmad_curve,
)
from visualization.airway_3d import plot_airway_3d_deposition
from visualization.safety_monitor import (
    create_efficacy_gauge,
    create_systemic_exposure_gauge,
    plot_efficacy_vs_exposure_timeseries,
)
from utils.export import export_simulation_dossier_csv, generate_simulation_dossier_dataframe

# Streamlit Page Setup
st.set_page_config(
    page_title="Inhaled Therapeutics Simulator",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Styling
st.markdown(
    """
    <style>
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    /* Section headers */
    h2, h3 {
        color: #1E293B;
        font-weight: 700;
    }
    /* Warning Banner Styling */
    .warning-banner {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        color: #991B1B;
    }
    .safe-banner {
        background-color: #ECFDF5;
        border: 1.5px solid #10B981;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
        color: #065F46;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("🫁 Inhaled Nanoparticle & Biological Respiratory Therapeutics Simulator")
st.caption(
    "Aerosol Deposition Physics • 4-Compartment Pulmonary Mucosal PK • Disease Pathological Modifiers • "
    "Local Efficacy & Safety Hazard Monitoring"
)

# Context / Educational Notice
with st.expander("ℹ️ Scientific Simulation & Illustrative Modeling Assumptions Notice", expanded=False):
    st.markdown(
        """
        **Scientific Simulation Notice:** This platform is an educational and computational simulation.
        All physiological presets, rate parameters ($k_{\\text{release}}, k_{\\text{clear}}, k_{\\text{diff}}$),
        mucus viscosity scaling factors ($V$), and safety hazard cutoffs ($D_b > 5.0\\ \\text{mg/L}$)
        are **illustrative simulation assumptions**. They do not constitute validated clinical dosing,
        pharmacotherapy protocols, or regulatory safety criteria.
        """
    )

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown("### ⚙️ Simulation Controls")

# 1. Clinical / Disease-State Simulation Preset at Top of Sidebar
preset_options = [
    "Healthy Adult",
    "Cystic Fibrosis (CF)",
    "Severe Asthma / COPD",
    "Infant / Pediatric",
    "Custom",
]

# Track preset changes in session state
if "current_preset" not in st.session_state:
    st.session_state["current_preset"] = "Healthy Adult"

selected_preset_name = st.sidebar.selectbox(
    "Simulation Preset",
    options=preset_options,
    index=preset_options.index(st.session_state["current_preset"]),
    help="Select a clinically inspired physiological scenario or customize parameters manually.",
)

preset_def = get_preset(selected_preset_name)

# If preset selection changed, update preset stored
if selected_preset_name != st.session_state["current_preset"]:
    st.session_state["current_preset"] = selected_preset_name
    st.session_state["mmad"] = preset_def.mmad
    st.session_state["dose"] = preset_def.inhaled_dose
    st.session_state["viscosity"] = preset_def.mucus_viscosity_factor
    st.session_state["clearance_factor"] = preset_def.mucociliary_clearance_factor
    st.session_state["airway_radius"] = preset_def.airway_radius_factor
    st.session_state["tidal_volume"] = preset_def.tidal_volume_factor
    st.session_state["k_release"] = preset_def.k_release
    st.session_state["safety_thresh"] = preset_def.safety_threshold_db

# Initialize session state values if not present
if "mmad" not in st.session_state:
    st.session_state["mmad"] = preset_def.mmad
if "dose" not in st.session_state:
    st.session_state["dose"] = preset_def.inhaled_dose
if "viscosity" not in st.session_state:
    st.session_state["viscosity"] = preset_def.mucus_viscosity_factor
if "clearance_factor" not in st.session_state:
    st.session_state["clearance_factor"] = preset_def.mucociliary_clearance_factor
if "airway_radius" not in st.session_state:
    st.session_state["airway_radius"] = preset_def.airway_radius_factor
if "tidal_volume" not in st.session_state:
    st.session_state["tidal_volume"] = preset_def.tidal_volume_factor
if "k_release" not in st.session_state:
    st.session_state["k_release"] = preset_def.k_release
if "safety_thresh" not in st.session_state:
    st.session_state["safety_thresh"] = preset_def.safety_threshold_db

st.sidebar.info(f"**Preset Info:** {preset_def.description}")

st.sidebar.markdown("---")
st.sidebar.markdown("**1. Aerosol Particle & Inhalation**")

mmad = st.sidebar.slider(
    "MMAD — Aerodynamic Diameter (μm)",
    min_value=0.5,
    max_value=10.0,
    value=float(st.session_state["mmad"]),
    step=0.1,
    help="Mass Median Aerodynamic Diameter. High values (>4 μm) undergo upper airway impaction; 1.5–3 μm reaches deep alveoli.",
)

inhaled_dose = st.sidebar.number_input(
    "Inhaled Dose (mg)",
    min_value=0.5,
    max_value=100.0,
    value=float(st.session_state["dose"]),
    step=0.5,
    help="Total nominal drug payload inhaled.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**2. Disease Physiological Modifiers**")

viscosity_factor = st.sidebar.slider(
    "Mucus Viscosity Factor (V)",
    min_value=0.5,
    max_value=10.0,
    value=float(st.session_state["viscosity"]),
    step=0.1,
    help="Viscosity relative to healthy mucus (1.0×). Impedes mucociliary clearance (k_clear/V) and tissue diffusion (k_diff/V^α).",
)

clearance_factor = st.sidebar.slider(
    "Mucociliary Clearance Factor",
    min_value=0.1,
    max_value=2.0,
    value=float(st.session_state["clearance_factor"]),
    step=0.05,
    help="Intrinsic ciliary transport velocity multiplier (1.0× normal, 0.2× in Cystic Fibrosis).",
)

airway_radius_factor = st.sidebar.slider(
    "Airway Radius Factor",
    min_value=0.3,
    max_value=1.5,
    value=float(st.session_state["airway_radius"]),
    step=0.05,
    help="Airway caliber ratio. Narrowed airways (e.g. 0.7× in Asthma) increase linear airflow velocity and inertial impaction.",
)

tidal_volume_factor = st.sidebar.slider(
    "Tidal Volume Factor",
    min_value=0.1,
    max_value=1.5,
    value=float(st.session_state["tidal_volume"]),
    step=0.05,
    help="Relative inhaled breath volume (1.0× adult reference, 0.25× infant). Affects deep peripheral lung penetration.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**3. Formulation & Safety Threshold**")

k_release = st.sidebar.slider(
    "Payload Release Rate k_release (1/h)",
    min_value=0.01,
    max_value=2.0,
    value=float(st.session_state["k_release"]),
    step=0.01,
    help="Rate of therapeutic release from nanoparticle carrier into mucosal fluid.",
)

safety_threshold_db = st.sidebar.number_input(
    "Systemic Safety Hazard Threshold Db (mg/L)",
    min_value=1.0,
    max_value=20.0,
    value=float(st.session_state["safety_thresh"]),
    step=0.5,
    help="Model-defined threshold for systemic blood exposure hazard warning (default 5.0 mg/L).",
)

# Advanced Baseline Parameters Expander
with st.sidebar.expander("🔬 Advanced Baseline Parameters"):
    st.markdown("*Baseline Physiological Rates (Normal V = 1.0)*")
    k_clear_normal = st.number_input("Normal Particle Clearance k_clear (1/h)", value=0.25, step=0.05)
    k_clear_drug_normal = st.number_input("Normal Free Drug Clearance k_clear,D (1/h)", value=0.40, step=0.05)
    k_diff_normal = st.number_input("Normal Epithelial Diffusion k_diff (1/h)", value=0.45, step=0.05)
    k_blood = st.number_input("Tissue to Blood Transfer k_blood (1/h)", value=0.15, step=0.05)
    k_tissue_loss = st.number_input("Tissue Metabolism Loss k_loss (1/h)", value=0.08, step=0.02)
    k_systemic_clear = st.number_input("Systemic Blood Elimination k_elim (1/h)", value=0.30, step=0.05)
    
    st.markdown("*Pathology Sensitivity*")
    alpha = st.slider("Diffusion Exponent (α)", min_value=0.1, max_value=1.5, value=0.70, step=0.05,
                      help="Impediment exponent: k_diff,eff = k_diff / V^α")
    
    st.markdown("*Pharmacodynamics (Emax / Hill)*")
    emax = st.number_input("Emax (%)", value=100.0, step=5.0)
    ec50 = st.number_input("EC50 (mg in tissue)", value=0.50, step=0.05)
    simulation_hours = st.slider("Simulation Timeline (hours)", min_value=12, max_value=72, value=24, step=6)

# ==========================================
# EXECUTE UNIFIED SIMULATION
# ==========================================
sim_result = run_simulation(
    mmad=mmad,
    inhaled_dose=inhaled_dose,
    mucus_viscosity_factor=viscosity_factor,
    mucociliary_clearance_factor=clearance_factor,
    airway_radius_factor=airway_radius_factor,
    tidal_volume_factor=tidal_volume_factor,
    k_release=k_release,
    k_clear_normal=k_clear_normal,
    k_clear_drug_normal=k_clear_drug_normal,
    k_diff_normal=k_diff_normal,
    k_blood=k_blood,
    k_tissue_loss=k_tissue_loss,
    k_systemic_clear=k_systemic_clear,
    diffusion_sensitivity_alpha=alpha,
    emax=emax,
    ec50=ec50,
    safety_threshold_db=safety_threshold_db,
    t_end=float(simulation_hours),
    time_points=300,
    preset_name=selected_preset_name,
)

metrics = sim_result["metrics"]
deposition = sim_result["deposition"]
effective_rates = sim_result["effective_rates"]

# ==========================================
# PROMINENT SYSTEMIC EXPOSURE SAFETY BANNER
# ==========================================
if metrics["is_safety_hazard"]:
    st.markdown(
        f"""
        <div class="warning-banner">
            <h4 style="margin:0 0 8px 0; color:#B91C1C; display:flex; align-items:center;">
                ⚠️ SIMULATION WARNING: Potential Systemic Exposure Hazard
            </h4>
            <p style="margin:0; font-size:0.95rem; line-height:1.5;">
                Predicted systemic blood concentration exceeds the configured model threshold of 
                <strong>{safety_threshold_db:.1f} mg/L</strong> (Peak simulated exposure: 
                <strong>{metrics['peak_blood_conc']:.2f} mg/L</strong>; Duration above threshold: 
                <strong>{metrics['time_above_safety_thresh']:.1f} hours</strong>).<br>
                <em>Notice: This indicates a simulated systemic-exposure hazard and does NOT represent a validated clinical toxicity limit.</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="safe-banner">
            <span style="font-weight:600;">✓ Systemic Safety Monitor:</span> 
            Simulated blood exposure remains below the configured threshold of 
            <strong>{safety_threshold_db:.1f} mg/L</strong> 
            (Peak simulated D<sub>b</sub>: <strong>{metrics['peak_blood_conc']:.2f} mg/L</strong>).
            <em>Simulation model parameter only.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================================
# KEY SIMULATION METRICS (4 CARDS)
# ==========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Card 1: Alveolar Deposition",
        value=f"{metrics['alveolar_deposition_pct']:.1f}%",
        delta=f"{deposition['alveolar_dose']:.2f} mg dose",
        delta_color="normal",
        help="Percentage of inhaled aerosol dose penetrating into deep pulmonary alveoli (G17–G23).",
    )

with col2:
    st.metric(
        label="Card 2: Peak Target Tissue (max Dt)",
        value=f"{metrics['peak_tissue_conc']:.3f} mg",
        delta=f"T_max @ {metrics['t_max_tissue']:.1f} h",
        delta_color="off",
        help="Maximum active drug mass accumulated in epithelial target tissue.",
    )

with col3:
    st.metric(
        label="Card 3: Max Pulmonary Efficacy",
        value=f"{metrics['max_efficacy']:.1f}%",
        delta=f"{metrics['time_above_50']:.1f} h >50% eff.",
        delta_color="normal",
        help="Peak anti-inflammatory effect calculated by the Emax Hill model.",
    )

with col4:
    loss_val = metrics["mucociliary_loss_pct"]
    st.metric(
        label="Card 4: Mucociliary Loss",
        value=f"{loss_val:.1f}%",
        delta=f"k_clear: {effective_rates.k_clear_eff:.3f} h⁻¹" if loss_val > 0 else "Normal clearance",
        delta_color="inverse" if loss_val > 0 else "off",
        help="Relative loss in clearance rate compared to healthy baseline (1.0× viscosity, 1.0× clearance).",
    )

st.markdown("---")

# ==========================================
# DASHBOARD TABS
# ==========================================
tab_3d, tab_safety, tab_pk, tab_aerosol, tab_export, tab_model = st.tabs([
    "🫁 3D Airway & Deposition Visualizer",
    "🛡️ Therapeutic Window & Safety Monitor",
    "📈 PK / PD Time-Series Simulation",
    "🎯 Aerosol Deposition & Particle Mechanics",
    "📥 Export Simulation Dossier (CSV)",
    "📚 Model & Biological Assumptions",
])

# ----------------------------------------------------
# TAB 1: 3D AIRWAY VISUALIZER
# ----------------------------------------------------
with tab_3d:
    st.markdown("### 🫁 Interactive 3D Bronchial & Alveolar Deposition Visualizer")
    st.caption(
        "Dichotomous branching respiratory tree spanning simplified anatomical generations: "
        "**G0–G4** (Upper Airways), **G5–G16** (Conducting Bronchial), and **G17–G23** (Deep Alveolar). "
        "Particle density and mechanisms are derived directly from the underlying aerosol model."
    )
    
    col_view, col_3d_info = st.columns([3, 1])
    
    with col_view:
        fig_3d = plot_airway_3d_deposition(sim_result)
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with col_3d_info:
        st.markdown("#### Airway Generation Legend")
        st.markdown(
            """
            - <span style="color:#EF4444; font-weight:700;">● Red (G0–G4):</span> **Inertial Impaction**
              Large particles collide at bifurcations and the oropharynx.
            - <span style="color:#F59E0B; font-weight:700;">◆ Amber (G5–G16):</span> **Bronchial Airways**
              Mixed sedimentation and secondary impaction in conducting zones.
            - <span style="color:#2563EB; font-weight:700;">● Blue (G17–G23):</span> **Gravitational Sedimentation**
              Fine particles settle into alveolar ducts and gas-exchange sacs.
            - <span style="color:#10B981; font-weight:700;">✕ Green:</span> **Exhaled Fraction**
              Non-deposited aerosol returning in expiratory air stream.
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("#### Anatomical Scale Settings")
        st.write(f"• **Airway Caliber:** {airway_radius_factor:.2f}×")
        st.write(f"• **Tidal Ventilation:** {tidal_volume_factor:.2f}×")
        st.write(f"• **Inhaled MMAD:** {mmad:.1f} μm")
        st.write(f"• **Preset:** {selected_preset_name}")
        st.info("💡 Rotate, zoom, and hover over individual particles to inspect generation number, size, and deposition mechanisms.")

# ----------------------------------------------------
# TAB 2: THERAPEUTIC WINDOW & SAFETY MONITOR
# ----------------------------------------------------
with tab_safety:
    st.markdown("### 🛡️ Therapeutic Window & Safety Hazard Monitor")
    st.caption(
        "Real-time evaluation of local pulmonary target efficacy vs. systemic blood exposure. "
        "All zones represent model-defined simulation criteria."
    )
    
    # Gauges Row
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_eff_gauge = create_efficacy_gauge(sim_result)
        st.plotly_chart(fig_eff_gauge, use_container_width=True)
        
    with col_g2:
        fig_exp_gauge = create_systemic_exposure_gauge(sim_result, safety_threshold_db)
        st.plotly_chart(fig_exp_gauge, use_container_width=True)
        
    # Safety Metrics Grid
    st.markdown("#### ⏱️ Exposure & Therapeutic Window Statistics")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    
    with m_col1:
        st.metric("Current Efficacy (24h)", f"{metrics['current_efficacy']:.1f}%")
    with m_col2:
        st.metric("Peak Efficacy", f"{metrics['max_efficacy']:.1f}%")
    with m_col3:
        st.metric("Current Blood Db (24h)", f"{metrics['current_blood_conc']:.3f} mg/L")
    with m_col4:
        st.metric(
            "Peak Blood Exposure",
            f"{metrics['peak_blood_conc']:.3f} mg/L",
            delta="Hazard Exceeded" if metrics["is_safety_hazard"] else "Within Limit",
            delta_color="inverse" if metrics["is_safety_hazard"] else "normal",
        )
    with m_col5:
        st.metric("Time Above Hazard Limit", f"{metrics['time_above_safety_thresh']:.1f} h")
        
    st.markdown("---")
    # Dual-Axis Time Series
    fig_safety_ts = plot_efficacy_vs_exposure_timeseries(sim_result, safety_threshold_db)
    st.plotly_chart(fig_safety_ts, use_container_width=True)
    
    st.markdown("#### Conceptual Therapeutic Window Criteria (Simulation Model)")
    st.markdown(
        """
        - **Sub-Therapeutic ($E < 50\\%$):** Insufficient local target engagement to elicit robust anti-inflammatory suppression.
        - **Therapeutic Simulation Window ($50\\% \\leq E \\leq 90\\%$):** Balanced target inhibition with acceptable systemic exposure ($D_b \\leq \\text{threshold}$).
        - **High-Efficacy Zone ($E > 90\\%$):** Near-saturation of target receptors. Requires monitoring to avoid excessive systemic spillover.
        """
    )

# ----------------------------------------------------
# TAB 3: PK/PD TIME SERIES
# ----------------------------------------------------
with tab_pk:
    st.markdown("### 📈 Four-Compartment PK & Anti-Inflammatory PD Time Series")
    st.caption("Coupled disposition of Nanoparticles ($N_m$), Free Drug in Mucus ($D_m$), Target Epithelial Tissue ($D_t$), and Systemic Blood ($D_b$).")
    
    fig_pk = plot_pk_profiles(sim_result)
    st.plotly_chart(fig_pk, use_container_width=True)
    
    fig_pd = plot_pd_response(sim_result)
    st.plotly_chart(fig_pd, use_container_width=True)
    
    # Detailed Pharmacometric Summary Table
    with st.expander("📊 Detailed Pharmacokinetic & Exposure Summary Table", expanded=True):
        summary_df = pd.DataFrame({
            "Parameter": [
                "Disease / Physiological Preset",
                "Inhaled Dose",
                "Deep Alveolar Dose",
                "Conducting Bronchial Dose",
                "Upper Airway Loss (Swallowed)",
                "Effective Mucociliary Clearance (k_clear,eff)",
                "Effective Tissue Diffusion Rate (k_diff,eff)",
                "Peak Tissue Exposure (Cmax, tissue)",
                "Time-to-Peak Tissue Exposure (Tmax, tissue)",
                "Tissue Exposure AUC (AUC_tissue)",
                "Systemic Blood AUC (AUC_blood)",
                "Peak Systemic Blood Concentration (Cmax, blood)",
                "Time in Therapeutic Window (50-90% E & Db <= Limit)",
                "Time Above Safety Hazard Limit",
            ],
            "Value": [
                selected_preset_name,
                f"{inhaled_dose:.2f} mg",
                f"{deposition['alveolar_dose']:.2f} mg ({metrics['alveolar_deposition_pct']:.1f}%)",
                f"{deposition['bronchial_dose']:.2f} mg ({metrics['bronchial_deposition_pct']:.1f}%)",
                f"{deposition['upper_dose']:.2f} mg ({metrics['upper_deposition_pct']:.1f}%)",
                f"{effective_rates.k_clear_eff:.4f} h⁻¹",
                f"{effective_rates.k_diff_eff:.4f} h⁻¹",
                f"{metrics['peak_tissue_conc']:.4f} mg",
                f"{metrics['t_max_tissue']:.1f} hours",
                f"{metrics['auc_tissue']:.3f} mg·h",
                f"{metrics['auc_blood']:.3f} mg·h",
                f"{metrics['peak_blood_conc']:.4f} mg/L",
                f"{metrics['time_in_therapeutic_window']:.1f} hours",
                f"{metrics['time_above_safety_thresh']:.1f} hours",
            ],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ----------------------------------------------------
# TAB 4: AEROSOL MECHANICS
# ----------------------------------------------------
with tab_aerosol:
    st.markdown("### 🎯 Aerosol Deposition & Particle Mechanics")
    st.caption("Quantifying inertial impaction, sedimentation, and mucosal deposition across particle aerodynamic sizes.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = plot_deposition_breakdown(deposition, inhaled_dose)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_b:
        fig_curve = plot_mmad_curve(mmad)
        st.plotly_chart(fig_curve, use_container_width=True)
        
    st.info(
        f"**Regional Aerosol Balance:** At **MMAD = {mmad:.1f} μm** under **{selected_preset_name}** airway conditions, "
        f"**{metrics['alveolar_deposition_pct']:.1f}%** ({deposition['alveolar_dose']:.2f} mg) reaches deep alveoli, "
        f"**{metrics['bronchial_deposition_pct']:.1f}%** ({deposition['bronchial_dose']:.2f} mg) deposits in conducting bronchioles, "
        f"and **{metrics['upper_deposition_pct']:.1f}%** ({deposition['upper_dose']:.2f} mg) impacts in upper airways."
    )

# ----------------------------------------------------
# TAB 5: EXPORT SIMULATION DOSSIER (CSV)
# ----------------------------------------------------
with tab_export:
    st.markdown("### 📥 Export Simulation Dossier (CSV)")
    st.caption(
        "Export complete high-resolution ODE numerical time-series ($N_m, D_m, D_t, D_b$), "
        "calculated pharmacodynamic efficacy values, and disease simulation parameters. "
        "Directly formatted for Excel, Python/Pandas, MATLAB, R, and Jupyter/Google Colab."
    )
    
    dossier_df = generate_simulation_dossier_dataframe(sim_result)
    csv_data, csv_filename = export_simulation_dossier_csv(sim_result)
    
    st.download_button(
        label=f"💾 Export Simulation Dossier ({csv_filename})",
        data=csv_data,
        file_name=csv_filename,
        mime="text/csv",
        help="Download the complete numerical dataset containing all ODE state variables and simulation parameters.",
        use_container_width=True,
    )
    
    st.markdown("#### 📄 Simulation Dossier Preview (First 15 Time Steps)")
    st.dataframe(dossier_df.head(15), use_container_width=True, hide_index=True)
    
    st.caption(
        f"Total simulated rows: {len(dossier_df)} time points spanning {float(simulation_hours):.0f} hours. "
        "Explicitly labeled as simulation export; not a validated clinical record."
    )

# ----------------------------------------------------
# TAB 6: MATHEMATICAL & BIOLOGICAL ASSUMPTIONS
# ----------------------------------------------------
with tab_model:
    st.markdown("### 📚 Mathematical Model & Coupled ODE Architecture")
    
    st.markdown("#### 1. Aerosol Deposition Mechanics")
    st.latex(r"F_{\text{upper}} + F_{\text{bronchial}} + F_{\text{alveolar}} = 1.0")
    st.write(
        "Aerosol transport is dictated by inertial impaction and sedimentation. "
        "Narrowed airway caliber increases the Stokes number ($Stk \\propto d^2 u / R$), "
        "exacerbating upper and bronchial impaction at the expense of alveolar penetration."
    )
    
    st.markdown("#### 2. Four-Compartment Pulmonary Mucosal PK Model")
    st.latex(r"\frac{dN_m}{dt} = -k_{\text{release}} N_m - k_{\text{clear,eff}} N_m")
    st.latex(r"\frac{dD_m}{dt} = k_{\text{release}} N_m - k_{\text{diff,eff}} D_m - k_{\text{clear},D,\text{eff}} D_m")
    st.latex(r"\frac{dD_t}{dt} = k_{\text{diff,eff}} D_m - k_{\text{blood}} D_t - k_{\text{tissue-loss}} D_t")
    st.latex(r"\frac{dD_b}{dt} = k_{\text{blood}} D_t - k_{\text{systemic-clear}} D_b")
    
    st.markdown("#### 3. Disease Pathological Modifiers")
    st.latex(r"k_{\text{clear,eff}} = \frac{k_{\text{clear,normal}} \times \text{Factor}_{\text{clear}}}{V}")
    st.latex(r"k_{\text{diff,eff}} = \frac{k_{\text{diff,normal}}}{V^\alpha}")
    st.latex(r"L_{\text{mucociliary}} = \left( 1 - \frac{k_{\text{clear,eff}}}{k_{\text{clear,normal}}} \right) \times 100\%")
    st.write(
        "Where $V$ is the Mucus Pathological Viscosity Factor (1.0 = normal, >1.0 = diseased), "
        "$\\alpha$ is the diffusion sensitivity exponent, and $\\text{Factor}_{\\text{clear}}$ "
        "is the intrinsic ciliary transport velocity modifier."
    )
    
    st.markdown("#### 4. Pharmacodynamics (Emax / Hill)")
    st.latex(r"E(D_t) = \frac{E_{\max} D_t}{EC_{50} + D_t}")
    st.write(
        "Evaluates local pulmonary anti-inflammatory suppression as a saturable function of active tissue amount."
    )

# Footer
st.markdown("---")
st.caption(
    "Inhaled Nanoparticle & Biological Respiratory Therapeutics Simulator | "
    "Computational / Educational Model. Solved with SciPy ODE integration (solve_ivp RK45), NumPy, Pandas, and Plotly."
)
