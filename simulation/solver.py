"""
Simulation Solver
Performs numerical integration of the 4-compartment pulmonary PK/PD ODE system
using scipy.integrate.solve_ivp across the 0-24 hour timeline.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Any

from models.aerosol import calculate_regional_deposition
from models.pharmacokinetics import PKParameters, compute_effective_rates, pk_derivatives
from models.pharmacodynamics import calculate_efficacy, calculate_auc

def run_simulation(
    mmad: float = 2.5,
    inhaled_dose: float = 10.0,
    mucus_viscosity_factor: float = 1.0,
    mucociliary_clearance_factor: float = 1.0,
    airway_radius_factor: float = 1.0,
    tidal_volume_factor: float = 1.0,
    k_release: float = 0.35,
    k_clear_normal: float = 0.25,
    k_clear_drug_normal: float = 0.40,
    k_diff_normal: float = 0.45,
    k_blood: float = 0.15,
    k_tissue_loss: float = 0.08,
    k_systemic_clear: float = 0.30,
    diffusion_sensitivity_alpha: float = 0.7,
    emax: float = 100.0,
    ec50: float = 0.5,
    safety_threshold_db: float = 5.0,
    t_end: float = 24.0,
    time_points: int = 300,
    preset_name: str = "Healthy Adult",
) -> Dict[str, Any]:
    """
    Executes the unified computational pipeline:
      Inputs & Presets -> Aerosol Deposition Model (MMAD + Airway caliber + Tidal volume)
      -> Regional Deposition Fractions (Upper G0-G4, Bronchial G5-G16, Alveolar G17-G23)
      -> Initial Pulmonary Reservoir -> 4-Compartment PK Model (solve_ivp RK45)
      -> Target Tissue Concentration -> Emax/Hill PD Model
      -> Systemic Blood Exposure & Safety Threshold -> Visualizations & CSV Export
    """
    # 1. Aerosol Deposition
    deposition = calculate_regional_deposition(
        mmad=mmad,
        inhaled_dose=inhaled_dose,
        airway_radius_factor=airway_radius_factor,
        tidal_volume_factor=tidal_volume_factor,
    )
    
    # Initial Conditions:
    # Conducting bronchial and deep alveolar deposition form the initial mucosal reservoir Nm(0).
    # Upper airway deposition is rapidly swallowed / cleared out of the respiratory tract.
    initial_nm = deposition["pulmonary_dose"]
    initial_dm = 0.0
    initial_dt = 0.0
    initial_db = 0.0
    y0 = [initial_nm, initial_dm, initial_dt, initial_db]
    
    # 2. Pharmacokinetic Parameters & Disease Modifiers
    pk_params = PKParameters(
        k_release=k_release,
        k_clear_normal=k_clear_normal,
        k_clear_drug_normal=k_clear_drug_normal,
        k_diff_normal=k_diff_normal,
        k_blood=k_blood,
        k_tissue_loss=k_tissue_loss,
        k_systemic_clear=k_systemic_clear,
        mucus_viscosity_factor=mucus_viscosity_factor,
        mucociliary_clearance_factor=mucociliary_clearance_factor,
        diffusion_sensitivity_alpha=diffusion_sensitivity_alpha,
    )
    effective_rates = compute_effective_rates(pk_params)
    
    # 3. ODE Integration using scipy.integrate.solve_ivp (RK45)
    t_eval = np.linspace(0.0, t_end, time_points)
    
    sol = solve_ivp(
        fun=lambda t, y: pk_derivatives(t, y, effective_rates),
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    
    times = sol.t
    nm = np.maximum(0.0, sol.y[0])
    dm = np.maximum(0.0, sol.y[1])
    dt = np.maximum(0.0, sol.y[2])
    db = np.maximum(0.0, sol.y[3])
    
    # 4. Pharmacodynamics Calculation
    efficacy = calculate_efficacy(dt, emax=emax, ec50=ec50)
    
    # 5. Key Pharmacometric & Safety Metrics
    alveolar_deposition_pct = deposition["f_alveolar"] * 100.0
    upper_deposition_pct = deposition["f_upper"] * 100.0
    bronchial_deposition_pct = deposition["f_bronchial"] * 100.0
    
    peak_tissue_conc = float(np.max(dt))
    t_max_tissue_idx = int(np.argmax(dt))
    t_max_tissue = float(times[t_max_tissue_idx])
    
    max_efficacy = float(np.max(efficacy))
    current_efficacy = float(efficacy[-1])
    
    peak_blood_conc = float(np.max(db))
    current_blood_conc = float(db[-1])
    t_max_blood_idx = int(np.argmax(db))
    t_max_blood = float(times[t_max_blood_idx])
    
    mucociliary_loss_pct = float(effective_rates.mucociliary_loss_pct)
    auc_tissue = calculate_auc(times, dt)
    auc_blood = calculate_auc(times, db)
    
    # Time above 50% efficacy threshold
    dt_step = times[1] - times[0] if len(times) > 1 else 0.0
    above_50 = efficacy >= 50.0
    time_above_50 = float(np.sum(above_50) * dt_step) if np.any(above_50) else 0.0
    
    # Time in Therapeutic Window (50% <= E <= 90% AND Db <= safety_threshold_db)
    in_therapeutic_window = (efficacy >= 50.0) & (efficacy <= 90.0) & (db <= safety_threshold_db)
    time_in_therapeutic_window = float(np.sum(in_therapeutic_window) * dt_step) if np.any(in_therapeutic_window) else 0.0
    
    # Safety Hazard Analysis: Db > safety_threshold_db
    above_safety_thresh = db > safety_threshold_db
    is_safety_hazard = bool(np.any(above_safety_thresh))
    time_above_safety_thresh = float(np.sum(above_safety_thresh) * dt_step) if is_safety_hazard else 0.0
    
    metrics = {
        "alveolar_deposition_pct": alveolar_deposition_pct,
        "upper_deposition_pct": upper_deposition_pct,
        "bronchial_deposition_pct": bronchial_deposition_pct,
        "peak_tissue_conc": peak_tissue_conc,
        "t_max_tissue": t_max_tissue,
        "max_efficacy": max_efficacy,
        "current_efficacy": current_efficacy,
        "mucociliary_loss_pct": mucociliary_loss_pct,
        "auc_tissue": auc_tissue,
        "auc_blood": auc_blood,
        "peak_blood_conc": peak_blood_conc,
        "current_blood_conc": current_blood_conc,
        "t_max_blood": t_max_blood,
        "time_above_50": time_above_50,
        "time_in_therapeutic_window": time_in_therapeutic_window,
        "safety_threshold_db": safety_threshold_db,
        "is_safety_hazard": is_safety_hazard,
        "time_above_safety_thresh": time_above_safety_thresh,
        "k_clear_eff": effective_rates.k_clear_eff,
        "k_diff_eff": effective_rates.k_diff_eff,
    }
    
    return {
        "times": times,
        "nm": nm,
        "dm": dm,
        "dt": dt,
        "db": db,
        "efficacy": efficacy,
        "deposition": deposition,
        "effective_rates": effective_rates,
        "metrics": metrics,
        "inputs": {
            "preset_name": preset_name,
            "mmad": mmad,
            "inhaled_dose": inhaled_dose,
            "mucus_viscosity_factor": mucus_viscosity_factor,
            "mucociliary_clearance_factor": mucociliary_clearance_factor,
            "airway_radius_factor": airway_radius_factor,
            "tidal_volume_factor": tidal_volume_factor,
            "k_release": k_release,
            "k_clear_normal": k_clear_normal,
            "k_clear_drug_normal": k_clear_drug_normal,
            "k_diff_normal": k_diff_normal,
            "k_blood": k_blood,
            "k_tissue_loss": k_tissue_loss,
            "k_systemic_clear": k_systemic_clear,
            "diffusion_sensitivity_alpha": diffusion_sensitivity_alpha,
            "emax": emax,
            "ec50": ec50,
            "safety_threshold_db": safety_threshold_db,
            "t_end": t_end,
        },
    }
