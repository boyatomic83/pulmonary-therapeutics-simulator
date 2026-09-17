"""
Simulation Dossier Export Utility
Packages the complete ODE time-series dataset and derived simulation variables
into a standardized Pandas DataFrame and CSV format compatible with
Excel, Python, MATLAB, R, and Jupyter/Colab.
Explicitly labeled as a simulation export, not a clinical record.
"""

import pandas as pd
from typing import Dict, Any, Tuple

def generate_simulation_dossier_dataframe(sim_result: Dict[str, Any]) -> pd.DataFrame:
    """
    Constructs a comprehensive tabular dataset of the ODE simulation trajectories
    and key parameter metadata.
    """
    times = sim_result["times"]
    nm = sim_result["nm"]
    dm = sim_result["dm"]
    dt = sim_result["dt"]
    db = sim_result["db"]
    efficacy = sim_result["efficacy"]
    
    metrics = sim_result["metrics"]
    deposition = sim_result["deposition"]
    inputs = sim_result["inputs"]
    
    df = pd.DataFrame({
        "time_hours": np_round(times, 4),
        "Nm_nanoparticles_in_mucus_mg": np_round(nm, 5),
        "Dm_free_drug_in_mucus_mg": np_round(dm, 5),
        "Dt_target_epithelial_tissue_mg": np_round(dt, 5),
        "Db_systemic_blood_mg_per_L": np_round(db, 5),
        "pulmonary_efficacy_pct": np_round(efficacy, 2),
        "total_pulmonary_mass_mg": np_round(nm + dm + dt, 5),
        # Static simulation parameters per row for seamless relational analytics / R / Python grouping
        "preset_name": inputs.get("preset_name", "Healthy Adult"),
        "mmad_um": inputs["mmad"],
        "inhaled_dose_mg": inputs["inhaled_dose"],
        "mucus_viscosity_factor_V": inputs["mucus_viscosity_factor"],
        "airway_radius_factor": inputs.get("airway_radius_factor", 1.0),
        "tidal_volume_factor": inputs.get("tidal_volume_factor", 1.0),
        "k_release_per_h": inputs["k_release"],
        "alveolar_deposition_pct": round(metrics["alveolar_deposition_pct"], 2),
        "bronchial_deposition_pct": round(metrics["bronchial_deposition_pct"], 2),
        "upper_airway_deposition_pct": round(metrics["upper_deposition_pct"], 2),
        "mucociliary_loss_pct": round(metrics["mucociliary_loss_pct"], 2),
        "safety_threshold_db_mg_per_L": inputs.get("safety_threshold_db", 5.0),
        "is_safety_hazard_exceeded": metrics.get("is_safety_hazard", False),
    })
    
    return df

def np_round(arr, decimals: int):
    """Helper to round float arrays or lists cleanly."""
    import numpy as np
    return np.round(np.array(arr, dtype=float), decimals)

def export_simulation_dossier_csv(sim_result: Dict[str, Any]) -> Tuple[str, str]:
    """
    Generates CSV text and a recommended filename for download.
    """
    df = generate_simulation_dossier_dataframe(sim_result)
    preset_slug = sim_result["inputs"].get("preset_name", "custom").lower().replace(" ", "_").replace("/", "_")
    mmad_val = sim_result["inputs"]["mmad"]
    
    filename = f"simulation_dossier_{preset_slug}_mmad{mmad_val:.1f}um.csv"
    csv_data = df.to_csv(index=False)
    
    return csv_data, filename
