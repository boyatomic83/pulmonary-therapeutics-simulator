"""
Aerosol Physics — Regional Particle Deposition Model
Calculates particle deposition fractions as a function of Mass Median Aerodynamic Diameter (MMAD).
Incorporates inertial impaction and gravitational sedimentation mechanisms, modified by
airway caliber (radius factor) and ventilation (tidal volume factor).

Enforces:
  F_upper + F_bronchial + F_alveolar + F_exhaled = 1.0 (or normalized regional fraction)
  F_upper + F_bronchial + F_alveolar = 1.0 (for deposited pulmonary dose calculations)
"""

import numpy as np
from typing import Dict, Any

def calculate_regional_deposition(
    mmad: float,
    inhaled_dose: float = 1.0,
    airway_radius_factor: float = 1.0,
    tidal_volume_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Computes regional deposition fractions based on empirical respiratory deposition principles
    (Heyder et al. / ICRP lung deposition models), modulated by airway geometry and tidal volume.
    
    Parameters:
        mmad (float): Mass Median Aerodynamic Diameter in micrometers (0.5 to 10.0 um).
        inhaled_dose (float): Total inhaled dose in mg.
        airway_radius_factor (float): Relative airway radius (1.0 = normal, <1.0 = constricted).
        tidal_volume_factor (float): Relative tidal volume (1.0 = normal adult, <1.0 = pediatric/hypoventilation).
        
    Returns:
        dict with regional fractions, doses, and generation-level breakdown.
    """
    d = float(np.clip(mmad, 0.2, 12.0))
    r_factor = float(np.clip(airway_radius_factor, 0.2, 2.0))
    vt_factor = float(np.clip(tidal_volume_factor, 0.1, 2.0))
    
    # Airway constriction increases linear velocity u ~ 1 / r^2, boosting Stokes number Stk ~ d^2 * u / r ~ d^2 / r^3
    impaction_enhancement = (1.0 / r_factor) ** 1.8
    
    # 1. Upper Airways (G0-G4): Dominated by inertial impaction
    raw_upper = ((d ** 2.5) / (d ** 2.5 + 3.3 ** 2.5)) * impaction_enhancement
    
    # 2. Conducting Bronchial Airways (G5-G16): Intermediate impaction & sedimentation
    raw_bronchial = ((0.85 * (d ** 1.3)) / (1.0 + (d / 3.8) ** 2.4)) * ((1.0 / r_factor) ** 0.8)
    
    # 3. Deep Alveolar (G17-G23): Dominated by sedimentation and diffusion.
    # Narrower airways and reduced tidal volume significantly reduce alveolar penetration.
    alveolar_reach = vt_factor ** 0.75 * (r_factor ** 0.5)
    raw_alveolar = ((1.55 * (d ** 1.8)) / (1.0 + (d / 2.3) ** 3.5)) * alveolar_reach
    
    # Total raw deposited fraction
    total_raw = raw_upper + raw_bronchial + raw_alveolar
    
    # Normalized regional fractions of deposited dose (sum = 1.0)
    f_upper = float(raw_upper / total_raw)
    f_bronchial = float(raw_bronchial / total_raw)
    f_alveolar = float(raw_alveolar / total_raw)
    
    # Exhaled fraction estimation (particles that are neither impacted nor settled, e.g. very fine 0.5 um particles)
    # Typically 10-25% of inhaled aerosol is exhaled without depositing
    f_exhaled_raw = 0.15 * np.exp(-((d - 0.5) / 1.5) ** 2)
    # Total nominal fraction of inhaled dose
    f_deposited_total = 1.0 - f_exhaled_raw
    f_exhaled = float(f_exhaled_raw)
    
    upper_dose = inhaled_dose * f_upper
    bronchial_dose = inhaled_dose * f_bronchial
    alveolar_dose = inhaled_dose * f_alveolar
    pulmonary_dose = bronchial_dose + alveolar_dose
    
    # Generation-level breakdown for scientific 3D visualization:
    # G0-G4: Upper airways (Trachea to main bronchi)
    # G5-G16: Conducting bronchial airways
    # G17-G23: Deep alveolar beds
    generation_fractions = {}
    
    # Distribute upper fraction across G0-G4 (exponential decay as impaction occurs at early bifurcations)
    g0_4_weights = np.array([0.40, 0.25, 0.18, 0.11, 0.06])
    for g, w in zip(range(0, 5), g0_4_weights):
        generation_fractions[f"G{g}"] = float(f_upper * w)
        
    # Distribute bronchial fraction across G5-G16 (bell-shaped across central bronchioles)
    g5_16_idx = np.arange(5, 17)
    g5_16_weights = np.exp(-0.5 * ((g5_16_idx - 10) / 2.8) ** 2)
    g5_16_weights /= np.sum(g5_16_weights)
    for g, w in zip(g5_16_idx, g5_16_weights):
        generation_fractions[f"G{g}"] = float(f_bronchial * w)
        
    # Distribute alveolar fraction across G17-G23 (increasing surface area towards G23)
    g17_23_idx = np.arange(17, 24)
    g17_23_weights = np.linspace(0.8, 1.4, len(g17_23_idx))
    g17_23_weights /= np.sum(g17_23_weights)
    for g, w in zip(g17_23_idx, g17_23_weights):
        generation_fractions[f"G{g}"] = float(f_alveolar * w)
        
    return {
        "f_upper": f_upper,
        "f_bronchial": f_bronchial,
        "f_alveolar": f_alveolar,
        "f_exhaled": f_exhaled,
        "f_deposited_total": f_deposited_total,
        "upper_dose": upper_dose,
        "bronchial_dose": bronchial_dose,
        "alveolar_dose": alveolar_dose,
        "pulmonary_dose": pulmonary_dose,
        "generation_fractions": generation_fractions,
        "airway_radius_factor": r_factor,
        "tidal_volume_factor": vt_factor,
        "impaction_enhancement": impaction_enhancement,
    }

def generate_deposition_curve(
    airway_radius_factor: float = 1.0,
    tidal_volume_factor: float = 1.0,
    points: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Generates regional deposition curves across MMAD from 0.5 to 10.0 um.
    """
    mmad_range = np.linspace(0.5, 10.0, points)
    upper_list = []
    bronchial_list = []
    alveolar_list = []
    
    for d in mmad_range:
        res = calculate_regional_deposition(
            mmad=d,
            inhaled_dose=1.0,
            airway_radius_factor=airway_radius_factor,
            tidal_volume_factor=tidal_volume_factor,
        )
        upper_list.append(res["f_upper"] * 100.0)
        bronchial_list.append(res["f_bronchial"] * 100.0)
        alveolar_list.append(res["f_alveolar"] * 100.0)
        
    return {
        "mmad": mmad_range,
        "upper_pct": np.array(upper_list),
        "bronchial_pct": np.array(bronchial_list),
        "alveolar_pct": np.array(alveolar_list),
    }
