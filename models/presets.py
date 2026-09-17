"""
Clinical and Disease-State Simulation Presets
Contains illustrative physiological parameter sets for:
  - Healthy Adult
  - Cystic Fibrosis (CF)
  - Severe Asthma / COPD
  - Infant / Pediatric
  - Custom
All parameters are simulation assumptions / illustrative model parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PhysiologicalPreset:
    id: str
    name: str
    description: str
    mucus_viscosity_factor: float      # V (1.0 = normal baseline)
    mucociliary_clearance_factor: float # baseline clearance multiplier
    airway_radius_factor: float        # airway lumen diameter ratio (affects impaction)
    tidal_volume_factor: float         # ventilation volume ratio (affects alveolar reach)
    mmad: float                        # recommended default aerosol particle size (um)
    inhaled_dose: float                # recommended default dose (mg)
    k_release: float                   # payload release rate (1/h)
    safety_threshold_db: float         # simulation systemic blood safety hazard threshold (mg/L or mg)

PRESETS: Dict[str, PhysiologicalPreset] = {
    "Healthy Adult": PhysiologicalPreset(
        id="healthy_adult",
        name="Healthy Adult",
        description="Illustrative baseline physiology with normal airway geometry, baseline mucociliary clearance, and standard tidal volume.",
        mucus_viscosity_factor=1.0,
        mucociliary_clearance_factor=1.0,
        airway_radius_factor=1.0,
        tidal_volume_factor=1.0,
        mmad=2.5,
        inhaled_dose=10.0,
        k_release=0.35,
        safety_threshold_db=5.0,
    ),
    "Cystic Fibrosis (CF)": PhysiologicalPreset(
        id="cystic_fibrosis",
        name="Cystic Fibrosis (CF)",
        description="Pathological model characterized by dehydrated, hyper-viscous mucus (5.0×) and severely compromised mucociliary clearance (0.2×), creating mucosal drug entrapment.",
        mucus_viscosity_factor=5.0,
        mucociliary_clearance_factor=0.2,
        airway_radius_factor=1.0,
        tidal_volume_factor=1.0,
        mmad=2.2,
        inhaled_dose=15.0,
        k_release=0.25,
        safety_threshold_db=5.0,
    ),
    "Severe Asthma / COPD": PhysiologicalPreset(
        id="asthma_copd",
        name="Severe Asthma / COPD",
        description="Obstructive airway model with bronchoconstriction (0.7× radius), moderate mucus hypersecretion (2.5× viscosity), and altered tidal volume.",
        mucus_viscosity_factor=2.5,
        mucociliary_clearance_factor=0.5,
        airway_radius_factor=0.7,
        tidal_volume_factor=0.85,
        mmad=2.8,
        inhaled_dose=12.0,
        k_release=0.30,
        safety_threshold_db=5.0,
    ),
    "Infant / Pediatric": PhysiologicalPreset(
        id="pediatric",
        name="Infant / Pediatric",
        description="Scaled pediatric physiology with narrower anatomical caliber (0.5× radius) and significantly reduced tidal volume (0.25×). Illustrative simulation scaling only.",
        mucus_viscosity_factor=1.0,
        mucociliary_clearance_factor=0.8,
        airway_radius_factor=0.5,
        tidal_volume_factor=0.25,
        mmad=1.8,
        inhaled_dose=2.5,
        k_release=0.40,
        safety_threshold_db=2.0,
    ),
    "Custom": PhysiologicalPreset(
        id="custom",
        name="Custom",
        description="Fully manual configuration of airway geometry, mucus viscosity, clearance factors, and dosing.",
        mucus_viscosity_factor=1.0,
        mucociliary_clearance_factor=1.0,
        airway_radius_factor=1.0,
        tidal_volume_factor=1.0,
        mmad=2.5,
        inhaled_dose=10.0,
        k_release=0.35,
        safety_threshold_db=5.0,
    ),
}

def get_preset(name: str) -> PhysiologicalPreset:
    """Returns preset definition, falling back to Healthy Adult."""
    return PRESETS.get(name, PRESETS["Healthy Adult"])
