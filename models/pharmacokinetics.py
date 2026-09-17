"""
Pulmonary Mucosal & Tissue Pharmacokinetic Engine
Four-compartment coupled differential equations with disease-specific mucus modifiers:
  1. Nm: Nanoparticle payload in mucus
  2. Dm: Free drug in mucus
  3. Dt: Target epithelial tissue
  4. Db: Systemic blood exposure
"""

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class PKParameters:
    k_release: float = 0.35          # 1/h: Payload release rate from nanoparticle
    k_clear_normal: float = 0.25     # 1/h: Baseline normal mucociliary clearance of nanoparticles
    k_clear_drug_normal: float = 0.40 # 1/h: Baseline clearance of free drug in mucus
    k_diff_normal: float = 0.45      # 1/h: Baseline drug diffusion rate from mucus to epithelial tissue
    k_blood: float = 0.15            # 1/h: Transfer rate from tissue to systemic blood
    k_tissue_loss: float = 0.08      # 1/h: Local tissue metabolism / non-systemic loss
    k_systemic_clear: float = 0.30   # 1/h: Systemic blood elimination rate
    
    # Disease modifiers
    mucus_viscosity_factor: float = 1.0        # V (1.0 = normal, >1.0 = pathological)
    mucociliary_clearance_factor: float = 1.0  # Pathological baseline multiplier (e.g. 0.2 in CF)
    diffusion_sensitivity_alpha: float = 0.7   # alpha exponent in k_diff_eff = k_diff / V^alpha

@dataclass
class EffectivePKRates:
    k_release: float
    k_clear_eff: float
    k_clear_drug_eff: float
    k_diff_eff: float
    k_blood: float
    k_tissue_loss: float
    k_systemic_clear: float
    mucociliary_loss_pct: float

def compute_effective_rates(params: PKParameters) -> EffectivePKRates:
    """
    Computes effective kinetic rates modified by mucus pathological viscosity factor V
    and mucociliary clearance factor:
      k_clear,eff = (k_clear,normal * clearance_factor) / V
      k_diff,eff = k_diff,normal / (V ^ alpha)
    """
    v = max(0.01, params.mucus_viscosity_factor)
    clear_factor = max(0.01, params.mucociliary_clearance_factor)
    alpha = params.diffusion_sensitivity_alpha
    
    k_clear_eff = (params.k_clear_normal * clear_factor) / v
    k_clear_drug_eff = (params.k_clear_drug_normal * clear_factor) / v
    k_diff_eff = params.k_diff_normal / (v ** alpha)
    
    # Loss in mucociliary clearance relative to normal reference (V = 1.0, factor = 1.0)
    mucociliary_loss_pct = max(0.0, (1.0 - (k_clear_eff / params.k_clear_normal)) * 100.0)
    
    return EffectivePKRates(
        k_release=params.k_release,
        k_clear_eff=k_clear_eff,
        k_clear_drug_eff=k_clear_drug_eff,
        k_diff_eff=k_diff_eff,
        k_blood=params.k_blood,
        k_tissue_loss=params.k_tissue_loss,
        k_systemic_clear=params.k_systemic_clear,
        mucociliary_loss_pct=mucociliary_loss_pct,
    )

def pk_derivatives(t: float, y: List[float], rates: EffectivePKRates) -> List[float]:
    """
    Evaluates the 4-compartment coupled ODE derivatives.
    
    dNm/dt = - k_release * Nm - k_clear,eff * Nm
    dDm/dt = k_release * Nm - k_diff,eff * Dm - k_clear,D,eff * Dm
    dDt/dt = k_diff,eff * Dm - k_blood * Dt - k_tissue_loss * Dt
    dDb/dt = k_blood * Dt - k_systemic_clear * Db
    """
    nm, dm, dt, db = y
    
    # Enforce non-negative amounts for derivative stability
    nm_val = max(0.0, nm)
    dm_val = max(0.0, dm)
    dt_val = max(0.0, dt)
    db_val = max(0.0, db)
    
    d_nm = -rates.k_release * nm_val - rates.k_clear_eff * nm_val
    d_dm = rates.k_release * nm_val - rates.k_diff_eff * dm_val - rates.k_clear_drug_eff * dm_val
    d_dt = rates.k_diff_eff * dm_val - rates.k_blood * dt_val - rates.k_tissue_loss * dt_val
    d_db = rates.k_blood * dt_val - rates.k_systemic_clear * db_val
    
    return [d_nm, d_dm, d_dt, d_db]
