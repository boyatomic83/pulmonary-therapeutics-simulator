"""
Pharmacodynamics — Local Pulmonary Anti-Inflammatory Efficacy
Emax Hill Model:
  E(Dt) = (Emax * Dt) / (EC50 + Dt)
"""

import numpy as np

def calculate_efficacy(dt_array: np.ndarray, emax: float = 100.0, ec50: float = 0.5) -> np.ndarray:
    """
    Computes anti-inflammatory efficacy percentage as a function of target tissue concentration Dt.
    
    Parameters:
        dt_array (np.ndarray or float): Target tissue concentration or amount Dt (mg).
        emax (float): Maximum achievable therapeutic effect (%) [default 100%].
        ec50 (float): Tissue concentration producing 50% of Emax (mg).
        
    Returns:
        np.ndarray: Anti-inflammatory efficacy E (%)
    """
    dt_clean = np.maximum(0.0, np.asarray(dt_array, dtype=float))
    denominator = ec50 + dt_clean
    # Avoid zero division
    denominator = np.where(denominator <= 0.0, 1e-9, denominator)
    return (emax * dt_clean) / denominator

def calculate_auc(time_grid: np.ndarray, values: np.ndarray) -> float:
    """
    Computes trapezoidal Area Under the Curve (AUC).
    """
    return float(np.trapz(values, time_grid))
