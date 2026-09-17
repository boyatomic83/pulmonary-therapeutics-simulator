import { RegionalDeposition } from '../types';

/**
 * Calculates regional aerosol deposition fractions as a function of
 * Mass Median Aerodynamic Diameter (MMAD in micrometers, 0.5 - 10.0 um).
 * Incorporates inertial impaction (dominant in upper airways for large particles)
 * and gravitational sedimentation (dominant in peripheral / alveolar airways).
 * 
 * Enforces: F_upper + F_bronchial + F_alveolar = 1.0
 */
export function calculateRegionalDeposition(
  mmad: number,
  inhaledDose: number
): RegionalDeposition {
  // Clamp MMAD within reasonable range
  const d = Math.max(0.2, Math.min(12.0, mmad));

  // Empirical mathematical model of deposition based on ICRP / Heyder lung deposition curves:
  // Upper airways: inertial impaction scales strongly with d^2.5
  const rawUpper = Math.pow(d, 2.5) / (Math.pow(d, 2.5) + Math.pow(3.3, 2.5));

  // Deep alveolar: optimal penetration occurs between 1.5 - 3.0 um, then declines due to upstream filtering
  const rawAlveolar = (1.55 * Math.pow(d, 1.8)) / (1.0 + Math.pow(d / 2.3, 3.5));

  // Conducting bronchial airways: intermediate impaction & sedimentation
  const rawBronchial = (0.85 * Math.pow(d, 1.3)) / (1.0 + Math.pow(d / 3.8, 2.4));

  // Exact normalization to guarantee sum = 1.0
  const sum = rawUpper + rawBronchial + rawAlveolar;
  const upper = rawUpper / sum;
  const bronchial = rawBronchial / sum;
  const alveolar = rawAlveolar / sum;

  const totalLungDose = inhaledDose * (bronchial + alveolar);
  const alveolarDose = inhaledDose * alveolar;
  const upperDose = inhaledDose * upper;

  return {
    upper,
    bronchial,
    alveolar,
    totalLungDose,
    alveolarDose,
    upperDose,
  };
}

/**
 * Generates continuous deposition curves across the 0.5 - 10.0 um range
 * for visualization and educational comparison.
 */
export function generateDepositionCurve(points: number = 80) {
  const curve: Array<{
    mmad: number;
    upperPct: number;
    bronchialPct: number;
    alveolarPct: number;
  }> = [];

  const minD = 0.5;
  const maxD = 10.0;
  const step = (maxD - minD) / (points - 1);

  for (let i = 0; i < points; i++) {
    const d = minD + i * step;
    const dep = calculateRegionalDeposition(d, 1.0);
    curve.push({
      mmad: parseFloat(d.toFixed(2)),
      upperPct: parseFloat((dep.upper * 100).toFixed(1)),
      bronchialPct: parseFloat((dep.bronchial * 100).toFixed(1)),
      alveolarPct: parseFloat((dep.alveolar * 100).toFixed(1)),
    });
  }

  return curve;
}
