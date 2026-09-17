/**
 * Pharmacodynamics — Local Anti-Inflammatory Efficacy
 * Computes therapeutic effect using the classical Emax Hill equation:
 * E(Dt) = (Emax * Dt) / (EC50 + Dt)
 */
export function calculateEfficacy(dt: number, emax: number, ec50: number): number {
  if (dt <= 0 || ec50 <= 0) return 0;
  return (emax * dt) / (ec50 + dt);
}

/**
 * Calculates trapezoidal Area Under the Curve (AUC) across an array of (time, value) pairs.
 */
export function calculateAUC(points: Array<{ time: number; val: number }>): number {
  let auc = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const dt = points[i + 1].time - points[i].time;
    const avgVal = (points[i].val + points[i + 1].val) / 2;
    auc += avgVal * dt;
  }
  return auc;
}
