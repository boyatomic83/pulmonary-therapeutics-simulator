import { SimulationParameters } from '../types';

export interface EffectiveRates {
  kClearEff: number;
  kClearDrugEff: number;
  kDiffEff: number;
  kRelease: number;
  kBlood: number;
  kTissueLoss: number;
  kSystemicClear: number;
  mucociliaryLossPct: number;
}

/**
 * Computes effective kinetic rate constants by applying the disease-specific
 * mucus pathological viscosity modifier V and sensitivity exponent alpha.
 */
export function computeEffectiveRates(params: SimulationParameters): EffectiveRates {
  const V = Math.max(0.1, params.mucusViscosityFactor);
  const alpha = params.diffusionSensitivityAlpha;

  // Higher viscosity reduces mucociliary velocity & clearance
  const kClearEff = params.kClearNormal / V;
  const kClearDrugEff = params.kClearDrugNormal / V;

  // Higher viscosity and dense mucin crosslinking impedes macromolecule / nanoparticle drug diffusion
  const kDiffEff = params.kDiffNormal / Math.pow(V, alpha);

  // Mucociliary loss percentage relative to normal baseline (V = 1.0)
  const mucociliaryLossPct = Math.max(0, (1 - kClearEff / params.kClearNormal) * 100);

  return {
    kClearEff,
    kClearDrugEff,
    kDiffEff,
    kRelease: params.kRelease,
    kBlood: params.kBlood,
    kTissueLoss: params.kTissueLoss,
    kSystemicClear: params.kSystemicClear,
    mucociliaryLossPct,
  };
}

/**
 * Evaluates the 4-compartment coupled ODE derivatives at a given state.
 * State vector: [Nm, Dm, Dt, Db]
 * 
 * dNm/dt = - k_release * Nm - k_clear * Nm
 * dDm/dt = k_release * Nm - k_diff * Dm - k_clear,D * Dm
 * dDt/dt = k_diff * Dm - k_blood * Dt - k_tissue-loss * Dt
 * dDb/dt = k_blood * Dt - k_systemic-clear * Db
 */
export function evaluateDerivatives(
  _t: number,
  state: [number, number, number, number],
  rates: EffectiveRates
): [number, number, number, number] {
  const [nm, dm, dt, db] = state;

  const dNm = -rates.kRelease * nm - rates.kClearEff * nm;
  const dDm = rates.kRelease * nm - rates.kDiffEff * dm - rates.kClearDrugEff * dm;
  const dDt = rates.kDiffEff * dm - rates.kBlood * dt - rates.kTissueLoss * dt;
  const dDb = rates.kBlood * dt - rates.kSystemicClear * db;

  return [dNm, dDm, dDt, dDb];
}
