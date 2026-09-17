import { SimulationParameters, TimePointResult, SimulationMetrics, RegionalDeposition } from '../types';
import { calculateRegionalDeposition } from '../models/aerosol';
import { computeEffectiveRates, evaluateDerivatives, EffectiveRates } from '../models/pharmacokinetics';
import { calculateEfficacy, calculateAUC } from '../models/pharmacodynamics';

export interface SimulationOutput {
  timeSeries: TimePointResult[];
  metrics: SimulationMetrics;
  deposition: RegionalDeposition;
  effectiveRates: EffectiveRates;
}

/**
 * High-precision 4th-Order Runge-Kutta (RK4) numerical ODE solver
 * designed to emulate scipy.integrate.solve_ivp across the 0-24h (or user defined) domain.
 */
export function runSimulation(params: SimulationParameters): SimulationOutput {
  // 1. Aerosol deposition calculation
  const deposition = calculateRegionalDeposition(params.mmad, params.inhaledDose);

  // Initial condition: Nanoparticles deposited in the respiratory tract
  // Upper airway fraction is swallowed/cleared directly via oropharynx;
  // conducting bronchial + deep alveolar fractions enter mucosal reservoir Nm(0).
  const initialNm = deposition.totalLungDose;
  const initialDm = 0;
  const initialDt = 0;
  const initialDb = 0;

  // 2. Compute modified kinetic rates based on disease factor V and alpha
  const effectiveRates = computeEffectiveRates(params);

  // 3. RK4 integration setup
  const tStart = 0;
  const tEnd = params.simulationHours || 24;
  const numSteps = 240; // 0.1h step for smooth 24h trajectory
  const h = (tEnd - tStart) / numSteps;

  const timeSeries: TimePointResult[] = [];

  let currentState: [number, number, number, number] = [initialNm, initialDm, initialDt, initialDb];
  let currentTime = tStart;

  // Record t = 0
  const initialEfficacy = calculateEfficacy(initialDt, params.emax, params.ec50);
  timeSeries.push({
    time: 0,
    nm: parseFloat(initialNm.toFixed(4)),
    dm: parseFloat(initialDm.toFixed(4)),
    dt: parseFloat(initialDt.toFixed(4)),
    db: parseFloat(initialDb.toFixed(4)),
    efficacy: parseFloat(initialEfficacy.toFixed(2)),
    totalPulmonaryDrug: parseFloat((initialNm + initialDm + initialDt).toFixed(4)),
  });

  for (let i = 0; i < numSteps; i++) {
    // RK4 step
    const k1 = evaluateDerivatives(currentTime, currentState, effectiveRates);

    const stateK2: [number, number, number, number] = [
      Math.max(0, currentState[0] + 0.5 * h * k1[0]),
      Math.max(0, currentState[1] + 0.5 * h * k1[1]),
      Math.max(0, currentState[2] + 0.5 * h * k1[2]),
      Math.max(0, currentState[3] + 0.5 * h * k1[3]),
    ];
    const k2 = evaluateDerivatives(currentTime + 0.5 * h, stateK2, effectiveRates);

    const stateK3: [number, number, number, number] = [
      Math.max(0, currentState[0] + 0.5 * h * k2[0]),
      Math.max(0, currentState[1] + 0.5 * h * k2[1]),
      Math.max(0, currentState[2] + 0.5 * h * k2[2]),
      Math.max(0, currentState[3] + 0.5 * h * k2[3]),
    ];
    const k3 = evaluateDerivatives(currentTime + 0.5 * h, stateK3, effectiveRates);

    const stateK4: [number, number, number, number] = [
      Math.max(0, currentState[0] + h * k3[0]),
      Math.max(0, currentState[1] + h * k3[1]),
      Math.max(0, currentState[2] + h * k3[2]),
      Math.max(0, currentState[3] + h * k3[3]),
    ];
    const k4 = evaluateDerivatives(currentTime + h, stateK4, effectiveRates);

    // Update state with non-negativity constraint
    currentState = [
      Math.max(0, currentState[0] + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])),
      Math.max(0, currentState[1] + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])),
      Math.max(0, currentState[2] + (h / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])),
      Math.max(0, currentState[3] + (h / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])),
    ];

    currentTime += h;

    const eff = calculateEfficacy(currentState[2], params.emax, params.ec50);

    timeSeries.push({
      time: parseFloat(currentTime.toFixed(2)),
      nm: parseFloat(currentState[0].toFixed(4)),
      dm: parseFloat(currentState[1].toFixed(4)),
      dt: parseFloat(currentState[2].toFixed(4)),
      db: parseFloat(currentState[3].toFixed(4)),
      efficacy: parseFloat(eff.toFixed(2)),
      totalPulmonaryDrug: parseFloat((currentState[0] + currentState[1] + currentState[2]).toFixed(4)),
    });
  }

  // Calculate summary metrics
  let peakTissueConcentration = 0;
  let tMaxTissue = 0;
  let maxPulmonaryEfficacy = 0;
  let cMaxBlood = 0;
  let timeAbove50PctEfficacy = 0;

  for (let i = 0; i < timeSeries.length; i++) {
    const pt = timeSeries[i];
    if (pt.dt > peakTissueConcentration) {
      peakTissueConcentration = pt.dt;
      tMaxTissue = pt.time;
    }
    if (pt.efficacy > maxPulmonaryEfficacy) {
      maxPulmonaryEfficacy = pt.efficacy;
    }
    if (pt.db > cMaxBlood) {
      cMaxBlood = pt.db;
    }
    if (pt.efficacy >= 50) {
      // trapezoidal time accumulation
      timeAbove50PctEfficacy += (i > 0 ? pt.time - timeSeries[i - 1].time : 0);
    }
  }

  const aucTissue = calculateAUC(timeSeries.map((t) => ({ time: t.time, val: t.dt })));
  const aucBlood = calculateAUC(timeSeries.map((t) => ({ time: t.time, val: t.db })));

  const metrics: SimulationMetrics = {
    alveolarDepositionPct: parseFloat((deposition.alveolar * 100).toFixed(1)),
    peakTissueConcentration: parseFloat(peakTissueConcentration.toFixed(3)),
    tMaxTissue: parseFloat(tMaxTissue.toFixed(1)),
    maxPulmonaryEfficacy: parseFloat(maxPulmonaryEfficacy.toFixed(1)),
    mucociliaryLossPct: parseFloat(effectiveRates.mucociliaryLossPct.toFixed(1)),
    aucTissue: parseFloat(aucTissue.toFixed(3)),
    aucBlood: parseFloat(aucBlood.toFixed(3)),
    timeAbove50PctEfficacy: parseFloat(timeAbove50PctEfficacy.toFixed(1)),
    cMaxBlood: parseFloat(cMaxBlood.toFixed(3)),
    effectiveKClear: parseFloat(effectiveRates.kClearEff.toFixed(3)),
    effectiveKDiff: parseFloat(effectiveRates.kDiffEff.toFixed(3)),
  };

  return {
    timeSeries,
    metrics,
    deposition,
    effectiveRates,
  };
}
