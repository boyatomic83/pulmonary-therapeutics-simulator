export interface SimulationParameters {
  // Particle Parameters
  mmad: number; // Mass Median Aerodynamic Diameter in micrometers (0.5 - 10.0)
  inhaledDose: number; // Inhaled dose in mg

  // Disease Modifier
  mucusViscosityFactor: number; // Viscosity factor V (1.0 = normal, >1.0 = pathological)
  diffusionSensitivityAlpha: number; // alpha exponent in k_diff,eff = k_diff / V^alpha

  // Nanoparticle Parameter
  kRelease: number; // Payload release rate (1/h)

  // Pharmacokinetic Rate Constants (Baseline)
  kClearNormal: number; // Normal mucociliary clearance of nanoparticles (1/h)
  kClearDrugNormal: number; // Clearance of free drug in mucus (1/h)
  kDiffNormal: number; // Normal diffusion rate from mucus to epithelial tissue (1/h)
  kBlood: number; // Transfer rate from tissue into systemic blood (1/h)
  kTissueLoss: number; // Local tissue metabolism / non-systemic loss (1/h)
  kSystemicClear: number; // Elimination rate from systemic circulation (1/h)

  // Pharmacodynamic Parameters
  emax: number; // Maximum achievable therapeutic effect (%)
  ec50: number; // Tissue amount/concentration producing 50% of Emax (mg)

  // Simulation Controls
  simulationHours: number; // Total simulation duration (hours, default 24)
}

export interface RegionalDeposition {
  upper: number; // Fraction in Upper Airways (0 - 1)
  bronchial: number; // Fraction in Conducting Bronchial Airways (0 - 1)
  alveolar: number; // Fraction in Deep Alveolar Region (0 - 1)
  totalLungDose: number; // mg reaching bronchial + alveolar
  alveolarDose: number; // mg reaching alveolar
  upperDose: number; // mg impacted in upper airways
}

export interface TimePointResult {
  time: number; // hours
  nm: number; // Nanoparticle payload in mucus (mg)
  dm: number; // Free drug in mucus (mg)
  dt: number; // Target epithelial tissue (mg)
  db: number; // Systemic blood exposure (mg)
  efficacy: number; // Anti-inflammatory efficacy E(Dt) in %
  totalPulmonaryDrug: number; // nm + dm + dt
}

export interface SimulationMetrics {
  alveolarDepositionPct: number; // %
  peakTissueConcentration: number; // max(Dt) in mg
  tMaxTissue: number; // time of peak tissue concentration in hours
  maxPulmonaryEfficacy: number; // max(E) in %
  mucociliaryLossPct: number; // % reduction in clearance
  aucTissue: number; // mg * h (area under Dt curve)
  aucBlood: number; // mg * h (area under Db curve)
  timeAbove50PctEfficacy: number; // hours where E >= 50%
  cMaxBlood: number; // max(Db) in mg
  effectiveKClear: number; // 1/h
  effectiveKDiff: number; // 1/h
}

export interface ClinicalPreset {
  id: string;
  name: string;
  badge: string;
  description: string;
  params: Partial<SimulationParameters>;
}
