# Inhaled Nanoparticle & Biological Respiratory Therapeutics Simulator

A biophysical and computational pharmacometrics simulation platform engineered in **Python, Streamlit, NumPy, SciPy, Plotly, and Pandas**.

The simulator integrates **aerosol deposition physics, pulmonary mucosal barrier kinetics, four-compartment pharmacokinetic (PK) ordinary differential equations (ODEs), disease-state pathophysiological modifiers, and local anti-inflammatory pharmacodynamics (PD)** into an interactive dashboard with 3D anatomical visualization and clinical safety hazard monitoring.

> **Scientific Modeling Notice:** This platform is an educational and computational simulation. All physiological presets, kinetic rate constants ($k_{\text{release}}, k_{\text{clear}}, k_{\text{diff}}$), mucus viscosity modifiers ($V$), and systemic hazard thresholds ($D_b > 5.0\ \text{mg/L}$) represent **illustrative simulation assumptions**. They do not constitute clinical dosing recommendations, medical guidance, or regulatory safety criteria.

---

## Table of Contents
1. [Executive Overview & Scientific Motivation](#1-executive-overview--scientific-motivation)
2. [Respiratory Anatomy & the Weibel Zonal Architecture (G0–G23)](#2-respiratory-anatomy--the-weibel-zonal-architecture-g0g23)
3. [The Pulmonary Mucosal Barrier & Cellular Microenvironment](#3-the-pulmonary-mucosal-barrier--cellular-microenvironment)
4. [Biophysics of Aerosol Deposition](#4-biophysics-of-aerosol-deposition)
   - [Aerodynamic Diameter Formulation](#aerodynamic-diameter-formulation)
   - [Inertial Impaction (Stokes Number)](#inertial-impaction-stokes-number)
   - [Gravitational Sedimentation](#gravitational-sedimentation)
   - [Brownian Diffusion](#brownian-diffusion)
   - [Airway Caliber & Ventilation Scaling](#airway-caliber--ventilation-scaling)
5. [Four-Compartment Coupled ODE Pharmacokinetic Engine](#5-four-compartment-coupled-ode-pharmacokinetic-engine)
   - [Compartmental State Variables](#compartmental-state-variables)
   - [System of Differential Equations](#system-of-differential-equations)
   - [Numerical Integration Scheme (RK45)](#numerical-integration-scheme-rk45)
6. [Disease-Specific Pathophysiology & Mathematical Modifiers](#6-disease-specific-pathophysiology--mathematical-modifiers)
   - [Healthy Adult Baseline](#healthy-adult-baseline)
   - [Cystic Fibrosis (CF)](#cystic-fibrosis-cf)
   - [Severe Asthma & COPD](#severe-asthma--copd)
   - [Pediatric / Infant Physiology](#pediatric--infant-physiology)
   - [Mucus Viscosity & Clearance Attenuation](#mucus-viscosity--clearance-attenuation)
7. [Pharmacodynamics: Local Target Inhibition & Safety Window](#7-pharmacodynamics-local-target-inhibition--safety-window)
   - [Emax Hill Model](#emax-hill-model)
   - [Model-Defined Efficacy Zones](#model-defined-efficacy-zones)
   - [Systemic Safety Hazard Monitoring](#systemic-safety-hazard-monitoring)
8. [Interactive 3D Airway Visualizer: How It Works](#8-interactive-3d-airway-visualizer-how-it-works)
9. [Simulation Dossier Export & Data Schema](#9-simulation-dossier-export--data-schema)
10. [Software Architecture & Module Map](#10-software-architecture--module-map)
11. [Installation & Local Execution Guide](#11-installation--local-execution-guide)

---

## 1. Executive Overview & Scientific Motivation

Inhalation drug delivery provides an attractive route for administering small molecules, monoclonal antibodies, peptides, and nanocarriers (lipid nanoparticles, polymeric micelles, solid lipid nanoparticles). Delivering therapeutics directly to the respiratory tract achieves **high local drug concentrations in target epithelial tissues** while minimizing systemic exposure and toxicity.

However, delivering inhaled biotherapeutics presents formidable physical and biological hurdles:

$$\boxed{\text{Inhaled Aerosol}} \xrightarrow{\text{Aerodynamic Deposition}} \boxed{\text{Mucus Gel Phase}} \xrightarrow{\text{Enzyme / Escalator Clearance}} \boxed{\text{Epithelial Uptake}} \xrightarrow{\text{Receptor Inhibition}} \boxed{\text{Capillary Spillover}}$$

1. **Aerodynamic Filtration:** Particles larger than $5\ \mu\text{m}$ impact prematurely in the oropharynx and larynx, resulting in swallowing and gastrointestinal loss. Particles below $1\ \mu\text{m}$ remain suspended and are largely exhaled.
2. **The Mucociliary Barrier:** Inhaled carriers deposit on an adhesive, moving viscoelastic mucus blanket propelled by coordinated ciliary beating toward the pharynx. If drug release is too slow, nanoparticles are swept away before the therapeutic cargo reaches the underlying epithelium.
3. **Steric & Hydrodynamic Hindrance:** In chronic respiratory diseases like Cystic Fibrosis (CF), Asthma, and Chronic Obstructive Pulmonary Disease (COPD), mucus hypersecretion and dehydration elevate viscosity up to 10-fold, severely impeding passive drug diffusion to the epithelium.
4. **Systemic Absorption & Off-Target Toxicity:** Unbound drug diffusing across the alveolar-capillary membrane enters the pulmonary microcirculation and reaches systemic circulation, presenting potential safety hazards if blood concentrations exceed tolerable thresholds.

This simulator mechanistically couples these interdependent biological, physical, and pharmacological processes into a unified computational model.

---

## 2. Respiratory Anatomy & the Weibel Zonal Architecture (G0–G23)

The human respiratory tree is structurally parameterized using E.R. Weibel's symmetric dichotomous branching model, encompassing 24 distinct branching generations ($G_0$ to $G_{23}$):

```
       [ Trachea (G0) ]
              │
      ┌───────┴───────┐
   Left Main       Right Main
     (G1)            (G1)
      │               │
   Lobar (G2-G4)   Lobar (G2-G4)       <-- UPPER AIRWAYS (G0 - G4)
      │               │
  Segmental & Terminal Bronchioles     <-- CONDUCTING ZONE (G5 - G16)
         (G5 - G16)
      │               │
  Respiratory Bronchioles & Alveoli    <-- RESPIRATORY / ACINAR ZONE (G17 - G23)
         (G17 - G23)
```

| Anatomical Zone | Generations | Primary Structures | Epithelial Morphology | Predominant Clearance Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Upper Airway Zone** | $G_0 - G_4$ | Trachea, main bronchi, lobar/segmental bronchi | Pseudostratified ciliated columnar with goblet cells | Fast mucociliary escalator ($u_c \approx 5\text{–}20\ \text{mm/min}$), cough |
| **Conducting Bronchial Zone** | $G_5 - G_{16}$ | Subsegmental bronchi, small bronchioles, terminal bronchioles | Simple ciliated cuboidal, club (Clara) cells | Moderate mucociliary transport ($u_c \approx 0.5\text{–}2\ \text{mm/min}$) |
| **Deep Alveolar Zone** | $G_{17} - G_{23}$ | Respiratory bronchioles, alveolar ducts, alveolar sacs | Type I (gas exchange) & Type II (surfactant) pneumocytes | Alveolar macrophage phagocytosis, surfactant recycling, lymphatic drainage (No cilia) |

- **Anatomical Dead Space:** Generations $G_0 - G_{16}$ contain $\approx 150\ \text{mL}$ of air where no gas exchange occurs; their primary function is conditioning, humidification, and particle filtration.
- **Surface Area Escalation:** As generation number increases from $G_0$ to $G_{23}$, total cross-sectional area increases exponentially from $\approx 2.5\ \text{cm}^2$ (trachea) to over $100\ \text{m}^2$ (alveoli). Consequently, bulk linear airflow velocity drops by more than three orders of magnitude in the deep lungs, transitioning particle transport from convective inertia to quiet sedimentation.

---

## 3. The Pulmonary Mucosal Barrier & Cellular Microenvironment

### The Two-Phase Viscoelastic Mucus Layer
The tracheobronchial airway surface liquid (ASL) consists of two distinct continuous micro-phases:
1. **The Periciliary Liquid (PCL) Layer ($\approx 7\ \mu\text{m}$ thick):** A low-viscosity, water-like aqueous sol bathing the cilia. The PCL contains tethered cell-surface mucins (MUC1, MUC4, MUC16) that form a protective steric brush, preventing the overlying gel from compressing the cilia.
2. **The Mucus Gel Blanket ($\approx 2\text{–}10\ \mu\text{m}$ thick):** A viscoelastic gel composed of high-molecular-weight polymeric mucins (**MUC5AC** produced by surface goblet cells and **MUC5B** secreted by submucosal glands). Disulfide-linked mucin monomers form a dynamic, cross-linked meshwork with pore sizes typically ranging from $100\ \text{nm}$ to $500\ \text{nm}$.

### The Ciliary Escalator
Airway epithelial cilia beat metachronously at **12–15 Hz** in a coordinated effective stroke. Ciliary tips engage the underside of the mucus gel layer during forward motion, propelling entrapped foreign debris, microorganisms, and insoluble nanoparticles upward toward the pharynx, where they are swallowed into the gastrointestinal tract.

### Epithelial Permeation vs. Systemic Loss
Beneath the mucus and PCL lies the epithelial cell monolayer:
- **Tight Junctions:** Claudins, occludins, and zonula occludens (ZO-1) form tight intercellular seals that restrict paracellular transport of charged biologicals and nanocarriers.
- **Transcellular Diffusion:** Hydrophobic small molecules and unencapsulated drugs partition across the apical membrane, diffuse through the cytoplasm, and exit into the interstitial space to engage tissue target receptors ($D_t$).
- **Vascular Uptake:** The pulmonary capillary network has an ultra-thin barrier ($0.2\text{–}0.5\ \mu\text{m}$ in alveoli), enabling rapid absorption of dissolved drug directly into the systemic circulation ($D_b$).

---

## 4. Biophysics of Aerosol Deposition

Aerosolized therapeutic particles inhale into the lung as an aerosol cloud. Regional deposition fraction depends on the particle's **Mass Median Aerodynamic Diameter (MMAD)**, inspiratory flow rate, airway lumen diameter, and lung volume.

### Aerodynamic Diameter Formulation
The aerodynamic diameter ($d_{ae}$) normalizes particles of arbitrary physical size ($d_{geo}$), density ($\rho_p$), and dynamic shape factor ($\chi$) to an equivalent unit-density sphere ($\rho_0 = 1.0\ \text{g/cm}^3$):

$$d_{ae} = d_{geo} \sqrt{\frac{\rho_p}{\rho_0 \chi}}$$

Nanoparticle therapeutics are rarely inhaled as isolated $50\ \text{nm}$ particles; instead, they are formulated as aerosolized micro-droplets (nebulizers) or porous micro-aggregates (dry powder inhalers) with aerodynamic diameters between **$0.5\ \mu\text{m}$ and $10.0\ \mu\text{m}$**.

```
Relative Deposition (%)
 100% ┌─────────────────────────────────────────────────────────────┐
      │                                    Upper Airway Impaction   │
  80% │                                          /                  │
      │                                         /                   │
  60% │            Alveolar Sedimentation      /                    │
      │                  /----\               /                     │
  40% │                 /      \             /                      │
      │   Exhaled      /        \-----\     /                       │
  20% │    Plume      /                \---/  Conducting Bronchial  │
      │    \         /                                              │
   0% └─────\───────/───────────────────────────────────────────────┘
     0.5   1.0     2.0    3.0    4.0    5.0   6.0   7.0   8.0  10.0
                              MMAD (μm)
```

### Inertial Impaction (Stokes Number)
Inertial impaction dominates in **Generations $G_0 - G_4$** where linear air velocity $u$ is high and airflow direction changes abruptly at bifurcations. A particle with excessive momentum cannot follow the curving streamlines and collides with the airway wall:

$$Stk = \frac{\rho_p d_{ae}^2 u C_c}{18 \mu D_{\text{airway}}}$$

Where:
- $\rho_p$ is particle density
- $u$ is mean linear airflow velocity
- $\mu$ is dynamic air viscosity ($1.81 \times 10^{-5}\ \text{Pa}\cdot\text{s}$)
- $D_{\text{airway}}$ is airway lumen diameter
- $C_c$ is the Cunningham slip correction factor

Impaction probability scales strongly with particle size squared ($d_{ae}^2$) and air velocity ($u$). For **$\text{MMAD} > 5\ \mu\text{m}$**, upper airway impaction accounts for over $80\%$ of deposited mass.

### Gravitational Sedimentation
In the **peripheral and alveolar regions ($G_{17} - G_{23}$)**, the massive expansion in airway cross-section reduces air velocity to near zero ($u \to 0$). In this quiet regime, gravitational settling dominates over residence times:

$$v_s = \frac{\rho_p d_{ae}^2 g C_c}{18 \mu}$$

Where $g$ is gravitational acceleration ($9.81\ \text{m/s}^2$). Sedimentation is most efficient for particles sized **$1.5\ \mu\text{m} - 3.5\ \mu\text{m}$**, where particles have sufficient mass to settle out before exhalation, yet are small enough to escape upper airway impaction.

### Brownian Diffusion
For sub-micron aerosols ($d_{ae} < 0.5\ \mu\text{m}$), particles undergo Brownian motion driven by thermal collisions with gas molecules. The diffusion coefficient is described by the Stokes-Einstein equation:

$$D_{\text{diff}} = \frac{k_B T C_c}{3 \pi \mu d_{geo}}$$

Where $k_B$ is the Boltzmann constant and $T$ is temperature ($310\ \text{K}$). Particles between $0.3\ \mu\text{m}$ and $0.8\ \mu\text{m}$ fall into a "deposition minimum" (neither heavy enough to settle rapidly nor light enough to diffuse quickly), resulting in high exhaled fractions ($\approx 15\text{–}25\%$).

### Airway Caliber & Ventilation Scaling
When airway caliber is reduced (e.g., in asthma or pediatric airways), linear velocity $u$ increases inversely with cross-sectional area:

$$u \propto \frac{1}{R_{\text{airway}}^2} \implies Stk \propto \frac{d_{ae}^2}{R_{\text{airway}}^3}$$

This cubic dependence on airway radius ($1/R^3$) means that **even mild airway constriction markedly amplifies inertial impaction** in conducting bronchi, shifting the deposition profile upward and depriving the peripheral alveoli of drug payload.

### Mathematical Mass Conservation
The model enforces strict conservation of mass across all deposition compartments:

$$F_{\text{upper}} + F_{\text{bronchial}} + F_{\text{alveolar}} = 1.0 \quad (\text{normalized deposited fraction})$$

$$F_{\text{upper}} + F_{\text{bronchial}} + F_{\text{alveolar}} + F_{\text{exhaled}} = 1.0 \quad (\text{total inhaled fraction})$$

$$\text{Dose}_{\text{inhaled}} = \text{Dose}_{\text{upper}} + \text{Dose}_{\text{bronchial}} + \text{Dose}_{\text{alveolar}} + \text{Dose}_{\text{exhaled}}$$

---

## 5. Four-Compartment Coupled ODE Pharmacokinetic Engine

Once deposited in the pulmonary tree, the therapeutic formulation undergoes dissolution, degradation, clearance, and transport across four coupled kinetic compartments:

```
           [ Inhaled Aerosol Dose ]
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
[ Upper Airway Impaction ]    [ Deposited in Mucus ]
 (G0 - G4: Swallowed)          (Bronchial & Alveolar)
                                    │
                                    ▼
                         ┌────────────────────┐
                         │  Nm: Nanoparticles │ ──(k_clear,eff)──> Swallowed / Cleared
                         │      in Mucus      │
                         └────────────────────┘
                                    │ (k_release)
                                    ▼
                         ┌────────────────────┐
                         │  Dm: Free Drug in  │ ──(k_clear,D)───> Swallowed / Cleared
                         │       Mucus        │
                         └────────────────────┘
                                    │ (k_diff,eff)
                                    ▼
                         ┌────────────────────┐
                         │ Dt: Target Tissue  │ ──(k_tissue-loss)─> Local Metabolism
                         │     Epithelium     │
                         └────────────────────┘
                                    │ (k_blood)
                                    ▼
                         ┌────────────────────┐
                         │ Db: Systemic Blood │ ──(k_systemic-clear)─> Renal / Hepatic
                         │     Circulation    │                        Elimination
                         └────────────────────┘
```

### Compartmental State Variables

1. **$N_m(t)$ [$\text{mg}$]: Nanoparticle Reservoir in Mucus**
   Represents unreleased therapeutic encapsulated inside intact nanoparticles residing in the bronchial and alveolar lining fluid.
   Initial condition: $N_m(0) = \text{Dose}_{\text{pulmonary}} = \text{Dose}_{\text{bronchial}} + \text{Dose}_{\text{alveolar}}$.

2. **$D_m(t)$ [$\text{mg}$]: Dissolved / Free Drug in Mucus**
   Represents molecular drug liberated from nanocarriers into the mucus blanket through diffusion, matrix erosion, or ester cleavage.
   Initial condition: $D_m(0) = 0$.

3. **$D_t(t)$ [$\text{mg}$]: Target Epithelial Tissue Concentration**
   Active therapeutic that has successfully diffused across the mucus and apical membranes into the airway epithelium and sub-epithelial parenchyma. This is the pharmacodynamically active driver of efficacy.
   Initial condition: $D_t(0) = 0$.

4. **$D_b(t)$ [$\text{mg/L}$ or $\text{mg}$]: Systemic Blood / Plasma Exposure**
   Therapeutic that has permeated into the bronchial circulation or alveolar capillary beds, distributing into the systemic vascular volume. High levels represent risk of systemic adverse events.
   Initial condition: $D_b(0) = 0$.

### System of Differential Equations

$$\frac{dN_m}{dt} = - k_{\text{release}} N_m - k_{\text{clear,eff}} N_m$$

$$\frac{dD_m}{dt} = k_{\text{release}} N_m - k_{\text{diff,eff}} D_m - k_{\text{clear},D,\text{eff}} D_m$$

$$\frac{dD_t}{dt} = k_{\text{diff,eff}} D_m - k_{\text{blood}} D_t - k_{\text{tissue-loss}} D_t$$

$$\frac{dD_b}{dt} = k_{\text{blood}} D_t - k_{\text{systemic-clear}} D_b$$

### Biological Rate Constants

| Parameter | Symbol | Nominal Baseline | Units | Biological Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **Payload Release Rate** | $k_{\text{release}}$ | $0.35$ | $\text{h}^{-1}$ | Rate of therapeutic release from the nanoparticle carrier into mucus. Formulations with low $k_{\text{release}}$ (e.g. $0.05\ \text{h}^{-1}$) exhibit sustained-release behavior. |
| **Mucociliary Particle Clearance** | $k_{\text{clear,normal}}$ | $0.25$ | $\text{h}^{-1}$ | Normal velocity of the mucociliary escalator removing intact carriers out of the lungs. |
| **Free Drug Mucosal Clearance** | $k_{\text{clear},D,\text{normal}}$ | $0.40$ | $\text{h}^{-1}$ | Rapid clearance of small molecules dissolved in moving mucus. |
| **Mucosal Epithelial Diffusion** | $k_{\text{diff,normal}}$ | $0.45$ | $\text{h}^{-1}$ | Fickian mass transfer rate of free drug diffusing through mucus into the epithelial cells. |
| **Tissue-to-Blood Absorption** | $k_{\text{blood}}$ | $0.15$ | $\text{h}^{-1}$ | Vascular uptake across the capillary endothelium into systemic circulation. |
| **Local Tissue Metabolism** | $k_{\text{tissue-loss}}$ | $0.08$ | $\text{h}^{-1}$ | Degradation by respiratory esterases/peptidases and cellular receptor internalization. |
| **Systemic Elimination** | $k_{\text{systemic-clear}}$ | $0.30$ | $\text{h}^{-1}$ | First-order renal filtration and hepatic metabolic clearance from systemic blood. |

### Numerical Integration Scheme (RK45)
The coupled linear system exhibits time-scale separation: rapid dissolution and clearance compete with slower systemic distribution. The system is numerically integrated using the **Runge-Kutta-Fehlberg 4(5) (RK45)** algorithm provided by `scipy.integrate.solve_ivp`:
- **Local Error Estimation:** Fifth-order step compared with embedded fourth-order predictor.
- **Tolerances:** Relative tolerance $\text{rtol} = 10^{-6}$; absolute tolerance $\text{atol} = 10^{-8}$.
- **Non-Negativity Constraint:** Derivatives enforce non-negative state variable boundaries ($N_m, D_m, D_t, D_b \ge 0$) to eliminate unphysical numerical oscillations near zero.

---

## 6. Disease-Specific Pathophysiology & Mathematical Modifiers

Pulmonary disease states drastically alter mucus rheology, epithelial barrier integrity, and airway geometry. The simulator models these changes using physiological scaling factors:

```
                             [ Pathological Viscosity Factor: V ]
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
      k_clear,eff = (k_clear * Factor) / V               k_diff,eff = k_diff / (V ^ α)
             (Escalator Stasis)                                (Steric Hindrance)
```

### Disease Presets Matrix

| Parameter | Healthy Adult | Cystic Fibrosis (CF) | Severe Asthma / COPD | Infant / Pediatric |
| :--- | :--- | :--- | :--- | :--- |
| **Mucus Viscosity Factor ($V$)** | $1.0\times$ (Normal) | **$5.0\times$** (Dehydrated gel) | **$2.5\times$** (Hypersecretion) | $1.0\times$ (Configurable) |
| **Mucociliary Clearance Multiplier** | $1.0\times$ (Standard) | **$0.2\times$** (Ciliary stasis) | **$0.5\times$** (Impaired velocity) | $0.8\times$ (Immature cilia) |
| **Airway Radius Factor ($R_{\text{airway}}$)** | $1.0\times$ (Normal lumen) | $1.0\times$ (Standard caliber) | **$0.7\times$** (Bronchoconstriction) | **$0.5\times$** (Pediatric caliber) |
| **Tidal Volume Factor ($V_T$)** | $1.0\times$ ($\approx 500\ \text{mL}$) | $1.0\times$ ($\approx 500\ \text{mL}$) | **$0.85\times$** (Air trapping) | **$0.25\times$** ($\approx 125\ \text{mL}$) |
| **Recommended MMAD** | $2.5\ \mu\text{m}$ | $2.2\ \mu\text{m}$ | $2.8\ \mu\text{m}$ | $1.8\ \mu\text{m}$ |
| **Nominal Inhaled Dose** | $10.0\ \text{mg}$ | $15.0\ \text{mg}$ | $12.0\ \text{mg}$ | $2.5\ \text{mg}$ |
| **Systemic Safety Limit ($D_b$)** | $5.0\ \text{mg/L}$ | $5.0\ \text{mg/L}$ | $5.0\ \text{mg/L}$ | $2.0\ \text{mg/L}$ |

### Biological Rationale for Disease States

#### Healthy Adult Baseline
Represents normal lung physiology with intact mucociliary clearance, hydrated mucus ($97\%$ water, $3\%$ solids), unimpeded mucosal diffusion, and symmetric branching geometry. Serves as the comparator for calculating mucociliary loss percentages.

#### Cystic Fibrosis (CF)
Caused by autosomal recessive mutations in the cystic fibrosis transmembrane conductance regulator (**CFTR**) gene. Defective epithelial chloride and bicarbonate secretion, coupled with unregulated sodium hyper-absorption via ENaC, leads to severe airway surface liquid dehydration.
- **Mucus Hyperviscosity ($V = 5.0\times$):** Mucus solids increase from $3\%$ to over $12\%$. Mucin polymers form dense, rigid mesh networks reinforced by DNA and actin filaments released from lysed necrotic neutrophils (neutrophil extracellular traps, NETs).
- **Ciliary Stasis ($0.2\times$ clearance):** The dehydrated mucus blanket collapses directly onto the ciliated epithelium, crushing the periciliary liquid layer and paralyzing ciliary strokes. While reduced clearance extends nanoparticle residence time in the lung, it severely traps carriers, creating an impenetrable diffusion barrier.

#### Severe Asthma & COPD
Characterized by chronic airway inflammation, goblet cell hyperplasia, and smooth muscle hypertrophy.
- **Bronchoconstriction ($R_{\text{airway}} = 0.7\times$):** Airway narrowing increases local linear airflow velocity, driving higher Stokes numbers and inertial impaction in large bronchi.
- **Mucus Hypersecretion ($V = 2.5\times$):** Overexpression of MUC5AC driven by IL-13 and TH2 cytokines produces sticky mucus plugs that occlude small conducting airways ($G_{10} - G_{14}$).

#### Pediatric / Infant Physiology
Children have much narrower airway calibers ($R_{\text{airway}} \approx 0.5\times$) and lower tidal breathing volumes ($V_T \approx 0.25\times$).
- Narrower airways mean that adult-sized aerosol particles ($3\text{–}5\ \mu\text{m}$) impact heavily in the upper throat and larynx, leaving very little drug to reach the alveolar spaces.
- Pediatric formulations require smaller aerodynamic diameters (**$\text{MMAD} \approx 1.5\text{–}2.0\ \mu\text{m}$**) and reduced dose payloads.

### Mucus Viscosity & Clearance Attenuation
The effective kinetic parameters are dynamically modulated according to:

$$k_{\text{clear,eff}} = \frac{k_{\text{clear,normal}} \times \text{Factor}_{\text{clear}}}{V}$$

$$k_{\text{diff,eff}} = \frac{k_{\text{diff,normal}}}{V^\alpha}$$

Where:
- $V$ is the Pathological Mucus Viscosity Factor ($V \ge 1.0$)
- $\text{Factor}_{\text{clear}}$ is the intrinsic disease clearance multiplier
- $\alpha$ is the diffusion sensitivity exponent ($0.1 \le \alpha \le 1.5$, nominal $\alpha = 0.70$) representing steric obstruction and electrostatic binding to mucin glycan chains

**Mucociliary Loss Percentage:**
$$L_{\text{mucociliary}} = \max\left(0,\ \left(1 - \frac{k_{\text{clear,eff}}}{k_{\text{clear,normal}}}\right) \times 100\%\right)$$

---

## 7. Pharmacodynamics: Local Target Inhibition & Safety Window

### Emax Hill Model
Local pharmacodynamic efficacy represents target receptor occupancy (such as glucocorticoid receptor activation, phosphodiesterase-4 inhibition, or cytokine neutralization) in respiratory epithelial tissue ($D_t$):

$$E(D_t) = \frac{E_{\max} \cdot D_t(t)}{EC_{50} + D_t(t)}$$

Where:
- $E_{\max}$ is the maximal attainable efficacy ($100\%$)
- $EC_{50}$ is the active tissue amount eliciting half-maximal response (nominal $0.50\ \text{mg}$)
- $D_t(t)$ is instantaneous drug mass in the epithelial target tissue

```
Pulmonary Efficacy (%)
 100% ┌─────────────────────────────────────────────────────────────┐
      │                                    High-Efficacy Zone (>90%)│
  90% ├- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -┤
      │                                                             │
      │                  Therapeutic Simulation Window (50% - 90%)   │
      │                                                             │
  50% ├- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -┤
      │                                                             │
      │                     Sub-Therapeutic Zone (<50%)             │
   0% └─────────────────────────────────────────────────────────────┘
      0.0       0.5 (EC50)      1.0            2.0           3.0
                         Target Tissue Drug Mass Dt (mg)
```

### Model-Defined Efficacy Zones
The dashboard partitions response into three distinct zones:
- **Sub-Therapeutic ($E < 50\%$):** Tissue drug mass is insufficient to saturate the target receptor, resulting in inadequate anti-inflammatory suppression.
- **Therapeutic Simulation Window ($50\% \le E \le 90\%$):** Optimal local efficacy achieved while systemic exposure remains below the safety hazard threshold.
- **High-Efficacy Zone ($E > 90\%$):** Receptor saturation reached. Further dosage increases yield diminishing returns in efficacy while increasing the risk of systemic spillover.

### Systemic Safety Hazard Monitoring
Inhaled therapeutics that permeate into pulmonary capillaries accumulate in systemic blood ($D_b$).
- **Configurable Safety Threshold:** A default safety threshold is set at **$D_b = 5.0\ \text{mg/L}$** (or user-configured).
- **Dynamic Warning Engine:** When the simulated blood curve crosses this threshold ($D_b(t) > 5.0\ \text{mg/L}$), the dashboard displays a prominent red warning banner detailing peak exposure and exposure duration. The warning banner clears automatically when $D_b$ returns below the threshold.
- **Exposure Area Under the Curve (AUC):** Evaluated across the 24-hour timeline using the trapezoidal rule:
  $$\text{AUC}_{0-t} = \sum_{i=0}^{n-1} \frac{C(t_i) + C(t_{i+1})}{2} \cdot (t_{i+1} - t_i)$$

---

## 8. Interactive 3D Airway Visualizer: How It Works

The 3D Airway & Deposition Visualizer (`visualization/airway_3d.py`) provides an intuitive spatial representation of aerosol deposition:

```
                         [ Dichotomous Branching Generator ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     Airway Geometry Backbone (G0-G16)               Cluster Endpoints (G17-G23)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                      [ Inhaled Aerosol Deposition Model ]
                       (Calculates f_upper, f_bronchial,
                        f_alveolar, f_exhaled)
                                         │
                                         ▼
                      [ Particle Generation & Placement ]
                       - Red Points: Upper Impaction (G0-G4)
                       - Amber Points: Bronchial Mixed (G5-G16)
                       - Blue Points: Alveolar Sedimentation (G17-G23)
                       - Green Points: Exhaled Plume (Airstream)
                                         │
                                         ▼
                      [ Interactive WebGL 3D Visualization ]
                       - Camera Orbit / Zoom / Pan
                       - Per-particle Physics & Clinical Telemetry
```

1. **Deterministic Branching Geometry:** A dichotomous tree algorithm generates 3D coordinates starting from the trachea ($G_0$: $z=10.0$) down through main bronchi ($G_1$), lobar divisions ($G_2-G_4$), conducting branches ($G_5-G_{16}$), and terminal alveolar acinar clusters ($G_{17}-G_{23}$).
2. **Coupled Particle Seeding:** The total visual particle count ($N=360$) is divided across generations in direct proportion to the physical fractions calculated by the aerosol model:
   - $N_{\text{upper}} \propto F_{\text{upper}}$
   - $N_{\text{bronchial}} \propto F_{\text{bronchial}}$
   - $N_{\text{alveolar}} \propto F_{\text{alveolar}}$
   - $N_{\text{exhaled}} \propto F_{\text{exhaled}}$
3. **Deposition Mechanism Color Coding:**
   - **Red (`#EF4444`):** High Inertial Impaction (concentrated in $G_0-G_4$).
   - **Amber (`#F59E0B`):** Conducting Bronchial Deposition (mixed impaction & sedimentation in $G_5-G_{16}$).
   - **Blue (`#2563EB`):** Gravitational Sedimentation (dominant in $G_{17}-G_{23}$).
   - **Green (`#10B981`):** Non-Deposited / Exhaled aerosol returning with the expired breath.
4. **Rich Hover Telemetry:** Hovering over any point displays its airway generation, particle aerodynamic diameter, deposition mechanism, and regional percentage.

---

## 9. Simulation Dossier Export & Data Schema

The **Export Simulation Dossier (CSV)** feature in `utils/export.py` converts the complete numerical trajectory into a standardized CSV dataset for external analysis.

### Data Schema

| Column Name | Data Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `time_hours` | `float` | $\text{hours}$ | Simulation time points from $0.0$ to $24.0\ \text{h}$ (300 steps) |
| `Nm_nanoparticles_in_mucus_mg` | `float` | $\text{mg}$ | Nanoparticle carrier payload in mucus layer |
| `Dm_free_drug_in_mucus_mg` | `float` | $\text{mg}$ | Unencapsulated free drug dissolved in mucus |
| `Dt_target_epithelial_tissue_mg` | `float` | $\text{mg}$ | Active drug mass accumulated in target epithelium |
| `Db_systemic_blood_mg_per_L` | `float` | $\text{mg/L}$ | Systemic blood concentration |
| `pulmonary_efficacy_pct` | `float` | $\%$ | Anti-inflammatory efficacy via Emax Hill model |
| `total_pulmonary_mass_mg` | `float` | $\text{mg}$ | Conservation check: $N_m + D_m + D_t$ |
| `preset_name` | `string` | — | Selected clinical/disease state scenario |
| `mmad_um` | `float` | $\mu\text{m}$ | Mass Median Aerodynamic Diameter |
| `inhaled_dose_mg` | `float` | $\text{mg}$ | Total nominal inhaled dose |
| `mucus_viscosity_factor_V` | `float` | — | Pathological viscosity factor relative to normal ($1.0\times$) |
| `airway_radius_factor` | `float` | — | Airway caliber multiplier ($1.0\times$ baseline, $0.7\times$ asthma) |
| `tidal_volume_factor` | `float` | — | Tidal breath volume multiplier ($1.0\times$ adult, $0.25\times$ infant) |
| `k_release_per_h` | `float` | $\text{h}^{-1}$ | Carrier payload release rate constant |
| `alveolar_deposition_pct` | `float` | $\%$ | Calculated alveolar deposition fraction |
| `bronchial_deposition_pct` | `float` | $\%$ | Calculated bronchial deposition fraction |
| `upper_airway_deposition_pct` | `float` | $\%$ | Calculated upper airway impaction fraction |
| `mucociliary_loss_pct` | `float` | $\%$ | Reduction in mucociliary clearance vs. healthy baseline |
| `safety_threshold_db_mg_per_L`| `float` | $\text{mg/L}$ | Configured systemic exposure safety limit |
| `is_safety_hazard_exceeded` | `boolean` | — | Flag indicating whether $D_b$ exceeded threshold |

### Compatibility
The exported CSV file is directly compatible with:
- **Python / Pandas:** `pd.read_csv("simulation_dossier.csv")`
- **R / Tidyverse:** `read_csv("simulation_dossier.csv")`
- **MATLAB:** `readtable('simulation_dossier.csv')`
- **Microsoft Excel / Google Sheets**
- **Jupyter Notebooks & Google Colab**

---

## 10. Software Architecture & Module Map

The codebase follows a modular Python architecture separating physics calculations, differential equations, data processing, and user interfaces:

```
respiratory-therapeutics-simulator/
│
├── app.py                          # Streamlit application UI, tabs & sidebar controls
│
├── models/
│   ├── aerosol.py                  # Aerosol deposition physics & MMAD curves
│   ├── pharmacokinetics.py         # 4-compartment ODE model & mucus viscosity modifiers
│   ├── pharmacodynamics.py         # Emax Hill efficacy equation & trapezoidal AUC
│   └── presets.py                  # Clinical & disease-state physiological presets
│
├── simulation/
│   └── solver.py                   # SciPy solve_ivp (RK45) integrator & metrics engine
│
├── visualization/
│   ├── airway_3d.py                # Interactive 3D branching airway & particle visualizer
│   ├── safety_monitor.py           # Efficacy gauge, systemic exposure gauge & dual-axis plot
│   └── plots.py                    # PK 4-compartment time series & PD efficacy figures
│
├── utils/
│   └── export.py                   # Simulation dossier CSV exporter (Pandas)
│
├── .streamlit/
│   └── config.toml                 # Streamlit server and iframe configuration
├── run_dev.sh                      # Development server runner script
├── requirements.txt                # Python scientific dependencies
└── README.md                       # Comprehensive documentation manual
```

---

## 11. Installation & Local Execution Guide

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- `pip` package manager

### 1. Clone or Navigate to Project Directory
```bash
cd respiratory-therapeutics-simulator
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

*Required packages include:*
- `streamlit >= 1.30.0`
- `numpy >= 1.24.0`
- `scipy >= 1.10.0`
- `plotly >= 5.15.0`
- `pandas >= 2.0.0`

### 3. Launch the Streamlit Simulator
```bash
streamlit run app.py
```
Or execute via the runner script:
```bash
./run_dev.sh
```

The simulator will launch and open in your default browser at `http://localhost:3000` (or `http://localhost:8501`).

---

## Scientific Modeling Disclaimer
This computational simulator is developed solely for **educational, exploratory, and computational research purposes**. The underlying mathematical relationships, rate constants, disease scaling multipliers, and exposure safety hazard thresholds are simplified mathematical models and illustrative parameters. They must **not** be used for clinical dosing calculations, medical diagnosis, patient care protocols, or regulatory safety determinations without thorough validation against clinical and experimental data.
