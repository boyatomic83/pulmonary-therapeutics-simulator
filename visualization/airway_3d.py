"""
Interactive 3D Bronchial & Alveolar Deposition Visualizer
Visualizes simplified 3D airway branching generations (G0 to G23):
  - G0–G4: Upper Airways (Trachea, main bronchi, lobar bronchi)
  - G5–G16: Conducting Bronchial Airways
  - G17–G23: Deep Alveolar Beds
Color-coded particles based on deposition mechanism:
  - Red: High Inertial Impaction (predominant in upper generations for larger MMAD)
  - Blue: Gravitational Sedimentation (predominant in alveolar generations)
  - Green: Exhaled / Non-deposited fraction
Interactive 3D rotation, zoom, pan, and detailed clinical/physics hover data.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List, Tuple

def generate_branching_tree() -> Tuple[List[Dict[str, Any]], Dict[int, List[np.ndarray]]]:
    """
    Generates a deterministic 3D dichotomous branching airway tree.
    Returns:
        branches: list of branch segments with start/end coordinates and generation
        generation_nodes: mapping of generation number to 3D endpoints
    """
    branches = []
    generation_nodes = {g: [] for g in range(24)}
    
    # Root: Trachea (Generation 0)
    root_start = np.array([0.0, 0.0, 10.0])
    root_end = np.array([0.0, 0.0, 7.5])
    branches.append({
        "start": root_start,
        "end": root_end,
        "gen": 0,
        "radius": 1.2,
    })
    generation_nodes[0].append(root_end)
    
    # Iteratively branch down to G5, then generate cluster nodes for G6-G23
    current_ends = [(root_end, np.array([0.0, 0.0, -1.0]), 1.0, 0.0)]
    
    # Build structural branches through G4 (Upper airways)
    for gen in range(1, 5):
        next_ends = []
        length = 2.4 * (0.75 ** gen)
        angle_spread = 0.55 + 0.05 * gen
        
        for p_end, p_dir, p_rad, azim in current_ends:
            # Branch left and right
            for sign, d_azim in [(-1, 0.3), (1, -0.3)]:
                theta = sign * angle_spread
                phi = azim + d_azim
                
                dx = length * np.sin(theta) * np.cos(phi)
                dy = length * np.sin(theta) * np.sin(phi)
                dz = -length * np.cos(theta)
                
                new_dir = np.array([dx, dy, dz])
                new_dir /= np.linalg.norm(new_dir)
                new_end = p_end + new_dir * length
                
                branches.append({
                    "start": p_end,
                    "end": new_end,
                    "gen": gen,
                    "radius": max(0.2, p_rad * 0.75),
                })
                generation_nodes[gen].append(new_end)
                next_ends.append((new_end, new_dir, p_rad * 0.75, phi))
                
        current_ends = next_ends
        
    # Subsample branches for G5-G16 (Conducting bronchial) to maintain responsive rendering
    np.random.seed(42)  # Deterministic spatial seed
    bronchial_ends = []
    for p_end, p_dir, p_rad, azim in current_ends:
        # Create 2 sub-branches per G4 terminal
        for b_idx in range(2):
            angle = (b_idx - 0.5) * 0.8
            length = 2.0
            vec = np.array([
                np.sin(angle) * 1.5 + np.random.uniform(-0.3, 0.3),
                np.cos(angle) * 1.5 + np.random.uniform(-0.3, 0.3),
                -length - np.random.uniform(0.2, 0.8),
            ])
            b_end = p_end + vec
            branches.append({
                "start": p_end,
                "end": b_end,
                "gen": 8,
                "radius": 0.35,
            })
            bronchial_ends.append(b_end)
            
    # Map G5-G16 along these paths
    for g in range(5, 17):
        alpha = (g - 5) / 11.0
        for b_idx, b_end in enumerate(bronchial_ends):
            p_start = branches[len(branches) - len(bronchial_ends) + b_idx]["start"]
            pt = p_start + alpha * (b_end - p_start) + np.random.normal(0, 0.15, 3)
            generation_nodes[g].append(pt)
            
    # Generate G17-G23 terminal alveolar clusters beneath bronchial terminals
    for g in range(17, 24):
        spread = 0.25 + 0.15 * (g - 17)
        depth_offset = -0.3 * (g - 17)
        for b_end in bronchial_ends:
            for _ in range(3):
                pt = b_end + np.array([
                    np.random.normal(0, spread),
                    np.random.normal(0, spread),
                    depth_offset + np.random.uniform(-0.4, 0.1),
                ])
                generation_nodes[g].append(pt)
                
    return branches, generation_nodes

# Precompute tree backbone
AIRWAY_BRANCHES, AIRWAY_GENERATION_NODES = generate_branching_tree()

def plot_airway_3d_deposition(sim_result: Dict[str, Any]) -> go.Figure:
    """
    Constructs the 3D Interactive Airway & Particle Deposition visualization.
    Integrates regional particle counts proportional to calculated deposition fractions.
    """
    deposition = sim_result["deposition"]
    inputs = sim_result["inputs"]
    mmad = inputs["mmad"]
    inhaled_dose = inputs["inhaled_dose"]
    preset_name = inputs.get("preset_name", "Healthy Adult")
    
    f_upper = deposition["f_upper"]
    f_bronchial = deposition["f_bronchial"]
    f_alveolar = deposition["f_alveolar"]
    f_exhaled = deposition.get("f_exhaled", 0.15)
    
    # 1. Construct Airway Tree Geometry Lines
    edge_x, edge_y, edge_z = [], [], []
    for b in AIRWAY_BRANCHES:
        p0, p1 = b["start"], b["end"]
        edge_x.extend([p0[0], p1[0], None])
        edge_y.extend([p0[1], p1[1], None])
        edge_z.extend([p0[2], p1[2], None])
        
    tree_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="#64748B", width=4),
        name="Airway Branching (G0–G16)",
        hoverinfo="skip",
    )
    
    # Total particle pool for visualization (e.g., 360 points)
    n_total = 360
    n_upper = max(8, int(round(n_total * f_upper)))
    n_bronchial = max(8, int(round(n_total * f_bronchial)))
    n_alveolar = max(8, int(round(n_total * f_alveolar)))
    n_exhaled = max(6, int(round(n_total * f_exhaled * 0.4)))
    
    np.random.seed(int(mmad * 100) + int(inhaled_dose * 10))
    
    # 2. Upper Airway Particles (G0-G4) -> Red: High Inertial Impaction
    upper_pts = []
    upper_meta = []
    for _ in range(n_upper):
        g = int(np.random.choice([0, 1, 2, 3, 4], p=[0.35, 0.28, 0.18, 0.12, 0.07]))
        nodes = AIRWAY_GENERATION_NODES[g]
        base_pt = nodes[np.random.randint(len(nodes))]
        jitter = np.random.normal(0, 0.15, 3)
        pt = base_pt + jitter
        upper_pts.append(pt)
        upper_meta.append([
            f"G{g}",
            mmad,
            "Upper Airways (Oropharynx / Trachea)",
            "Inertial Impaction",
            f_upper * 100.0,
        ])
        
    # 3. Deep Alveolar Particles (G17-G23) -> Blue: Gravitational Sedimentation
    alveolar_pts = []
    alveolar_meta = []
    for _ in range(n_alveolar):
        g = int(np.random.randint(17, 24))
        nodes = AIRWAY_GENERATION_NODES[g]
        base_pt = nodes[np.random.randint(len(nodes))]
        jitter = np.random.normal(0, 0.22, 3)
        pt = base_pt + jitter
        alveolar_pts.append(pt)
        alveolar_meta.append([
            f"G{g}",
            mmad,
            "Deep Alveolar Region",
            "Gravitational Sedimentation",
            f_alveolar * 100.0,
        ])
        
    # 4. Conducting Bronchial Particles (G5-G16) -> Cyan/Intermediate
    bronchial_pts = []
    bronchial_meta = []
    for _ in range(n_bronchial):
        g = int(np.random.randint(5, 17))
        nodes = AIRWAY_GENERATION_NODES[g]
        base_pt = nodes[np.random.randint(len(nodes))]
        jitter = np.random.normal(0, 0.18, 3)
        pt = base_pt + jitter
        bronchial_pts.append(pt)
        mech = "Sedimentation / Impaction" if mmad <= 3.5 else "Inertial Impaction"
        bronchial_meta.append([
            f"G{g}",
            mmad,
            "Conducting Bronchial Airways",
            mech,
            f_bronchial * 100.0,
        ])
        
    # 5. Exhaled / Non-deposited Particles -> Green
    exhaled_pts = []
    exhaled_meta = []
    for _ in range(n_exhaled):
        # Positioned along the central core and exhaled upward
        z = np.random.uniform(2.0, 11.0)
        x = np.random.normal(0, 0.45)
        y = np.random.normal(0, 0.45)
        exhaled_pts.append(np.array([x, y, z]))
        exhaled_meta.append([
            "Exhaled Stream",
            mmad,
            "Non-Deposited Fraction",
            "Exhaled in Breath Plume",
            f_exhaled * 100.0,
        ])
        
    upper_pts = np.array(upper_pts) if len(upper_pts) > 0 else np.empty((0, 3))
    alveolar_pts = np.array(alveolar_pts) if len(alveolar_pts) > 0 else np.empty((0, 3))
    bronchial_pts = np.array(bronchial_pts) if len(bronchial_pts) > 0 else np.empty((0, 3))
    exhaled_pts = np.array(exhaled_pts) if len(exhaled_pts) > 0 else np.empty((0, 3))
    
    hovertemplate = (
        "<b>Airway Generation:</b> %{customdata[0]}<br>"
        "<b>Particle Size:</b> %{customdata[1]:.1f} μm<br>"
        "<b>Deposition Region:</b> %{customdata[2]}<br>"
        "<b>Deposition Mechanism:</b> %{customdata[3]}<br>"
        "<b>Relative Deposition:</b> %{customdata[4]:.1f}%"
        "<extra></extra>"
    )
    
    # Trace: Red (High Impaction / Upper)
    trace_impaction = go.Scatter3d(
        x=upper_pts[:, 0] if len(upper_pts) else [],
        y=upper_pts[:, 1] if len(upper_pts) else [],
        z=upper_pts[:, 2] if len(upper_pts) else [],
        mode="markers",
        marker=dict(
            size=6.5,
            color="#EF4444",
            opacity=0.9,
            symbol="circle",
            line=dict(color="#991B1B", width=1),
        ),
        name=f"Upper Airways (G0–G4): Impaction ({f_upper*100:.1f}%)",
        customdata=upper_meta,
        hovertemplate=hovertemplate,
    )
    
    # Trace: Cyan/Orange (Conducting Bronchial)
    trace_bronchial = go.Scatter3d(
        x=bronchial_pts[:, 0] if len(bronchial_pts) else [],
        y=bronchial_pts[:, 1] if len(bronchial_pts) else [],
        z=bronchial_pts[:, 2] if len(bronchial_pts) else [],
        mode="markers",
        marker=dict(
            size=5.5,
            color="#F59E0B",
            opacity=0.85,
            symbol="diamond",
            line=dict(color="#B45309", width=1),
        ),
        name=f"Bronchial (G5–G16): Mixed ({f_bronchial*100:.1f}%)",
        customdata=bronchial_meta,
        hovertemplate=hovertemplate,
    )
    
    # Trace: Blue (Sedimentation / Alveolar)
    trace_sedimentation = go.Scatter3d(
        x=alveolar_pts[:, 0] if len(alveolar_pts) else [],
        y=alveolar_pts[:, 1] if len(alveolar_pts) else [],
        z=alveolar_pts[:, 2] if len(alveolar_pts) else [],
        mode="markers",
        marker=dict(
            size=5.0,
            color="#2563EB",
            opacity=0.85,
            symbol="circle",
            line=dict(color="#1E40AF", width=1),
        ),
        name=f"Deep Alveolar (G17–G23): Sedimentation ({f_alveolar*100:.1f}%)",
        customdata=alveolar_meta,
        hovertemplate=hovertemplate,
    )
    
    # Trace: Green (Exhaled / Non-deposited)
    trace_exhaled = go.Scatter3d(
        x=exhaled_pts[:, 0] if len(exhaled_pts) else [],
        y=exhaled_pts[:, 1] if len(exhaled_pts) else [],
        z=exhaled_pts[:, 2] if len(exhaled_pts) else [],
        mode="markers",
        marker=dict(
            size=4.5,
            color="#10B981",
            opacity=0.7,
            symbol="cross",
        ),
        name=f"Exhaled Fraction ({f_exhaled*100:.1f}%)",
        customdata=exhaled_meta,
        hovertemplate=hovertemplate,
    )
    
    fig = go.Figure(data=[
        tree_trace,
        trace_impaction,
        trace_bronchial,
        trace_sedimentation,
        trace_exhaled,
    ])
    
    fig.update_layout(
        title=dict(
            text=f"<b>Interactive 3D Airway Deposition Tree</b> (MMAD = {mmad:.1f} μm | {preset_name})",
            x=0.02,
            font=dict(size=15, color="#0F172A"),
        ),
        scene=dict(
            xaxis=dict(title="Lateral (X)", showgrid=True, zeroline=False, showbackground=False),
            yaxis=dict(title="Anteroposterior (Y)", showgrid=True, zeroline=False, showbackground=False),
            zaxis=dict(title="Cranial-Caudal Depth (Z)", showgrid=True, zeroline=False, showbackground=False),
            camera=dict(
                eye=dict(x=1.35, y=-1.5, z=0.7),
                up=dict(x=0, y=0, z=1),
            ),
            aspectratio=dict(x=1.1, y=1.0, z=1.35),
        ),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=540,
        paper_bgcolor="#FFFFFF",
    )
    
    return fig
