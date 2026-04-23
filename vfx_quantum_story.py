#!/usr/bin/env python3
"""
VFX-first quantum narrative: three acts in Plotly 3D (Dash).
Aesthetic: pure black canvas, neon accents (Raylib story / QUBO vis vibe).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

# -----------------------------------------------------------------------------
# Story controller (mirrors StoryStep pattern in main.cpp)
# -----------------------------------------------------------------------------


class StoryStep(IntEnum):
    COHERENT_SOURCE = 0
    PRISM_SPLIT = 1
    ZENO_BLOCKADE = 2


@dataclass(frozen=True)
class ActTiming:
    play_s: float  # scripted motion within the act
    explore_s: float  # hold: user can orbit / zoom freely while scene is static-ish

    @property
    def total(self) -> float:
        return self.play_s + self.explore_s


ACTS: Tuple[ActTiming, ...] = (
    ActTiming(play_s=7.0, explore_s=4.0),
    ActTiming(play_s=9.0, explore_s=5.0),
    ActTiming(play_s=8.0, explore_s=6.0),
)

ACT_BOUNDARIES = np.cumsum([0.0] + [a.total for a in ACTS])
TOTAL_STORY_S = float(ACT_BOUNDARIES[-1])

# Neon palette (electric, high contrast on #000)
NEON_CYAN = "rgb(0, 255, 255)"
NEON_MAGENTA = "rgb(255, 0, 220)"
NEON_YELLOW = "rgb(255, 245, 60)"
NEON_LIME = "rgb(160, 255, 40)"
PRISM_EDGE = "rgb(120, 200, 255)"


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _ease_out_cubic(t: float) -> float:
    t = _clamp01(t)
    u = 1.0 - t
    return 1.0 - u * u * u


def story_step_at_global_time(t_s: float) -> Tuple[StoryStep, float, float, bool]:
    """
    Returns (step, u_play_in_act, explore_u, in_explore).
    During scripted motion u_play eases 0→1 and in_explore is False.
    During the hold window u_play stays 1 and in_explore is True (orbit/zoom/pan beat).
    """
    t_s = float(t_s % TOTAL_STORY_S) if TOTAL_STORY_S > 0 else 0.0
    for i, act in enumerate(ACTS):
        a0 = ACT_BOUNDARIES[i]
        a1 = ACT_BOUNDARIES[i + 1]
        if t_s < a1 or i == len(ACTS) - 1:
            local = max(0.0, min(act.total, t_s - a0))
            if local < act.play_s:
                u_play = _ease_out_cubic(local / max(1e-6, act.play_s))
                return StoryStep(i), u_play, 0.0, False
            explore_u = (local - act.play_s) / max(1e-6, act.explore_s)
            return StoryStep(i), 1.0, explore_u, True
    return StoryStep(len(ACTS) - 1), 1.0, 1.0, True


# -----------------------------------------------------------------------------
# Mesh builders: thick cylinders / cones / box / slab
# -----------------------------------------------------------------------------


def _orthonormal_frame(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    z = axis / n
    if abs(z[2]) < 0.9:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
    x = np.cross(z, tmp)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def mesh_cylinder(
    a: Sequence[float],
    b: Sequence[float],
    radius: float,
    *,
    n_seg: int = 36,
    cap_bottom: bool = True,
    cap_top: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    axis = b - a
    h = float(np.linalg.norm(axis))
    if h < 1e-9:
        raise ValueError("degenerate cylinder")
    ex, ey, _ez = _orthonormal_frame(axis)
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    ring = (np.cos(angles)[:, None] * ex + np.sin(angles)[:, None] * ey) * radius
    bottom = a + ring
    top = b + ring
    x = np.concatenate([bottom[:, 0], top[:, 0]])
    y = np.concatenate([bottom[:, 1], top[:, 1]])
    z = np.concatenate([bottom[:, 2], top[:, 2]])
    i_list: List[int] = []
    j_list: List[int] = []
    k_list: List[int] = []
    for s in range(n_seg):
        s2 = (s + 1) % n_seg
        i_list += [s, s, s2]
        j_list += [s2, s + n_seg, s2 + n_seg]
        k_list += [s + n_seg, s2 + n_seg, s2]

    idx = 2 * n_seg
    if cap_bottom:
        x = np.append(x, a[0])
        y = np.append(y, a[1])
        z = np.append(z, a[2])
        c0 = idx
        idx += 1
        for s in range(n_seg):
            s2 = (s + 1) % n_seg
            i_list.append(c0)
            j_list.append(s)
            k_list.append(s2)
    if cap_top:
        x = np.append(x, b[0])
        y = np.append(y, b[1])
        z = np.append(z, b[2])
        c1 = idx
        for s in range(n_seg):
            s2 = (s + 1) % n_seg
            i_list.append(c1)
            j_list.append(s + n_seg)
            k_list.append(s2 + n_seg)

    return x, y, z, np.array(i_list), np.array(j_list), np.array(k_list)


def mesh_cone(
    base_center: Sequence[float],
    tip: Sequence[float],
    radius: float,
    *,
    n_seg: int = 28,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_center = np.asarray(base_center, dtype=float)
    tip = np.asarray(tip, dtype=float)
    axis = tip - base_center
    h = float(np.linalg.norm(axis))
    if h < 1e-9:
        raise ValueError("degenerate cone")
    ex, ey, _ez = _orthonormal_frame(axis)
    angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    ring = (np.cos(angles)[:, None] * ex + np.sin(angles)[:, None] * ey) * radius
    base = base_center + ring
    x = np.concatenate([base[:, 0], [tip[0]]])
    y = np.concatenate([base[:, 1], [tip[1]]])
    z = np.concatenate([base[:, 2], [tip[2]]])
    apex = n_seg
    i_list: List[int] = []
    j_list: List[int] = []
    k_list: List[int] = []
    for s in range(n_seg):
        s2 = (s + 1) % n_seg
        i_list += [apex, s]
        j_list += [s, s2]
        k_list += [s2, apex]
    # base cap
    bc = n_seg + 1
    x = np.append(x, base_center[0])
    y = np.append(y, base_center[1])
    z = np.append(z, base_center[2])
    for s in range(n_seg):
        s2 = (s + 1) % n_seg
        i_list.append(bc)
        j_list.append(s)
        k_list.append(s2)
    return x, y, z, np.array(i_list), np.array(j_list), np.array(k_list)


def mesh_box(
    center: Sequence[float],
    hx: float,
    hy: float,
    hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, cz = center
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=float,
    )
    # 12 triangles for axis-aligned box
    faces = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    x = verts[:, 0]
    y = verts[:, 1]
    z = verts[:, 2]
    i, j, k = zip(*faces)
    return x, y, z, np.array(i), np.array(j), np.array(k)


def add_mesh(
    fig: go.Figure,
    x,
    y,
    z,
    i,
    j,
    k,
    *,
    color: str,
    opacity: float = 1.0,
    name: str | None = None,
    show: bool = True,
    lighting_ambient: float = 0.35,
    lighting_diffuse: float = 0.95,
    lighting_specular: float = 1.2,
    flatshading: bool = True,
) -> None:
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            name=name,
            showlegend=bool(name),
            visible=show,
            color=color,
            opacity=opacity,
            flatshading=flatshading,
            lighting=dict(ambient=lighting_ambient, diffuse=lighting_diffuse, specular=lighting_specular),
            lightposition=dict(x=100, y=-200, z=200),
        )
    )


def add_glow_shell(
    fig: go.Figure,
    a: Sequence[float],
    b: Sequence[float],
    r_inner: float,
    r_outer: float,
    *,
    color: str,
    opacity: float,
    n_seg: int = 28,
    show: bool = True,
) -> None:
    xi, yi, zi, ii, ji, ki = mesh_cylinder(a, b, r_outer, n_seg=n_seg, cap_bottom=False, cap_top=False)
    xo, yo, zo, io, jo, ko = mesh_cylinder(a, b, r_inner, n_seg=n_seg, cap_bottom=False, cap_top=False)
    x = np.concatenate([xi, xo])
    y = np.concatenate([yi, yo])
    z = np.concatenate([zi, zo])
    off = len(xi)
    ii2 = np.concatenate([ii, io + off])
    ji2 = np.concatenate([ji, jo + off])
    ki2 = np.concatenate([ki, ko + off])
    add_mesh(fig, x, y, z, ii2, ji2, ki2, color=color, opacity=opacity, flatshading=False, show=show)


# -----------------------------------------------------------------------------
# Figure construction
# -----------------------------------------------------------------------------


def _fan_directions() -> List[np.ndarray]:
    """Four unit directions fanning in +x with spread in yz (fiber bundle)."""
    outs = []
    for ang_deg, scale in [(-34.0, 1.0), (-11.0, 0.92), (11.0, 0.92), (34.0, 1.0)]:
        th = np.radians(ang_deg)
        # rotate base in yz
        v = np.array([1.0, scale * 0.55 * np.sin(th), scale * 0.55 * np.cos(th)], dtype=float)
        v /= np.linalg.norm(v)
        outs.append(v)
    return outs


def _split_segment_at_x(
    p0: np.ndarray, p1: np.ndarray, xp: float
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], Optional[Tuple[np.ndarray, np.ndarray]]]:
    x0, x1 = float(p0[0]), float(p1[0])
    if max(x0, x1) <= xp:
        return (p0, p1), None
    if min(x0, x1) >= xp:
        return None, (p0, p1)
    if abs(x1 - x0) < 1e-9:
        return (p0, p1), None
    t = (xp - x0) / (x1 - x0)
    pm = p0 + t * (p1 - p0)
    if x0 < x1:
        return (p0, pm), (pm, p1)
    return (pm, p0), (pm, p1)


def build_figure(
    step: StoryStep,
    u_play: float,
    beam_yaw_deg: float,
    beam_pitch_deg: float,
    zeno_gamma: float,
) -> go.Figure:
    fig = go.Figure()

    # --- Act 1: coherent beam ---
    yaw = np.radians(beam_yaw_deg)
    pitch = np.radians(beam_pitch_deg)
    rot_y = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rot_x = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    R = rot_y @ rot_x
    origin0 = np.array([-9.0, 0.0, 0.0], dtype=float)
    origin = R @ origin0
    beam_len = 2.5 + 9.5 * _clamp01(u_play)
    dir_x = np.array([1.0, 0.0, 0.0], dtype=float)
    beam_end = origin + dir_x * beam_len
    tip = beam_end + dir_x * 0.85

    show_act1 = step == StoryStep.COHERENT_SOURCE
    show_act2 = step == StoryStep.PRISM_SPLIT
    show_act3 = step == StoryStep.ZENO_BLOCKADE

    # Main beam + glow
    if show_act1:
        xc, yc, zc, ic, jc, kc = mesh_cylinder(origin, beam_end, 0.42, n_seg=40)
        add_mesh(fig, xc, yc, zc, ic, jc, kc, color=NEON_CYAN, opacity=1.0, name="Coherent beam", show=True)
        add_glow_shell(fig, origin, beam_end, 0.42, 0.78, color=NEON_CYAN, opacity=0.14, n_seg=32, show=True)
        xk, yk, zk, ik, jk, kk = mesh_cone(beam_end, tip, 0.55, n_seg=26)
        add_mesh(fig, xk, yk, zk, ik, jk, kk, color=NEON_CYAN, opacity=0.92, name="Beam head", show=True)

    # --- Prism + split tubes (act 2 onward) ---
    prism_center = np.array([2.0, 0.0, 0.0], dtype=float)
    dirs = _fan_directions()
    colors = [NEON_CYAN, NEON_MAGENTA, NEON_YELLOW, NEON_LIME]

    split_u = _clamp01(u_play) if show_act2 else 1.0
    if show_act2 or show_act3:
        bx, by, bz, bi, bj, bk = mesh_box(prism_center, 0.95, 0.95, 0.95)
        add_mesh(
            fig,
            bx,
            by,
            bz,
            bi,
            bj,
            bk,
            color=PRISM_EDGE,
            opacity=0.22,
            name="Prism",
            show=True,
        )
        # faint edge accent (second slightly larger shell, very transparent)
        bx2, by2, bz2, bi2, bj2, bk2 = mesh_box(prism_center, 1.02, 1.02, 1.02)
        add_mesh(fig, bx2, by2, bz2, bi2, bj2, bk2, color=NEON_CYAN, opacity=0.06, name=None, show=True)

    inlet = prism_center + np.array([1.0, 0.0, 0.0])
    tube_r = 0.18
    zeno_plane_x = 7.5
    gamma = _clamp01(zeno_gamma)

    if show_act2 and split_u < 0.98:
        stub_end = prism_center - np.array([1.1, 0.0, 0.0]) + np.array([0.4 * split_u, 0, 0])
        stub_start = stub_end - np.array([4.0 * (1.0 - 0.2 * split_u), 0, 0])
        xs, ys, zs, ins, jns, kns = mesh_cylinder(stub_start, stub_end, tube_r * 1.05, n_seg=28)
        add_mesh(fig, xs, ys, zs, ins, jns, kns, color=NEON_CYAN, opacity=0.85, show=True)

    if show_act2 or show_act3:
        for idx, (dv, col) in enumerate(zip(dirs, colors)):
            spread = np.array([0.0, dv[1], dv[2]], dtype=float) * (2.8 * split_u)
            end = inlet + dv * (3.0 + 7.0 * split_u) + spread * 1.15
            t0 = inlet + dv * 0.15
            if show_act3:
                pre_seg, post_seg = _split_segment_at_x(t0, end, zeno_plane_x)
                if col == NEON_LIME:
                    o_post = 1.0
                else:
                    o_post = float(max(0.04, (1.0 - gamma) ** 1.65))
                if pre_seg is not None:
                    pa, pb = pre_seg
                    if np.linalg.norm(pb - pa) > 0.08:
                        xa, ya, za, ia, ja, ka = mesh_cylinder(pa, pb, tube_r, n_seg=26)
                        add_mesh(fig, xa, ya, za, ia, ja, ka, color=col, opacity=1.0, show=True)
                        add_glow_shell(fig, pa, pb, tube_r, tube_r * 1.55, color=col, opacity=0.08, show=True)
                if post_seg is not None:
                    qa, qb = post_seg
                    if np.linalg.norm(qb - qa) > 0.08:
                        xb, yb, zb, ib, jb, kb = mesh_cylinder(qa, qb, tube_r, n_seg=26)
                        add_mesh(fig, xb, yb, zb, ib, jb, kb, color=col, opacity=o_post, show=True)
                        add_glow_shell(
                            fig,
                            qa,
                            qb,
                            tube_r,
                            tube_r * 1.55,
                            color=col,
                            opacity=0.07 * o_post,
                            show=True,
                        )
            else:
                xa, ya, za, ia, ja, ka = mesh_cylinder(t0, end, tube_r, n_seg=26)
                add_mesh(fig, xa, ya, za, ia, ja, ka, color=col, opacity=0.95, name=f"Mode {idx + 1}", show=True)
                add_glow_shell(fig, t0, end, tube_r, tube_r * 1.55, color=col, opacity=0.09, show=True)

    # Zeno plane (act 3)
    if show_act3:
        pc = np.array([zeno_plane_x, 0.0, 0.0], dtype=float)
        px, py, pz, pi, pj, pk = mesh_box(pc, 0.06, 4.2, 4.2)
        add_mesh(fig, px, py, pz, pi, pj, pk, color=NEON_MAGENTA, opacity=0.18, name="Zeno filter", show=True)

    # Camera / scene styling
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=0, r=0, t=28, b=0),
        title=dict(
            text="",
            font=dict(color="rgb(200, 220, 255)", size=14),
        ),
        showlegend=False,
        uirevision="scene",  # preserve orbit / zoom / pan across timed updates
        scene=dict(
            bgcolor="#000000",
            aspectmode="data",
            xaxis=dict(visible=False, showbackground=False, range=[-14, 16]),
            yaxis=dict(visible=False, showbackground=False, range=[-6, 6]),
            zaxis=dict(visible=False, showbackground=False, range=[-6, 6]),
            camera=dict(eye=dict(x=1.45, y=-1.65, z=0.85)),
        ),
    )
    return fig


# -----------------------------------------------------------------------------
# Dash app
# -----------------------------------------------------------------------------

EQ_ACT1 = r"""
**Act I — Coherent source**

$$
i\hbar \dot{\psi} = \hat{H}\psi
$$
"""

EQ_ACT2 = r"""
**Act II — Superposition split**

$$
\sum_i \alpha_i | w_i \rangle
$$
"""

EQ_ACT3 = r"""
**Act III — Zeno blockade (Lindblad)**

$$
\dot{\rho} = -\frac{i}{\hbar}[H,\rho]
+ \sum_k \gamma_k \Big(
L_k \rho L_k^\dagger
- \tfrac{1}{2} \{ L_k^\dagger L_k, \rho \}
\Big)
$$
"""


def _equation_md(step: StoryStep) -> str:
    if step == StoryStep.COHERENT_SOURCE:
        return EQ_ACT1
    if step == StoryStep.PRISM_SPLIT:
        return EQ_ACT2
    return EQ_ACT3


def _status_line(step: StoryStep, u_play: float, in_explore: bool, t_global: float) -> str:
    phase = "explore — orbit / zoom / pan" if in_explore else "play — scripted beat"
    return (
        f"Time {t_global:5.1f}s / {TOTAL_STORY_S:.1f}s · "
        f"{step.name.replace('_', ' ')} · {phase} "
        f"(motion {_clamp01(u_play):.2f})"
    )


app = Dash(__name__, title="VFX Quantum Story")
app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "row",
        "height": "100vh",
        "backgroundColor": "#000000",
        "color": "#e8f0ff",
        "fontFamily": "'SF Pro Text', 'Segoe UI', system-ui, sans-serif",
    },
    children=[
        html.Div(
            style={"flex": "3", "minWidth": 0, "display": "flex", "flexDirection": "column"},
            children=[
                html.Div(
                    style={"padding": "10px 16px 0 16px"},
                    children=[
                        html.Div(id="status", style={"fontSize": "13px", "opacity": 0.85}),
                        dcc.Interval(id="clock", interval=50, n_intervals=0, max_intervals=-1),
                    ],
                ),
                dcc.Graph(
                    id="graph",
                    style={"flex": "1", "height": "100%"},
                    config={
                        "scrollZoom": True,
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["toImage"],
                    },
                ),
                html.Div(
                    style={"padding": "8px 20px 16px 20px"},
                    children=[
                        html.Div(
                            "Act I · beam origin (yaw / pitch)",
                            style={"fontSize": "12px", "marginBottom": "4px", "opacity": 0.75},
                        ),
                        dcc.Slider(
                            id="beam-yaw",
                            min=-55,
                            max=55,
                            step=1,
                            value=0,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        dcc.Slider(
                            id="beam-pitch",
                            min=-40,
                            max=40,
                            step=1,
                            value=0,
                            marks=None,
                        ),
                        html.Div(
                            "Act III · Zeno strength Γ (post-filter survival of non-lime modes)",
                            style={
                                "fontSize": "12px",
                                "marginTop": "14px",
                                "marginBottom": "4px",
                                "opacity": 0.75,
                            },
                        ),
                        dcc.Slider(
                            id="zeno-gamma",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.35,
                            marks={0: "0", 0.5: "½", 1: "1"},
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            style={
                "flex": "1",
                "minWidth": "300px",
                "maxWidth": "420px",
                "borderLeft": "1px solid rgba(120, 200, 255, 0.25)",
                "padding": "20px 22px",
                "background": "linear-gradient(180deg, rgba(0,30,45,0.5) 0%, #000000 100%)",
                "boxSizing": "border-box",
                "overflowY": "auto",
            },
            children=[
                html.Div(
                    "Physics panel",
                    style={
                        "fontSize": "11px",
                        "letterSpacing": "0.12em",
                        "textTransform": "uppercase",
                        "color": "rgb(120, 200, 255)",
                        "marginBottom": "12px",
                    },
                ),
                dcc.Markdown(id="eq-md", mathjax=True, className="eq-panel"),
            ],
        ),
    ],
)


@app.callback(
    Output("graph", "figure"),
    Output("eq-md", "children"),
    Output("status", "children"),
    Input("clock", "n_intervals"),
    Input("beam-yaw", "value"),
    Input("beam-pitch", "value"),
    Input("zeno-gamma", "value"),
)
def refresh(n_tick: int, yaw, pitch, gamma):
    dt = 0.05
    t_global = (n_tick or 0) * dt
    step, u_play, _explore_u, in_explore = story_step_at_global_time(t_global)
    fig = build_figure(step, u_play, float(yaw or 0), float(pitch or 0), float(gamma or 0))
    return fig, _equation_md(step), _status_line(step, u_play, in_explore, t_global)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
