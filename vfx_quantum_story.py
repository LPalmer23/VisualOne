#!/usr/bin/env python3
"""
Story Mode: two scenes — coherent propagating wave (white) and prism spectral scatter.
Dash + Plotly 3D. Pure black void, stable time via wall clock modulo + pause.
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html

T_LOOP = 20.0

PRISM_CENTER = np.array([6.0, 0.0, 0.0], dtype=float)
PRISM_HALF = 0.95
WHITE = "rgb(248, 252, 255)"
NEON_CYAN = "rgb(0, 255, 255)"
NEON_MAGENTA = "rgb(255, 0, 220)"
NEON_YELLOW = "rgb(255, 240, 60)"
NEON_LIME = "rgb(160, 255, 40)"
PRISM_GLASS = "rgb(140, 200, 255)"


class Scene(IntEnum):
    COHERENT = 1
    PRISM = 2


def _sigmoid01(t: float, k: float = 11.0) -> float:
    t = float(max(0.0, min(1.0, t)))
    return float(1.0 / (1.0 + np.exp(-k * (t - 0.5))))


def _wave_dz(x: float, t: float, amp: float, freq: float, speed: float) -> float:
    """z = Amplitude * sin(2*pi * (Frequency*x - Speed*t))."""
    if amp <= 0.0:
        return 0.0
    return float(amp * np.sin(2.0 * np.pi * (freq * x - speed * t)))


def _beam_points(
    x0: float,
    x1: float,
    y0: float,
    z0: float,
    n: int,
    t: float,
    amp: float,
    freq: float,
    speed: float,
) -> np.ndarray:
    pts = []
    for xw in np.linspace(x0, x1, max(2, n)):
        z = z0 + _wave_dz(float(xw), t, amp, freq, speed)
        pts.append([float(xw), float(y0), float(z)])
    return np.array(pts, dtype=float)


def _orthonormal_frame(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    z = axis / n
    tmp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(z, tmp)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def mesh_cylinder(
    a: Sequence[float],
    b: Sequence[float],
    radius: float,
    *,
    n_seg: int = 28,
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
            color=color,
            opacity=opacity,
            flatshading=True,
            lighting=dict(ambient=0.28, diffuse=0.92, specular=1.2),
            lightposition=dict(x=80, y=-160, z=200),
        )
    )


def add_polyline_tube(
    fig: go.Figure,
    points: np.ndarray,
    radius: float,
    *,
    color: str,
    opacity: float,
    n_ring: int = 26,
    name: str | None = None,
) -> None:
    if points.shape[0] < 2:
        return
    n = points.shape[0] - 1
    for i in range(n):
        a, b = points[i], points[i + 1]
        if np.linalg.norm(b - a) < 1e-7:
            continue
        cap_b = i == 0
        cap_t = i == n - 1
        xc, yc, zc, ic, jc, kc = mesh_cylinder(a, b, radius, n_seg=n_ring, cap_bottom=cap_b, cap_top=cap_t)
        add_mesh(fig, xc, yc, zc, ic, jc, kc, color=color, opacity=opacity, name=name if i == 0 else None)


def _fan_unit_vectors(scatter_width: float) -> List[np.ndarray]:
    """Peacock fan in yz; scatter_width in [0,1] scales angular spread."""
    w = max(0.0, min(1.0, float(scatter_width)))
    base_deg = 12.0 + w * 38.0
    outs = []
    for ang_deg, sc in [(-1.0, 1.0), (-0.33, 0.95), (0.33, 0.95), (1.0, 1.0)]:
        th = np.radians(ang_deg * base_deg)
        v = np.array([1.0, sc * 0.62 * np.sin(th), sc * 0.62 * np.cos(th)], dtype=float)
        v /= np.linalg.norm(v)
        outs.append(v)
    return outs


def _chord_with_wave(
    a: np.ndarray,
    b: np.ndarray,
    n: int,
    t: float,
    amp: float,
    freq: float,
    speed: float,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = []
    for u in np.linspace(0.0, 1.0, max(2, n)):
        p = a + u * (b - a)
        p = p.copy()
        p[2] += _wave_dz(float(p[0]), t, amp, freq, speed)
        out.append(p)
    return np.stack(out, axis=0)


def _axis_void() -> dict:
    return dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showspikes=False,
        zeroline=False,
        showticklabels=False,
        title=dict(text=""),
        gridcolor="rgba(0,0,0,0)",
        backgroundcolor="rgba(0,0,0,0)",
        color="rgba(0,0,0,0)",
    )


def _advance_time_state(ts: Optional[Dict[str, Any]], paused: bool) -> Tuple[Dict[str, Any], float]:
    now = time.time()
    st: Dict[str, Any] = dict(ts) if ts else {}
    ws = float(st.get("ws", now))
    freeze = st.get("freeze")
    if paused:
        if freeze is None:
            freeze = (now - ws) % T_LOOP
        return {"ws": ws, "freeze": float(freeze)}, float(freeze)
    if freeze is not None:
        ws = now - float(freeze)
        freeze = None
    t = (now - ws) % T_LOOP
    return {"ws": ws, "freeze": None}, float(t)


def build_figure(
    scene: int,
    t: float,
    amp: float,
    freq: float,
    wave_speed: float,
    scatter_width: float,
) -> go.Figure:
    fig = go.Figure()
    amp = max(0.0, float(amp))
    freq = max(0.02, float(freq))
    wave_speed = float(wave_speed)

    if scene == Scene.COHERENT:
        # Single thick white propagating tube along +x
        pts = _beam_points(-10.0, 8.0, 0.0, 0.0, 40, t, amp, freq, wave_speed)
        add_polyline_tube(fig, pts, 0.38, color=WHITE, opacity=1.0, n_ring=24, name="Beam")
        # High-density centerline glow (Scatter3d) for crisp highlight
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines",
                line=dict(color=WHITE, width=10),
                opacity=0.95,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=-1.0, y=0.0, z=0.0),
            eye=dict(x=0.72, y=0.52, z=0.28),
        )
        dragmode = False
    else:
        # Scene 2: white inbound → prism → four neon rays (smooth fan + wave on each chord)
        x_face_l = PRISM_CENTER[0] - PRISM_HALF
        x_face_r = PRISM_CENTER[0] + PRISM_HALF
        pts_in = _beam_points(-10.0, x_face_l, 0.0, 0.0, 44, t, amp, freq, wave_speed)
        add_polyline_tube(fig, pts_in, 0.34, color=WHITE, opacity=0.98, n_ring=28, name="Inbound")

        bx, by, bz, bi, bj, bk = mesh_box(PRISM_CENTER, PRISM_HALF, PRISM_HALF, PRISM_HALF)
        add_mesh(fig, bx, by, bz, bi, bj, bk, color=PRISM_GLASS, opacity=0.2, name="Prism")
        bx2, by2, bz2, bi2, bj2, bk2 = mesh_box(PRISM_CENTER, PRISM_HALF + 0.06, PRISM_HALF + 0.06, PRISM_HALF + 0.06)
        add_mesh(fig, bx2, by2, bz2, bi2, bj2, bk2, color=NEON_CYAN, opacity=0.05)

        dirs = _fan_unit_vectors(scatter_width)
        cols = [NEON_CYAN, NEON_MAGENTA, NEON_YELLOW, NEON_LIME]
        start_pt = np.array([x_face_r + 0.05, 0.0, float(pts_in[-1, 2])], dtype=float)
        ray_len = 9.5
        n_ray = 20
        for dv, col in zip(dirs, cols):
            pts_ray = []
            for i, u in enumerate(np.linspace(0.0, 1.0, n_ray)):
                w = _sigmoid01(u * 1.15)
                d = np.array([1.0, 0.0, 0.0], dtype=float) * (1.0 - w) + dv * w
                d = d / max(1e-9, np.linalg.norm(d))
                p = start_pt + d * (u * ray_len)
                p = p.copy()
                p[2] += _wave_dz(float(p[0]), t, amp, freq, wave_speed)
                pts_ray.append(p)
            poly = np.stack(pts_ray, axis=0)
            add_polyline_tube(fig, poly, 0.2, color=col, opacity=0.96, n_ring=16)

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=6.0, y=0.0, z=0.1),
            eye=dict(x=0.95, y=0.9, z=0.48),
        )
        dragmode = "orbit"

    ax = _axis_void()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision=f"scene-{scene}",
        scene=dict(
            bgcolor="#000000",
            aspectmode="data",
            xaxis={**ax, "range": [-12, 18]},
            yaxis={**ax, "range": [-7, 7]},
            zaxis={**ax, "range": [-4, 4]},
            dragmode=dragmode,
            camera=camera,
        ),
    )
    return fig


EQ1 = r"$$i\hbar \frac{\partial}{\partial t}|\psi\rangle = \hat{H}|\psi\rangle$$"
EQ2 = r"$$|\psi\rangle = \sum_i c_i |w_i\rangle$$"

app = Dash(__name__, title="Story Mode — Quantum wave")
app.layout = html.Div(
    style={
        "display": "flex",
        "flexDirection": "row",
        "height": "100vh",
        "backgroundColor": "#000",
        "color": "#e8f0ff",
        "fontFamily": "system-ui, sans-serif",
    },
    children=[
        html.Div(
            style={
                "width": "260px",
                "minWidth": "260px",
                "padding": "18px 16px",
                "borderRight": "1px solid rgba(120,200,255,0.25)",
                "background": "linear-gradient(180deg, #0a1018 0%, #000 100%)",
            },
            children=[
                html.Div("Story Mode", style={"fontSize": "11px", "letterSpacing": "0.15em", "color": PRISM_GLASS}),
                html.Div("Scene", style={"marginTop": "14px", "fontSize": "12px", "opacity": 0.85}),
                dcc.Dropdown(
                    id="scene-select",
                    options=[
                        {"label": "1 · Coherent propagation", "value": 1},
                        {"label": "2 · Spectral scattering (prism)", "value": 2},
                    ],
                    value=1,
                    clearable=False,
                    style={"marginTop": "6px", "color": "#111"},
                ),
                html.Button(
                    "Next scene →",
                    id="btn-next",
                    n_clicks=0,
                    style={
                        "marginTop": "10px",
                        "width": "100%",
                        "padding": "8px",
                        "borderRadius": "8px",
                        "border": "1px solid rgba(120,200,255,0.4)",
                        "background": "rgba(0,80,120,0.35)",
                        "color": "#e8f0ff",
                        "cursor": "pointer",
                    },
                ),
                html.Div("Pause", style={"marginTop": "22px", "fontSize": "12px", "opacity": 0.85}),
                dcc.Checklist(
                    id="pause",
                    options=[{"label": " Pause time", "value": "p"}],
                    value=[],
                    style={"fontSize": "13px"},
                    inputStyle={"marginRight": "8px"},
                ),
                html.Div("Wave (Scene 1 & inbound in 2)", style={"marginTop": "20px", "fontSize": "11px", "color": PRISM_GLASS}),
                html.Div("Amplitude", style={"fontSize": "12px", "marginTop": "8px", "opacity": 0.8}),
                dcc.Slider(id="s-amp", min=0, max=0.55, step=0.02, value=0.22),
                html.Div("Frequency", style={"fontSize": "12px", "marginTop": "10px", "opacity": 0.8}),
                dcc.Slider(id="s-freq", min=0.05, max=0.55, step=0.01, value=0.16),
                html.Div("Wave speed", style={"fontSize": "12px", "marginTop": "10px", "opacity": 0.8}),
                dcc.Slider(id="s-speed", min=0, max=2.2, step=0.02, value=0.7),
                html.Div("Scattering width (Scene 2)", style={"marginTop": "18px", "fontSize": "11px", "color": PRISM_GLASS}),
                dcc.Slider(id="s-scatter", min=0.05, max=1.0, step=0.02, value=0.65),
            ],
        ),
        html.Div(
            style={"flex": "1", "minWidth": 0, "display": "flex", "flexDirection": "column"},
            children=[
                html.Div(
                    style={"padding": "8px 12px", "fontSize": "12px", "opacity": 0.85},
                    children=[html.Span(id="status")],
                ),
                html.Div(
                    style={"position": "relative", "flex": "1", "minHeight": 0},
                    children=[
                        dcc.Graph(
                            id="graph",
                            style={"position": "absolute", "inset": 0, "width": "100%", "height": "100%"},
                            config={"scrollZoom": True, "displaylogo": False},
                        ),
                        html.Div(
                            style={
                                "position": "absolute",
                                "top": "12px",
                                "right": "12px",
                                "maxWidth": "min(400px, 44vw)",
                                "padding": "14px 18px",
                                "borderRadius": "10px",
                                "border": "1px solid rgba(200,220,255,0.35)",
                                "background": "rgba(0,4,12,0.88)",
                                "pointerEvents": "none",
                                "zIndex": 5,
                            },
                            children=[dcc.Markdown(id="eq-box", mathjax=True)],
                        ),
                    ],
                ),
                dcc.Interval(id="clock", interval=50, n_intervals=0),
            ],
        ),
        dcc.Store(id="time-state", data=None),
    ],
)


@app.callback(
    Output("scene-select", "value"),
    Input("btn-next", "n_clicks"),
    State("scene-select", "value"),
    prevent_initial_call=True,
)
def cycle_scene(n_clicks, current):
    if not n_clicks:
        return current
    cur = int(current) if current is not None else 1
    return 2 if cur == 1 else 1


@app.callback(
    Output("time-state", "data"),
    Output("graph", "figure"),
    Output("eq-box", "children"),
    Output("status", "children"),
    Input("clock", "n_intervals"),
    Input("scene-select", "value"),
    Input("pause", "value"),
    Input("s-amp", "value"),
    Input("s-freq", "value"),
    Input("s-speed", "value"),
    Input("s-scatter", "value"),
    State("time-state", "data"),
)
def frame(_, scene, pause_v, amp, freq, spd, scat, ts):
    paused = pause_v is not None and "p" in pause_v
    st_out, t = _advance_time_state(ts, paused)
    sc = int(scene) if scene is not None else 1
    fig = build_figure(sc, t, float(amp or 0), float(freq or 0.1), float(spd or 0), float(scat or 0.5))
    eq = EQ1 if sc == Scene.COHERENT else EQ2
    stat = f"Scene {sc}  ·  t = {t:5.2f}s (mod {T_LOOP:.0f}s)  ·  {'PAUSED' if paused else 'LIVE'}"
    return st_out, fig, eq, stat


if __name__ == "__main__":
    app.run(debug=True, port=8050)
