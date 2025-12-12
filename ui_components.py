import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Callable
import numpy as np

from types_module import SimulationStep, AdvancedAnalysis, PatientProfile
from constants import COLORS, DRUG_INFO

# --- Helpers ---
CHART_MARGIN = dict(l=20, r=20, t=40, b=20)

def render_chart_container(title: str, subtitle: str, fig: go.Figure):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:0.8em;color:{COLORS['textMuted']}'>{subtitle}</span>", font=dict(size=14, family='sans-serif')),
        margin=CHART_MARGIN,
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        showlegend=False,
        height=250
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- Widgets ---
def render_shock_state_tile(current_step: SimulationStep):
    if current_step["map"] < 65: state, color = "UNCOMPENSATED", COLORS["crit"]
    elif current_step["ci"] < 2.2 and current_step["svri"] > 2000: state, color = "CARDIOGENIC", COLORS["crit"]
    elif current_step["ci"] > 2.2 and current_step["svri"] < 1200: state, color = "DISTRIBUTIVE", COLORS["alert"]
    else: state, color = "STABLE", COLORS["ok"]
    st.markdown(f"""
    <div style="background:white; border-top: 5px solid {color}; padding:10px; border-radius:5px; text-align:center; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
        <p style="font-size:0.7em; color:#64748b; margin:0; font-weight:bold;">PHENOTYPE</p>
        <h3 style="margin:0; color:{color};">{state}</h3>
    </div>
    """, unsafe_allow_html=True)

def render_vital_card(label: str, value: float, unit: str, threshold: dict = None):
    is_crit = threshold and (value < threshold["min"] or value > threshold["max"])
    st.metric(label, f"{value:.1f if label in ['CI','Lactate','CPO'] else int(value)} {unit}", delta="CRITICAL" if is_crit else None, delta_color="inverse")

def render_waveform_monitor(current_step: SimulationStep):
    # Only render if data exists (it should due to sim engine fix)
    if not current_step["ecgWave"]:
        st.info("Initializing Monitor...")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    
    ecg_y = [p['y'] for p in current_step["ecgWave"]]
    art_y = [p['y'] for p in current_step["artWave"]]
    
    fig.add_trace(go.Scatter(y=ecg_y, mode='lines', line=dict(color=COLORS["ecg"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=art_y, mode='lines', line=dict(color=COLORS["art"], width=2)), row=2, col=1)
    
    fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_infusion_pump(drug_name: str, callback: Callable):
    info = DRUG_INFO.get(drug_name, {})
    val = st.session_state.params["drugs"][drug_name]
    st.slider(
        f"{info.get('label', drug_name)} ({info.get('unit','')})", 
        0.0, info.get('max_dose', 1.0), val, info.get('step', 0.01),
        key=f"drug_{drug_name}", on_change=callback, args=(drug_name,)
    )

def render_fluid_manager():
    c1, c2 = st.columns(2)
    if c1.button("💧 250mL Crystal"): st.session_state.params["fluids"] += 250
    if c2.button("🩸 1 Unit pRBC"): st.session_state.params["prbc"] += 300
    st.slider("Diuresis (mL)", 0, 5000, st.session_state.params["diuresisVolume"], 100, key="diuresis_slider", 
              on_change=lambda: st.session_state.params.update({"diuresisVolume": st.session_state.diuresis_slider}))

# --- Chart Renderers (Fully Implemented) ---

def render_forrester_plot(data):
    curr = data[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d['ci'] for d in data], y=[d['svri'] for d in data], mode='lines', line_color=COLORS['textMuted']))
    fig.add_trace(go.Scatter(x=[curr['ci']], y=[curr['svri']], mode='markers', marker=dict(color=COLORS['cardiac'], size=10)))
    fig.add_shape(type="rect", x0=2.2, y0=1600, x1=4.0, y1=2400, fillcolor="green", opacity=0.1, line_width=0)
    fig.update_layout(xaxis_title="CI", yaxis_title="SVRI", xaxis_range=[0,6], yaxis_range=[0,4000])
    render_chart_container("Forrester Class", "Perfusion State", fig)

def render_map_trend_plot(data):
    fig = go.Figure(go.Scatter(x=[d['time'] for d in data], y=[d['map'] for d in data], mode='lines', line_color=COLORS['hemo'], fill='tozeroy'))
    fig.add_hline(y=65, line_color='red', line_dash='dash')
    fig.update_layout(yaxis_range=[40, 120])
    render_chart_container("MAP Trend", "Target > 65", fig)

def render_guyton_plot(curr):
    # Simplified Guyton Lines
    pmsf = curr['pmsf']
    cvp = np.linspace(0, 20, 20)
    vr = np.maximum(0, (pmsf - cvp) / 2) # VR Curve
    co = np.minimum(10, cvp * curr['contractility']) # Function Curve
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cvp, y=vr, name='VR', line_color='blue'))
    fig.add_trace(go.Scatter(x=cvp, y=co, name='CO', line_color='red'))
    fig.add_trace(go.Scatter(x=[curr['cvp']], y=[curr['ci']], mode='markers', marker=dict(size=10, color='black')))
    fig.update_layout(xaxis_title="CVP", yaxis_title="Flow (L/min)", xaxis_range=[0,20], yaxis_range=[0,10])
    render_chart_container("Guyton", "Venous Return", fig)

def render_coupling_plot(data):
    curr = data[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[curr['contractility']], y=[curr['ea']], mode='markers', marker=dict(size=12, color='purple')))
    fig.add_shape(type="line", x0=0, y0=0, x1=3, y1=3, line=dict(dash='dash', color='gray'))
    fig.update_layout(xaxis_title="Ees (Contractility)", yaxis_title="Ea (Afterload)")
    render_chart_container("V-A Coupling", "Optimization", fig)

def render_pv_loop_plot(curr):
    fig = go.Figure()
    if curr['pvLoop']:
        loop = curr['pvLoop']
        fig.add_trace(go.Scatter(x=[p['volume'] for p in loop], y=[p['pressure'] for p in loop], fill='toself', line_color=COLORS['cardiac']))
    fig.update_layout(xaxis_title="Volume", yaxis_title="Pressure", xaxis_range=[0,250], yaxis_range=[0,150])
    render_chart_container("PV Loop", "Mechanics", fig)

def render_frank_starling_plot(curr):
    pl = np.linspace(0, 200, 50)
    sv = (120 * curr['contractility'] * pl) / (40 + pl)
    fig = go.Figure(go.Scatter(x=pl, y=sv, mode='lines', line_color=COLORS['ok']))
    fig.add_trace(go.Scatter(x=[curr['preload']], y=[curr['sv']], mode='markers', marker=dict(size=10, color='black')))
    fig.update_layout(xaxis_title="Preload", yaxis_title="Stroke Volume")
    render_chart_container("Frank-Starling", "Fluid Response", fig)

def render_lactate_trend_plot(data):
    fig = go.Figure(go.Scatter(x=[d['time'] for d in data], y=[d['lactate'] for d in data], fill='tozeroy', line_color=COLORS['metabolic']))
    fig.update_layout(yaxis_title="Lactate")
    render_chart_container("Metabolic", "Lactate Clearance", fig)

# --- Main Views ---
def render_chart_tabs():
    return st.radio("Views", ["Hemo", "Mech", "Meta"], horizontal=True, label_visibility="collapsed")

def render_main_chart_view(tab, curr, hist):
    c1, c2 = st.columns(2)
    if tab == "Hemo":
        with c1: render_forrester_plot(hist)
        with c2: render_map_trend_plot(hist)
        with c1: render_guyton_plot(curr)
        with c2: render_coupling_plot(hist)
    elif tab == "Mech":
        with c1: render_pv_loop_plot(curr)
        with c2: render_frank_starling_plot(curr)
    elif tab == "Meta":
        with c1: render_lactate_trend_plot(hist)
        with c2: render_vital_card("DO2", curr['do2'], "ml/min")
