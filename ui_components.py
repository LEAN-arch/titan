"""
=============================================================================
UI Components Module for TITAN L8 (Ultimate, Complete Version)
=============================================================================
This file contains all functions for rendering the application's UI, from
cards to complex Plotly charts, mirroring a professional component architecture.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Callable
import numpy as np

# --- Core Module Imports ---
from types_module import SimulationStep, AdvancedAnalysis, PatientProfile, RiskFactor, Prescription
from constants import COLORS, CLINICAL_DEFINITIONS, DRUG_INFO

# =============================================================================
# --- 1. SHARED & HELPER COMPONENTS ---
# =============================================================================

CHART_MARGIN = dict(l=40, r=20, t=50, b=40)
AXIS_STYLE = dict(showgrid=True, gridcolor='#e2e8f0', gridwidth=1, zeroline=False, linecolor='#d1d5db', linewidth=1)
TOOLTIP_FONT = dict(family="JetBrains Mono", size=12, color="white")

def render_chart_container(title: str, subtitle: str, fig: go.Figure):
    """A reusable container for displaying a Plotly chart with a standardized title."""
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:0.8em;color:{COLORS['textMuted']}'>{subtitle}</span>", font=dict(size=14, family='Inter')),
        margin=CHART_MARGIN,
        paper_bgcolor='white',
        plot_bgcolor='#fafbfd',
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# =============================================================================
# --- 2. TOP-LEVEL & CARD COMPONENTS ---
# =============================================================================

def render_shock_state_tile(current_step: SimulationStep):
    if current_step["map"] < 65: state, color = "UNCOMPENSATED", COLORS["crit"]
    elif current_step["ci"] < 2.2 and current_step["svri"] > 2000: state, color = "CARDIOGENIC", COLORS["crit"]
    elif current_step["ci"] > 2.2 and current_step["svri"] < 1200: state, color = "DISTRIBUTIVE", COLORS["alert"]
    else: state, color = "STABLE", COLORS["ok"]
    st.markdown(f"""<div class="shock-tile" style="border-top: 5px solid {color};"><p class="tile-label">DIAGNOSIS</p><h3 style="color: {color};">{state}</h3></div>""", unsafe_allow_html=True)

def render_waveform_monitor(current_step: SimulationStep):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0)
    fig.add_trace(go.Scatter(y=[d['y'] for d in current_step["ecgWave"]], mode='lines', line=dict(color=COLORS["ecg"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=[d['y'] for d in current_step["artWave"]], mode='lines', line=dict(color=COLORS["art"], width=2)), row=2, col=1)
    fig.update_layout(height=240, margin=dict(l=0, r=0, t=0, b=0), showlegend=False, plot_bgcolor='#0f172a', paper_bgcolor='#0f172a')
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[-0.5, 1.5], row=1, col=1)
    fig.update_yaxes(range=[0, 180], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def render_vital_card(label: str, value: float, unit: str, threshold: dict = None):
    is_critical = threshold and (value < threshold["min"] or value > threshold["max"])
    st.metric(label, f"{value:.1f if label in ['CI', 'CPO', 'Lactate'] else int(value)} {unit}", delta="CRITICAL" if is_critical else None, delta_color="inverse")

def render_predictive_horizon(history: list, forecast: list, label: str):
    if not forecast:
        st.info("Insufficient data to generate forecast.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d['time'] for d in history], y=[d['map'] for d in history], mode='lines', line=dict(color=COLORS["hemo"], width=2), name='History'))
    fig.add_trace(go.Scatter(x=[d['time'] for d in forecast], y=[d['value'] for d in forecast], mode='lines', line=dict(color=COLORS["hemo"], width=2, dash='dash'), name='Forecast'))
    fig.add_trace(go.Scatter(x=[d['time'] for d in forecast] + forecast[::-1], y=[d['upper'] for d in forecast] + [d['lower'] for d in forecast][::-1], fill='toself', fillcolor=f'rgba({int(COLORS["hemo"][1:3],16)},{int(COLORS["hemo"][3:5],16)},{int(COLORS["hemo"][5:7],16)},0.2)', line=dict(width=0), name='Confidence'))
    render_chart_container("Predictive Horizon", label, fig)

def render_preventive_sentinel(analysis: AdvancedAnalysis):
    with st.container(border=True, height=300):
        st.subheader("Sentinel Monitor")
        if not analysis["risks"]:
            st.success("System Stable")
        for risk in analysis["risks"]:
            st.warning(f"**{risk['label']}**: {risk['reasoning']}")

def render_prescriptive_plan(analysis: AdvancedAnalysis):
    # ... (Implementation from previous correct step)
    pass

def render_prescriptive_plan(analysis: AdvancedAnalysis):
    with st.container(border=True, height=300):
        st.subheader("Protocol Engine")
        if not analysis["prescriptions"]:
            st.info("Targets met")
        for rx in analysis["prescriptions"]:
            st.info(f"**{rx['action']}**: {rx['rationale']}")

def render_treatment_header(vis: float, weight: float):
    c1, c2 = st.columns(2)
    c1.metric("Vasoactive-Inotropic Score (VIS)", f"{vis:.1f}")
    c2.metric("Patient Weight", f"{weight} kg")

def render_infusion_pump(drug_name: str, callback: Callable):
    info = DRUG_INFO.get(drug_name, {})
    st.slider(info.get("label", drug_name.upper()), 0.0, info.get("max_dose", 1.0), st.session_state.params["drugs"][drug_name], info.get("step", 0.01), key=f"drug_{drug_name}", on_change=callback, args=(drug_name,), format="%.3f")

def render_ventilator_panel(profile: PatientProfile, callback: Callable):
    st.slider("FiO2", 0.21, 1.0, st.session_state.params["resp"]["fio2"], 0.05, key="resp_fio2", on_change=callback, args=("fio2",))
    st.slider("PEEP (cmH2O)", 0, 24, st.session_state.params["resp"]["peep"], 1, key="resp_peep", on_change=callback, args=("peep",))
    st.slider("Rate (/min)", 8, 40, st.session_state.params["resp"]["rr"], 1, key="resp_rr", on_change=callback, args=("rr",))
    st.slider("Tidal Volume (mL)", 250, 800, st.session_state.params["resp"]["tv"], 10, key="resp_tv", on_change=callback, args=("tv",))

def render_fluid_manager():
    if st.button("+250mL Bolus"): st.session_state.params["fluids"] += 250
    if st.button("+1 Unit pRBC"): st.session_state.params["prbc"] += 300

# =============================================================================
# --- 4. CHARTING WORKSPACE ---
# =============================================================================

def render_chart_tabs() -> str:
    return st.radio("Select View", ['hemo', 'mech', 'resp', 'meta', 'sofa', 'neuro'], format_func=lambda x: x.upper(), horizontal=True, label_visibility="collapsed")

def render_main_chart_view(active_tab: str, current: SimulationStep, history: list):
    col1, col2 = st.columns(2)
    chart_map = {
        'hemo': [render_forrester_plot, render_guyton_plot, render_map_trend_plot, render_coupling_plot],
        'mech': [render_pv_loop_plot, render_frank_starling_plot, render_energetics_plot, render_energetic_phase_plot],
        'resp': [render_vent_scalars, render_protective_vent_radar, render_pf_ratio_gauge, render_mechanical_power_gauge],
        'meta': [render_metabolic_radar, render_o2er_gauge, render_oxygen_balance_plot, render_lactate_trend_plot],
        'sofa': [render_sofa_radar_plot, render_sofa_trend_plot],
        'neuro': [render_cerebral_auto_plot, render_intracranial_compliance_plot, None, render_oxy_hemo_dissociation_plot]
    }
    charts_to_render = chart_map.get(active_tab, [])
    
    with col1:
        if charts_to_render: charts_to_render[0](history if 'data' in charts_to_render[0].__code__.co_varnames else current)
        if len(charts_to_render) > 2: charts_to_render[2](history if 'data' in charts_to_render[2].__code__.co_varnames else current)
    with col2:
        if len(charts_to_render) > 1: charts_to_render[1](history if 'data' in charts_to_render[1].__code__.co_varnames else current)
        if len(charts_to_render) > 3: charts_to_render[3](history if 'data' in charts_to_render[3].__code__.co_varnames else current)

# --- INDIVIDUAL CHART FUNCTIONS (COMPLETE IMPLEMENTATIONS) ---

def render_forrester_plot(data: List[SimulationStep]):
    """Renders the Forrester Plot for hemodynamic state classification."""
    current = data[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d['ci'] for d in data], y=[d['svri'] for d in data], mode='lines+markers', marker_size=3, line_color=COLORS["textMuted"], name="Trajectory"))
    fig.add_trace(go.Scatter(x=[current['ci']], y=[current['svri']], mode='markers', marker=dict(color=COLORS["cardiac"], size=12, symbol='x'), name='Current'))
    fig.update_layout(xaxis_title="Cardiac Index (L/min/m²)", yaxis_title="SVRI (dyn·s/cm⁵/m²)", xaxis_range=[0,6], yaxis_range=[0,4000])
    fig.add_shape(type="rect", x0=2.5, y0=1600, x1=4.0, y1=2400, fillcolor=COLORS["ok"], opacity=0.2, layer="below", line_width=0)
    fig.add_hline(y=2000, line_dash="dash", line_color=COLORS["textMuted"], opacity=0.5)
    fig.add_vline(x=2.2, line_dash="dash", line_color=COLORS["textMuted"], opacity=0.5)
    render_chart_container("Hemo-Target", "Perfusion States", fig)

def render_guyton_plot(current: SimulationStep):
    """Renders the Guyton Diagram of venous return and cardiac function."""
    vr_cvp = np.arange(0, current['pmsf'] + 2, 0.5)
    vr_flow = (current['pmsf'] - vr_cvp) / max(0.1, current['vrResistance'])
    
    cf_cvp = np.arange(0, 20, 0.5)
    sv = (120 * current['contractility'] * cf_cvp) / (4 + cf_cvp)
    cf_flow = (sv * current['hr']) / 1000

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vr_cvp, y=vr_flow, mode='lines', name='Venous Return', line_color=COLORS['hemo']))
    fig.add_trace(go.Scatter(x=cf_cvp, y=cf_flow, mode='lines', name='Cardiac Function', line_color=COLORS['purple']))
    fig.add_trace(go.Scatter(x=[current['cvp']], y=[current['ci']], mode='markers', marker=dict(color='black', size=10), name='Operating Point'))
    fig.update_layout(xaxis_title="CVP (mmHg)", yaxis_title="Flow (L/min)", xaxis_range=[0,25], yaxis_range=[0,10], showlegend=True)
    render_chart_container("Volume", "Guyton Diagram", fig)

def render_map_trend_plot(data: List[SimulationStep]):
    """Renders the historical trend of Mean Arterial Pressure."""
    fig = go.Figure(go.Scatter(x=[d['time'] for d in data], y=[d['map'] for d in data], mode='lines', line_color=COLORS['hemo']))
    fig.add_hline(y=65, line_dash="dash", line_color=COLORS['crit'], annotation_text="MAP Target", annotation_position="bottom right")
    fig.update_layout(yaxis_title="MAP (mmHg)", yaxis_range=[40, 140])
    render_chart_container("Perfusion", "MAP Trend", fig)

def render_coupling_plot(data: List[SimulationStep]):
    """Renders the Ventriculo-Arterial Coupling (Ea vs Ees) plot."""
    current = data[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d['contractility'] for d in data], y=[d['ea'] for d in data], mode='lines+markers', marker_size=3, line_color=COLORS['textMuted']))
    fig.add_trace(go.Scatter(x=[current['contractility']], y=[current['ea']], mode='markers', marker=dict(color=COLORS['purple'], size=10)))
    fig.add_shape(type="line", x0=0, y0=0, x1=4, y1=4, line=dict(color=COLORS['ok'], width=2, dash="dash"))
    fig.update_layout(xaxis_title="Ees (Contractility)", yaxis_title="Ea (Afterload)", xaxis_range=[0,4], yaxis_range=[0,4])
    render_chart_container("Coupling", "Ea vs Ees", fig)

def render_pv_loop_plot(current: SimulationStep):
    """Renders the Pressure-Volume Loop of the left ventricle."""
    fig = go.Figure()
    if current['pvLoop']:
        fig.add_trace(go.Scatter(x=[p['volume'] for p in current['pvLoop']], y=[p['pressure'] for p in current['pvLoop']], fill="toself", fillcolor=f'rgba({int(COLORS["cardiac"][1:3], 16)},{int(COLORS["cardiac"][3:5], 16)},{int(COLORS["cardiac"][5:7], 16)},0.2)', line_color=COLORS["cardiac"]))
    fig.update_layout(xaxis_title="LV Volume (mL)", yaxis_title="LV Pressure (mmHg)", xaxis_range=[0,250], yaxis_range=[0,180])
    render_chart_container("Mechanics", "PV Loop", fig)

def render_frank_starling_plot(current: SimulationStep):
    """Renders the Frank-Starling curve and the current operating point."""
    preload_range = np.arange(0, 201, 10)
    max_sv = 150 * current['contractility']
    k_sq = np.power(40 if current['contractility'] > 1 else 60, 2)
    sv_curve = (max_sv * np.power(preload_range, 2)) / (k_sq + np.power(preload_range, 2))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preload_range, y=sv_curve, mode='lines', line_color=COLORS['cardiac'], name='Function'))
    fig.add_trace(go.Scatter(x=[current['preload']], y=[current['sv']], mode='markers', marker=dict(color='black', size=10), name='Current State'))
    fig.update_layout(xaxis_title="Preload (LVEDV, mL)", yaxis_title="Stroke Volume (mL)", xaxis_range=[0,150], yaxis_range=[0,120])
    render_chart_container("Preload", "Responsiveness", fig)

def render_energetics_plot(current: SimulationStep):
    """Renders a stacked bar chart of myocardial efficiency."""
    efficiency = current['efficiency'] * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(y=['Energy'], x=[current['strokeWork']], name='Stroke Work (SW)', orientation='h', marker_color=COLORS['cardiac']))
    fig.add_trace(go.Bar(y=['Energy'], x=[current['potentialEnergy']], name='Potential Energy (PE)', orientation='h', marker_color=COLORS['alert']))
    fig.update_layout(barmode='stack', yaxis_title="", xaxis_title="Joules", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.add_annotation(text=f"Efficiency: {efficiency:.0f}%", align='left', showarrow=False, xref='paper', yref='paper', x=0.05, y=0.1)
    render_chart_container("Energetics", "Myocardial Efficiency", fig)
    
def render_energetic_phase_plot(data: List[SimulationStep]):
    """Renders the 'Survival Matrix' of CPO vs. Lactate."""
    current = data[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d['cpo'] for d in data], y=[d['lactate'] for d in data], mode='lines+markers', marker_size=3, line_color=COLORS['textMuted']))
    fig.add_trace(go.Scatter(x=[current['cpo']], y=[current['lactate']], mode='markers', marker=dict(color=COLORS['purple'], size=10)))
    fig.add_shape(type="rect", x0=0, y0=2, x1=0.6, y1=10, fillcolor=COLORS["crit"], opacity=0.2, layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0.6, y0=0, x1=1.5, y1=2, fillcolor=COLORS["ok"], opacity=0.2, layer="below", line_width=0)
    fig.update_layout(xaxis_title="Cardiac Power Output (W)", yaxis_title="Lactate (mmol/L)", xaxis_range=[0,1.5], yaxis_range=[0,10])
    render_chart_container("Matrix", "Power vs Debt", fig)

def render_vent_scalars(current: SimulationStep):
    """Renders ventilator pressure and flow waveforms."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(y=[p['y'] for p in current['ventPressure']], name='Pressure', line_color=COLORS['respiratory']), row=1, col=1)
    fig.add_trace(go.Scatter(y=[f['y'] for f in current['ventFlow']], name='Flow', line_color=COLORS['alert']), row=2, col=1)
    fig.update_layout(height=300, showlegend=False, margin=CHART_MARGIN)
    fig.update_yaxes(title_text="Pressure (cmH2O)", row=1, col=1)
    fig.update_yaxes(title_text="Flow (L/min)", row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color='black', row=2, col=1)
    render_chart_container("Waveforms", "Scalar Analysis", fig)

def render_protective_vent_radar(current: SimulationStep):
    """Renders the VILI Risk Monitor radar chart."""
    categories = ['Pplat', 'ΔP', 'Power', 'RSBI']
    values = [
        (current['platPressure'] / 30) * 100,
        (current['drivingPressure'] / 15) * 100,
        (current['mechanicalPower'] / 17) * 100,
        (current['rsbi'] / 105) * 100
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', fillcolor=f'rgba({int(COLORS["respiratory"][1:3], 16)},{int(COLORS["respiratory"][3:5], 16)},{int(COLORS["respiratory"][5:7], 16)},0.4)', line_color=COLORS["respiratory"], name='Patient'))
    fig.add_trace(go.Scatterpolar(r=[100, 100, 100, 100], theta=categories, mode='lines', line=dict(color=COLORS['crit'], dash='dash'), name='Limit'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 120])))
    render_chart_container("Safety", "Lung Protection", fig)

def render_pf_ratio_gauge(current: SimulationStep):
    """Renders a gauge for the PaO2/FiO2 ratio."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current['pfRatio'],
        title = {'text': "P/F Ratio (Gas Exchange)"},
        gauge = {
            'axis': {'range': [0, 500]},
            'steps': [
                {'range': [0, 100], 'color': COLORS['crit']},
                {'range': [100, 200], 'color': COLORS['alert']},
                {'range': [200, 300], 'color': '#facc15'}, # yellow-400
                {'range': [300, 500], 'color': COLORS['ok']}],
            'bar': {'color': 'black'}
        }))
    fig.update_layout(height=300, margin=dict(l=30,r=30,t=30,b=30))
    st.plotly_chart(fig, use_container_width=True)
    
def render_mechanical_power_gauge(current: SimulationStep):
    """Renders a gauge for Mechanical Power."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current['mechanicalPower'],
        title = {'text': "Mechanical Power (J/min)"},
        gauge = {'axis': {'range': [0, 25]}, 'steps': [{'range': [17, 25], 'color': COLORS['crit']}], 'bar': {'color': 'black'}}))
    fig.update_layout(height=300, margin=dict(l=30,r=30,t=30,b=30))
    st.plotly_chart(fig, use_container_width=True)

def render_metabolic_radar(current: SimulationStep):
    """Renders the Resuscitation Diamond radar chart."""
    categories = ['Flow (CI)', 'Extraction (SvO2)', 'Microcirc. (Pv-aCO2)', 'Metabolism (Lactate)']
    values = [
        min(1, current['ci'] / 3.0) * 100,
        min(1, current['svo2'] / 70) * 100,
        min(1, 6 / max(1, current['pvaCO2'])) * 100,
        min(1, 1 / max(0.5, current['lactate'])) * 100
    ]
    fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', fillcolor=f'rgba({int(COLORS["purple"][1:3], 16)},{int(COLORS["purple"][3:5], 16)},{int(COLORS["purple"][5:7], 16)},0.4)', line_color=COLORS["purple"]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])))
    render_chart_container("Status", "Resuscitation Diamond", fig)

def render_o2er_gauge(current: SimulationStep):
    """Renders a gauge for Oxygen Extraction Ratio."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current['o2er'],
        title = {'text': "Oxygen Extraction (%)"},
        gauge = {'axis': {'range': [0, 100]}, 'steps': [{'range': [50, 100], 'color': COLORS['crit']}], 'bar': {'color': 'black'}}))
    fig.update_layout(height=300, margin=dict(l=30,r=30,t=30,b=30))
    st.plotly_chart(fig, use_container_width=True)

def render_oxygen_balance_plot(data: List[SimulationStep]):
    """Renders the Oxygen Flux (DO2 vs VO2) chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d['time'] for d in data], y=[d['do2'] for d in data], fill='tozeroy', name='DO2 (Supply)', line_color=COLORS['hemo']))
    fig.add_trace(go.Scatter(x=[d['time'] for d in data], y=[d['vo2'] for d in data], fill='tozeroy', name='VO2 (Demand)', line_color=COLORS['metabolic']))
    fig.update_layout(yaxis_title="mL/min/m²", showlegend=True)
    render_chart_container("Flux", "Supply vs Demand", fig)

def render_lactate_trend_plot(data: List[SimulationStep]):
    """Renders the historical trend of Lactate."""
    fig = go.Figure(go.Scatter(x=[d['time'] for d in data], y=[d['lactate'] for d in data], fill='tozeroy', line_color=COLORS['metabolic']))
    fig.add_hline(y=2.0, line_dash="dash", line_color=COLORS['alert'])
    fig.update_layout(yaxis_title="Lactate (mmol/L)")
    render_chart_container("Kinetics", "Lactate Washout", fig)

def render_sofa_radar_plot(current: SimulationStep):
    """Renders the SOFA score as a radar chart."""
    categories = ['Resp', 'Coag', 'Liver', 'CV', 'CNS', 'Renal']
    values = [current['sofa'][cat.lower()] for cat in categories]
    fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color=COLORS['alert']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 4])))
    render_chart_container("Failure", "SOFA Score", fig)

def render_sofa_trend_plot(data: List[SimulationStep]):
    """Renders the historical trend of the total SOFA score."""
    y_values = [d['sofa']['total'] for d in data]
    fig = go.Figure(go.Scatter(x=[d['time'] for d in data], y=y_values, mode='lines', line_shape='hv', line_color=COLORS['metabolic']))
    fig.update_layout(yaxis_title="Total SOFA Score", yaxis_range=[0, 24])
    render_chart_container("Trend", "Organ Failure", fig)
    
def render_cerebral_auto_plot(current: SimulationStep):
    """Renders the Cerebral Autoregulation (Lassen) curve."""
    map_range = np.arange(30, 161, 5)
    flow_curve = np.full_like(map_range, 50.0)
    flow_curve[map_range < 60] = 50 * (map_range[map_range < 60] / 60)
    flow_curve[map_range > 150] = 50 + (map_range[map_range > 150] - 150)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=map_range, y=flow_curve, mode='lines', line_color=COLORS['neuro']))
    fig.add_trace(go.Scatter(x=[current['map']], y=[50], mode='markers', marker=dict(color='black', size=10)))
    fig.add_shape(type="rect", x0=60, y0=0, x1=150, y1=100, fillcolor=COLORS["ok"], opacity=0.1, layer="below", line_width=0)
    fig.update_layout(xaxis_title="MAP (mmHg)", yaxis_title="CBF (%)", yaxis_range=[0,100])
    render_chart_container("Neuro", "Lassen Curve", fig)

def render_intracranial_compliance_plot(current: SimulationStep):
    """Renders the Intracranial Pressure-Volume (Monroe-Kellie) curve."""
    vol_idx = np.arange(0, 81, 2)
    icp_curve = 5 + 0.1 * np.exp(vol_idx * 0.08)
    current_x = np.log(max(0.1, (current['icp'] - 5) * 10)) / 0.08
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vol_idx, y=icp_curve, mode='lines', line_color=COLORS['neuro']))
    fig.add_trace(go.Scatter(x=[current_x], y=[current['icp']], mode='markers', marker=dict(color='black', size=10)))
    fig.add_hline(y=20, line_dash="dash", line_color=COLORS['crit'])
    fig.update_layout(xaxis_title="Volume (Index)", yaxis_title="ICP (mmHg)", yaxis_range=[0,60])
    render_chart_container("Compliance", "Monroe-Kellie", fig)

def render_oxy_hemo_dissociation_plot(current: SimulationStep):
    """Renders the Oxyhemoglobin Dissociation Curve (Bohr Effect)."""
    def get_sat(po2, ph, temp):
        shift = ((ph - 7.4) * 0.4) - ((temp - 37) * 0.05)
        shifted_po2 = po2 * np.power(10, shift)
        s = np.power(shifted_po2, 3) + 150 * shifted_po2
        return (s / (s + 23400)) * 100
        
    po2_range = np.arange(0, 121, 5)
    std_curve = get_sat(po2_range, 7.4, 37)
    patient_curve = get_sat(po2_range, current['ph'], current['temp'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=po2_range, y=std_curve, mode='lines', line=dict(color=COLORS['textMuted'], dash='dash'), name='Standard'))
    fig.add_trace(go.Scatter(x=po2_range, y=patient_curve, mode='lines', line=dict(color=COLORS['respiratory'], width=2), name='Patient'))
    fig.add_trace(go.Scatter(x=[current['paco2']], y=[current['spo2']], mode='markers', marker=dict(color='black', size=10), name='Current'))
    fig.update_layout(xaxis_title="PaO2 (mmHg)", yaxis_title="SaO2 (%)", showlegend=True)
    render_chart_container("Hgb Affinity", "Dissociation Curve", fig)

# RVSWIGauge and MechanicalPowerGauge are simple metrics, can be included directly in app.py
# OrganTile is also a layout of metrics, better handled in app.py
