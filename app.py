import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os
import google.generativeai as genai

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="TITAN L8 | ICU Command",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for the ICU "Dark Mode" Aesthetic
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .css-1d391kg { background-color: #1e293b; } /* Sidebar */
    .stMetric { background-color: #1e293b; border: 1px solid #334155; padding: 10px; border-radius: 8px; }
    .stMetricLabel { color: #94a3b8 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.1em; }
    .stMetricValue { color: #f8fafc !important; font-family: 'JetBrains Mono', monospace; font-weight: 700; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #f8fafc; }
    .critical-alert { border-left: 4px solid #ef4444; background: #450a0a; padding: 1rem; border-radius: 4px; }
    .safe-zone { border-left: 4px solid #10b981; background: #064e3b; padding: 1rem; border-radius: 4px; }
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA MODELS & CONSTANTS
# ==========================================

@dataclass
class DrugDosages:
    norepi: float = 0.0
    vaso: float = 0.0
    phenyl: float = 0.0
    epi: float = 0.0
    angiotensin: float = 0.0
    dobu: float = 0.0
    milrinone: float = 0.0
    nicardipine: float = 0.0
    esm: float = 0.0
    propofol: float = 0.0
    fentanyl: float = 0.0
    cisatracurium: float = 0.0

@dataclass
class RespiratorySettings:
    fio2: float = 0.4
    peep: float = 5.0
    rr: float = 14.0
    tv: float = 450.0
    iepo: bool = False

@dataclass
class PatientProfile:
    id: str
    label: str
    svr_b: float
    co_b: float
    map_b: float
    vo2: float
    hb: float
    weight: float
    height: float
    shunt: float
    copd: float
    compliance: float

PATIENT_PROFILES = {
    "sepsis": PatientProfile("sepsis", "24F Septic Shock", 600, 6.5, 45, 180, 11, 60, 165, 0.20, 1.0, 30),
    "cabg": PatientProfile("cabg", "65M Post-CABG (Low Output)", 1800, 3.2, 75, 140, 10, 85, 175, 0.10, 1.0, 50),
    "hfpef": PatientProfile("hfpef", "82M HFpEF (Diastolic Failure)", 1600, 3.0, 60, 130, 9, 70, 170, 0.20, 1.5, 40),
    "trauma": PatientProfile("trauma", "50M Polytrauma (Hemorrhage)", 2400, 3.5, 65, 150, 7, 90, 180, 0.05, 1.0, 60),
    "ards": PatientProfile("ards", "45M Severe ARDS", 900, 5.0, 65, 160, 12, 80, 178, 0.40, 1.0, 20),
    "neuro": PatientProfile("neuro", "35M Neurostorm (TBI)", 3000, 2.5, 160, 200, 15, 75, 180, 0.05, 1.0, 55),
    "rvi": PatientProfile("rvi", "55M RV Infarct", 1400, 2.2, 55, 130, 14, 82, 176, 0.10, 1.0, 50),
}

DRUG_PK = {
    'norepi': {'ke0': 0.2, 'ec50': 0.3, 'n': 1.5, 'emax': {'svr': 2500.0, 'map': 80.0, 'hr': 15.0, 'contractility': 0.2}},
    'vaso': {'ke0': 0.05, 'ec50': 0.04, 'n': 2.0, 'emax': {'svr': 3500.0, 'map': 90.0, 'co': -0.5}},
    'phenyl': {'ke0': 0.3, 'ec50': 1.5, 'n': 1.5, 'emax': {'svr': 2000.0, 'map': 60.0, 'hr': -10.0}},
    'epi': {'ke0': 0.2, 'ec50': 0.15, 'n': 1.5, 'emax': {'svr': 1500.0, 'hr': 40.0, 'contractility': 1.5, 'co': 3.0}},
    'angiotensin': {'ke0': 0.15, 'ec50': 20.0, 'n': 1.8, 'emax': {'svr': 4000.0, 'map': 100.0}},
    'dobu': {'ke0': 0.15, 'ec50': 5.0, 'n': 1.2, 'emax': {'svr': -400.0, 'co': 4.0, 'hr': 30.0, 'contractility': 1.2}},
    'milrinone': {'ke0': 0.05, 'ec50': 0.5, 'n': 1.0, 'emax': {'svr': -800.0, 'co': 2.5, 'contractility': 0.8, 'map': -10.0}},
    'nicardipine': {'ke0': 0.1, 'ec50': 5.0, 'n': 1.2, 'emax': {'svr': -1500.0, 'map': -50.0}},
    'esm': {'ke0': 0.5, 'ec50': 0.1, 'n': 1.0, 'emax': {'hr': -50.0, 'contractility': -0.4, 'map': -20.0}},
    'propofol': {'ke0': 0.4, 'ec50': 30.0, 'n': 1.5, 'emax': {'svr': -1000.0, 'contractility': -0.5, 'map': -40.0}},
    'fentanyl': {'ke0': 0.1, 'ec50': 200.0, 'n': 1.0, 'emax': {'hr': -10.0, 'svr': -100.0}},
    'cisatracurium': {'ke0': 0.05, 'ec50': 2.0, 'n': 2.0, 'emax': {}}
}

# ==========================================
# 3. SIMULATION ENGINE
# ==========================================

def hill_effect(ce, emax, ec50, n):
    if ce <= 0 or not emax: return 0
    num = emax * math.pow(ce, n)
    den = math.pow(ec50, n) + math.pow(ce, n)
    return num / den

def run_simulation(case_id: str, drugs: DrugDosages, fluid_vol: float, prbc_vol: float, diuresis: float, resp: RespiratorySettings):
    profile = PATIENT_PROFILES[case_id]
    bsa = math.sqrt((profile.height * profile.weight) / 3600)
    
    steps = 100 # Reduced from 300 for Streamlit performance
    data = []
    
    # Init State
    curr_map = profile.map_b
    curr_hr = 90.0
    if case_id == 'anaphylaxis': curr_hr = 120
    
    # Volume Logic
    total_vol_given = fluid_vol + prbc_vol
    effective_vol_add = total_vol_given - diuresis
    curr_vol = (effective_vol_add / 250) + (70 if case_id == 'hfpef' else 50)
    
    # Hemoglobin
    curr_hb = profile.hb
    if case_id in ['trauma', 'liver']:
        dilution = fluid_vol / 5000
        curr_hb = max(4, curr_hb - dilution)
        curr_hb += (prbc_vol / 300)

    # Initial concentrations
    ce = {k: 0.0 for k in vars(drugs)}
    
    # Noise generators
    noise_map = np.random.normal(0, 1, steps)
    noise_hr = np.random.normal(0, 1.5, steps)
    
    # Physics Constants
    ATM = 760
    H2O = 47
    RQ = 0.8
    
    for t in range(steps):
        # 1. Pharmacokinetics
        for k, dose in vars(drugs).items():
            pk = DRUG_PK.get(k)
            if pk:
                decay = math.exp(-pk['ke0'] * 1.0)
                ce[k] = ce[k] * decay + dose * (1 - decay)

        # 2. Pharmacodynamics
        effects = {'svr': 0, 'map': 0, 'co': 0, 'hr': 0, 'contractility': 0}
        for k, conc in ce.items():
            pk = DRUG_PK.get(k)
            if pk:
                for target, max_effect in pk['emax'].items():
                    effects[target] += hill_effect(conc, max_effect, pk['ec50'], pk['n'])

        # 3. Pathophysiology
        svr_base = profile.svr_b
        vol_effective = curr_vol
        contractility = 1.0 + effects['contractility']
        
        # Profile modifiers
        if case_id in ['sepsis', 'anaphylaxis', 'liver']:
            svr_base *= 0.6 # Vasoplegia
            vol_effective *= 0.9 # Capillary Leak
        
        # Neurostorm
        if case_id == 'neuro':
            curr_map = max(curr_map, 140) # Baseline HTN

        # 4. Mechanics
        preload = max(10, vol_effective)
        k_curve = 70 if case_id == 'hfpef' else 40
        sv = (130 * contractility * math.pow(preload, 2)) / (math.pow(k_curve, 2) + math.pow(preload, 2))
        
        total_svr = max(100, (svr_base / bsa) + effects['svr'])
        
        # HR Logic
        target_hr = 80 + effects['hr']
        if case_id in ['sepsis', 'trauma']: target_hr = 110
        curr_hr = curr_hr + 0.1 * (target_hr - curr_hr) + noise_hr[t]
        
        # Neurostorm Cushing's
        icp = 10
        if case_id == 'neuro':
            icp = 25
            if curr_map > 120:
                curr_hr = max(45, curr_hr - 20) # Reflex Brady
        
        curr_co = (sv * curr_hr) / 1000
        ci = curr_co / bsa
        
        target_map = (curr_co * total_svr / 80.0) + 5
        curr_map = curr_map + 0.2 * (target_map - curr_map) + noise_map[t]
        
        # 5. Respiratory & Oxygenation
        vco2 = profile.vo2 / 0.8
        va = (max(1, resp.rr) * (resp.tv/1000)) * (1 - (0.3 * profile.copd))
        paco2 = (0.863 * vco2) / max(0.5, va)
        
        p_ideal = (resp.fio2 * (ATM - H2O)) - (paco2 / RQ)
        eff_shunt = profile.shunt * math.exp(-0.08 * resp.peep)
        pao2 = max(0, p_ideal * (1 - eff_shunt * 2))
        pf_ratio = pao2 / resp.fio2
        
        # Saturation (Hill)
        s = pao2
        spo2 = 100 * (s**3 + 150*s) / (s**3 + 150*s + 23400)
        
        # Delivery/Consumption
        do2 = curr_co * curr_hb * 1.34 * (spo2/100) * 10
        vo2 = profile.vo2 # Simplified
        o2er = min(1.0, vo2 / max(1, do2))
        svo2 = spo2 * (1 - o2er)
        
        # Lactate Model
        lactate = 1.0
        if o2er > 0.3: lactate += (o2er - 0.3) * 10
        if case_id == 'sepsis': lactate += 1.5
        if case_id == 'liver': lactate *= 1.5

        # Mechanics
        esv = sv / 0.8
        esp = curr_map * 0.9 + 20 # Approx Systolic
        edp = 8 # Approx CVP
        cpo = curr_map * curr_co / 451
        
        # PV Loop Data (Last step only)
        pv_loop = []
        if t == steps - 1:
            # Simplified box loop for plotting
            pv_loop = [
                {'v': esv + sv, 'p': edp}, # EDV, EDP
                {'v': esv + sv, 'p': esp}, # Isovolumetric contraction
                {'v': esv, 'p': esp},      # ESV, ESP
                {'v': esv, 'p': edp},      # Isovolumetric relaxation
                {'v': esv + sv, 'p': edp}  # Close loop
            ]

        step_data = {
            'time': t,
            'map': curr_map, 'hr': curr_hr, 'ci': ci, 'svri': total_svr * bsa,
            'cvp': max(2, edp), 'lactate': lactate, 'svo2': svo2, 'spo2': spo2,
            'do2': do2/bsa, 'o2er': o2er * 100,
            'cpo': cpo, 'icp': icp, 'cpp': curr_map - icp,
            'pf_ratio': pf_ratio, 'paco2': paco2,
            'pv_loop': pv_loop
        }
        data.append(step_data)
        
    return pd.DataFrame(data)

# ==========================================
# 4. AI SERVICE
# ==========================================
def get_ai_analysis(current_data, api_key):
    if not api_key:
        return "⚠️ API Key Missing"
    
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        SYSTEM_ROLE: You are TITAN, an autonomous Critical Care AI.
        
        LIVE TELEMETRY:
        - MAP: {current_data['map']:.0f} mmHg
        - CI: {current_data['ci']:.2f} L/min/m2
        - SVR: {current_data['svri']:.0f}
        - Lactate: {current_data['lactate']:.1f}
        - CPO: {current_data['cpo']:.2f} W
        - O2ER: {current_data['o2er']:.0f}%
        
        OUTPUT FORMAT:
        1. PHENOTYPE: [Specific Shock Type]
        2. TRAJECTORY: [Improving/Stable/Decompensating] - Why?
        3. CRITICAL ACTION: The single most effective intervention right now.
        
        Keep it under 75 words.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# ==========================================
# 5. UI COMPONENTS & CHARTS
# ==========================================

def plot_pv_loop(loop_data):
    df = pd.DataFrame(loop_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['v'], y=df['p'], fill='toself', fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='#3b82f6', width=3)))
    fig.update_layout(
        title="PV Loop",
        xaxis_title="Volume (mL)",
        yaxis_title="Pressure (mmHg)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=250
    )
    return fig

def plot_trend(df, col, color, title, target_line=None):
    fig = px.line(df, x='time', y=col, title=title)
    fig.update_traces(line=dict(color=color, width=2))
    if target_line:
        fig.add_hline(y=target_line, line_dash="dash", line_color="white", opacity=0.5)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=150,
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(showgrid=True, gridcolor='#334155', title=None)
    )
    return fig

def plot_cpo_lactate(current):
    fig = go.Figure()
    # Quadrants
    fig.add_shape(type="rect", x0=0, y0=2, x1=0.6, y1=10, fillcolor="rgba(239, 68, 68, 0.2)", line_width=0)
    fig.add_shape(type="rect", x0=0.6, y0=0, x1=2.0, y1=2, fillcolor="rgba(16, 185, 129, 0.2)", line_width=0)
    
    fig.add_trace(go.Scatter(
        x=[current['cpo']], y=[current['lactate']],
        mode='markers', marker=dict(size=15, color='#a855f7', line=dict(width=2, color='white'))
    ))
    fig.update_layout(
        title="Survival Matrix",
        xaxis_title="Cardiac Power (W)",
        yaxis_title="Lactate (mmol/L)",
        xaxis=dict(range=[0, 1.5]),
        yaxis=dict(range=[0, 10]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=20, r=20, t=30, b=20),
        height=250,
        showlegend=False
    )
    return fig

# ==========================================
# 6. MAIN APP LOGIC
# ==========================================

def main():
    # --- Sidebar Controls ---
    st.sidebar.title("TITAN L8 Controls")
    
    # 1. Profile Selection
    selected_profile_id = st.sidebar.selectbox("Simulation Scenario", list(PATIENT_PROFILES.keys()), format_func=lambda x: PATIENT_PROFILES[x].label)
    profile = PATIENT_PROFILES[selected_profile_id]
    
    st.sidebar.markdown("---")
    
    # 2. Drugs (Infusion Pumps)
    st.sidebar.subheader("Infusion Pumps")
    with st.sidebar.expander("Vasopressors", expanded=True):
        d_norepi = st.slider("Norepinephrine (mcg/kg/min)", 0.0, 2.0, 0.0, 0.05)
        d_vaso = st.slider("Vasopressin (u/min)", 0.0, 0.1, 0.0, 0.01)
        d_epi = st.slider("Epinephrine (mcg/kg/min)", 0.0, 1.0, 0.0, 0.05)
    
    with st.sidebar.expander("Inotropes"):
        d_dobu = st.slider("Dobutamine (mcg/kg/min)", 0.0, 20.0, 0.0, 0.5)
        d_mil = st.slider("Milrinone (mcg/kg/min)", 0.0, 0.75, 0.0, 0.05)
        
    with st.sidebar.expander("Sedation & Other"):
        d_prop = st.slider("Propofol (mcg/kg/min)", 0.0, 80.0, 0.0, 5.0)
        d_fent = st.slider("Fentanyl (mcg/kg/hr)", 0.0, 200.0, 0.0, 10.0)

    # 3. Ventilator
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ventilator")
    v_fio2 = st.sidebar.number_input("FiO2", 0.21, 1.0, 0.4, 0.05)
    v_peep = st.sidebar.number_input("PEEP", 0, 24, 5)
    v_rr = st.sidebar.number_input("RR", 8, 40, 14)
    v_tv = st.sidebar.number_input("Tidal Vol (mL)", 200, 800, 450, 10)
    
    # 4. Fluids
    st.sidebar.markdown("---")
    st.sidebar.subheader("Fluids")
    col_f1, col_f2 = st.sidebar.columns(2)
    vol_crys = col_f1.number_input("Crystalloid (mL)", 0, 10000, 0, 250)
    vol_blood = col_f2.number_input("PRBC (mL)", 0, 5000, 0, 300)
    vol_out = st.sidebar.number_input("Diuresis (mL)", 0, 5000, 0, 250)

    # --- Run Simulation ---
    drugs = DrugDosages(norepi=d_norepi, vaso=d_vaso, epi=d_epi, dobu=d_dobu, milrinone=d_mil, propofol=d_prop, fentanyl=d_fent)
    resp = RespiratorySettings(fio2=v_fio2, peep=v_peep, rr=v_rr, tv=v_tv)
    
    # Run the math
    sim_df = run_simulation(selected_profile_id, drugs, vol_crys, vol_blood, vol_out, resp)
    current = sim_df.iloc[-1] # Latest step

    # --- Main Dashboard ---
    
    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("TITAN L8 | Clinical Command")
        st.caption(f"Active Scenario: {profile.label}")
    with col_h2:
        vis_score = (d_norepi*100) + (d_epi*100) + (d_vaso*10000) + (d_mil*10) + d_dobu
        st.metric("VIS Score", f"{vis_score:.1f}", delta="Critical" if vis_score > 20 else None, delta_color="inverse")

    # Top Row: Vital Signs (KPI Cards)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("MAP", f"{current['map']:.0f}", "mmHg")
    k2.metric("CI", f"{current['ci']:.2f}", "L/min/m²")
    k3.metric("SVR", f"{current['svri']:.0f}", "dyn·s")
    k4.metric("Lactate", f"{current['lactate']:.1f}", "mmol/L")
    k5.metric("SpO2", f"{current['spo2']:.0f}%", f"PaO2 {current['map']*0.8:.0f}") # Mock PaO2 correlation for space
    k6.metric("CPO", f"{current['cpo']:.2f}", "W")

    # Middle Row: Tabs for details
    tab_hemo, tab_resp, tab_neuro, tab_ai = st.tabs(["Hemodynamics", "Respiratory", "Neuro/Trauma", "AI Consultant"])
    
    with tab_hemo:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(plot_trend(sim_df, 'map', '#ef4444', "MAP Trend", target_line=65), use_container_width=True)
            st.plotly_chart(plot_trend(sim_df, 'ci', '#3b82f6', "Cardiac Index Trend", target_line=2.2), use_container_width=True)
        with c2:
            st.plotly_chart(plot_pv_loop(current['pv_loop']), use_container_width=True)
            st.plotly_chart(plot_cpo_lactate(current), use_container_width=True)

    with tab_resp:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("P/F Ratio", f"{current['pf_ratio']:.0f}")
            st.plotly_chart(plot_trend(sim_df, 'spo2', '#a855f7', "SpO2 Trend"), use_container_width=True)
        with c2:
            st.metric("O2 Extraction", f"{current['o2er']:.0f}%", "Normal: 25-30%")
            st.plotly_chart(plot_trend(sim_df, 'do2', '#10b981', "Oxygen Delivery (DO2)"), use_container_width=True)

    with tab_neuro:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("ICP", f"{current['icp']:.0f}", "mmHg", delta_color="inverse")
        with c2:
            st.metric("CPP", f"{current['cpp']:.0f}", "mmHg (MAP - ICP)")
        
        st.info("Neurostorm Logic: If ICP > 20, Cushing's Reflex activates (HTN + Bradycardia).")

    with tab_ai:
        st.subheader("TITAN Diagnostic Sentinel")
        api_key = os.getenv("API_KEY") 
        if not api_key:
            st.warning("Please set API_KEY environment variable for AI analysis.")
            st.text_input("Or enter API Key here:", key="user_api_key", type="password")
            if st.session_state.get("user_api_key"):
                api_key = st.session_state.user_api_key
        
        if st.button("Generate Clinical Analysis", type="primary"):
            with st.spinner("Analyzing phenotype..."):
                analysis = get_ai_analysis(current, api_key)
                st.markdown(f"""
                <div style="background-color: #1e1b4b; padding: 15px; border-radius: 8px; border: 1px solid #4f46e5;">
                    <h4 style="color: #818cf8; margin-top:0;">TITAN ASSESSMENT</h4>
                    <p style="font-family: monospace; white-space: pre-line;">{analysis}</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
