import streamlit as st
import time
import os

# --- Module Imports ---
from constants import PATIENT_PROFILES, CLINICAL_DEFINITIONS
from types_module import SimulationParams, DrugDosages, RespiratorySettings
from simulation_engine import run_simulation
from analysis_service import analyze_data
from mock_ai_service import get_clinical_analysis
import ui_components as ui

# --- Page Configuration ---
st.set_page_config(
    page_title="TITAN L8 | ICU Command",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Injection ---
with open("style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Session State Initialization ---
def init_state():
    if "params" not in st.session_state:
        # Default to the first profile (Sepsis)
        default_profile = PATIENT_PROFILES[0]
        st.session_state.params = SimulationParams(
            caseId=default_profile["id"],
            drugs=DrugDosages(norepi=0.0, vaso=0.0, phenyl=0.0, epi=0.0, angiotensin=0.0,
                              dobu=0.0, milrinone=0.0, nicardipine=0.0, esm=0.0, 
                              propofol=10.0, fentanyl=25.0, cisatracurium=0.0),
            fluids=0,
            prbc=0,
            diuresisVolume=0,
            resp=RespiratorySettings(fio2=0.4, peep=5, rr=16, tv=450, iepo=False),
            isPaced=False
        )

init_state()

# --- Callback Functions for UI Interactivity ---
def update_drug(drug_name):
    val = st.session_state[f"drug_{drug_name}"]
    st.session_state.params["drugs"][drug_name] = val

def update_resp(param_name):
    val = st.session_state[f"resp_{param_name}"]
    st.session_state.params["resp"][param_name] = val

def update_profile():
    new_id = st.session_state.selected_profile_id
    st.session_state.params["caseId"] = new_id
    # Reset fluid/blood on profile switch for a fresh start
    st.session_state.params["fluids"] = 0
    st.session_state.params["prbc"] = 0

# =============================================================================
# --- MAIN APPLICATION LAYOUT ---
# =============================================================================

def main():
    # 1. Sidebar Controls
    with st.sidebar:
        st.markdown('<div class="header"><div class="logo">L8</div><div><h1>TITAN</h1><h2>Life Support</h2></div></div>', unsafe_allow_html=True)
        
        # Profile Selector
        st.selectbox(
            "CLINICAL SCENARIO", 
            options=[p["id"] for p in PATIENT_PROFILES], 
            format_func=lambda x: next(p["label"] for p in PATIENT_PROFILES if p["id"] == x),
            key="selected_profile_id",
            on_change=update_profile
        )
        
        st.markdown("---")
        
        # Infusion Pumps
        with st.expander("INFUSION PUMPS", expanded=True):
            tabs = st.tabs(["PRESSORS", "INOTROPES", "SEDATION"])
            with tabs[0]:
                ui.render_infusion_pump("norepi", update_drug)
                ui.render_infusion_pump("vaso", update_drug)
                ui.render_infusion_pump("epi", update_drug)
                ui.render_infusion_pump("phenyl", update_drug)
            with tabs[1]:
                ui.render_infusion_pump("dobu", update_drug)
                ui.render_infusion_pump("milrinone", update_drug)
            with tabs[2]:
                ui.render_infusion_pump("propofol", update_drug)
                ui.render_infusion_pump("fentanyl", update_drug)
        
        # Ventilator
        with st.expander("MECHANICAL VENTILATION", expanded=False):
            ui.render_ventilator_panel(st.session_state.params["caseId"], update_resp)
            
        # Fluids
        with st.expander("FLUID RESUSCITATION", expanded=False):
            ui.render_fluid_manager()
            st.metric("Net Fluid Balance", f"{st.session_state.params['fluids'] + st.session_state.params['prbc'] - st.session_state.params['diuresisVolume']} mL")

    # 2. Simulation & Analysis Core
    # We run the simulation *every* rerun to reflect the current state immediately (React pattern)
    sim_history = run_simulation(st.session_state.params, steps=100)
    current_step = sim_history[-1]
    
    # Run Advanced Analytics
    advanced_analysis = analyze_data(sim_history, st.session_state.params)
    
    # Run Mock AI Diagnosis
    ai_diagnosis = get_clinical_analysis(current_step, sim_history)

    # 3. Main Dashboard UI
    
    # Top Row: Header & VIS
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("Clinical Command Center")
    with col_h2:
        ui.render_treatment_header(current_step['vis'], 70) # Assuming 70kg standard for now

    # Row 1: KPI Cards (Vitals)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: ui.render_vital_card("MAP", current_step['map'], "mmHg", {"min":65, "max":110})
    with k2: ui.render_vital_card("CI", current_step['ci'], "L/min/m²", {"min":2.2, "max":4.0})
    with k3: ui.render_vital_card("SVR", current_step['svri'], "dyn·s", {"min":1600, "max":2400})
    with k4: ui.render_vital_card("Lactate", current_step['lactate'], "mmol/L", {"min":0, "max":2.0})
    with k5: ui.render_vital_card("SpO2", current_step['spo2'], "%", {"min":92, "max":100})
    with k6: ui.render_vital_card("CPO", current_step['cpo'], "W", {"min":0.6, "max":1.5})

    # Row 2: Diagnostics & Waveforms
    d1, d2, d3 = st.columns([1, 2, 1])
    with d1:
        ui.render_shock_state_tile(current_step)
        st.markdown(f"<div style='background:#f1f5f9;padding:10px;border-radius:5px;font-family:monospace;font-size:0.8em;white-space:pre-line;margin-top:10px;'>{ai_diagnosis}</div>", unsafe_allow_html=True)
    with d2:
        ui.render_waveform_monitor(current_step)
    with d3:
        ui.render_preventive_sentinel(advanced_analysis)

    # Row 3: Prescriptions & Trends
    p1, p2 = st.columns([1, 1])
    with p1:
        ui.render_prescriptive_plan(advanced_analysis)
    with p2:
        ui.render_predictive_horizon(sim_history, advanced_analysis["forecastMAP"], "MAP Trajectory")

    # Row 4: Detailed Charting Workspace
    st.markdown("### Advanced Physiology")
    active_tab = ui.render_chart_tabs()
    ui.render_main_chart_view(active_tab, current_step, sim_history)

if __name__ == "__main__":
    main()
