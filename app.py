import streamlit as st
import time

# --- Setup must be first ---
st.set_page_config(page_title="TITAN L8", layout="wide", initial_sidebar_state="expanded")

# --- Imports ---
from constants import PATIENT_PROFILES
from types_module import SimulationParams, DrugDosages, RespiratorySettings
from simulation_engine import run_simulation
from analysis_service import analyze_data
from mock_ai_service import get_clinical_analysis
import ui_components as ui

# --- Styles ---
try:
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("style.css not found. UI will be unstyled.")

# --- State ---
if "params" not in st.session_state:
    st.session_state.params = SimulationParams(
        caseId="sepsis",
        drugs=DrugDosages(norepi=0.0, vaso=0.0, phenyl=0.0, epi=0.0, angiotensin=0.0,
                          dobu=0.0, milrinone=0.0, nicardipine=0.0, esm=0.0, 
                          propofol=0.0, fentanyl=0.0, cisatracurium=0.0),
        fluids=0, prbc=0, diuresisVolume=0,
        resp=RespiratorySettings(fio2=0.4, peep=5, rr=16, tv=450, iepo=False),
        isPaced=False
    )

# --- Callbacks ---
def update_drug(name):
    st.session_state.params["drugs"][name] = st.session_state[f"drug_{name}"]

def update_profile():
    st.session_state.params["caseId"] = st.session_state.selected_profile
    st.session_state.params["fluids"] = 0 # Reset volume on case switch

# --- Main ---
def main():
    # Sidebar
    with st.sidebar:
        st.header("TITAN L8")
        st.selectbox("Scenario", [p["id"] for p in PATIENT_PROFILES], key="selected_profile", on_change=update_profile)
        
        with st.expander("Pressors", expanded=True):
            ui.render_infusion_pump("norepi", update_drug)
            ui.render_infusion_pump("vaso", update_drug)
            ui.render_infusion_pump("epi", update_drug)
        
        with st.expander("Inotropes"):
            ui.render_infusion_pump("dobu", update_drug)
            ui.render_infusion_pump("milrinone", update_drug)

        with st.expander("Fluids"):
            ui.render_fluid_manager()
            st.metric("Net Balance", f"{st.session_state.params['fluids'] + st.session_state.params['prbc'] - st.session_state.params['diuresisVolume']} mL")

    # Simulation Logic (Fast)
    history = run_simulation(st.session_state.params, steps=60)
    current = history[-1]
    analysis = analyze_data(history, st.session_state.params)
    ai_text = get_clinical_analysis(current, history)

    # Dashboard
    st.title(f"ICU Command | {current['time']}s")
    
    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1: ui.render_vital_card("MAP", current['map'], "mmHg", {"min":65, "max":110})
    with k2: ui.render_vital_card("CI", current['ci'], "L/min", {"min":2.2, "max":4.0})
    with k3: ui.render_vital_card("SVR", current['svri'], "dyn", {"min":1600, "max":2400})
    with k4: ui.render_vital_card("Lactate", current['lactate'], "mmol", {"min":0, "max":2.0})
    with k5: ui.render_vital_card("SpO2", current['spo2'], "%", {"min":92, "max":100})
    with k6: ui.render_vital_card("CPO", current['cpo'], "W", {"min":0.6, "max":1.5})

    # Middle Section
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1: 
        ui.render_shock_state_tile(current)
        st.caption(ai_text.split('\n')[1]) # Show simple phenotype
    with c2: ui.render_waveform_monitor(current)
    with c3: ui.render_preventive_sentinel(analysis)

    # Charts
    st.divider()
    tab = ui.render_chart_tabs()
    ui.render_main_chart_view(tab, current, history)

if __name__ == "__main__":
    main()
