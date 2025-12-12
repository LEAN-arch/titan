import numpy as np
from typing import List, Dict
from types_module import SimulationParams, SimulationStep, SOFAScore
from constants import PATIENT_PROFILES, DRUG_PK

def _hill_effect(ce: float, emax: float, ec50: float, n_hill: float) -> float:
    """Calculates drug effect with zero-division safety."""
    if ce <= 0 or emax == 0 or ec50 <= 0: return 0.0
    cn = float(ce) ** n_hill
    en = float(ec50) ** n_hill
    denom = en + cn
    if denom == 0: return 0.0
    return (emax * cn) / denom

def _generate_waveforms(hr: float, sys: float, dia: float):
    """Generates a snapshot of waveform data for the monitor."""
    points = 100
    if hr <= 10: hr = 60
    
    # Vectorized generation
    phase = np.linspace(0, 1, points)
    
    # Synthetic ECG
    # P-wave, QRS, T-wave approximation
    y_ecg = (0.1 * np.exp(-500 * (phase - 0.1)**2) +  # P
             1.5 * np.exp(-1000 * (phase - 0.38)**2) - # R
             0.3 * np.exp(-800 * (phase - 0.42)**2) +  # S
             0.3 * np.exp(-200 * (phase - 0.7)**2))    # T
             
    # Synthetic Arterial Line
    # Dicrotic notch logic
    t_sys = 0.35
    y_art = np.zeros(points)
    
    # Systole
    mask_sys = phase < t_sys
    y_art[mask_sys] = dia + (sys - dia) * np.sin(phase[mask_sys] * (np.pi / t_sys))
    
    # Diastole with exponential decay
    mask_dia = phase >= t_sys
    decay = np.exp(-3 * (phase[mask_dia] - t_sys))
    notch = 0.1 * (sys - dia) * np.exp(-100 * (phase[mask_dia] - t_sys - 0.05)**2)
    y_art[mask_dia] = dia + ((sys - dia) * 0.5 * decay) + notch

    # Convert to lightweight list of dicts for Plotly
    ecg_data = [{"t": i, "y": float(y)} for i, y in enumerate(y_ecg)]
    art_data = [{"t": i, "y": float(y)} for i, y in enumerate(y_art)]
    
    return ecg_data, art_data

def run_simulation(params: SimulationParams, steps: int = 60) -> List[SimulationStep]:
    """
    Main physics engine. 
    OPTIMIZATION: Waveforms are only generated for the final step.
    """
    profile = next((p for p in PATIENT_PROFILES if p["id"] == params["caseId"]), PATIENT_PROFILES[0])
    
    # 1. Setup Initial Physiology
    bsa = np.sqrt((profile["height"] * profile["weight"]) / 3600)
    map_val = float(profile["map_b"])
    hr = 80.0 if profile["id"] != 'sepsis' else 110.0
    lactate = 1.0
    
    # 2. Volume Status logic
    total_fluid_in = params["fluids"] + params["prbc"]
    net_vol = total_fluid_in - params["diuresisVolume"]
    # Simple curve: 1000mL = 10% preload increase
    preload_mod = 1.0 + (net_vol / 5000.0)
    
    # 3. Drug concentrations (Steady state approx for responsiveness)
    # In a real PK model, we'd integrate over time. Here we map sliders directly to Ce for UI responsiveness.
    drugs = params["drugs"]
    
    # 4. Loop
    data_history = []
    
    # Pre-calculate Drug Effects (Vectorized outside loop for speed)
    d_svr = 0.0
    d_cont = 0.0
    d_hr = 0.0
    vis = 0.0
    
    # VIS Score
    vis = (drugs["norepi"] * 100) + (drugs["epi"] * 100) + (drugs["vaso"] * 10000) + (drugs["milrinone"] * 10) + drugs["dobu"]

    for drug_name, dose in drugs.items():
        pk = DRUG_PK.get(drug_name)
        if pk:
            emax = pk["emax"]
            if "svr" in emax: d_svr += _hill_effect(dose, emax["svr"], pk["ec50"], pk["n"])
            if "contractility" in emax: d_cont += _hill_effect(dose, emax["contractility"], pk["ec50"], pk["n"])
            if "hr" in emax: d_hr += _hill_effect(dose, emax["hr"], pk["ec50"], pk["n"])

    # Baseline physics
    svr_base = profile["svr_b"]
    vo2 = profile["vo2"]
    
    for t in range(steps):
        # Apply Physiology
        svr_final = max(200, svr_base + d_svr)
        cont_final = max(0.2, 1.0 + d_cont)
        hr_final = max(30, min(200, hr + d_hr))
        
        # Frank-Starling Relationship
        # SV depends on Preload (Volume) and Contractility
        preload = 100 * preload_mod
        sv = (120 * cont_final * preload) / (40 + preload)
        sv = max(10, min(200, sv)) # Safety clamps
        
        # Cardiac Output
        co = (sv * hr_final) / 1000.0
        ci = co / bsa
        
        # Pressure Generation (MAP = CO * SVR) + CVP
        # Damping factor creates a trend rather than instant jumps
        target_map = (co * svr_final / 80.0) + 5
        map_val = map_val + 0.15 * (target_map - map_val)
        
        # Oxygenation
        hb = profile["hb"] + (params["prbc"] / 300.0)
        do2 = co * hb * 1.34 * 0.98 * 10
        o2er = min(0.99, vo2 / max(1.0, do2))
        
        # Lactate Logic (Anaerobic threshold)
        lac_change = 0.0
        if o2er > 0.35: lac_change = (o2er - 0.35) * 0.2
        else: lac_change = -0.05 * lactate # Clearance
        lactate = max(0.5, lactate + lac_change)
        
        # Derived
        svri = svr_final * bsa
        cpo = map_val * co / 451.0
        
        # Waveforms - ONLY GENERATE ON LAST STEP
        # This fixes the "Oven" / Loading issue
        ecg = []
        art = []
        pv_loop = []
        
        if t == steps - 1:
            ecg, art = _generate_waveforms(hr_final, map_val + 15, map_val - 15)
            # Simple PV Loop box for visualization
            esv = sv * 0.6
            edv = esv + sv
            esp = map_val + 10
            edp = 8
            pv_loop = [
                {"volume": edv, "pressure": edp},
                {"volume": edv, "pressure": esp},
                {"volume": esv, "pressure": esp},
                {"volume": esv, "pressure": edp},
                {"volume": edv, "pressure": edp}
            ]

        # Pack Data
        step: SimulationStep = {
            "time": t, "hr": hr_final, "map": map_val, "ci": ci, "svri": svri,
            "lactate": lactate, "spo2": 100 - (o2er*20), "svo2": (1-o2er)*100,
            "cpo": cpo, "vis": vis, "do2": do2, "vo2": vo2, "o2er": o2er*100,
            "ecgWave": ecg, "artWave": art, "pvLoop": pv_loop,
            "cvp": 8, "sys": map_val+15, "dia": map_val-15, "temp": 37,
            "pfRatio": 300, "hb": hb, "etco2": 35, "ph": 7.35, "paco2": 40,
            "creatinine": 1.0, "urineOutput": 1.0, "bilirubin": 0.5, "platelets": 150,
            "gcs": 15, "sofa": {"total": 0}, "sv": sv, "ppv": 5, "shockIndex": hr_final/map_val,
            "contractility": cont_final, "preload": preload, "ea": 1.5, "vac": 1.0,
            "pvaCO2": 4, "strokeWork": 0, "potentialEnergy": 0, "efficiency": 0.25,
            "pmsf": 15, "vrResistance": 2, "icp": 10, "cpp": map_val-10,
            "peakPressure": 20, "platPressure": 18, "drivingPressure": 10,
            "mechanicalPower": 10, "rsbi": 40, "cumulativeO2Debt": 0,
            "ventPressure": [{"t":0, "y":5}], "ventFlow": [{"t":0, "y":0}]
        }
        data_history.append(step)
        
    return data_history
