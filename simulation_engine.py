import numpy as np
from typing import List, Dict
from types_module import SimulationParams, SimulationStep, SOFAScore
from constants import PATIENT_PROFILES, DRUG_PK, PHYSICS

def _hill_effect(ce: float, emax: float, ec50: float, n_hill: float) -> float:
    if ce <= 0 or not emax or ec50 <= 0: return 0
    cn = np.power(ce, n_hill)
    en = np.power(ec50, n_hill)
    if (en + cn) == 0: return 0
    return (emax * cn) / (en + cn)

def _calculate_sofa(vals: dict, drugs: dict, weight: float) -> SOFAScore:
    # Simplified mock SOFA calculation
    return {"respiratory": 0, "coagulation": 0, "liver": 0, "cardiovascular": 0, "cns": 0, "renal": 0, "total": 0}

def _generate_waveforms(hr: float, sys: float, dia: float, ppv_mag: float, rr: float):
    points = 100
    if hr <= 0: hr = 60
    beat_dur = 60 / hr
    phase = np.linspace(0, 1, points)
    
    # ECG Logic
    y_ecg = np.zeros(points)
    y_ecg += 1.0 * np.exp(-1000 * (phase - 0.38)**2) # R-wave
    ecg_wave = [{"t": i, "y": float(val)} for i, val in enumerate(y_ecg)]
    
    # Arterial Logic
    pp = sys - dia
    y_art = dia + pp * np.sin(phase * np.pi) * (phase < 0.5)
    art_wave = [{"t": i, "y": float(val)} for i, val in enumerate(y_art)]
    
    return ecg_wave, art_wave

def run_simulation(params: SimulationParams, steps: int = 100) -> List[SimulationStep]:
    profile = next((p for p in PATIENT_PROFILES if p["id"] == params["caseId"]), PATIENT_PROFILES[0])
    bsa = np.sqrt((profile["height"] * profile["weight"]) / 3600)
    
    # Init State Variables
    map_val = float(profile["map_b"])
    hr = 80.0
    if profile["id"] == 'sepsis': hr = 110.0
    lactate = 1.0
    
    # Volume Status
    total_fluid_in = params["fluids"] + params["prbc"]
    net_vol = total_fluid_in - params["diuresisVolume"]
    vol_status_modifier = 1.0 + (net_vol / 5000) # Simple volume expansion model
    
    data_history = []
    
    # Pre-calculate drug concentrations (steady state approximation for responsiveness)
    ce = {}
    for drug, dose in params["drugs"].items():
        ce[drug] = dose # Simplified PK for immediate UI response
        
    # Effect Summation
    d_svr, d_cont, d_hr = 0.0, 0.0, 0.0
    vis = 0.0
    
    # VIS Score Calculation
    vis += params["drugs"]["norepi"] * 100
    vis += params["drugs"]["epi"] * 100
    vis += params["drugs"]["vaso"] * 10000
    vis += params["drugs"]["milrinone"] * 10
    vis += params["drugs"]["dobu"]
    
    for drug, dose in params["drugs"].items():
        pk = DRUG_PK.get(drug)
        if pk:
            emax = pk["emax"]
            if "svr" in emax: d_svr += _hill_effect(dose, emax["svr"], pk["ec50"], pk["n"])
            if "contractility" in emax: d_cont += _hill_effect(dose, emax["contractility"], pk["ec50"], pk["n"])
            if "hr" in emax: d_hr += _hill_effect(dose, emax["hr"], pk["ec50"], pk["n"])

    # Loop for trend generation (short history)
    for t in range(steps):
        # Physiology
        svr = profile["svr_b"] + d_svr
        contractility = 1.0 + d_cont
        final_hr = hr + d_hr
        
        # Hemodynamics
        sv = 70 * contractility * vol_status_modifier
        co = (sv * final_hr) / 1000
        svr = max(100, svr) # Safety clamp
        
        # MAP Calculation (Ohms Law)
        target_map = (co * svr) / 80
        # Smooth approach to target
        map_val = map_val + 0.1 * (target_map - map_val)
        
        # Oxygenation
        hb = profile["hb"] + (params["prbc"] / 300)
        do2 = co * hb * 1.34 * 0.98 * 10
        vo2 = profile["vo2"]
        o2er = min(0.99, vo2 / max(1, do2))
        
        # Lactate Kinetics
        lac_prod = 0.0
        if o2er > 0.4: lac_prod = (o2er - 0.4) * 0.5
        lactate = lactate + lac_prod - (lactate * 0.05) # Clearance
        lactate = max(0.5, lactate)

        # Derived metrics
        ci = co / bsa
        svri = svr * bsa
        cpo = map_val * co / 451
        
        ecg, art = _generate_waveforms(final_hr, map_val + 10, map_val - 10, 0, 12)

        step: SimulationStep = {
            "time": t, "hr": final_hr, "map": map_val, "ci": ci, "svri": svri,
            "lactate": lactate, "spo2": 98.0, "svo2": (1-o2er)*100, "cpo": cpo,
            "vis": vis, "do2": do2, "vo2": vo2, "o2er": o2er*100,
            "ecgWave": ecg, "artWave": art,
            "cvp": 8, "sys": map_val+10, "dia": map_val-10, "temp": 37,
            "pfRatio": 300, "hb": hb, "etco2": 35, "ph": 7.35, "paco2": 40,
            "creatinine": 1.0, "urineOutput": 1.0, "bilirubin": 0.5, "platelets": 150,
            "gcs": 15, "sofa": {}, "sv": sv, "ppv": 5, "shockIndex": final_hr/map_val,
            "contractility": contractility, "preload": 100, "ea": 1.5, "vac": 1.0,
            "pvaCO2": 4, "strokeWork": 0, "potentialEnergy": 0, "efficiency": 0.25,
            "pmsf": 15, "vrResistance": 2, "icp": 10, "cpp": map_val-10,
            "peakPressure": 20, "platPressure": 18, "drivingPressure": 10,
            "mechanicalPower": 10, "rsbi": 40, "cumulativeO2Debt": 0,
            "pvLoop": [{"volume": 50, "pressure": 10}, {"volume": 120, "pressure": 120}], # Mock loop
            "ventPressure": [{"t":0, "y":5}], "ventFlow": [{"t":0, "y":0}]
        }
        data_history.append(step)
        
    return data_history
