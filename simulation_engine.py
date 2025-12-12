"""
=============================================================================
SIMULATION ENGINE for TITAN L8 (Ultimate, Complete Version)
=============================================================================
This file contains the core physiological simulation logic, translated to
Python and heavily optimized with NumPy for high-performance computation.
"""

import numpy as np
from typing import List, Dict

# --- Core Module Imports ---
from types_module import SimulationParams, SimulationStep, DrugDosages, PVPoint, WaveformPoint, SOFAScore
from constants import PATIENT_PROFILES, DRUG_PK, PHYSICS

# =============================================================================
# --- 1. UTILITY & MODEL FUNCTIONS ---
# =============================================================================

def _generate_pink_noise(n: int) -> np.ndarray:
    """Generates a pink noise array for biological variability."""
    white = np.random.randn(n)
    return np.convolve(white, np.array([0.05, 0.95]), 'full')[:n]

def _hill_effect(ce: float, emax: float, ec50: float, n_hill: float) -> float:
    """Calculates the sigmoidal Emax drug effect."""
    if ce <= 0 or not emax or ec50 <= 0: return 0
    cn = np.power(ce, n_hill)
    en = np.power(ec50, n_hill)
    return (emax * cn) / (en + cn)

# =============================================================================
# --- 2. SUB-MODEL CALCULATORS ---
# =============================================================================

def _calculate_sofa(vals: dict, drugs: DrugDosages, weight: float) -> SOFAScore:
    s = {k: 0 for k in ["respiratory", "coagulation", "liver", "cardiovascular", "cns", "renal"]}
    
    pf = vals.get("pfRatio", 400)
    if pf < 100: s["respiratory"] = 4
    elif pf < 200: s["respiratory"] = 3
    elif pf < 300: s["respiratory"] = 2
    elif pf < 400: s["respiratory"] = 1

    norepi_eq = drugs.get("norepi", 0) + drugs.get("epi", 0)
    if norepi_eq > 0.1: s["cardiovascular"] = 4
    elif norepi_eq > 0 or drugs.get("dobu", 0) > 0: s["cardiovascular"] = 3
    elif vals.get("map", 80) < 70: s["cardiovascular"] = 2
    
    s["total"] = sum(s.values())
    return s

def _generate_waveforms(hr: float, sys: float, dia: float, ppv_mag: float, rr: float):
    beat_dur = 60 / hr
    points = 100
    phase = np.linspace(0, 1, points)
    
    # ECG
    y_ecg = np.zeros(points)
    y_ecg += 0.15 * np.sin((phase - 0.05) * 10 * np.pi) * ((phase >= 0.05) & (phase < 0.15))
    y_ecg -= 0.15 * np.sin((phase - 0.35) * 33 * np.pi) * ((phase >= 0.35) & (phase < 0.38))
    y_ecg += 1.0 * np.sin((phase - 0.38) * 25 * np.pi) * ((phase >= 0.38) & (phase < 0.42))
    y_ecg -= 0.25 * np.sin((phase - 0.42) * 33 * np.pi) * ((phase >= 0.42) & (phase < 0.45))
    y_ecg += 0.25 * np.sin((phase - 0.6) * 5 * np.pi) * ((phase >= 0.6) & (phase < 0.8))
    ecg_wave = [{"t": i, "y": val + np.random.normal(0, 0.01)} for i, val in enumerate(y_ecg)]
    
    # Arterial
    pp = sys - dia
    resp_phase = (np.arange(points) / points * beat_dur) % (60 / rr) / (60/rr)
    ppv_factor = 1 + (np.sin(resp_phase * 2 * np.pi) * (ppv_mag / 100))
    current_pp = pp * ppv_factor
    
    y_art = np.full(points, dia)
    systolic_mask = phase < 0.3
    diastolic_mask = ~systolic_mask
    
    y_art[systolic_mask] += current_pp[systolic_mask] * np.sin(phase[systolic_mask] * 3.33 * np.pi / 2)
    
    t_diastole = phase[diastolic_mask] - 0.3
    decay = current_pp[diastolic_mask] * np.exp(-4 * t_diastole)
    notch = (current_pp[diastolic_mask] * 0.1) * np.exp(-100 * np.power(t_diastole - 0.1, 2))
    y_art[diastolic_mask] = dia + decay + notch
    
    art_wave = [{"t": i, "y": val} for i, val in enumerate(y_art)]
    
    return ecg_wave, art_wave

def run_simulation(params: SimulationParams, steps: int = 300) -> List[SimulationStep]:
    profile = next((p for p in PATIENT_PROFILES if p["id"] == params["caseId"]), PATIENT_PROFILES[0])
    bsa = np.sqrt((profile["height"] * profile["weight"]) / 3600)
    
    map_val, hr, lactate, o2_debt = profile["map_b"], 80.0, 1.0, 0.0
    if profile["id"] in ['sepsis', 'anaphylaxis']: hr, lactate = 110.0, 4.0
    if profile["id"] == 'trauma': hr, lactate = 120.0, 5.0
    if profile["id"] == 'neuro': map_val, hr = 110.0, 60.0
    
    blood_vol = profile["weight"] * 0.07
    if profile["id"] in ['sepsis', 'anaphylaxis']: blood_vol *= 0.85
    if profile["id"] == 'trauma': blood_vol *= 0.65
    blood_vol += (params["fluids"] + params["prbc"] - params["diuresisVolume"]) / 1000
    
    ce = {k: 0.0 for k in DRUG_PK.keys()}
    noise_hr = _generate_pink_noise(steps)
    noise_map = _generate_pink_noise(steps)
    data_history: List[SimulationStep] = []

    for t in range(steps):
        for drug, pk in DRUG_PK.items():
            decay = np.exp(-pk["ke0"])
            ce[drug] = ce[drug] * decay + params["drugs"][drug] * (1 - decay)

        dSVR, dContractility, dHR, dVeno = 0.0, 0.0, 0.0, 0.0
        for drug, pk in DRUG_PK.items():
            emax = pk["emax"]
            if "svr" in emax: dSVR += _hill_effect(ce[drug], emax["svr"], pk["ec50"], pk["n"])
            if "contractility" in emax: dContractility += _hill_effect(ce[drug], emax["contractility"], pk["ec50"], pk["n"])
            if "hr" in emax: dHR += _hill_effect(ce[drug], emax["hr"], pk["ec50"], pk["n"])
            if drug in ["norepi", "phenyl", "vaso"]: dVeno += _hill_effect(ce[drug], 4.0, pk["ec50"], pk["n"])
        
        contractility = 1.0 + (0.0 if profile["id"] != 'sepsis' else -0.2) + dContractility
        Ees = 2.0 * contractility
        
        pmsf = 7 + (blood_vol - profile["weight"] * 0.07) / 0.25 + dVeno
        svr = profile["svr_b"] + dSVR
        if profile["id"] in ['sepsis', 'liver', 'anaphylaxis']: svr = max(400, svr)
        rvr = (svr / 80) * 0.05
        
        cvp, co, sv = 8.0, 5.0, 70.0
        for _ in range(5):
            sv_max = 120 * contractility
            sv = np.clip(sv_max * cvp / (4 + cvp), 20, 150)
            sed_effect = (ce["propofol"] * 0.5) + (ce["fentanyl"] * 0.2)
            final_hr = hr + dHR - sed_effect + noise_hr[t] * 2 + (80 - map_val) * 0.5
            hr = np.clip(final_hr, 30, 200)
            co = (sv * hr) / 1000
            cvp = (cvp + (pmsf - co * rvr)) / 2
        
        map_val = (co * svr / 80) + cvp + noise_map[t] * 2
        
        hb = np.clip(profile["hb"] + (params["prbc"]/250)*1.0 - (params["fluids"]/1000)*0.5, 4, 18)
        do2 = co * hb * PHYSICS["HB_CONVERSION"] * (0.98 if params["resp"]["fio2"] > 0.21 else 0.95) * 10
        vo2 = profile["vo2"] * (1 + (hr - 80)/100)
        o2er = np.clip(vo2 / do2, 0.1, 0.9)
        if vo2 > do2: o2_debt += (vo2 - do2) / 60
        
        lac_prod = 0.5 if do2 < vo2 * 1.5 else 0
        if profile["id"] == 'liver': lac_prod += 0.2
        lactate = np.clip(lactate + lac_prod - (0.1 * lactate) + np.random.normal(0, 0.025), 0.5, 25)
        
        sys = map_val + (2/3) * (sv / 1.5)
        dia = map_val - (1/3) * (sv / 1.5)
        cpo = map_val * co / 451
        ea = (0.9 * sys) / sv
        vac = ea / Ees if Ees > 0 else 0
        pvaCO2 = np.clip(2 + (6 / (co/bsa)), 2, 20)
        edv = 60 * (cvp / 5)
        
        ecg, art = _generate_waveforms(hr, sys, dia, np.clip((1 - blood_vol / (profile["weight"] * 0.07)) * 30, 0, 40), params["resp"]["rr"])

        current_step: SimulationStep = {
            "time": t, "hr": hr, "map": map_val, "sys": sys, "dia": dia, "ci": co / bsa,
            "svri": svr * bsa, "cvp": cvp, "temp": 37.0, "lactate": lactate, "spo2": np.clip(100 - (o2er * 25), 70, 100),
            "svo2": 100 * (1 - o2er), "pfRatio": 400, "hb": hb, "etco2": 35, "ph": 7.35, "paco2": 40,
            "creatinine": 0.8, "urineOutput": 1.0, "bilirubin": 0.5, "platelets": 250, "gcs": 15,
            "sofa": _calculate_sofa({}, params["drugs"], profile["weight"]), "sv": sv, "cpo": cpo, "ppv": 0, "shockIndex": hr / sys,
            "contractility": Ees, "preload": edv, "ea": ea, "vac": vac, "pvaCO2": pvaCO2, "strokeWork": 0,
            "potentialEnergy": 0, "efficiency": 0, "vis": 0, "pmsf": pmsf, "vrResistance": rvr, "icp": 10,
            "cpp": map_val - 10, "peakPressure": 30, "platPressure": 25, "drivingPressure": 10, "mechanicalPower": 12,
            "rsbi": 80, "do2": do2/bsa, "vo2": vo2/bsa, "o2er": o2er * 100, "cumulativeO2Debt": o2_debt,
            "pvLoop": [], "ecgWave": ecg, "artWave": art, "ventPressure": [], "ventFlow": []
        }
        data_history.append(current_step)

    return data_history
