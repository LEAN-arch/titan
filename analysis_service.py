"""
=============================================================================
ANALYSIS SERVICE for TITAN L8 (Ultimate, Complete Version)
=============================================================================
This file contains the logic for advanced data analysis, including risk
assessment, prescription generation, and forecasting.
"""

import numpy as np
from typing import List

# --- Core Module Imports ---
from types_module import SimulationStep, AdvancedAnalysis, RiskFactor, Prescription, Prediction, SimulationParams

# =============================================================================
# --- 1. MAIN ANALYSIS ORCHESTRATOR ---
# =============================================================================

def analyze_data(history: List[SimulationStep], params: SimulationParams) -> AdvancedAnalysis:
    """
    The main entry point for the analysis service. Orchestrates all sub-modules.
    """
    if not history:
        return {"risks": [], "prescriptions": [], "forecastMAP": [], "forecastLactate": []}

    current = history[-1]
    
    return {
        "risks": _assess_risks(current, params),
        "prescriptions": _generate_prescriptions(current, params),
        "forecastMAP": _generate_forecast(history, 'map', 5.0), 
        "forecastLactate": _generate_forecast(history, 'lactate', 0.2),
    }

# =============================================================================
# --- 2. RISK ASSESSMENT SUB-SYSTEM ---
# =============================================================================

def _assess_risks(current: SimulationStep, params: SimulationParams) -> List[RiskFactor]:
    """Assesses all clinical risks based on the current physiological state."""
    risks: List[RiskFactor] = []
    
    # Hemodynamic Risks
    if current['map'] < 65:
        risks.append({
            "id": 'hypotension', "label": 'Critical Hypoperfusion',
            "severity": 'critical' if current['map'] < 55 else 'high',
            "reasoning": f"MAP of {current['map']:.0f} risks end-organ ischemia (AKI, MINS)."
        })
    if current['cpo'] < 0.6:
        risks.append({
            "id": 'low_power', "label": 'Cardiogenic Collapse Risk',
            "severity": 'critical',
            "reasoning": f"Cardiac Power Output {current['cpo']:.2f}W is below the 0.6W survival threshold."
        })

    # Respiratory Risks
    if current['drivingPressure'] > 15:
        risks.append({
            "id": 'vili_dp', "label": 'Lung Injury (High Strain)',
            "severity": 'high',
            "reasoning": f"Driving Pressure of {current['drivingPressure']:.0f} > 15 cmH2O is a primary driver of VILI."
        })
    if current['mechanicalPower'] > 17:
        risks.append({
            "id": 'vili_power', "label": 'Lung Injury (High Power)',
            "severity": 'critical',
            "reasoning": f"Mechanical Power of {current['mechanicalPower']:.1f} > 17 J/min causes biotrauma."
        })

    # Neurological Risks (only for relevant cases)
    if params['caseId'] == 'neuro' and current['cpp'] < 60:
        risks.append({
            "id": 'ischemia_neuro', "label": 'Cerebral Ischemia',
            "severity": 'critical',
            "reasoning": f"CPP of {current['cpp']:.0f} < 60 mmHg compromises brain perfusion."
        })

    # Sort risks by severity for prioritized display
    severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    risks.sort(key=lambda r: severity_order[r['severity']], reverse=True)
    
    return risks

# =============================================================================
# --- 3. PRESCRIPTIVE ACTION SUB-SYSTEM ---
# =============================================================================

def _generate_prescriptions(current: SimulationStep, params: SimulationParams) -> List[Prescription]:
    """Generates actionable therapeutic suggestions based on the current state."""
    rx: List[Prescription] = []

    # --- Priority 1: Manage Critical Hypotension ---
    if current['map'] < 65:
        if current['svri'] < 1600:
            rx.append({"id": 'start_ne', "action": 'Titrate Vasopressors', "rationale": f"MAP is low with vasodilation (SVRI {current['svri']:.0f}).", "urgency": 'stat', "category": 'pressor'})
        elif current['ci'] < 2.2:
            rx.append({"id": 'start_inotrope', "action": 'Start Inotrope', "rationale": f"MAP is low with pump failure (CI {current['ci']:.1f}).", "urgency": 'stat', "category": 'inotropes'})
        elif current['ppv'] > 13 and current['cvp'] < 14:
             rx.append({"id": 'fluid_bolus', "action": 'Fluid Challenge', "rationale": f"High PPV ({current['ppv']:.0f}%) suggests preload responsiveness.", "urgency": 'stat', "category": 'fluid'})
        return rx # Prioritize shock management over other suggestions

    # --- Priority 2: Lung Protective Ventilation ---
    if current['drivingPressure'] > 15:
        rx.append({"id": 'drop_vt', "action": 'Reduce Tidal Volume', "rationale": f"Driving Pressure > 15 cmH2O. Lower Vt to target 4-6 mL/kg to mitigate VILI.", "urgency": 'urgent', "category": 'vent'})
    
    # --- Priority 3: Weaning Protocols ---
    if current['map'] > 80 and params['drugs']['norepi'] > 0.05:
        rx.append({"id": 'wean_ne', "action": 'Wean Norepinephrine', "rationale": f"MAP is above target. Reduce catecholamine exposure.", "urgency": 'routine', "category": 'weaning'})

    return rx

# =============================================================================
# --- 4. FORECASTING SUB-SYSTEM ---
# =============================================================================

def _generate_forecast(history: List[SimulationStep], key: str, volatility: float) -> List[Prediction]:
    """
    PERFORMANCE: Uses NumPy's polyfit for a fast and numerically stable linear regression.
    """
    lookback = min(len(history), 30)
    if lookback < 5: return []

    recent_history = history[-lookback:]
    y_values = np.array([step[key] for step in recent_history])
    x_values = np.arange(lookback)

    # Use NumPy's polynomial fit for linear regression (degree 1)
    slope, intercept = np.polyfit(x_values, y_values, 1)

    horizon = 60
    future_x = np.arange(lookback, lookback + horizon)
    projected_values = intercept + slope * future_x

    predictions: List[Prediction] = []
    last_time = recent_history[-1]['time']

    for i in range(horizon):
        uncertainty = volatility * np.sqrt(i + 1) # Uncertainty grows over time
        predictions.append({
            "time": last_time + i + 1,
            "value": projected_values[i],
            "lower": projected_values[i] - uncertainty,
            "upper": projected_values[i] + uncertainty,
        })
        
    return predictions```

I have delivered the fully optimized `analysis_service.py` file. Please ask for the next file when you are ready.
