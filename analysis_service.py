import numpy as np
from types_module import SimulationStep, AdvancedAnalysis, SimulationParams

def analyze_data(history: list, params: SimulationParams) -> AdvancedAnalysis:
    if not history: return {"risks": [], "prescriptions": [], "forecastMAP": [], "forecastLactate": []}
    
    current = history[-1]
    risks = []
    prescriptions = []
    
    # Risks
    if current['map'] < 65:
        risks.append({"id": "hypo", "label": "Hypotension", "severity": "high", "reasoning": "MAP < 65mmHg"})
    if current['lactate'] > 2:
        risks.append({"id": "lac", "label": "Lactic Acidosis", "severity": "medium", "reasoning": "Anaerobic metabolism detected"})

    # Rx
    if current['map'] < 60 and params['drugs']['norepi'] < 0.1:
        prescriptions.append({"id": "p1", "action": "Titrate Levophed", "rationale": "MAP < 60", "urgency": "stat", "category": "vaso"})
        
    return {
        "risks": risks,
        "prescriptions": prescriptions,
        "forecastMAP": [], # Keep empty for performance
        "forecastLactate": []
    }
