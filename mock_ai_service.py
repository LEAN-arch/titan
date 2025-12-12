"""
=============================================================================
MOCK AI SERVICE for TITAN L8 (Ultimate, Complete Version)
=============================================================================
This self-reliant module uses a rule-based clinical heuristic engine to
generate a realistic 'AI Diagnosis' without any external API calls. The logic
is designed to be clinically nuanced and context-aware.
"""

from typing import List
from types_module import SimulationStep

# =============================================================================
# --- HEURISTIC SUB-ROUTINES ---
# =============================================================================

def _determine_phenotype(current_step: SimulationStep) -> str:
    """Determines the dominant shock phenotype using a scoring system."""
    scores = {
        "Vasoplegic": 0,
        "Cardiogenic": 0,
        "Hypovolemic": 0,
    }

    # Score vasoplegia based on low SVR and high CI
    if current_step['svri'] < 1400: scores["Vasoplegic"] += (1400 - current_step['svri']) / 200
    if current_step['ci'] > 3.5: scores["Vasoplegic"] += (current_step['ci'] - 3.5) * 2

    # Score cardiogenic shock based on low CI and high CVP
    if current_step['ci'] < 2.2: scores["Cardiogenic"] += (2.2 - current_step['ci']) * 4
    if current_step['cvp'] > 12: scores["Cardiogenic"] += (current_step['cvp'] - 12)

    # Score hypovolemia based on low CVP and high SVR (compensatory)
    if current_step['cvp'] < 4: scores["Hypovolemic"] += (4 - current_step['cvp']) * 2
    if current_step['svri'] > 2400: scores["Hypovolemic"] += (current_step['svri'] - 2400) / 400

    # If no scores are significant, the patient is stable
    if all(score < 2 for score in scores.values()):
        return "Hemodynamically Stable"

    # Determine the winner
    dominant_phenotype = max(scores, key=scores.get)
    
    # Check for mixed shock state
    sorted_scores = sorted(scores.values(), reverse=True)
    if sorted_scores[0] > 0 and sorted_scores[1] > (sorted_scores[0] * 0.5):
        return f"Mixed Shock ({dominant_phenotype}-dominant)"
        
    return f"{dominant_phenotype} Shock"

def _assess_perfusion(current_step: SimulationStep) -> str:
    """Assesses the adequacy of tissue perfusion from multiple angles."""
    if current_step['map'] < 60:
        return "Impaired (Macrocirculatory Failure)"
    if current_step['lactate'] > 4.0:
        return "Impaired (Severe Metabolic Dysoxia)"
    if current_step['pvaCO2'] > 8:
        return "Impaired (Microcirculatory Stagnation)"
    if current_step['svo2'] < 60:
        return "Impaired (High Extraction)"
    if current_step['lactate'] > 2.0:
        return "Stressed Perfusion"
    
    return "Compensated"

def _generate_directive(current_step: SimulationStep, phenotype: str) -> str:
    """Generates the single most critical, high-yield intervention."""
    
    # Priority 1: Treat life-threatening hypotension
    if current_step['map'] < 60:
        if "Vasoplegic" in phenotype: return "Titrate Vasopressors to MAP > 65"
        if "Cardiogenic" in phenotype: return "Initiate Inotropic Support"
        if "Hypovolemic" in phenotype: return "Administer Volume Challenge"
        return "Restore MAP to > 65 mmHg" # Default for mixed/unclear

    # Priority 2: Address clear physiological derangements
    if "Cardiogenic" in phenotype and current_step['ci'] < 2.2:
        return "Optimize Inotropy to improve CI"
    if current_step['ppv'] > 13 and "Hypovolemic" in phenotype:
        return "Consider Fluid Challenge for Preload"
    if current_step['vac'] > 1.3:
        return "Reduce Afterload to improve V-A Coupling"

    return "Optimize and Monitor Trend"

# =============================================================================
# --- MAIN SERVICE FUNCTION ---
# =============================================================================

def get_clinical_analysis(current_step: SimulationStep, history: List[SimulationStep]) -> str:
    """
    Analyzes the current patient state and returns a formatted diagnostic string,
    mimicking the output of a generative AI model.
    """
    
    phenotype = _determine_phenotype(current_step)
    perfusion = _assess_perfusion(current_step)
    directive = _generate_directive(current_step, phenotype)

    # Format the final string exactly as the UI expects
    return f"""
PHENOTYPE: {phenotype}
PERFUSION: {perfusion}
DIRECTIVE: {directive}
"""
