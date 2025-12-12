"""
=============================================================================
TYPES MODULE for TITAN L8
=============================================================================
This file defines the data structures for the entire application using Python's
TypedDict, providing clarity, type safety, and editor autocompletion.
"""

from typing import TypedDict, List, Literal

# --- Simulation & Pharmacology ---

class DrugDosages(TypedDict):
    norepi: float
    vaso: float
    phenyl: float
    epi: float
    angiotensin: float
    dobu: float
    milrinone: float
    nicardipine: float
    esm: float
    propofol: float
    fentanyl: float
    cisatracurium: float

class RespiratorySettings(TypedDict):
    fio2: float
    peep: int
    rr: int
    tv: int
    iepo: bool

class SimulationParams(TypedDict):
    caseId: str
    drugs: DrugDosages
    fluids: int
    prbc: int
    diuresisVolume: int
    resp: RespiratorySettings
    isPaced: bool

# --- Physiology & Data Models ---

class PVPoint(TypedDict):
    volume: float
    pressure: float

class WaveformPoint(TypedDict):
    t: int
    y: float

class SOFAScore(TypedDict):
    respiratory: int
    coagulation: int
    liver: int
    cardiovascular: int
    cns: int
    renal: int
    total: int

class SimulationStep(TypedDict):
    time: int
    # Core Vitals
    hr: float
    map: float
    sys: float
    dia: float
    ci: float
    svri: float
    cvp: float
    temp: float
    # Metabolic
    lactate: float
    spo2: float
    svo2: float
    pfRatio: float
    hb: float
    etco2: float
    ph: float
    paco2: float
    # Organ Function
    creatinine: float
    urineOutput: float
    bilirubin: float
    platelets: int
    gcs: int
    sofa: SOFAScore
    # Advanced Hemodynamics
    sv: float
    cpo: float
    ppv: float
    shockIndex: float
    contractility: float
    preload: float
    ea: float
    vac: float
    pvaCO2: float
    strokeWork: float
    potentialEnergy: float
    efficiency: float
    vis: float
    # Guytonian
    pmsf: float
    vrResistance: float
    # Neuro
    icp: float
    cpp: float
    # Respiratory
    peakPressure: float
    platPressure: float
    drivingPressure: float
    mechanicalPower: float
    rsbi: float
    # Oxygen Flux
    do2: float
    vo2: float
    o2er: float
    cumulativeO2Debt: float
    # Visualization Data
    pvLoop: List[PVPoint]
    ecgWave: List[WaveformPoint]
    artWave: List[WaveformPoint]
    ventPressure: List[WaveformPoint]
    ventFlow: List[WaveformPoint]

# --- Static Definitions & Profiles ---

class PatientProfile(TypedDict):
    id: str
    label: str
    svr_b: int
    co_b: float
    map_b: int
    vo2: int
    hb: float
    weight: int
    height: int
    compliance: int

# --- AI & Advanced Analysis ---

AIStatus = Literal['idle', 'loading', 'success', 'error']

class AIAnalysis(TypedDict):
    status: AIStatus
    text: str
    timestamp: float

class Prediction(TypedDict):
    time: int
    value: float
    lower: float
    upper: float

class RiskFactor(TypedDict):
    id: str
    label: str
    severity: Literal['low', 'medium', 'high', 'critical']
    reasoning: str

class Prescription(TypedDict):
    id: str
    action: str
    rationale: str
    urgency: Literal['routine', 'urgent', 'stat']
    category: str

class AdvancedAnalysis(TypedDict):
    risks: List[RiskFactor]
    prescriptions: List[Prescription]
    forecastMAP: List[Prediction]
    forecastLactate: List[Prediction]
