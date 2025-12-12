"""
=============================================================================
CONSTANTS MODULE for TITAN L8 (Ultimate, Complete Version)
=============================================================================
This file serves as the single source of truth for all static data used in the
application, such as patient profiles, drug parameters, and UI definitions.
"""
from types_module import PatientProfile, DrugDosages

PATIENT_PROFILES: list[PatientProfile] = [
  { "id": "sepsis", "label": "24F Septic Shock (Vasoplegic)", "svr_b": 500, "co_b": 7.5, "map_b": 45, "vo2": 180, "hb": 11, "weight": 60, "height": 165, "compliance": 30 },
  { "id": "cabg", "label": "65M Post-CABG (Cardiogenic)", "svr_b": 2200, "co_b": 2.5, "map_b": 70, "vo2": 140, "hb": 9.5, "weight": 85, "height": 175, "compliance": 45 },
  { "id": "hfpef", "label": "82M HFpEF (Diastolic Failure)", "svr_b": 1800, "co_b": 3.5, "map_b": 85, "vo2": 110, "hb": 9, "weight": 70, "height": 170, "compliance": 35 },
  { "id": "trauma", "label": "50M Polytrauma (Hemorrhagic)", "svr_b": 2800, "co_b": 2.8, "map_b": 55, "vo2": 160, "hb": 6.5, "weight": 90, "height": 180, "compliance": 50 },
  { "id": "ards", "label": "45M Severe ARDS (Hypoxic)", "svr_b": 1000, "co_b": 5.5, "map_b": 75, "vo2": 160, "hb": 12, "weight": 80, "height": 178, "compliance": 18 },
  { "id": "asthma", "label": "22F Status Asthmaticus", "svr_b": 1200, "co_b": 4.5, "map_b": 80, "vo2": 170, "hb": 13, "weight": 55, "height": 162, "compliance": 55 },
  { "id": "rvi", "label": "55M RV Infarct (Preload Dependent)", "svr_b": 1500, "co_b": 2.5, "map_b": 60, "vo2": 130, "hb": 14, "weight": 82, "height": 176, "compliance": 50 },
  { "id": "liver", "label": "70F Cirrhotic (Hepatorenal)", "svr_b": 400, "co_b": 8.5, "map_b": 50, "vo2": 140, "hb": 7.0, "weight": 65, "height": 160, "compliance": 40 },
  { "id": "neuro", "label": "35M Neurostorm (TBI)", "svr_b": 3200, "co_b": 3.0, "map_b": 130, "vo2": 200, "hb": 15, "weight": 75, "height": 180, "compliance": 50 },
  { "id": "anaphylaxis", "label": "60M Anaphylaxis", "svr_b": 300, "co_b": 9.0, "map_b": 40, "vo2": 150, "hb": 14, "weight": 88, "height": 182, "compliance": 45 },
  { "id": "normal", "label": "Healthy Control (Calibration)", "svr_b": 1000, "co_b": 5.0, "map_b": 70, "vo2": 120, "hb": 14, "weight": 70, "height": 175, "compliance": 60 },
]

DRUG_PK: dict[str, dict] = {
  "norepi": { "ke0": 0.25, "ec50": 0.25, "n": 1.5, "emax": { "svr": 3000.0, "hr": 10.0, "contractility": 0.3 } },
  "vaso": { "ke0": 0.1, "ec50": 0.03, "n": 3.0, "emax": { "svr": 2000.0, "co": -0.5 } },
  "phenyl": { "ke0": 0.4, "ec50": 1.0, "n": 1.5, "emax": { "svr": 2500.0, "hr": -15.0, "co": -1.0 } },
  "epi": { "ke0": 0.3, "ec50": 0.1, "n": 1.5, "emax": { "svr": 1500.0, "hr": 50.0, "contractility": 2.0, "co": 4.0 } },
  "angiotensin": { "ke0": 0.2, "ec50": 10.0, "n": 2.0, "emax": { "svr": 4000.0 } },
  "dobu": { "ke0": 0.15, "ec50": 5.0, "n": 1.2, "emax": { "svr": -400.0, "co": 3.5, "hr": 35.0, "contractility": 1.5 } },
  "milrinone": { "ke0": 0.05, "ec50": 0.4, "n": 1.0, "emax": { "svr": -800.0, "co": 2.0, "contractility": 1.0 } },
  "nicardipine": { "ke0": 0.15, "ec50": 5.0, "n": 1.5, "emax": { "svr": -2000.0 } },
  "esm": { "ke0": 0.8, "ec50": 50.0, "n": 1.5, "emax": { "hr": -60.0, "contractility": -0.8 } },
  "propofol": { "ke0": 0.5, "ec50": 30.0, "n": 1.8, "emax": { "svr": -1200.0, "contractility": -0.6 } },
  "fentanyl": { "ke0": 0.2, "ec50": 100.0, "n": 1.2, "emax": { "hr": -15.0, "svr": -200.0 } },
  "cisatracurium": { "ke0": 0.1, "ec50": 2.0, "n": 5.0, "emax": {} },
}

DRUG_INFO = {
    "norepi": {"label": "Norepinephrine", "max_dose": 2.0, "step": 0.01, "unit": "mcg/kg/min"},
    "vaso": {"label": "Vasopressin", "max_dose": 0.1, "step": 0.01, "unit": "u/min"},
    "phenyl": {"label": "Phenylephrine", "max_dose": 5.0, "step": 0.1, "unit": "mcg/kg/min"},
    "epi": {"label": "Epinephrine", "max_dose": 1.0, "step": 0.01, "unit": "mcg/kg/min"},
    "angiotensin": {"label": "Angiotensin II", "max_dose": 80.0, "step": 5.0, "unit": "ng/kg/min"},
    "dobu": {"label": "Dobutamine", "max_dose": 20.0, "step": 0.5, "unit": "mcg/kg/min"},
    "milrinone": {"label": "Milrinone", "max_dose": 1.0, "step": 0.05, "unit": "mcg/kg/min"},
    "nicardipine": {"label": "Nicardipine", "max_dose": 15.0, "step": 1.0, "unit": "mg/hr"},
    "esm": {"label": "Esmolol", "max_dose": 300.0, "step": 10.0, "unit": "mcg/kg/min"},
    "propofol": {"label": "Propofol", "max_dose": 80.0, "step": 5.0, "unit": "mcg/kg/min"},
    "fentanyl": {"label": "Fentanyl", "max_dose": 300.0, "step": 10.0, "unit": "mcg/hr"},
    "cisatracurium": {"label": "Cisatracurium", "max_dose": 10.0, "step": 0.5, "unit": "mcg/kg/min"},
}

COLORS = {
  "hemo": "#06b6d4", "cardiac": "#3b82f6", "respiratory": "#8b5cf6",
  "metabolic": "#ef4444", "alert": "#f59e0b", "crit": "#dc2626",
  "ok": "#10b981", "purple": "#a855f7", "neuro": "#d946ef",
  "ecg": "#10b981", "art": "#ef4444",
  "textMain": "#0f172a", "textMuted": "#64748b", "grid": "#e2e8f0",
}

PHYSICS = {
  "ATM_PRESSURE": 760,
  "H2O_PRESSURE": 47,
  "R_QUOTIENT": 0.8,
  "HB_CONVERSION": 1.34,
}

CLINICAL_DEFINITIONS = {
  "MAP": { "label": "Mean Arterial Pressure", "description": "Average arterial pressure; the primary driver of organ perfusion. A MAP < 65 mmHg is a common threshold for defining shock.", "normalRange": "65-110 mmHg", "target": "> 65 mmHg" },
  "CI": { "label": "Cardiac Index", "description": "Cardiac Output indexed to Body Surface Area. The gold standard for assessing blood flow.", "normalRange": "2.5-4.0 L/min/m²", "target": "> 2.2" },
  "SVRI": { "label": "SVR Index", "description": "Systemic Vascular Resistance Index. The 'afterload' or resistance the heart pumps against.", "normalRange": "1600-2400", "target": "1600-2400" },
  "CVP": { "label": "Central Venous Pressure", "description": "Pressure in the right atrium. Indicates right-sided preload and tolerance to volume.", "normalRange": "2-6 mmHg", "target": "Avoid extremes" },
  "SvO2": { "label": "Mixed Venous Saturation", "description": "Oxygen saturation of blood returning to the heart. A key indicator of the global O2 supply vs. demand balance.", "normalRange": "65-75%", "target": "> 65%" },
  "CPO": { "label": "Cardiac Power Output", "description": "MAP x CO / 451. The 'horsepower' of the heart. The strongest predictor of mortality in cardiogenic shock.", "normalRange": "0.8-1.2 W", "target": "> 0.6 W" },
  "Lactate": { "label": "Serum Lactate", "description": "A marker of anaerobic metabolism, indicating cellular hypoperfusion or mitochondrial dysfunction.", "normalRange": "0.5-1.5 mmol/L", "target": "< 2.0" },
  "Vac": { "label": "V-A Coupling", "description": "Ventriculo-Arterial Coupling (Ea/Ees). Measures cardiac efficiency. A value > 1.3 suggests afterload mismatch.", "normalRange": "0.8-1.2", "target": "~ 1.0" },
  "pvaCO2": { "label": "Pv-aCO2 Gap", "description": "Venous-to-Arterial CO2 difference. A gap > 6 mmHg suggests microcirculatory stagnation.", "normalRange": "2-5 mmHg", "target": "< 6 mmHg" },
  "ICP": { "label": "Intracranial Pressure", "description": "Pressure within the skull. A value > 22 mmHg is a neurosurgical emergency and risks herniation.", "normalRange": "5-15 mmHg", "target": "< 20 mmHg" },
  "CPP": { "label": "Cerebral Perfusion Pressure", "description": "MAP - ICP. The net pressure gradient driving blood flow to the brain.", "normalRange": "70-90 mmHg", "target": "> 60 mmHg" },
  "DrivingPressure": { "label": "Driving Pressure", "description": "Plateau Pressure - PEEP. The strain applied to the lung. A value > 14 cmH2O is a major risk factor for VILI.", "normalRange": "< 14 cmH2O", "target": "< 14" },
  "MechanicalPower": { "label": "Mechanical Power", "description": "Total energy delivered to the lungs per minute. Power > 17 J/min is strongly associated with VILI and mortality.", "normalRange": "< 12 J/min", "target": "< 12" },
}
