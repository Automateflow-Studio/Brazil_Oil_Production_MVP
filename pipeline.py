import pandas as pd
import numpy as np

def classify_petroleum_mechanisms(df: pd.DataFrame, window: int = 6) -> dict:
    """
    Expert System Architecture mimicking a Senior Petroleum/Reservoir Engineer (40+ yrs exp).
    Analyzes physical reservoir mechanisms based on co-trends of Oil, Water, Gas, GOR, and Water Cut.
    """
    if len(df) < 3:
        return {"primary_diagnosis": "INSUFFICIENT_DATA", "confidence": 0.0, "details": {}}

    recent = df.tail(window).copy().reset_index(drop=True)
    
    # Calculate key petroleum engineering metrics
    recent["water_cut"] = recent["water"] / (recent["oil"] + recent["water"] + 1e-9)
    recent["gor"] = recent["gas"] / (recent["oil"] + 1e-9)

    latest = recent.iloc[-1]
    prev = recent.iloc[-2]

    # Percentage changes (MoM)
    oil_chg = (latest["oil"] - prev["oil"]) / (prev["oil"] + 1e-9)
    water_chg = (latest["water"] - prev["water"]) / (prev["water"] + 1e-9)
    gas_chg = (latest["gas"] - prev["gas"]) / (prev["gas"] + 1e-9)
    wc_chg = latest["water_cut"] - prev["water_cut"]
    gor_chg = (latest["gor"] - prev["gor"]) / (prev["gor"] + 1e-9)

    # Multi-month trends (Slope Analysis)
    t = np.arange(len(recent))
    oil_slope = np.polyfit(t, recent["oil"], 1)[0]
    wc_slope = np.polyfit(t, recent["water_cut"], 1)[0]
    gor_slope = np.polyfit(t, recent["gor"], 1)[0]

    diagnoses = []

    # 1. Operational Shut-in / Mechanical Choke Change
    if oil_chg < -0.25 and water_chg < -0.25 and gas_chg < -0.25:
        diagnoses.append({
            "code": "OPERATIONAL_SHUTIN_OR_CHOKE_RESTRICTION",
            "factor": "Operational Interruption or Surface Choke Reduction",
            "category": "Operational",
            "confidence": 0.90,
            "evidence": f"Simultaneous sharp decline in Oil ({oil_chg*100:.1f}%), Water ({water_chg*100:.1f}%), and Gas ({gas_chg*100:.1f}%). Reservoir properties are stable, indicating surface/mechanical restriction or well shut-in."
        })

    # 2. Water Breakthrough / Channeling
    elif wc_slope > 0.01 or (oil_chg < -0.10 and wc_chg > 0.05):
        confidence = 0.85 if wc_chg > 0.1 else 0.70
        diagnoses.append({
            "code": "WATER_BREAKTHROUGH",
            "factor": "Water Breakthrough / Encroachment from Aquifer or Injection",
            "category": "Reservoir Dynamics",
            "confidence": confidence,
            "evidence": f"Water cut increased to {latest['water_cut']*100:.1f}% (+{wc_chg*100:.1f}% MoM). Oil rate is being displaced by water production, characteristic of water sweep front arrival or fingering."
        })

    # 3. Gas Coning / Reservoir Depletion below Bubble Point
    elif gor_slope > 0.05 or (oil_chg < -0.10 and gor_chg > 0.20):
        diagnoses.append({
            "code": "GAS_CONING_OR_PRESSURE_DEPLETION",
            "factor": "Gas Coning or Reservoir Pressure Dropping Below Bubble Point",
            "category": "Reservoir Dynamics",
            "confidence": 0.80,
            "evidence": f"Gas-Oil Ratio (GOR) expanded significantly to {latest['gor']:.2f} (MoM change: +{gor_chg*100:.1f}%). Indicates free gas cap expansion, gas coning due to high drawdown, or solution gas liberation."
        })

    # 4. Mechanical Formation Damage / Scaling / Wellbore Restricting
    elif oil_slope < 0 and abs(wc_slope) < 0.005 and abs(gor_slope) < 0.01:
        diagnoses.append({
            "code": "FORMATION_DAMAGE_OR_RESTRICTION",
            "factor": "Inflow Performance Decline (Formation Damage, Scale, or Tubing Restricting)",
            "category": "Wellbore Integrity",
            "confidence": 0.75,
            "evidence": "Steady decline in oil production while Water Cut and GOR remain constant. Suggests increasing skin factor, inorganic scale deposition, or tubing restriction."
        })

    # 5. Successful Workover / Stimulation / Artificial Lift Optimization
    elif oil_chg > 0.20:
        if wc_chg < 0:
            evidence_str = f"Oil rate jumped by {oil_chg*100:.1f}% with dropping Water Cut ({wc_chg*100:.1f}%). Suggests successful acidizing, hydraulic fracturing, or recompletion in a cleaner zone."
        else:
            evidence_str = f"Oil rate surged by {oil_chg*100:.1f}%. Consistent with artificial lift optimization (e.g., frequency increase on ESP) or choke enlargement."
            
        diagnoses.append({
            "code": "WELL_STIMULATION_OR_OPTIMIZATION",
            "factor": "Well Intervention, Stimulation, or Artificial Lift Upgrade",
            "category": "Production Optimization",
            "confidence": 0.85,
            "evidence": evidence_str
        })

    # Fallback for Normal or Steady Decline
    if not diagnoses:
        if oil_slope < 0:
            diagnoses.append({
                "code": "NATURAL_RESERVOIR_DECLINE",
                "factor": "Normal Boundary-Dominated Reservoir Pressure Decline",
                "category": "Reservoir Dynamics",
                "confidence": 0.80,
                "evidence": "Decline follows expected natural depletion dynamics without anomalous water or gas influx."
            })
        else:
            diagnoses.append({
                "code": "STEADY_PRODUCTION",
                "factor": "Stable Production Baseline",
                "category": "Baseline",
                "confidence": 0.90,
                "evidence": "Production parameters remain within normal statistical fluctuation margins."
            })

    return {
        "status": "success",
        "primary_diagnosis": diagnoses[0]["code"],
        "contributing_factors": diagnoses
    }

def analyze_well(well_id: str, horizon_months: int = 3, as_of_date: str | None = None) -> dict:
    dummy_df = pd.DataFrame([
        {"oil": 120, "water": 10, "gas": 500},
        {"oil": 115, "water": 12, "gas": 510},
        {"oil": 110, "water": 15, "gas": 520},
        {"oil": 100, "water": 25, "gas": 540},
        {"oil": 80, "water": 50, "gas": 560},
        {"oil": 60, "water": 85, "gas": 590}
    ])
    res = classify_petroleum_mechanisms(dummy_df)
    res["well_id"] = well_id
    res["as_of_date"] = as_of_date
    return res