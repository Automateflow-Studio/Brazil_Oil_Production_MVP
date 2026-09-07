import pandas as pd
import numpy as np


def classify_petroleum_mechanisms(df: pd.DataFrame, window: int = 6) -> dict:
    """
    Analyze petroleum production behavior using the data supplied in df.

    IMPORTANT:
    - This function does NOT create or replace the input data.
    - If the upstream pipeline already calculated an expected oil rate
      (column: expected_production or expected_oil), it is exposed in the API.
    - No fake expected-production value is generated here.
    """
    required_columns = {"oil", "water", "gas"}
    missing = required_columns - set(df.columns)

    if missing:
        return {
            "status": "error",
            "primary_diagnosis": "INVALID_DATA",
            "confidence": 0.0,
            "details": {
                "message": "Missing required production columns.",
                "missing_columns": sorted(missing),
            },
        }

    if len(df) < 3:
        return {
            "status": "success",
            "primary_diagnosis": "INSUFFICIENT_DATA",
            "confidence": 0.0,
            "details": {},
        }

    recent = df.tail(window).copy().reset_index(drop=True)

    # Calculate key petroleum engineering metrics
    recent["water_cut"] = recent["water"] / (
        recent["oil"] + recent["water"] + 1e-9
    )
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
            "evidence": (
                f"Simultaneous sharp decline in Oil ({oil_chg*100:.1f}%), "
                f"Water ({water_chg*100:.1f}%), and Gas ({gas_chg*100:.1f}%). "
                "Reservoir properties are stable, indicating surface/mechanical "
                "restriction or well shut-in."
            ),
        })

    # 2. Water Breakthrough / Channeling
    elif wc_slope > 0.01 or (oil_chg < -0.10 and wc_chg > 0.05):
        confidence = 0.85 if wc_chg > 0.1 else 0.70
        diagnoses.append({
            "code": "WATER_BREAKTHROUGH",
            "factor": "Water Breakthrough / Encroachment from Aquifer or Injection",
            "category": "Reservoir Dynamics",
            "confidence": confidence,
            "evidence": (
                f"Water cut increased to {latest['water_cut']*100:.1f}% "
                f"(+{wc_chg*100:.1f}% MoM). Oil rate is being displaced by "
                "water production, characteristic of water sweep front arrival "
                "or fingering."
            ),
        })

    # 3. Gas Coning / Reservoir Depletion below Bubble Point
    elif gor_slope > 0.05 or (oil_chg < -0.10 and gor_chg > 0.20):
        diagnoses.append({
            "code": "GAS_CONING_OR_PRESSURE_DEPLETION",
            "factor": "Gas Coning or Reservoir Pressure Dropping Below Bubble Point",
            "category": "Reservoir Dynamics",
            "confidence": 0.80,
            "evidence": (
                f"Gas-Oil Ratio (GOR) expanded significantly to "
                f"{latest['gor']:.2f} (MoM change: +{gor_chg*100:.1f}%). "
                "Indicates free gas cap expansion, gas coning due to high "
                "drawdown, or solution gas liberation."
            ),
        })

    # 4. Mechanical Formation Damage / Scaling / Wellbore Restriction
    elif oil_slope < 0 and abs(wc_slope) < 0.005 and abs(gor_slope) < 0.01:
        diagnoses.append({
            "code": "FORMATION_DAMAGE_OR_RESTRICTION",
            "factor": (
                "Inflow Performance Decline "
                "(Formation Damage, Scale, or Tubing Restricting)"
            ),
            "category": "Wellbore Integrity",
            "confidence": 0.75,
            "evidence": (
                "Steady decline in oil production while Water Cut and GOR "
                "remain constant. Suggests increasing skin factor, inorganic "
                "scale deposition, or tubing restriction."
            ),
        })

    # 5. Successful Workover / Stimulation / Artificial Lift Optimization
    elif oil_chg > 0.20:
        if wc_chg < 0:
            evidence_str = (
                f"Oil rate jumped by {oil_chg*100:.1f}% with dropping "
                f"Water Cut ({wc_chg*100:.1f}%). Suggests successful acidizing, "
                "hydraulic fracturing, or recompletion in a cleaner zone."
            )
        else:
            evidence_str = (
                f"Oil rate surged by {oil_chg*100:.1f}%. Consistent with "
                "artificial lift optimization (e.g., frequency increase on ESP) "
                "or choke enlargement."
            )

        diagnoses.append({
            "code": "WELL_STIMULATION_OR_OPTIMIZATION",
            "factor": "Well Intervention, Stimulation, or Artificial Lift Upgrade",
            "category": "Production Optimization",
            "confidence": 0.85,
            "evidence": evidence_str,
        })

    # Fallback for Normal or Steady Decline
    if not diagnoses:
        if oil_slope < 0:
            diagnoses.append({
                "code": "NATURAL_RESERVOIR_DECLINE",
                "factor": "Normal Boundary-Dominated Reservoir Pressure Decline",
                "category": "Reservoir Dynamics",
                "confidence": 0.80,
                "evidence": (
                    "Decline follows expected natural depletion dynamics "
                    "without anomalous water or gas influx."
                ),
            })
        else:
            diagnoses.append({
                "code": "STEADY_PRODUCTION",
                "factor": "Stable Production Baseline",
                "category": "Baseline",
                "confidence": 0.90,
                "evidence": (
                    "Production parameters remain within normal statistical "
                    "fluctuation margins."
                ),
            })

    # ---------------------------------------------------------------
    # Structured numeric metrics for the API/dashboard.
    # These values come from the same dataframe used by the diagnosis.
    # ---------------------------------------------------------------
    current_production = float(latest["oil"])
    previous_production = float(prev["oil"])

    # Expected production must come from an upstream model (e.g. Arps).
    # We expose it if it already exists in the dataframe; otherwise None.
    expected_production = None
    if "expected_production" in latest.index and pd.notna(latest["expected_production"]):
        expected_production = float(latest["expected_production"])
    elif "expected_oil" in latest.index and pd.notna(latest["expected_oil"]):
        expected_production = float(latest["expected_oil"])

    deviation_percent = None
    if expected_production is not None and abs(expected_production) > 1e-9:
        deviation_percent = (
            (current_production - expected_production)
            / expected_production
        ) * 100.0

    metrics = {
        "current_production": round(current_production, 2),
        "previous_production": round(previous_production, 2),
        "expected_production": (
            round(expected_production, 2)
            if expected_production is not None
            else None
        ),
        "deviation_percent": (
            round(float(deviation_percent), 2)
            if deviation_percent is not None
            else None
        ),
        "oil_change_percent": round(float(oil_chg * 100), 2),
        "water_change_percent": round(float(water_chg * 100), 2),
        "gas_change_percent": round(float(gas_chg * 100), 2),
        "water_cut_percent": round(float(latest["water_cut"] * 100), 2),
        "previous_water_cut_percent": round(float(prev["water_cut"] * 100), 2),
        "gor": round(float(latest["gor"]), 2),
        "previous_gor": round(float(prev["gor"]), 2),
        "water_cut_change_percent_points": round(float(wc_chg * 100), 2),
        "gor_change_percent": round(float(gor_chg * 100), 2),
    }

    return {
        "status": "success",
        "primary_diagnosis": diagnoses[0]["code"],
        "confidence": diagnoses[0]["confidence"],
        "metrics": metrics,
        "contributing_factors": diagnoses,
    }


def analyze_well(
    well_id: str,
    horizon_months: int = 3,
    as_of_date: str | None = None,
) -> dict:
    # NOTE:
    # This block is still the existing demo/static source from the uploaded
    # project. It is intentionally NOT used to claim live/real well data.
    #
    # Replace ONLY this data-loading section with your actual DB/API/CSV
    # loader. The analysis and API response structure above can remain intact.
    dummy_df = pd.DataFrame([
        {"oil": 120, "water": 10, "gas": 500},
        {"oil": 115, "water": 12, "gas": 510},
        {"oil": 110, "water": 15, "gas": 520},
        {"oil": 100, "water": 25, "gas": 540},
        {"oil": 80, "water": 50, "gas": 560},
        {"oil": 60, "water": 85, "gas": 590},
    ])

    res = classify_petroleum_mechanisms(dummy_df)
    res["well_id"] = well_id
    res["as_of_date"] = as_of_date
    return res
