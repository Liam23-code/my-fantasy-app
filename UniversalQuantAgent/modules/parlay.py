"""Simple, explainable parlay evaluation. This is analysis, not betting advice."""
from __future__ import annotations
from typing import Any
import re
from modules.data_quality import safe_number

def build_parlay(props_list: list[dict]) -> dict[str, Any]:
    if not props_list: raise ValueError("Select at least one prop for the parlay.")
    warnings, injury_warnings = [], []
    for index, first in enumerate(props_list):
        for second in props_list[index+1:]:
            if first.get("player") == second.get("player"):
                warnings.append(f"Multiple legs use {first['player']}; their outcomes may be correlated.")
            if first.get("team") and first.get("team") == second.get("team"):
                warnings.append(f"Same-team legs for {first['team']} share pace and blowout risk.")
        for driver in first.get("key_drivers", []):
            if any(word in driver.lower() for word in ("injury", "questionable", "out", "availability fallback")):
                injury_warnings.append(f"{first.get('player')}: {driver}")
    combined_edge = sum(abs(safe_number(prop.get("edge"))) for prop in props_list)
    reliability_values = []
    for prop in props_list:
        if "reliability_score" in prop:
            reliability_values.append(safe_number(prop["reliability_score"],50))
            continue
        driver_text = " ".join(str(item) for item in prop.get("key_drivers",[]))
        match = re.search(r"Quant reliability:\s*([\d.]+)", driver_text, re.I)
        reliability_values.append(safe_number(match.group(1),50) if match else 50.0)
    average_reliability = sum(reliability_values)/len(reliability_values)
    average_width = sum(max(0, float(p.get("confidence_high",0))-float(p.get("confidence_low",0))) for p in props_list) / len(props_list)
    correlation_penalty = min(35.0, len(set(warnings))*8.0)
    injury_penalty = min(30.0, len(set(injury_warnings))*10.0)
    size_penalty = max(0, len(props_list)-2)*7.0
    reliability_penalty = max(0.0,(100.0-average_reliability)*.20)
    risk = min(100.0, 25.0 + average_width*2.0 + correlation_penalty + injury_penalty + size_penalty + reliability_penalty)
    same_team_pairs = sum(1 for i,a in enumerate(props_list) for b in props_list[i+1:] if a.get("team") and a.get("team")==b.get("team"))
    pace_correlation = min(100.0, same_team_pairs*25.0)
    blowout_mentions = sum(any("blowout" in str(d).lower() for d in p.get("key_drivers",[])) for p in props_list)
    blowout_risk = min(100.0, 25.0 + blowout_mentions*20.0)
    # Even-money probability proxy: edge improves 50% baseline, risk/correlation reduce it.
    win_probability = max(.02, min(.90, .5 + combined_edge*.015 - risk*.003)) ** len(props_list)
    decimal_payout = 1.91 ** len(props_list)
    expected_value = (win_probability * (decimal_payout-1) - (1-win_probability)) * 100
    summary = (f"{len(props_list)} legs combine for {combined_edge:.1f} units of absolute edge. "
               f"Average projection reliability is {average_reliability:.0f}/100 and heuristic risk is {risk:.0f}/100; "
               "review correlated legs and live availability before use.")
    return {"props":props_list, "combined_edge":round(combined_edge,2), "risk_score":round(risk,1),
            "correlation_warnings":list(dict.fromkeys(warnings)),
            "injury_warnings":list(dict.fromkeys(injury_warnings)),
            "pace_correlation":round(pace_correlation,1), "blowout_risk":round(blowout_risk,1),
            "expected_value":round(expected_value,2), "summary":summary}
