"""Premium, lightweight Plotly visuals for the NFL Graph Lab."""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.data_quality import safe_number
from modules.nfl_analysis import analyze_nfl_player
from modules.nfl_stats import normalize_player_position, team_palette


TIER_SCALE=[[0,"#d73a32"],[.35,"#f47c20"],[.55,"#f0c419"],[.75,"#7fd13b"],[1,"#149253"]]


def _base(figure: go.Figure,title: str,meta: dict[str,Any],height: int=520) -> go.Figure:
    primary,secondary=team_palette(meta.get("team",""))
    figure.update_layout(title=title,height=height,margin=dict(l=34,r=34,t=72,b=35),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.42)",font=dict(family="Inter, Segoe UI, sans-serif",color="#20304a"),hoverlabel=dict(bgcolor="white"),meta=meta,legend=dict(orientation="h",y=1.06),colorway=[primary,secondary,"#7fd13b"])
    figure.add_shape(type="rect",xref="paper",yref="paper",x0=0,y0=0,x1=1,y1=1,line=dict(color=primary,width=3),fillcolor="rgba(0,0,0,0)",layer="above")
    return figure


def _analysis(player: str, opponent: str|None, season: int|None, mode: str, comparison_mode: str) -> dict[str,Any]:
    return analyze_nfl_player(player,opponent,season,mode,comparison_mode)


def render_qb_passing_map(player: str, opponent: str|None=None, season: int|None=None, pocket: str="Clean pocket", mode: str="Adjusted", comparison_mode: str="League") -> go.Figure:
    data=_analysis(player,opponent,season,mode,comparison_mode)
    if data["position"]!="QB": raise ValueError("QB Passing Map requires a quarterback.")
    attrs={item["metric"]:item.get("display_value",item["selected_percentile"]) for item in data["attributes"]}
    deep=safe_number(attrs.get("deep_accuracy"),50); base=54+(deep-50)*.12
    if pocket=="Pressure": base-=9
    zones=np.array([[base-5,base+1,base-4],[base+3,base+7,base+2],[base+10,base+13,base+8]])
    primary,secondary=team_palette(data["team"])
    fig=go.Figure(go.Heatmap(z=zones,x=["Left","Middle","Right"],y=["Deep 20+","Intermediate 10-19","Short 0-9"],zmin=35,zmax=80,colorscale=[[0,"rgba(215,58,50,.45)"],[.5,"rgba(240,196,25,.48)"],[1,primary]],text=np.round(zones,1),texttemplate="%{text:.1f}%",hovertemplate="%{y} / %{x}<br>Completion %{z:.1f}%<extra></extra>",colorbar=dict(title="Comp %")))
    for i,(label,value) in enumerate([("Short",zones[2].mean()),("Intermediate",zones[1].mean()),("Deep",zones[0].mean())]):
        fig.add_trace(go.Scatter(x=["Left","Middle","Right"],y=[label+" 0-9" if label=="Short" else label+" 10-19" if label=="Intermediate" else label+" 20+"],mode="lines",line=dict(color=secondary,width=max(2,(value-35)/7),shape="spline"),opacity=.55,hovertemplate=f"{label} arc: {value:.1f}%<extra></extra>",showlegend=False))
    explosives=max(1,int(safe_number(attrs.get("explosive_pass_rate"),50)//20))
    fig.add_trace(go.Scatter(x=["Middle"]*explosives,y=["Deep 20+"]*explosives,mode="markers",marker=dict(size=11,color=secondary,symbol="star",line=dict(color="white",width=1)),name="Explosive-pass markers",hovertemplate="Explosive pass probability<extra></extra>"))
    meta={**data,"view":"QB Passing Map","pocket":pocket}
    return _base(fig,f"{data['player']} · {pocket.lower()} passing map",meta)


def render_route_tree(player: str, opponent: str|None=None, season: int|None=None, mode: str="Adjusted", comparison_mode: str="League") -> go.Figure:
    data = _analysis(player, opponent, season, mode, comparison_mode)
    position = normalize_player_position(data.get("position")) or ""
    route_stats = data.get("raw_stats", {})
    route_data_exists = (
        safe_number(route_stats.get("targets")) > 0
        and (
            safe_number(route_stats.get("routes_run")) > 0
            or safe_number(route_stats.get("yards_per_route_run")) > 0
            or safe_number(route_stats.get("target_share")) > 0
        )
    )
    # Requested diagnostics stay concise and make provider routing observable.
    print(f"NFL Route Tree player.position={position}")
    print(f"NFL Route Tree route_data_exists={route_data_exists}")
    if position not in {"WR", "TE"}:
        raise ValueError("Route Tree requires a wide receiver or tight end.")

    meta = {**data, "view": "WR/TE Route Tree", "route_data_exists": route_data_exists}
    if not route_data_exists:
        figure = go.Figure()
        figure.add_annotation(
            x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
            text="No route data available for this player yet.",
            font=dict(size=18, color="#64748b"),
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return _base(figure, f"{data['player']} · route identity", meta, 420)

    attrs = {
        item["metric"]: item.get("display_value", item["selected_percentile"])
        for item in data["attributes"]
    }
    primary, secondary = team_palette(data["team"])
    routes = {
        "Go": ([0,0,0], [0,12,28]), "Post": ([0,0,10], [0,13,27]),
        "Out": ([0,0,-12], [0,10,10]), "Slant": ([0,3,13], [0,5,10]),
        "Curl": ([0,0,-2], [0,13,10]),
    }
    figure = go.Figure()
    target = safe_number(attrs.get("target_share"), 50)
    success = safe_number(attrs.get("route_win_rate"), 50)
    explosive = safe_number(attrs.get("explosive_play_rate"), 50)
    for index, (name, (x, y)) in enumerate(routes.items()):
        frequency = max(8, target - index * 6)
        color = primary if success - index * 3 >= 65 else secondary
        figure.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color, width=8 + frequency / 8, shape="spline"),
            opacity=.14, hoverinfo="skip", showlegend=False,
        ))
        figure.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            line=dict(color=color, width=2 + frequency / 18, shape="spline"),
            marker=dict(size=[5,5,10], symbol=["circle","circle","star" if explosive >= 75 else "circle"]),
            name=name,
            hovertemplate=f"{name}<br>Frequency index {frequency:.0f}<br>Success percentile {max(0,success-index*3):.0f}<br>Explosive percentile {explosive:.0f}<extra></extra>",
        ))
    figure.add_annotation(x=0, y=-2, text="LOS", showarrow=False)
    figure.update_xaxes(range=[-18,18], visible=False)
    figure.update_yaxes(range=[-4,31], title="Yards downfield", gridcolor="rgba(100,116,139,.16)")
    return _base(figure, f"{data['player']} · route identity", meta)


def render_rb_usage_funnel(player: str, opponent: str|None=None, season: int|None=None, mode: str="Adjusted", comparison_mode: str="League") -> go.Figure:
    data=_analysis(player,opponent,season,mode,comparison_mode)
    if data["position"]!="RB": raise ValueError("RB Usage Funnel requires a running back.")
    usage=data["usage_profile"]; efficiency=data["efficiency_profile"]
    values=[100,safe_number(usage["snap_share"]),min(100,safe_number(usage["volume_per_game"])*3.5),safe_number(usage["red_zone_share"]),safe_number(efficiency["explosive_play_probability"])]
    labels=["Team opportunities","Snap share","Carries + targets","Red-zone touches","Explosive-run rate"]
    colors=["rgba(20,146,83,.58)","rgba(127,209,59,.58)","rgba(240,196,25,.58)","rgba(244,124,32,.58)","rgba(215,58,50,.58)"]
    fig=go.Figure(go.Funnel(y=labels,x=values,texttemplate="%{label}<br>%{value:.1f}",marker=dict(color=colors,line=dict(color="rgba(255,255,255,.72)",width=2)),connector=dict(line=dict(color="rgba(100,116,139,.35)",width=2)),hovertemplate="%{label}<br>Usage index %{value:.1f}<extra></extra>"))
    return _base(fig,f"{data['player']} · usage funnel",{**data,"view":"RB Usage Funnel"},500)


def render_defensive_pressure_map(team_or_player: str, opponent: str|None=None, season: int|None=None, mode: str="Adjusted", comparison_mode: str="League") -> go.Figure:
    # Defensive units are represented by a synthetic DEF row when available;
    # offensive players use their opponent matchup context as the pressure lens.
    data=_analysis(team_or_player,opponent,season,mode,comparison_mode)
    difficulty=safe_number(data["matchup"]["difficulty"],50); pressure=safe_number(data["matchup"]["pressure_rate"],25)
    z=np.array([[pressure*.8,pressure*1.05,pressure*.75],[pressure*1.15,pressure*1.3,pressure],[difficulty*.45,difficulty*.5,difficulty*.4]])
    primary,_=team_palette(data["matchup"].get("team") or data["team"])
    fig=go.Figure(go.Heatmap(z=z,x=["Left edge","Interior","Right edge"],y=["Blitz","Four-man rush","Coverage pressure"],colorscale=[[0,"rgba(215,58,50,.35)"],[.5,"rgba(240,196,25,.5)"],[1,primary]],text=np.round(z,1),texttemplate="%{text:.1f}",hovertemplate="%{y} · %{x}<br>Pressure index %{z:.1f}<extra></extra>",colorbar=dict(title="Pressure")))
    return _base(fig,f"{data['matchup'].get('team') or 'Opponent'} · pressure map",{**data,"view":"Defensive Pressure Map"},470)


def render_pace_play_volume(player: str, opponent: str|None=None, season: int|None=None, mode: str="Adjusted", comparison_mode: str="League") -> go.Figure:
    data=_analysis(player,opponent,season,mode,comparison_mode); primary,secondary=team_palette(data["team"])
    projected_seconds=safe_number(data["pace"]["projected_seconds_per_play"],28.5); plays=round(3600/projected_seconds*.52,1); weather=safe_number(data["weather"]["factor"],1)
    labels=["Team baseline","Opponent blend","Weather adjusted"]; volume=[plays*.98,plays,plays*weather]; pass_ratio=58 if data["position"] in {"QB","WR","TE"} else 48
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=labels,y=volume,name="Projected plays",marker=dict(color=[primary,primary,secondary],opacity=.58),hovertemplate="%{x}<br>%{y:.1f} plays<extra></extra>"),secondary_y=False)
    fig.add_trace(go.Scatter(x=labels,y=[pass_ratio,pass_ratio+2,pass_ratio+2-(1-weather)*30],name="Pass ratio",mode="lines+markers",line=dict(color=secondary,width=4,shape="spline"),marker=dict(size=9),hovertemplate="%{x}<br>Pass rate %{y:.1f}%<extra></extra>"),secondary_y=True)
    fig.update_yaxes(title="Projected plays",secondary_y=False); fig.update_yaxes(title="Pass rate %",range=[25,75],secondary_y=True)
    return _base(fig,f"{data['team']} vs {opponent or 'opponent'} · pace and play volume",{**data,"view":"Pace & Play Volume"},470)
