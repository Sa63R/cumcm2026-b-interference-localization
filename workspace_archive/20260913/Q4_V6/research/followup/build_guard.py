from pathlib import Path
r=Path(__file__).parent
(r/'q4_v6_unprotected.py').write_text((r/'q4_v6.py').read_text())
s=(r/'q4_learned_route.py').read_text()
s=s.replace('def solve_learned_route(device, model=', 'def solve_guarded(device, model=')
s=s.replace('v5=True, transit=None, transit_threshold=0., maxstops=16):', 'v5=True, transit=None, transit_threshold=0., maxstops=16, minimum_stations=4):')
s=s.replace('critic=Critic(model);tc=', 'critic=Critic(model) if model else None;tc=')
s=s.replace('last=None;chosen=[]', 'last=None;chosen=[];guarded_steps=0;activation=None')
s=s.replace('order=engine.ordered_actions();items=route_features(engine,order)\n        if items:', 'order=engine.ordered_actions()\n        active=len(engine.state.visited)>=minimum_stations\n        if not active:guarded_steps+=1\n        elif activation is None:activation=dict(visited_stations=len(engine.state.visited),known_sources=len(engine.state.pending)+len(engine.state.cleared))\n        items=route_features(engine,order) if active and critic else []\n        if items:')
s=s.replace('if tc and stops<maxstops', 'if active and tc and stops<maxstops')
s=s.replace('planner_calls=calls,planner_accepts=accepts,planner_seconds=elapsed)', 'planner_calls=calls,planner_accepts=accepts,planner_seconds=elapsed,\n                guarded_steps=guarded_steps,activation=activation,minimum_stations=minimum_stations)')
s='''"""Delay learned decisions until several distinct certified scan sites are visited.
This preserves the early V5 action prefix and does NOT use the hidden source
count. It is a tested risk-control heuristic, not a pointwise dominance theorem.
The underlying finite-action geometry and verified completion feedback remain.
"""\n'''+s[s.index('import random,time'):]
(r/'q4_guarded_solver.py').write_text(s)
s=(r/'q4_v6.py').read_text()
s=s.replace('from q4_learned_route import solve_learned_route','from q4_guarded_solver import solve_guarded')
s=s.replace('    max_transit_stops: int = 16','    max_transit_stops: int = 16\n    minimum_stations: int = 4')
s=s.replace("    if not all(math.isfinite", "    if not isinstance(c.minimum_stations,int) or not 0<=c.minimum_stations<=21:\n        raise ValueError('minimum_stations must be an integer in [0,21]')\n    if not all(math.isfinite")
start=s.index('    if c.route_ranking:')
end=s.index("    r['algorithm']",start)
s=s[:start]+'''    if c.route_ranking or c.transit_sensing:
        r=solve_guarded(device,c.route_model if c.route_ranking else None,c.route_threshold,
              v5=c.use_v5_local,transit=c.transit_model if c.transit_sensing else None,
              transit_threshold=c.transit_threshold,maxstops=c.max_transit_stops,
              minimum_stations=c.minimum_stations)
    else:
        if c.use_v5_local:r=solve_v5(device)
        else:
            from q4_state import Engine
            r=Engine(device).run_base()
'''+s[end:]
s=s.replace("r['algorithm']='Q4_V6'", "r['algorithm']='Q4_V6_GUARDED'")
s=s.replace('Q4 V6: transit sensing and learned completion-cost action ranking.', 'Q4 V6: guarded transit sensing and learned completion-cost action ranking.')
(r/'q4_v6.py').write_text(s)
