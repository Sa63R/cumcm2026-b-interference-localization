"""Illustrate explicit probe costs in two constructed observation histories.

No scenario generator, truth log, model checkpoint or official simulator is used.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from localization.omni import OmniCandidateRegion
from simulator_client.state import Position
from strategies.geometric_probe_cost import observation_worlds, primary_cost_samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/geometric_joint/probe_cost_construction_v1')
    args = parser.parse_args()
    long = OmniCandidateRegion().observe((0., 0.), 0.)
    small = OmniCandidateRegion().observe((-1400., 0.), 0.).observe((0., -1400.), 90.)
    specifications = [('One initial bearing', long, (0., 0.), (-300., -150., 0., 150., 300.)),
                      ('Two previous crossing bearings', small, (-100., 0.), (-50., -25., 0., 25., 50.))]
    panels = []
    for name, region, start, offsets in specifications:
        circle = region.enclosing_disk()
        worlds = observation_worlds(region)
        seen = {(round(o.position[0], 6), round(o.position[1], 6)) for o in region.observations}
        rows = []
        for offset in offsets:
            q = Position(circle.center[0], circle.center[1]+offset)
            samples = primary_cost_samples(region, worlds, Position(*start), 1, 1, q, seen)
            if samples is None:
                raise RuntimeError('Constructed continuation did not complete')
            rows.append(dict(offset_m=offset, position=[q.x, q.y],
                mean_cost_s=statistics.fmean(s['cost_s'] for s in samples),
                mean_components_s={k: statistics.fmean(s['components_s'][k] for s in samples)
                                   for k in samples[0]['components_s']},
                mean_measurements=statistics.fmean(sum(a['action']=='measure' for a in s['actions']) for s in samples)))
        panels.append(dict(name=name, start=start, original_circle=asdict(circle),
                           region_vertices=region.vertices, worlds=len(worlds), rows=rows))
    report = dict(scope='Synthetic observation-only geometries; finite proxy costs, not actual benchmark or guaranteed expected costs.',
                  source_sha256=hashlib.sha256((ROOT/'src/strategies/geometric_probe_cost.py').read_bytes()).hexdigest(),
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), constructions=panels)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'construction.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'q3-explicit-probe-cost-v1'})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.1), layout='constrained')
    polygon = np.asarray(long.vertices)
    axes[0].fill(polygon[:, 0], polygon[:, 1], color='#56b7b0', alpha=.5, label='Observed feasible polygon')
    axes[0].scatter([0.], [0.], marker='s', color='#444444', label='Previous observation')
    for row in panels[0]['rows']:
        x, y = row['position']
        axes[0].scatter([x], [y], color='#d66140' if row['offset_m'] == 0 else '#216b9c', s=24)
        axes[0].plot([0., x], [0., y], color='#666666', alpha=.2, lw=.7)
    axes[0].annotate('Original MEC probe', panels[0]['original_circle']['center'], xytext=(770., 65.), fontsize=9)
    axes[0].set(title='A  Reception-safe transverse candidates', xlabel='x / m', ylabel='y / m',
                xlim=(-30., 1580.), ylim=(-600., 600.))
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend(loc='lower right', fontsize=8)
    for ax, panel, title in zip(axes[1:], panels, ('B  Long wedge: parallax can pay', 'C  Small region: avoid the detour')):
        rows = panel['rows']
        x = [r['offset_m'] for r in rows]
        movement = np.array([r['mean_components_s']['movement_s'] for r in rows])
        detection = np.array([r['mean_components_s']['detection_s'] for r in rows])
        other = np.array([r['mean_cost_s'] for r in rows])-movement-detection
        width = (x[1]-x[0])*.68
        ax.bar(x, movement, width, label='Movement', color='#216b9c')
        ax.bar(x, detection, width, bottom=movement, label='Measurement', color='#56b7b0')
        ax.bar(x, other, width, bottom=movement+detection, label='Clear / other', color='#d6a552')
        for row in rows:
            ax.text(row['offset_m'], row['mean_cost_s']+1, f"{row['mean_cost_s']:.1f}", ha='center', fontsize=8)
        ax.set(title=title, xlabel='Transverse displacement from MEC / m', ylabel='Complete local proxy cost / s', xticks=x)
        ax.set_ylim(0, max(r['mean_cost_s'] for r in rows)*1.13)
        ax.grid(axis='y', alpha=.15)
    axes[2].set_ylim(0, max(row['mean_cost_s'] for row in panels[1]['rows'])*1.55)
    axes[2].legend(loc='upper right', fontsize=8)
    fig.savefig(args.output/'construction.png', dpi=170, facecolor='white')
    fig.savefig(args.output/'construction.svg', metadata={'Date': None}, facecolor='white')
    plt.close(fig)
    print(json.dumps({p['name']: [(r['offset_m'], r['mean_cost_s']) for r in p['rows']] for p in panels}, indent=2))


if __name__ == '__main__':
    main()
