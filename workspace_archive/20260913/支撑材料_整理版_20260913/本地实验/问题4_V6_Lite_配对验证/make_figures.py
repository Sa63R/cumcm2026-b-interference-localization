"""Standalone comparison plots; values come only from completed frozen tests."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

s = json.loads((ROOT / 'summary.json').read_text())
cpu = json.loads((ROOT / 'serial/summary.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout='constrained')
methods = ['v4', 'full', 'lite']
labels = ['V4', 'V6', 'V6 Lite']
colors = ['#a4adba', '#405f95', '#138b83']
for axis, group, title in [(axes[0], 'new_practice', '1,000 new random scenarios'),
                            (axes[1], 'boundary_outward_r1000', '70 outward boundary scenarios')]:
    values = [s['groups'][group]['methods'][m]['mean_seconds_per_source'] for m in methods]
    axis.bar(labels, values, color=colors, width=.62)
    axis.bar_label(axis.containers[0], labels=[f'{v:.2f}' for v in values], padding=5)
    axis.set_ylim(0, max(values) * 1.14)
    axis.set_ylabel('Virtual task seconds / source')
    axis.set_title(title, fontsize=11)
    axis.grid(axis='y', alpha=.15)
    axis.set_axisbelow(True)
values = [cpu['methods'][m]['cpu_seconds'] for m in methods]
axes[2].bar(labels, values, color=colors, width=.62)
axes[2].bar_label(axes[2].containers[0], labels=[f'{v:.3f}' for v in values], padding=5)
axes[2].set_ylim(0, max(values) * 1.18)
axes[2].set_ylabel('CPU seconds / complete run')
axes[2].set_title('50 paired scenarios, one worker', fontsize=11)
axes[2].grid(axis='y', alpha=.15)
axes[2].set_axisbelow(True)
for axis in axes:
    axis.spines[['top', 'right']].set_visible(False)
fig.suptitle('V6 Lite: task time and computation time', fontsize=15)
out = ROOT / 'figures'
out.mkdir(exist_ok=True)
for suffix in ['png', 'pdf']:
    fig.savefig(out / f'lite_comparison.{suffix}', dpi=170)
plt.close(fig)

rows = [json.loads(line) for line in (ROOT / 'main/records.jsonl').read_text().splitlines()]
index = {(r['case_key'], r['method']): r for r in rows if r['group'] == 'new_practice'}
keys = sorted({key for key, _ in index})
fig, axis = plt.subplots(figsize=(8, 4.6), layout='constrained')
for method, label, color in [('full', 'V6 vs V4', colors[1]), ('lite', 'V6 Lite vs V4', colors[2])]:
    values = np.sort([100 * (index[key, method]['virtual_time_s'] / index[key, 'v4']['virtual_time_s'] - 1) for key in keys])
    axis.step(values, 100 * np.arange(1, len(values) + 1) / len(values), where='post', label=label, color=color)
axis.axvline(0, color='#777777', linewidth=.8, linestyle='--')
axis.set(xlabel='Task slowdown relative to V4 (%) — negative is faster',
         ylabel='Cumulative share of scenarios (%)',
         title='Observed distribution on 1,000 new random scenarios')
axis.legend(frameon=False)
axis.grid(alpha=.15)
axis.spines[['top', 'right']].set_visible(False)
for suffix in ['png', 'pdf']:
    fig.savefig(out / f'lite_tail_distribution.{suffix}', dpi=170)
