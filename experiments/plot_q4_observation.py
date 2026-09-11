"""Show development versus independent effects using saved paired statistics."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'research/q4_r4_observation'


def main():
    paths = [RESEARCH / 'development-decision.json', RESEARCH / 'qualification.json']
    docs = [json.loads(p.read_bytes()) for p in paths]
    stages = [('development', 'Development: random (24)', docs[0]),
              ('development-stress', 'Development: stress (14)', docs[0]),
              ('confirmation', 'Independent: random (64)', docs[1]),
              ('stress', 'Independent: stress (42)', docs[1])]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10., 'text.color': '#27364b'})
    fig, (ax, detail) = plt.subplots(1, 2, figsize=(12., 5.4), gridspec_kw={'width_ratios': [1.8, 1.15]})
    fig.subplots_adjust(left=.22, right=.985, top=.75, bottom=.20, wspace=.09)
    for y, (name, label, doc) in zip((3, 2, 1, 0), stages):
        pair = doc['comparisons_vs_incumbent']['compact_observation_d1'][name]
        rows = doc['summaries'][name]
        old, new = rows['compact_combo'], rows['compact_observation_d1']
        if not pair['all_clear'] or old['mean_lower_bound_s'] != new['mean_lower_bound_s']:
            raise ValueError('Need complete matched pairs and the same lower bound')
        mean = pair['mean_saved_s']; lo, hi = pair['saving_ci95_s']
        color = '#4177a0' if y >= 2 else '#ad6838'
        ax.errorbar(mean, y, xerr=[[mean-lo], [hi-mean]], fmt='o', color=color, capsize=5, linewidth=2, markersize=7)
        ax.text(mean, y+.20, f'{mean:.2f} s', ha='center', fontsize=10, color=color)
        detail.text(0., y+.12, f"{old['mean_time_s']:,.2f} -> {new['mean_time_s']:,.2f} s", fontsize=10)
        detail.text(0., y-.13, f"LB {new['mean_lower_bound_s']:,.2f} s | {old['mean_time_over_mean_lower_bound']:.3f} -> {new['mean_time_over_mean_lower_bound']:.3f} x", fontsize=9, color='#617188')
    ax.axvline(0, color='#576575', linewidth=1)
    for a in (ax, detail):
        a.set_ylim(-.55, 3.65)
        a.axhline(1.5, color='#d9e0e8', linewidth=.8, linestyle='--')
        for spine in a.spines.values(): spine.set_visible(False)
    ax.set(yticks=(3, 2, 1, 0), yticklabels=[s[1] for s in stages], xlim=(-65, 310), xlabel='Time saved vs Combo (s); positive is faster')
    ax.tick_params(length=0)
    ax.grid(axis='x', color='#e7ebf0', linewidth=.65)
    ax.set_axisbelow(True)
    detail.set_xlim(0, 1); detail.axis('off')
    ax.set_title('Paired mean saving and 95% bootstrap interval', fontsize=10.5, pad=15)
    detail.set_title('Mean time, mean LB and ratio of means', fontsize=10.5, loc='left', pad=15)
    fig.suptitle('Q4 observation tree: a stress benefit, no general promotion', y=.96, fontsize=15, fontweight='bold')
    fig.text(.5, .865, 'Depth 1 was selected before independent testing. The random confirmation interval includes zero.', ha='center', fontsize=10)
    fig.text(.5, .065, 'Local synthetic cases only. All runs cleared every source. Historical LB; ratios are not means of per-case ratios.', ha='center', fontsize=9, color='#617188')
    output = RESEARCH / 'effects.png'
    fig.savefig(output, dpi=200, facecolor='white', metadata={'InputsSHA256': json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}), 'GeneratorSHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    plt.close(fig)
    print(output)


if __name__ == '__main__': main()
