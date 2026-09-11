"""Public geometry only: fixed cover routes and radial boundary observability."""
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from planning.q4_directional_cover import certified_cover_points
from planning.observation_cover_route import observation_cover_route


def main():
    old, cert = certified_cover_points('compact_22')
    data = [('R12: 22 stations', old, cert['route_length_m'], 7, 1800 / math.cos(math.pi/14) + 5)]
    for name, k, rho in [('ring_28', 9, 1920), ('ring_31', 10, 1900)]:
        pts, new_cert, metadata = observation_cover_route(name)
        assert new_cert['passed'] and len(pts) == 1+3*k
        data.append((name.replace('_', ' '), pts, metadata['route_length_m'], k, rho))
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.3))
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10})
    manifest = []
    for ax, (title, pts, length, k, rho) in zip(axes, data):
        ax.add_patch(Circle((0, 0), 1800, color='#eef3f7', zorder=0))
        ax.add_patch(Circle((0, 0), 1800, fill=False, edgecolor='#8795a1', lw=1))
        ax.plot([p.x for p in pts], [p.y for p in pts], color='#8d99a5', lw=1.2, zorder=1)
        for a, b in zip(pts, pts[1:]):
            ax.annotate('', xy=(a.x+.61*(b.x-a.x), a.y+.61*(b.y-a.y)),
                        xytext=(a.x+.48*(b.x-a.x), a.y+.48*(b.y-a.y)),
                        arrowprops=dict(arrowstyle='->', color='#596778', lw=.9))
        inner = [p for p in pts if 1 < math.hypot(p.x,p.y) < 1500]
        outer = [p for p in pts if math.hypot(p.x,p.y) > 1500]
        ax.scatter([p.x for p in inner], [p.y for p in inner], c='#2374a5', s=33, zorder=3)
        ax.scatter([p.x for p in outer], [p.y for p in outer], c='#d87828', s=33, zorder=3)
        ax.scatter([0], [0], c='#172d45', marker='s', s=35, zorder=4)
        alpha = min(math.acos(1800/rho), math.acos((rho*rho+1800**2-1000**2)/(2*rho*1800)))
        fraction = min(1., max(0., 2*alpha/(math.pi/k)-1))
        minimum = math.floor(2*alpha/(math.pi/k))
        ax.set_title(f'{title}\n{length/5:,.1f} s pure movement', fontsize=13, pad=14)
        ax.set_xlim(-2200, 2200); ax.set_ylim(-2200, 2200)
        ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_xticks([-2000,0,2000]); ax.set_yticks([-2000,0,2000])
        ax.text(.5, -.19, f'Radial boundary: min {minimum} outer receivers\n'
                f'Angle share with 2+ receivers: {fraction:.1%}', transform=ax.transAxes,
                ha='center', va='top', fontsize=10)
        ax.spines[['top','right']].set_visible(False)
        manifest.append(dict(title=title, ordered_points=[[p.x,p.y] for p in pts],
                             route_length_m=length, radial_boundary_min=minimum,
                             radial_boundary_fraction_two=fraction))
    fig.suptitle('More complementary observations, at the cost of more travel and scans', fontsize=15, y=.98)
    fig.text(.5, .025, 'Boundary model: source radius 1800 m, emission exactly outward, reception radius 1000 m.\n'
             'This geometry does not guarantee two observations elsewhere, successful localization, or lower total time.',
             ha='center', va='bottom', fontsize=10, color='#455361')
    fig.tight_layout(rect=(0, .17, 1, .95))
    dest=Path(__file__).parent/'figures'; dest.mkdir(exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(dest/f'observation_geometry.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    source_files = ['src/planning/q4_directional_cover.py', 'src/planning/observation_cover_route.py',
                    'research/q4_observation_cover/plot_geometry.py']
    record=dict(scope='Public fixed geometry, no scenes or simulator history', layouts=manifest,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files})
    (dest/'geometry_plot_inputs.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Saved public geometry PNG, PDF and input manifest')


if __name__=='__main__':
    main()
