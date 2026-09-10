"""A constructed geometric example, with no task or held-out simulation."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT if (ROOT / 'src').is_dir() else ROOT / 'q3-state-search'
sys.path.insert(0, str(SOURCE / 'src'))
from planning.coverage import omni_coverage_points
from planning.disk_cover import disk_cover_radius
from simulator_client.state import Position

OUT = Path(__file__).resolve().parent
font_path = Path('C:/Windows/Fonts/msyh.ttc')
if font_path.exists():
    plt.rcParams['font.family'] = FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update({'font.size': 10, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
sites = list(omni_coverage_points(1150))
old = sites[1]
new = Position(1700, 0)
fixed = sites[:1] + sites[2:]
p, q = Position(1700, 300), Position(1700, -300)
old_radius = disk_cover_radius(fixed + [old])
new_radius = disk_cover_radius(fixed + [new])
assert old_radius < 1000 - 1e-5 and new_radius < 1000 - 1e-5
old_distance = p.distance_to(old) + old.distance_to(q)
new_distance = p.distance_to(new) + new.distance_to(q)

# Shade only tested grid cells. Each actual policy relocation is separately
# certified by the full disk-cover oracle; this raster is an illustration.
xs, ys = np.linspace(850, 1800, 77), np.linspace(-550, 550, 81)
xx, yy = np.meshgrid(xs, ys)
mask = np.zeros_like(xx, dtype=float)
started = time.perf_counter()
for index in np.ndindex(xx.shape):
    point = Position(float(xx[index]), float(yy[index]))
    if math.hypot(point.x, point.y) <= 1800 and disk_cover_radius(fixed + [point]) <= 1000-1e-5:
        mask[index] = 1

fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), gridspec_kw={'width_ratios': [1.1, 1]})
blue, orange, green = '#3568a8', '#c85a19', '#337d64'
ax = axes[0]
arena = Circle((0, 0), 1800, facecolor='#f7f9fc', edgecolor='#253448', linewidth=1.6)
ax.add_patch(arena)
for site in fixed:
    circle = Circle((site.x, site.y), 1000, fill=False, edgecolor='#aeb9c8', linewidth=.9)
    circle.set_clip_path(arena)
    ax.add_patch(circle)
    ax.scatter(site.x, site.y, s=22, color='#65768e', zorder=4)
for point, color, style, label in ((old, blue, '--', '原扫描点：x = 1150 m'),
                                   (new, orange, '-', '移动后：x = 1700 m')):
    circle = Circle((point.x, point.y), 1000, fill=False, edgecolor=color,
                    linestyle=style, linewidth=2, label=label)
    circle.set_clip_path(arena)
    ax.add_patch(circle)
    ax.scatter(point.x, point.y, color=color, s=48, zorder=6)
ax.annotate('', xy=(new.x, new.y), xytext=(old.x, old.y),
            arrowprops={'arrowstyle': '->', 'color': orange, 'lw': 1.7})
ax.text(0, -2110, '其余六个扫描点固定；整个半径 1800 m 圆盘仍被覆盖', ha='center', fontsize=10)
ax.set(xlim=(-2000, 2000), ylim=(-2200, 2000), aspect='equal', xlabel='x / m', ylabel='y / m')
ax.set_title('A  保留完整覆盖，释放扫描点的位置自由度', loc='left', pad=12)
ax.legend(loc='upper right', framealpha=.95, fontsize=9)
ax.grid(alpha=.13)

ax = axes[1]
ax.contourf(xx, yy, mask, levels=[.5, 1.5], colors=['#dfeee7'], alpha=.95)
ax.contour(xx, yy, mask, levels=[.5], colors=[green], linewidths=.8)
ax.plot([p.x, old.x, q.x], [p.y, old.y, q.y], '--', color=blue, linewidth=2,
        label=f'原两段路程：{old_distance:.1f} m')
ax.plot([p.x, new.x, q.x], [p.y, new.y, q.y], '-', color=orange, linewidth=2.3,
        label=f'移动后：{new_distance:.1f} m')
for point, label, color, offset in ((p, '当前位置 p', '#253448', (-115, 50)),
                                   (q, '下一任务 q', '#253448', (-115, -90)),
                                   (old, '旧扫描点', blue, (-170, 50)),
                                   (new, '新扫描点', orange, (-180, 50))):
    ax.scatter(point.x, point.y, s=45, color=color, zorder=5)
    ax.annotate(label, (point.x, point.y), xytext=offset, textcoords='offset pixels',
                fontsize=9, color=color)
ax.text(885, -470, '绿色区域：图示网格中通过完整覆盖检验的位置', fontsize=8.4, color=green)
ax.set(xlim=(800, 1850), ylim=(-575, 575), aspect='equal', xlabel='x / m', ylabel='y / m')
ax.set_title('B  构造例：相同前后任务，减少覆盖绕行', loc='left', pad=12)
ax.legend(loc='upper left', framealpha=.96, fontsize=9)
ax.grid(alpha=.13)
fig.suptitle(f'扫描点可移动 550 m；此例两段移动时间减少 {(old_distance-new_distance)/5:.1f} 秒',
             fontsize=14, y=.98)
fig.text(.5, .035, f'移动前后覆盖半径均为 {old_radius:.3f} m < 1000 m。该图是几何构造例，不是单局实测或全局最优证明。',
         ha='center', fontsize=9, color='#536174')
fig.subplots_adjust(left=.065, right=.98, bottom=.18, top=.86, wspace=.23)
for suffix in ('png', 'svg'):
    fig.savefig(OUT / ('movable-cover-principle.' + suffix), dpi=200, bbox_inches='tight')
plt.close(fig)
metrics = {'scope': 'Constructed geometry only, zero simulated episodes',
           'source_commit': '8aa620649b609876b612d307b5be71ab588acc08',
           'disk_cover_source_sha256': hashlib.sha256((SOURCE / 'src/planning/disk_cover.py').read_bytes()).hexdigest(),
           'old_cover_radius_m': old_radius, 'new_cover_radius_m': new_radius,
           'old_distance_m': old_distance, 'new_distance_m': new_distance,
           'constructed_saving_s': (old_distance-new_distance)/5,
           'feasible_grid_shape': list(mask.shape), 'feasible_grid_points': int(mask.sum()),
           'illustration_compute_s': time.perf_counter()-started}
(OUT / 'movable-cover-principle.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print(json.dumps(metrics))
