from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
from matplotlib.font_manager import FontProperties
root=Path(__file__).resolve().parent
p=root/'official_observation_inventory.json'
j=json.loads(p.read_text())
font=FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc')
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(9,8.7),layout='constrained')
colors=['#2c6da6','#188675','#8061a8','#bf7334']
clean=[]
for i,(ax,w,col) in enumerate(zip(axes.flat,j['worlds'],colors),1):
 pts=w['source_estimates']
 ax.add_patch(Circle((0,0),1800,fill=False,color='#667180',lw=1.3))
 ax.add_patch(Circle((0,0),900,fill=False,color='#a4afbb',ls='--',lw=1))
 ax.axhline(0,color='#e3e7ec',lw=.8,zorder=0);ax.axvline(0,color='#e3e7ec',lw=.8,zorder=0)
 for s in pts:
  ax.add_patch(Polygon(s['candidate_vertices'],color=col,alpha=.4,zorder=3))
 ax.scatter([s['center'][0] for s in pts],[s['center'][1] for s in pts],s=24,color=col,zorder=4)
 ax.scatter([0],[0],marker='+',color='#6a7380',s=65,zorder=4)
 ax.set(xlim=(-1900,1900),ylim=(-1900,1900),aspect='equal',xticks=[-1800,-900,0,900,1800],yticks=[-1800,-900,0,900,1800])
 ax.set_title(f'场景 {i} · {len(pts)} 个干扰源',fontproperties=font,fontsize=13,pad=10)
 ax.set_xlabel('x / m');ax.set_ylabel('y / m')
 for spine in ax.spines.values():spine.set_visible(False)
 clean.append({'scene':i,'n':len(pts),'sources':[{'channel':s['channel'],'estimated_center_m':s['center'],'enclosure_radius_m':s['enclosure_radius_m'],'candidate_vertices_m':s['candidate_vertices']} for s in pts]})
fig.suptitle('从 4 局官方观测恢复的位置分布',fontproperties=font,fontsize=18,color='#25344b')
fig.supxlabel('虚线圈半径 900 m，占全圆面积的 1/4；48 个源中有 13 个估计位置在圈内。\n点为位置估计，保守误差半径不超过 19.25 m；这些场景不能代表完整数据库。',fontproperties=font,fontsize=10,color='#536175')
fig.savefig(root/'official_maps.png',dpi=170)
fig.savefig(root/'official_maps.svg')
(root/'estimated_maps.json').write_text(json.dumps({'origin':'4 completed official Q3 practice sessions; reconstructed from public responses','source_inventory_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ground_truth':False,'maps':clean},ensure_ascii=False,indent=2)+'\n')
print(root/'official_maps.png')
