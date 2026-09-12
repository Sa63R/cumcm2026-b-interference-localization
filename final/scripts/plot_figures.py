"""Publication figures from saved experiments; no simulator or policy imports.

Uses installed Scientific Visualization skill (pinned in sources/).
"""
from pathlib import Path
import os, sys, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parents[1]
SKILL=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'skills/scientific-visualization'
sys.path.insert(0,str(SKILL/'scripts'))
from style_presets import style_context
from figure_export import export_figure

DATA=OUT/'data'; FIG=OUT/'figures'
COL={'baseline':'#555555','state':'#0072B2','rl':'#D55E00','geo':'#009E73'}
LAB={'baseline':'前瞻基线','state':'状态搜索','rl':'强化学习','geo':'几何法'}
ORDER=['baseline','state','rl','geo']
def read(n): return json.loads((DATA/n).read_text(encoding='utf-8'))
def save(fig,name,source,description):
    prov={'raw_data':source,'transformations':[description],
          'destination':'国赛论文第二三问初稿，通用学术排版，非期刊合规认证',
          'skill_commit':(OUT/'sources/visualization_skill_commit.txt').read_text().strip(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    export_figure(fig,FIG/name,formats=['pdf','svg','png'],dpi=400,
        bbox_inches=None,overwrite=True,facecolor='white',font_mode='current',
        provenance=prov,write_manifest=True)
    plt.close(fig)

def comparison():
    c=read('four_methods.json')['random']
    fig,axs=plt.subplots(1,2,figsize=(6.7,2.62),layout='constrained',gridspec_kw={'width_ratios':[1,1.1]})
    ax=axs[0]; vals=[c['methods'][k]['raw_mean_total_time_s'] for k in ORDER]
    bars=ax.bar(np.arange(4),vals,color=[COL[k] for k in ORDER],width=.63,edgecolor='black',linewidth=.4)
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v+50,f'{v:.1f}',ha='center',fontsize=7)
    ax.set(xticks=range(4),xticklabels=[LAB[k] for k in ORDER],ylim=(0,3900),ylabel='平均计费用时 / s')
    ax.set_title('(a) 四方法均时  n = 256',loc='left',pad=9)
    ax.yaxis.grid(True,color='.9',lw=.5); ax.tick_params(axis='x',length=0,labelsize=7)
    ax=axs[1]
    for y,k in enumerate(['state','rl','geo']):
        m=c['comparisons'][k]; v=m['mean_seconds_saved']; lo,hi=m['saving_ci95_s']
        ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt=['o','s','^'][y],color=COL[k],capsize=3,ms=5,lw=1.4)
        ax.text(v,y+.21,f'{v:.1f} [{lo:.1f}, {hi:.1f}]',ha='center',fontsize=6.6)
    ax.axvline(0,color='.6',lw=.8,ls='--')
    ax.set(yticks=range(3),yticklabels=[LAB[k] for k in ['state','rl','geo']],xlim=(-10,335),ylim=(2.65,-.55),xlabel='相对基线节省 / s')
    ax.set_title('(b) 配对均值与95%置信区间',loc='left',pad=9)
    ax.xaxis.grid(True,color='.9',lw=.5)
    save(fig,'fig03_comparison','data/four_methods.json','全部256配对场景；条形从零起；区间直接引用原10000次配对bootstrap结果。')

def costs():
    c=read('four_methods.json')['random']['methods']
    fig,ax=plt.subplots(figsize=(6.7,2.5),layout='constrained')
    labels=['移动','测量','换频','光学与移除']; colors=['#0072B2','#E69F00','#999999','#555555']
    left=np.zeros(4)
    for i,(lab,color) in enumerate(zip(labels,colors)):
        vals=[]
        for k in ORDER:
            d=c[k]['mean_components_s']
            vals.append([d['movement_s'],d['detection_s'],d['switching_s'],d['optical_s']+d['removal_s']][i])
        bars=ax.barh(range(4),vals,left=left,label=lab,color=color,height=.57,edgecolor='white',lw=.5,hatch=['','//','','..'][i])
        if i<2:
            for y,(l,v) in enumerate(zip(left,vals)):ax.text(l+v/2,y,f'{v:.1f}',ha='center',va='center',fontsize=7,color='white' if i==0 else 'black')
        left+=vals
    for y,v in enumerate(left): ax.text(v+40,y,f'{v:.1f}',va='center',fontsize=7)
    ax.set(yticks=range(4),yticklabels=[LAB[k] for k in ORDER],xlim=(0,3760),xlabel='平均计费用时 / s')
    ax.invert_yaxis();ax.xaxis.grid(True,color='.92',lw=.5)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.24),fontsize=7)
    save(fig,'fig04_costs','data/four_methods.json','同一256随机场景的费用分项均值；光学与移除费用合并显示，其余未变换。')

def trajectory(label,number):
    c=read(f'trajectory_{label}.json')
    fig,axs=plt.subplots(2,2,figsize=(6.7,6.6),layout='constrained')
    for i,(ax,k) in enumerate(zip(axs.flat,ORDER)):
        a=c['plotted'][k]; row=a['row']; route=np.asarray(a['route']); measures=np.asarray(a['measures'])
        ax.add_patch(Circle((0,0),1800,fill=False,color='.55',lw=.7))
        ax.plot(route[:,0],route[:,1],color=COL[k],lw=1,zorder=2)
        for j in range(len(route)-1):
            d=route[j+1]-route[j]
            if np.linalg.norm(d)>150:
                p=route[j]+.52*d; q=route[j]+.68*d
                ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=6,color=COL[k],lw=.6,zorder=3))
        ax.scatter(measures[:,0],measures[:,1],s=10,marker='o',facecolors='white',edgecolors=COL[k],lw=.6,zorder=4)
        for s in c['ground_truth']['sources']:
            ax.scatter(s['x'],s['y'],s=18,marker='x',color='black',lw=.9,zorder=6)
            ax.annotate(str(s['channel']),(s['x'],s['y']),xytext=(4,3),textcoords='offset points',fontsize=6,color='black',zorder=7)
        ax.scatter([0],[0],marker='*',s=60,color='#882255',zorder=8)
        ax.scatter([route[-1,0]],[route[-1,1]],marker='s',s=27,facecolors='none',edgecolors='#882255',lw=1,zorder=8)
        clearpts=np.array([x['position'] for x in a['clears']]); ax.scatter(clearpts[:,0],clearpts[:,1],marker='+',s=28,color='#D55E00',lw=.9,zorder=7)
        order='→'.join(str(x['channel']) for x in a['clears'])
        ax.text(.01,.01,'清除顺序 '+order,transform=ax.transAxes,fontsize=5.7,va='bottom',bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':1})
        ax.set_title(f'({chr(97+i)}) {LAB[k]}   T = {row["virtual_time_s"]:.1f} s\n移动 {row["movement_s"]:.1f} s   测量 {row["measurement_count"]} 次',loc='left',fontsize=7.5,pad=6)
        ax.set(xlim=(-2050,2050),ylim=(-2050,2050),aspect='equal',xticks=[-1800,0,1800],yticks=[-1800,0,1800],xlabel='x / m',ylabel='y / m')
        ax.grid(True,color='.93',lw=.5)
    handles=[Line2D([],[],marker='o',ls='',mfc='white',mec='.3',label='测点'),Line2D([],[],marker='x',ls='',color='black',label='源与频道'),Line2D([],[],marker='+',ls='',color='#D55E00',label='实际清除点'),Line2D([],[],marker='*',ls='',color='#882255',label='起点'),Line2D([],[],marker='s',ls='',mfc='none',mec='#882255',label='终点')]
    fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=7)
    save(fig,f'fig{number:02d}_trajectory_{label}',f'data/trajectory_{label}.json',f'种子{c["seed"]}，四方法相同case哈希与坐标比例；源真值仅事后显示；选择规则见配套JSON。')

def geometry():
    eps=np.deg2rad(1.005); a=np.linspace(-eps,eps,160)
    sector=np.vstack([[0,0],np.c_[1500*np.cos(a),1500*np.sin(a)]])
    fig,axs=plt.subplots(1,2,figsize=(6.7,2.82),layout='constrained')
    ax=axs[0]
    # Conservative guaranteed reception: max distance to the outer polygon <=1000.
    xs=np.linspace(-250,1800,410); ys=np.linspace(-900,900,360); X,Y=np.meshgrid(xs,ys)
    vertices=np.array([[0,0],[1500,1500*np.tan(eps)],[1500,-1500*np.tan(eps)]])
    dmax=np.maximum.reduce([(X-p[0])**2+(Y-p[1])**2 for p in vertices])
    ax.contourf(X,Y,dmax,levels=[0,1000**2],colors=['#DDEAF4'],zorder=0)
    ax.contour(X,Y,dmax,levels=[1000**2],colors=['#0072B2'],linewidths=.8)
    ax.add_patch(Polygon(sector,facecolor='#E69F00',alpha=.6,edgecolor='#AA7000',lw=.8,zorder=2))
    ax.plot([0,1500],[0,0],'--',color='.4',lw=.7)
    points=[(0,0,r'$x_1$'),(750,250,r'$q_{\perp}$'),(750,0,r'$q_{\parallel}$')]
    for x,y,t in points:
        ax.plot(x,y,'o',ms=3,color='black'); ax.annotate(t,(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
    ax.annotate('$C_f$',(1280,18),xytext=(1420,260),arrowprops={'arrowstyle':'-','lw':.6},fontsize=9)
    ax.text(710,-350,'保证接收域',ha='center',fontsize=7,color='#005988')
    ax.set(xlim=(-150,1700),ylim=(-650,650),aspect='equal',xlabel='x / m',ylabel='y / m')
    ax.set_title('(a) 首测源域与充分安全测点域',loc='left',pad=8)
    ax=axs[1]
    # Purely geometric illustration with true angular error, not an experimental run.
    p=np.array([1050,0]); q=np.array([750,250]); beta=np.arctan2(*(p-q)[::-1])
    for sign in [-1,1]:
        u=np.array([np.cos(beta+sign*eps),np.sin(beta+sign*eps)])
        ax.plot([q[0],q[0]+1500*u[0]],[q[1],q[1]+1500*u[1]],color='#0072B2',lw=.8)
    ax.add_patch(Polygon(sector,facecolor='#E69F00',alpha=.3,edgecolor='#AA7000',lw=.6))
    X,Y=np.meshgrid(np.linspace(600,1500,600),np.linspace(-100,350,300))
    da=np.arctan2(Y-q[1],X-q[0])-beta; da=(da+np.pi)%(2*np.pi)-np.pi
    mask=(np.abs(np.arctan2(Y,X))<=eps)&(X*X+Y*Y<=1500**2)&(np.abs(da)<=eps)
    ax.contourf(X,Y,mask.astype(float),levels=[.5,1.5],colors=['#0072B2'])
    ax.plot(*q,'o',color='black',ms=3);ax.annotate(r'$q_{\perp}$',q,xytext=(6,2),textcoords='offset points',fontsize=8)
    ax.plot(*p,'x',color='black',ms=5);ax.annotate('$p$',p,xytext=(7,-13),textcoords='offset points',fontsize=8)
    ax.annotate('两次测向交会区域',(1050,0),xytext=(1130,160),arrowprops={'arrowstyle':'->','lw':.6},fontsize=7)
    ax.set(xlim=(650,1470),ylim=(-80,320),aspect='equal',xlabel='x / m',ylabel='y / m')
    ax.set_title('(b) 横向基线的交会示意',loc='left',pad=8)
    save(fig,'fig01_q2_geometry','scripts/plot_figures.py:geometry','解析示意，保守半角1.005°对应题面±1°加舍入余量；左图三角外包给充分接收域，右图p仅用于解释几何，非策略已知真值或性能实验。')

def framework():
    fig,axs=plt.subplots(1,2,figsize=(6.7,2.7),layout='constrained',gridspec_kw={'width_ratios':[1,1.25]})
    ax=axs[0];ang=np.arange(6)*np.pi/3
    stations=np.vstack([[0,0],np.c_[1150*np.cos(ang),1150*np.sin(ang)]])
    for i,p in enumerate(stations):
        ax.add_patch(Circle(p,1000,fill=False,ec='#0072B2',lw=.65,alpha=.65))
        ax.plot(*p,'o',color='#0072B2',ms=3)
        ax.text(p[0]+55,p[1]+40,str(i),fontsize=7)
    ax.add_patch(Circle((0,0),1800,fill=False,ec='black',lw=1))
    ax.set(xlim=(-2250,2250),ylim=(-2250,2250),aspect='equal',xlabel='x / m',ylabel='y / m',xticks=[-1800,0,1800],yticks=[-1800,0,1800])
    ax.set_title('(a) 初始七站与1000 m覆盖圆',loc='left',pad=8)
    ax=axs[1];ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes=[(.53,.86,'公开观测与几何可行域'),(.53,.65,'覆盖任务与已发现源任务'),(.53,.43,'状态压缩与有预算 A*'),(.53,.21,'执行覆盖扫描或单源定位清除')]
    for x,y,t in boxes:
        ax.add_patch(Rectangle((.14,y-.075),.78,.15,facecolor='white',edgecolor='.3',lw=.7))
        ax.text(x,y,t,ha='center',va='center',fontsize=8)
    for (_,y,_),(_,z,_) in zip(boxes,boxes[1:]):ax.annotate('',(.5,z+.077),(.5,y-.077),arrowprops={'arrowstyle':'->','lw':.8})
    ax.annotate('',(.125,.86),(.125,.21),arrowprops={'arrowstyle':'->','connectionstyle':'bar,fraction=-.1','lw':.8,'color':'#0072B2'})
    ax.text(.045,.53,'更\n新\n与\n重\n规\n划',ha='center',va='center',fontsize=6.5,color='#0072B2')
    ax.text(.5,.035,'所有源已清除且终止条件成立 → 退出',ha='center',fontsize=7)
    ax.set_title('(b) 观测驱动的滚动决策',loc='left',pad=8)
    save(fig,'fig02_framework','q3-state-search final v1 parameters','七点初始布局解析示意；方法框图概括真实模块，不将模型预测与实际反馈混淆。')

def ablations():
    doc=json.loads((OUT/'sources/evidence_experiments.json').read_text(encoding='utf-8'))
    e={x['id']:x for x in doc['entries']}
    state=[]
    for key,label in [('E05','上限静默  64局'),('E06','覆盖移站  64局')]:
        p=e[key]['pair'];state.append((label,p['mean_saved_s'],p['paired_bootstrap_95ci_saved_s']))
    for key,label in [('relative_only','相对静默  256局'),('cap_only','16源上限  256局'),('derived_silence','两者组合  256局')]:
        p=e['E07']['comparisons'][key];state.append((label,p['mean_saved_s'],p['ci95_saved_s']))
    rl=[(label,p['mean_saved_s'],p['ci95_saved_s']) for label,p in zip(['GAE 0.95 / 等预算λ1','GAE 0.95 / 父模型','注意力 / 父模型'],e['E09']['rows'][:3])]
    fig,axs=plt.subplots(2,1,figsize=(6.7,4.1),layout='constrained',gridspec_kw={'height_ratios':[1.35,1]})
    for ax,rows,col,title,lim in [(axs[0],state,COL['state'],'(a) 状态搜索组件  各行使用对应参照',(-8,68)),(axs[1],rl,COL['rl'],'(b) 强化学习配置  48局开发场景',(-100,130))]:
        for y,(lab,v,(lo,hi)) in enumerate(rows):
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',ms=4,color=col,capsize=3)
            ax.text(lim[1],y,f'{v:.2f} [{lo:.2f}, {hi:.2f}]',va='center',ha='right',fontsize=6.5)
        # Value labels occupy a dedicated right-hand region beyond interval endpoints.
        span=lim[1]-lim[0];ax.set_xlim(lim[0],lim[1]+span*.55)
        for t in ax.texts:t.set_x(lim[1]+span*.52)
        ax.set(yticks=range(len(rows)),yticklabels=[r[0] for r in rows],ylim=(len(rows)-.4,-.6),xlabel='相对各自参照节省 / s  点为均值，横线为95%配对区间')
        ax.axvline(0,ls='--',color='.55',lw=.7);ax.xaxis.grid(True,color='.92',lw=.5)
        ax.set_title(title,loc='left',fontsize=8,pad=7)
    save(fig,'fig05_ablations','sources/evidence_experiments.json:E05,E06,E07,E09','原报告点估计与95%配对区间；独立实验分组使用各自参照，不能加和节省或跨组直接比较算法性能。')

def main():
    with style_context('default',palette_name='okabe_ito_on_white'):
        plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'font.size':8,
             'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'path',
             'figure.constrained_layout.w_pad':.035,'figure.constrained_layout.h_pad':.035})
        geometry();framework();comparison();costs();ablations()
        trajectory('median',6);trajectory('state_worst',7);trajectory('rl_worst',8)
    print('Exported eight publication figures in PDF/SVG/PNG with provenance manifests.')
if __name__=='__main__':main()
