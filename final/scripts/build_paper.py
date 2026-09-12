"""Build the editable manuscript and PDF from reviewed section sources."""
from pathlib import Path
import re, json, subprocess, shutil
import pypandoc

OUT=Path(__file__).resolve().parents[1]
SRC=OUT/'sources'; QA=OUT/'qa/latex';QA.mkdir(parents=True,exist_ok=True)
def source(name):return (SRC/name).read_text(encoding='utf-8')
def clean(s):
    s=re.sub(r'<!--.*?-->','',s,flags=re.S)
    s=re.sub(r'\[E\d+[^\]]*\]','',s)
    return s.strip()
def figure(name,caption,width='100%'):
    return f'\n\n![{caption}](figures/{name}.pdf){{width={width}}}\n\n'

q2=clean(source('q2_methods.md'))
q2=q2.replace('## 问题二：','# 问题二 ').replace('### 2.','## 2.')
q2=q2.split('FIG:q2_geometry')[0]
q2+=figure('fig01_q2_geometry','图1 首测可行域及横向交会的解析示意。半角采用1.005°，蓝色测点域由三角外包构造，属于充分保证接收域；示意源p仅用于解释交会几何，不作为规划输入。右图展示另一比例下的局部几何，不能作为算法性能比较。')
q2+='\n以上构造同时给出了第二检测点的候选区域与可执行选择规则。其理论保证在于真实源保留、候选点可接收以及清除条件可靠；选点效率仍须由实际后续测量检验。本稿已有实验主要评估完整第三问，尚不把整局节省量当成第二检测点单独优越性的证据。\n'

q3=clean(source('q3_methods.md'))
q3=q3.replace('## 问题三：','# 问题三 ').replace('### 3.','## 3.')
q3=q3.split('FIG:framework')[0]
q3=q3.replace('将不同访问顺序压缩成相同的','借鉴子集动态规划[1]，将不同访问顺序压缩成相同的')
q3=q3.replace('为优先探索有希望的前缀','采用A*启发式图搜索[2]，为优先探索有希望的前缀')
insert=figure('fig02_framework','图2 初始发现覆盖与滚动决策框架。七个1000 m接收保证圆覆盖半径1800 m场地；未来站移动后仍需重新通过全域覆盖核验。')
q3=q3.replace('### 3.3','### 3.3') # Sections have already been normalized.
q3=q3.replace('## 3.3',insert+'## 3.3',1)
methods=clean(source('comparison_methods.md')).replace('### 3.7','## 3.7')

exp=clean(source('q3_experiments.md'))
exp=exp[exp.index('## 实验设计'):]
exp=exp.split('## 配图与证据衔接')[0]
exp=exp.replace('## 实验设计与比较方法','## 3.8 实验设计与评价指标')
exp=exp.replace('## 四种方法的独立比较','## 3.9 四种方法的独立比较')
exp=exp.replace('## 哪些改进获得了独立证据','## 3.10 核心策略的消融分析')
exp=exp.replace('## 强化学习的训练与负结果','## 3.11 强化学习的训练与探索结果')
exp=exp.replace('## 行为诊断与官方演练','## 3.12 学习行为与官方演练验证')
exp=exp.replace('以完整清除率和失败清除次数检查可行性','以全清局数占总场景数的比例和失败清除次数检查可行性')
exp=exp.replace('正值表示候选较参照节省时间。','正值表示候选较参照节省时间。区间描述冻结策略面对场景抽样的变动，并不涵盖重新训练网络的随机性；这一范围与强化学习评估中的不确定性要求一致[5]。')
exp=exp.replace('## 3.9',r'''为便于区分指标，令第 $i$ 个场景的基线与候选耗时分别为 $T_i^{(b)}$ 和 $T_i^{(m)}$，定义

$$
\overline T_m=\frac1n\sum_{i=1}^nT_i^{(m)},\qquad
\overline\Delta_m=\frac1n\sum_{i=1}^n(T_i^{(b)}-T_i^{(m)}),\qquad
G_m=\frac{\overline\Delta_m}{\overline T_b}.
\tag{3-17}
$$

全清失败或超时必须计入失败记录，不能只对成功子集报告速度优势。表中“284/284”是256个随机场景与28个压力场景全部全清，两种分布的时间统计仍分别计算。

## 3.9''',1)
exp=exp.replace('算法计算速度与机器人执行效率需要区分。',figure('fig03_comparison','图3 首版四方法在同一256随机场景上的均时与配对节省。右图为相对前瞻基线的95%配对bootstrap区间，不能当作左图各均时的区间。')+figure('fig04_costs','图4 首版随机集的平均费用分解。分项之和等于整局计费用时；移动是主要成本，但扫描与换频的节省同样影响总结果。')+'算法计算速度与机器人执行效率需要区分。')
exp=exp.replace('## 3.11',figure('fig05_ablations','图5 核心组件及训练配置的消融结果。正值表示节省，各行以各自冻结参照计算；上部的64局和256局来自不同实验批次，下部为48局开发结果。不同批次收益不作相加。')+'## 3.11',1)
exp=exp.replace('每次继承仅迁移权重并重建优化器，不能写成同一个优化器连续训练。','')
exp=exp.replace('本批固定跑满 50 局，不包含此前按源数组织的 40 局。','本批预定跑满50局。')
exp=exp.replace('；若先计算每局 T/N 再等权平均，则为 229.84 s/源','')
exp=exp.replace('官方表中下界采用历史观测条件下的区域路线松弛，并含空频道必要动作项，与合成配对的纯先知物理下界不同；本文主比较采用实际耗时，不混用下界比值。','本文统一报告可直接核验的实际耗时；各历史报告的下界口径在辅助证据索引中保留，不用于跨批次排名。')

trajectory=r'''
## 3.13 同场景轨迹与退化机制

为了将费用差异对应到具体行动，图6选取首版基线用时最接近256局中位数的场景800035；选择仅依据基线，不依据其他方法的输赢。四种方法使用相同的源位置、接收半径与噪声场，全部清除11个源。在此例中PPO用时2943.02 s，短于状态搜索的3118.22 s，与总体均值的排序不同。这说明代表性轨迹用于解释行为差异，不能替代全部场景上的统计判断。

状态搜索和PPO采用了不同的清除顺序与测点。PPO在该例移动耗时2205.02 s、测量116次；状态搜索移动耗时2311.22 s、测量127次。两项费用均有降低，因而共同解释本例PPO的优势。轨迹中的源坐标只在策略终止后用于展示，规划时不使用这些真值。
'''+figure('fig06_trajectory_median','图6 基线中位耗时附近场景800035的四方法真实轨迹。统一坐标比例，箭头表示移动方向，源旁数字为频道，面板底部列出实际清除顺序。同坐标多次测量合并显示位置，但测量次数保留原始计数。')+r'''
为展示方法的边界，附录还给出各方法相对基线退化最大的案例。状态搜索在800015场景中增加152.08 s移动费用，虽少用25 s测量和2 s换频，最终仍多用125.08 s；PPO在800081场景中多用340.45 s移动、20 s测量和5 s换频，最终多用365.45 s。这些事后选择的反例揭示了长距离折返的代价，不用于估计其总体发生频率，也不据此声称已经识别出网络内部决策的因果原因。

## 3.14 结果讨论与适用范围

现有结果支持在本题静止、全向、有限频道且存在可靠几何约束的条件下，以结构化规划作为有效主方案。状态压缩合并当前任务访问顺序，主动定位控制单源信息成本，静默证书删去可证明冗余的查询，覆盖约束负责完整发现。主实验中这一组合取得较低平均任务用时；后期PPO接近状态搜索，则表明学习调度仍具有竞争力。

这种优势不能归结为“固定点可以普遍代替强化学习”。主状态搜索已允许连续移动覆盖站，PPO也依赖人工候选；二者同时利用了题目结构。更大的状态表示、更充分的训练、不同源分布或移动源环境仍可能改变结论。有限任务模型的最优性、连续在线策略的最优性和官方演练的成功记录属于不同层次，本稿只在各自证据范围内提出判断。
'''

intro=r'''
# 符号说明

本文针对全向干扰源，将第二问的主动检测点选择作为第三问多源搜索的局部模块。先建立可保证接收和可靠清除的几何条件，再研究覆盖任务、已知源处理及测量费用的联合调度。下文的源位置可行域由观测构造，所有实验表中的用时均为题目动作计费时间，程序计算耗时单独说明。

| 符号 | 含义 | 符号 | 含义 |
|---|---|---|---|
| $\Omega$ | 半径1800 m目标圆域 | $f$ | 测向频道，1至20 |
| $p$ | 源坐标 | $x_t$ | 当前机器人位置 |
| $R_f$ | 固定但未知的接收半径 | $\bar\varepsilon$ | 含舍入余量的测角误差界 |
| $\widehat C_f$ | 源位置的凸外包区域 | $Q_f$ | 保证再次接收的测点域 |
| $z(C),\rho(C)$ | 最小包围圆的圆心与半径 | $v$ | 移速，5 m/s |
| $\mathcal D_t,\mathcal K_t$ | 已发现及已清除频道集合 | $M,j$ | 任务位掩码与最后任务 |
| $L,T$ | 实际路长与计费用时 | $U_t,L_t$ | 冻结规划问题的上界与下界 |

'''

refs=r'''
# 参考文献

[1] Held M, Karp R M. A Dynamic Programming Approach to Sequencing Problems. Journal of the Society for Industrial and Applied Mathematics, 1962, 10(1):196–210. doi:10.1137/0110015.

[2] Hart P E, Nilsson N J, Raphael B. A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Transactions on Systems Science and Cybernetics, 1968, 4(2):100–107. doi:10.1109/TSSC.1968.300136.

[3] Schulman J, Wolski F, Dhariwal P, et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.

[4] Schulman J, Moritz P, Levine S, et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. International Conference on Learning Representations, 2016.

[5] Agarwal R, Schwarzer M, Castro P S, et al. Deep Reinforcement Learning at the Edge of the Statistical Precipice. Advances in Neural Information Processing Systems, 2021, 34:29304–29320.

# 附录A 实验来源与复核口径

各批次的代码、权重身份与来源SHA256见辅助材料 `sources/evidence_experiments.json`。下表仅用于定位证据，原始研究目录保留完整动作记录；本轮论文整理未重新运行策略或训练模型。

| 证据 | 实验批次与样本 | 用途与边界 |
|---|---|---|
| E01至E03 | 首版256随机与28压力 | 四方法完整方案对比，分布分开 |
| E04 | 每方法32局串行计时 | 已用开发场景上的计算开销 |
| E05、E06 | 各64局冻结后确认 | 上限静默、覆盖移站单组件增益 |
| E07 | 新256随机及28压力 | 相对静默、16源上限及组合 |
| E08至E10 | 训练链及开发端点 | PPO预算、GAE与失败探索 |
| E11、E12 | 后期48开发场景 | RL配对结果与原点全扫诊断 |
| E13 | 既有50局官方演练 | 冻结组合版的接口与执行验证 |

图1、图2为解析或算法示意；图3至图8来源于既有数值记录。图6按基线中位耗时选取，图7和图8为事后最大退化例，不混入新增独立测试结论。上述随机评估沿用其原始冻结与数据分割记录；不将已经用于开发的场景重新称为未见测试。

写作组织参考了官方公开的2018年国赛B217与A440两篇匿名展示论文，其出处与获奖展示依据保存在 `references/references.md`。它们只作为表达范例，本题公式和成绩均以本项目实现及实验为依据。绘图使用Matplotlib及Scientific Visualization skill，其版本、软件文献和导出清单见配套README与参考资料。

# 附录B 退化案例的轨迹对照

图7为状态搜索相对前瞻基线退化最大的首版随机场景。图8为PPO相对前瞻基线退化最大的场景。两图均显示全部四方法，不因展示负例而省略其他方法结果。
'''+figure('fig07_trajectory_state_worst','图7 状态搜索最大退化场景800015。状态搜索虽减少检测，却因移动增长而较基线慢125.08 s；此例为事后诊断选择。')+figure('fig08_trajectory_rl_worst','图8 PPO最大退化场景800081。该例主要额外代价来自移动，PPO较基线慢365.45 s；此例不表示总体发生频率。')

content='\n\n'.join([intro,q2,q3,methods,exp,trajectory,refs])
content=content.replace('x∈A','x 属于 A').replace('`sources/evidence_experiments.json`','结构化实验证据索引')
(OUT/'第二问与第三问论文稿.md').write_text('# 全向干扰源的主动定位与状态搜索\n\n'+content,encoding='utf-8')
body=pypandoc.convert_text(content,'latex',format='markdown+tex_math_dollars+implicit_figures',extra_args=['--wrap=none','--syntax-highlighting=none'])
body=body.replace('\r\n','\n').replace('\r','\n')
body=body.replace(r'\def\LTcaptype{none}', '')
# Keep citations as prose labels instead of ordered lists; preserve explicit equation tags.
body=body.replace('width=1\\linewidth','width=\\linewidth')
preamble=r'''\documentclass[UTF8,zihao=-4,fontset=windows]{ctexart}
\usepackage[a4paper,top=23mm,bottom=23mm,left=23mm,right=23mm]{geometry}
\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage{graphicx,xcolor,longtable,booktabs,array,calc,multirow}
\usepackage{caption,float,placeins,fancyhdr,enumitem,fvextra}
\usepackage[hidelinks,unicode]{hyperref}
\usepackage{microtype}
\setmainfont{Times New Roman}
\setmonofont{Consolas}
\setCJKmainfont{SimSun}[BoldFont=SimHei]
\setCJKsansfont{Microsoft YaHei}
\setCJKmonofont{FangSong}
\linespread{1.18}
\setlength{\parindent}{2em}
\setlength{\parskip}{1.5pt}
\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.25}
\setlength{\LTpre}{7pt}\setlength{\LTpost}{7pt}
\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}
\renewcommand{\topfraction}{0.92}\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.06}\renewcommand{\floatpagefraction}{0.65}
\makeatletter
\setlength{\@fptop}{0pt}\setlength{\@fpsep}{14pt}\setlength{\@fpbot}{0pt plus 1fil}
\makeatother
\captionsetup{font=small,labelformat=empty,skip=6pt,justification=justified}
\setlist{nosep,leftmargin=2em}
\ctexset{section={format=\Large\bfseries,beforeskip=16pt,afterskip=9pt},subsection={format=\large\bfseries,beforeskip=12pt,afterskip=6pt}}
\setcounter{secnumdepth}{-1}
\pagestyle{fancy}\fancyhf{}
\fancyhead[C]{\small 全向干扰源的主动定位与状态搜索}
\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0.3pt}
\setlength{\headheight}{15pt}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\DefineVerbatimEnvironment{verbatim}{Verbatim}{fontsize=\small,breaklines=true,breakanywhere=true}
\AtBeginEnvironment{longtable}{\small}
\begin{document}
\begin{center}
{\LARGE\bfseries 全向干扰源的主动定位与状态搜索}\\[7pt]
{\large 第二问与第三问论文稿}
\end{center}
\vspace{4pt}
'''
# Before new major sections flush floats; preserve connected prose within each problem.
body=body.replace('\\section{参考文献}', '\\FloatBarrier\n\\section{参考文献}')
body=body.replace('\\begin{figure}', '\\begin{figure}[!htbp]')
body=body.replace('\\section{附录B', '\\clearpage\n\\section{附录B')
tex=preamble+body+'\n\\FloatBarrier\n\\end{document}\n'
(OUT/'第二问与第三问论文稿.tex').write_text(tex,encoding='utf-8')
cmd=['xelatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',f'-output-directory={QA}',str(OUT/'第二问与第三问论文稿.tex')]
for i in range(2):
    run=subprocess.run(cmd,cwd=OUT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (OUT/f'qa/latex_run_{i+1}.txt').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
    if run.returncode:print(run.stdout[-7000:]);raise SystemExit(run.returncode)
shutil.copy2(QA/'第二问与第三问论文稿.pdf',OUT/'第二问与第三问论文稿.pdf')
print('Built manuscript MD, editable LaTeX and PDF.')
