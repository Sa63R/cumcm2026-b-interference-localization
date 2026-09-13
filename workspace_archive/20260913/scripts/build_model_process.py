"""Build a narrative edition with interleaved geometric process illustrations."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tempfile

import rebuild_model_pdf as markdown

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/新版_过程讲解图'
OLD=ROOT/'output/pdf/问题一_定位区域模型.tex'
SOURCE=ROOT/'问题一_定位区域模型.md'
TEX=OUT/'问题一_过程讲解版.tex'


def figure(command, lead, caption, label):
    return ('\n% BEGIN PROCESS FIGURE\n'
            + r'\begin{center}\begin{minipage}{\linewidth}' + '\n'
            + r'\noindent '+lead.replace('图示', '图~\\ref{'+label+'}~') + '\n'
            + r'\par\medskip\centering' + '\n'
            + '\\'+command+'\n'
            + r'\captionof{figure}{'+caption+'}\n'
            + r'\label{'+label+'}\n'
            + r'\end{minipage}\end{center}'+'\n% END PROCESS FIGURE\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    previous=[OLD,OLD.with_suffix('.pdf'),ROOT/'output/pdf/新版_含示意图/问题一_新版_含示意图.tex',ROOT/'output/pdf/新版_含示意图/问题一_新版_含示意图.pdf']
    preserved={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in previous}
    snapshot=OUT/'问题一_原稿.md'
    if not snapshot.exists():shutil.copy2(SOURCE,snapshot)
    source=snapshot.read_text()
    body=markdown.convert(source)
    for line in source.splitlines():
        if line.startswith('> '):
            body=body.replace(markdown.inline(line),r'\begin{quote}'+'\n'+markdown.inline(line[2:])+'\n'+r'\end{quote}')
    clean=re.sub(r'\\addcontentsline\{toc\}\{(?:sub)?section\}\{[^{}]*\}','',body)
    han=lambda t:''.join(re.findall('[\u4e00-\u9fff]',t))
    assert han(source.split('\n',1)[1])==han(clean)

    def before(anchor, block):
        nonlocal body
        assert anchor in body,anchor
        body=body.replace(anchor,block+'\n'+anchor,1)

    before(markdown.inline('将式 (1) 展开，一次测向对应两个线性不等式：'),
        figure('DirectionToConstraints','图示将示向度、误差角域与两条约束对应起来。',
               '从示向度到半平面约束的转换。误差角作放大示意；实际计算仍取 $\\varepsilon=1^\\circ$。', 'fig:direction-process'))
    before(r'\subsection*{5.2 定位区域及其直径的求解}',
        figure('ProgressiveIntersection','图示按测向信息增加的顺序，展示公共区域怎样形成并继续收缩。',
               '逐次求交的过程：每次只保留已有区域中同时满足新约束的部分。图中角度和坐标为过程示意。','fig:intersection-process'))
    before(markdown.inline('**第二步，求顶点。**'),
        figure('FeasibilityCases','图示说明为什么必须先分类，再进入有限直径的计算。',
               '可行性与有界性决定下一步操作。空集与无界集都不能直接套用有限顶点对的直径公式。','fig:feasibility-process'))
    before(markdown.inline('**第三步，求直径。**'),
        figure('VertexExtraction','图示把交点处理展开为“枚举、约束检验、去重”三个动作。',
               '顶点提取的三个步骤。图中用一个示意多边形展示绘图窗口内的候选交点；橙色空心点为候选，红色叉号为不合格点，绿色实心点为保留点。','fig:vertex-process'))
    before(markdown.inline('综上，问题一的求解流程为：'),
        figure('DiameterReduction','图示将式 (5) 的结论转化为可以逐项执行的距离比较。',
               '由连续点对转为有限顶点对。灰线表示待比较的顶点连线，红线表示最远顶点对，其长度即区域直径。','fig:diameter-process'))
    before(markdown.inline('式 (6) 是否总成立？我们用下面的算例说明它可以不成立。取两个检测点'),
        figure('CoverageSteps','图示将覆盖判据与失败后的处理连在一起；最小包围圆的求法在第 5.4 节展开。',
               '覆盖判断的操作顺序：先锁定圆心，再检查每个顶点；若漏点，则改求最小包围圆。几何形状为定性示意，不采用表 1 的数值比例。','fig:coverage-process'))

    body=body.replace(r'\section*{一、问题背景与重述}',r'\clearpage'+'\n'+r'\section*{一、问题背景与重述}',1)
    body=body.replace(r'\section*{三、模型假设}',r'\Needspace{10\baselineskip}'+'\n'+r'\section*{三、模型假设}',1)
    body=body.replace(r'\section*{六、问题一小结}',r'\Needspace{14\baselineskip}'+'\n'+r'\section*{六、问题一小结}',1)

    # Exactly clip P2 by the third illustrative wedge; all three stages are consistent.
    poly=[(2,.6),(3.2,.96),(2,2.4),(.8,.96)]
    def clip(p,a,b,c):
        result=[]
        for s,e in zip(p,p[1:]+p[:1]):
            fs=a*s[0]+b*s[1]-c;fe=a*e[0]+b*e[1]-c
            if fs<=1e-9 and fe<=1e-9:result.append(e)
            elif (fs<=1e-9)!=(fe<=1e-9):
                t=fs/(fs-fe);result.append((s[0]+t*(e[0]-s[0]),s[1]+t*(e[1]-s[1])))
                if fe<=1e-9:result.append(e)
        return result
    p3=clip(clip(poly,.08,-1,-1.16),-.15,1,1.3)
    third='--'.join(f'({x:.10f},{y:.10f})' for x,y in p3)+'--cycle'
    assert len(p3)==4
    data=r'\def\ThirdIntersectionPath{'+third+'}\n'

    title=source.splitlines()[0][2:]
    preamble=OLD.read_text().split(r'\begin{document}',1)[0]
    preamble=re.sub(r'\\hypersetup\{pdftitle=\{[^{}]*\}\}',lambda _:r'\hypersetup{pdftitle={'+title+'}}',preamble)
    preamble=re.sub(r'\\fancyhead\[L\]\{\\small [^{}]*\}',lambda _:r'\fancyhead[L]{\small 无源测向快速定位模型}',preamble)
    preamble+='\n% BEGIN PROCESS DRAWING DEFINITIONS\n'+data+(ROOT/'scripts/model_process_figures.tex').read_text()+'\n% END PROCESS DRAWING DEFINITIONS\n'
    title_tex=markdown.inline(title).replace('的无源',r'的\par\vspace{0.25em}无源',1)
    tex=(preamble+'\n'+r'\begin{document}\thispagestyle{plain}'+'\n'
         +r'\begin{center}{\LARGE\bfseries '+title_tex+r'\par}\end{center}'+'\n'
         +'% BEGIN ORIGINAL BODY WITH PROCESS FIGURES\n'+body+'\n% END ORIGINAL BODY\n'+r'\end{document}'+'\n')
    formulas=re.findall(r'\$\$\s*\n(.*?)\n\$\$',source,re.S)
    assert all(f in tex for f in formulas)
    assert re.findall(r'\\tag\{(\d+)\}',source)==re.findall(r'\\tag\{(\d+)\}',tex)
    TEX.write_text(tex)
    scratch=Path(tempfile.mkdtemp(prefix='model-process-',dir='/private/tmp'))
    print('QA directory:',scratch,flush=True)
    args=['/Library/TeX/texbin/xelatex','-interaction=nonstopmode','-halt-on-error','-file-line-error','-output-directory='+str(scratch),str(TEX)]
    for n in (1,2):
        with (scratch/f'build-{n}.txt').open('w') as out:
            result=subprocess.run(args,cwd=OUT,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:
            print((scratch/f'build-{n}.txt').read_text()[-5000:]);raise RuntimeError('LaTeX failed')
    log=(scratch/(TEX.stem+'.log')).read_text()
    issues=re.findall(r'^.*(?:Overfull|Underfull|Missing character|Warning|undefined).*$' ,log,re.M)
    print('\n'.join(issues))
    pdf=scratch/TEX.with_suffix('.pdf').name
    subprocess.run(['/opt/homebrew/bin/pdftoppm','-r','100','-png',str(pdf),str(scratch/'page')],check=True)
    subprocess.run(['/opt/homebrew/bin/pdftotext',str(pdf),str(scratch/'extracted.txt')],check=True)
    for rel,digest in preserved.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest
    shutil.copy2(pdf,TEX.with_suffix('.pdf'))
    report=dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),preserved_files=preserved,
                figure_count=6,formula_count=len(formulas),warnings=issues,qa_directory=str(scratch),
                diagrams='geometric process illustrations interleaved with narrative',third_intersection=p3)
    (OUT/'build_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
