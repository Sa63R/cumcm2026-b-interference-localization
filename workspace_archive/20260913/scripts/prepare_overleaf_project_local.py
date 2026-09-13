"""Prepare, but do not push, the Q1 update against the configured MCP checkout."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = '6aa28083141219d8d2d1bdb1'
BASE = Path('/Users/zephyrr/.codex/mcp/overleaf-git-mcp/repos') / PROJECT_ID
OUT = ROOT / 'output/overleaf/项目本地待同步'
STANDALONE = ROOT / 'output/pdf/当前版'
QA = ROOT / 'tmp/pdfs/overleaf_project_preview'
sha = lambda data: hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    baseline = {name: (BASE / name).read_bytes() for name in ('main.tex', 'q1.tex', 'q2.tex')}
    assert not subprocess.check_output(['git', '-C', str(BASE), 'status', '--porcelain']).strip(), 'MCP checkout contains existing work.'
    base_commit = subprocess.check_output(['git', '-C', str(BASE), 'rev-parse', 'HEAD'], text=True).strip()

    standalone = (STANDALONE / 'main.tex').read_text()
    def section(name):
        start = standalone.index(r'\section*{' + name + '}')
        end = standalone.find(r'\section*{', start + 1)
        result = standalone[start:end if end >= 0 else standalone.index(r'\end{document}')]
        result = result.split('\n', 1)[1]
        result = re.sub(r'\\addcontentsline\{toc\}\{(?:sub)?section\}\{[^{}]*\}\n?', '', result)
        result = re.sub(r'\\Needspace\{\d+\\baselineskip\}\s*$', '', result)
        return result.strip()

    model = section('五、模型的建立与求解')
    headings = {
        '5.1 从测向数据到定位区域': ('从测向数据到定位区域', 'sec:q1-1'),
        '5.2 求定位区域的直径': ('求定位区域的直径', 'sec:q1-2'),
        '5.3 直径圆能否覆盖定位区域': ('直径圆能否覆盖定位区域', 'sec:q1-3'),
        '5.4 最小包围圆模型': ('最小包围圆模型', 'sec:q1-4'),
        '5.5 补充说明': ('补充说明', 'sec:q1-supplement'),
    }
    for original, (heading, label) in headings.items():
        assert r'\subsection*{' + original + '}' in model
        model = model.replace(r'\subsection*{' + original + '}',
                              r'\subsubsection{' + heading + r'}\label{' + label + '}')
    def equation(match):
        contents = match[1]
        tag = re.search(r'\\tag\{([1-8])\}', contents)
        if not tag:
            return match[0]
        contents = re.sub(r'\s*\\tag\{[1-8]\}', '', contents).strip()
        return r'\begin{equation}\label{eq:q1-' + tag[1] + '}\n' + contents + '\n' + r'\end{equation}'
    model = re.sub(r'\\\[(.*?)\\\]', equation, model, flags=re.S)
    model = re.sub(r'(式|模型)\s*\(([1-8])\)', lambda m: m[1] + r'\eqref{eq:q1-' + m[2] + '}', model)
    old_caption = r'\textbf{表 1 反例的检测数据（误差界 $\pm1^\circ$）}'
    assert old_caption in model
    model = model.replace(old_caption,
        r'\captionof{table}{反例的检测数据（误差界 $\pm1^\circ$）}\label{tab:q1-counterexample}')
    model = model.replace('表 1', r'表\ref{tab:q1-counterexample}')
    model = model.replace('第 5.4 节', r'第\ref{sec:q1-4}节')
    model = model.replace('fig:counterexample', 'fig:q1-counterexample')
    # Retain the integrated project's degree/radian distinction.
    model = model.replace(r'$\varepsilon=1^\circ=\pi/180$', r'$\delta=\pi\varepsilon/180=\pi/180$')
    model = model.replace(r'\varphi_i-\varepsilon', r'\varphi_i-\delta').replace(r'\varphi_i+\varepsilon', r'\varphi_i+\delta')
    q1 = ('% Updated from current Markdown; old standalone TeX is preserved.\n'
          + r'\subsection{问题一模型的建立与求解}\label{sec:q1}' + '\n'
          + model + '\n\n' + r'\subsubsection{问题一小结}\label{sec:q1-5}' + '\n'
          + section('六、问题一小结') + '\n')

    main_tex = baseline['main.tex'].decode()
    main_tex = main_tex.replace('% Working manuscript: Q1 integrated from teammate upload; Q2 is a draft.',
                                '% Working manuscript: Q1 updated from current Markdown; Q2 is preserved.')
    main_tex = main_tex.replace(r'\usepackage{tikz}', r'''\usepackage{tikz}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=quad,hypcap=false}
\input{figures/q1_case}''', 1)
    analysis = section('二、问题一的分析').replace('见 5.2 节', r'见第\ref{sec:q1-2}节')
    main_tex = re.sub(r'(\\subsection\{问题一的分析\}\n).*?(?=\\subsection\{问题二的分析\})',
                      lambda m: m[1] + analysis + '\n\n', main_tex, count=1, flags=re.S)
    start = main_tex.index(r'\section{模型假设}')
    stop = main_tex.index(r'\section{符号说明}')
    old_assumptions = main_tex[start:stop]
    items = re.findall(r'\\item .*?(?=\\item |\\end\{enumerate\})', old_assumptions, re.S)
    assumptions = section('三、模型假设')
    assumptions = assumptions.replace(r'\end{enumerate}', ''.join(items[3:]) + r'\end{enumerate}')
    main_tex = main_tex[:start] + r'\section{模型假设}' + '\n' + assumptions + '\n\n' + main_tex[stop:]

    (OUT / 'main.tex').write_text(main_tex)
    (OUT / 'q1.tex').write_text(q1)
    (OUT / 'q2.tex').write_bytes(baseline['q2.tex'])
    shutil.copy2(STANDALONE / 'figures/q1_case.tex', OUT / 'figures/q1_case.tex')
    shutil.copy2(STANDALONE / 'main.tex', OUT / '问题一_当前版.tex')
    changed = ('main.tex', 'q1.tex', 'figures/q1_case.tex', '问题一_当前版.tex')
    patch = ''.join(''.join(difflib.unified_diff(baseline.get(name, b'').decode().splitlines(True),
                     (OUT / name).read_text().splitlines(True), fromfile='a/' + name, tofile='b/' + name))
                    for name in changed)
    (OUT / '本地更新.diff').write_text(patch)

    args = ['/Library/TeX/texbin/xelatex', '-interaction=nonstopmode', '-halt-on-error',
            '-file-line-error', '-output-directory=' + str(QA), 'main.tex']
    for n in (1, 2):
        r = subprocess.run(args, cwd=OUT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (QA / f'build-{n}.txt').write_text(r.stdout)
        if r.returncode:
            print(r.stdout[-4000:]); raise RuntimeError('Integrated project failed to compile')
    log = (QA / 'main.log').read_text()
    warnings = re.findall(r'^.*(?:Overfull|Underfull|Missing character|Warning|undefined).*$', log, re.M)
    preview = ROOT / 'output/pdf/当前版/整合主稿_本地预览.pdf'
    shutil.copy2(QA / 'main.pdf', preview)
    subprocess.run(['/opt/homebrew/bin/pdftoppm', '-r', '90', '-png', str(preview), str(QA / 'page')], check=True)
    assert (OUT / 'q2.tex').read_bytes() == baseline['q2.tex']
    for name, content in baseline.items():
        assert (BASE / name).read_bytes() == content, 'MCP checkout was changed.'
    manifest = {
        'project_id': PROJECT_ID, 'base_commit': base_commit,
        'baseline_sha256': {name: sha(content) for name, content in baseline.items()},
        'update_files': {name: sha((OUT / name).read_bytes()) for name in changed},
        'q2_unchanged': True, 'mcp_checkout_unchanged': True, 'pushed': False,
        'warnings': warnings, 'visual_review_passed': False,
        'sync_note': 'Use configured Overleaf MCP; refresh remote before applying this local change set.'
    }
    (OUT / '同步清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'README.md').write_text('''# Overleaf 项目本地待同步版本

此目录尚未推送。主文档为 main.tex，编译器为 XeLaTeX。

已按既有项目结构更新问题一模型、问题一分析和对应假设；q2.tex 原样保留。
算例图在 figures/q1_case.tex 中用 TikZ 绘制，全部待同步文件均为 MCP 支持的 .tex。
问题一_当前版.tex 是完整 Markdown 转换的独立版本，原有问题一_定位区域模型.tex 保留。
原稿半径精确式笔误已修正为 65/6 m，根目录 Markdown 未改动。

同步前，通过既有 Overleaf MCP 读取最新 main.tex 与 q1.tex，与同步清单中的基线比较；
如远端已经变化，应先合并，再仅写入清单中的四个文件。验证差异后再调用 push_changes。
''')
    archive = ROOT / 'output/overleaf/Overleaf项目_本地待同步.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in (*changed, 'q2.tex', 'README.md', '同步清单.json', '本地更新.diff'):
            z.write(OUT / name, name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
