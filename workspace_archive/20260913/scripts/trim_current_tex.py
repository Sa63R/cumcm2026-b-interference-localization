"""Apply the requested TeX-only reduction to Section 5.1--5.3."""
from pathlib import Path
from datetime import datetime
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/当前版'
PROJECT = ROOT / 'output/overleaf/项目本地待同步'
MCP = Path('/Users/zephyrr/.codex/mcp/overleaf-git-mcp/repos/6aa28083141219d8d2d1bdb1')
SENTENCE = r'楔形张角仅 $2^\circ$，检测点背后的方向不满足这两个不等式，不需要另行排除。'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def compile_pdf(directory, name, result):
    qa = ROOT / 'tmp/pdfs' / name
    qa.mkdir(parents=True, exist_ok=True)
    for pass_number in (1, 2):
        r = subprocess.run(['/Library/TeX/texbin/xelatex', '-interaction=nonstopmode',
            '-halt-on-error', '-file-line-error', '-output-directory=' + str(qa), 'main.tex'],
            cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (qa / f'build-{pass_number}.txt').write_text(r.stdout)
        if r.returncode:
            print(r.stdout[-4000:]); raise RuntimeError('Compilation failed')
    log = (qa / 'main.log').read_text()
    warnings = re.findall(r'^.*(?:Overfull|Underfull|Missing character|Warning|undefined).*$', log, re.M)
    assert not warnings, warnings
    shutil.copy2(qa / 'main.pdf', result)
    subprocess.run(['/opt/homebrew/bin/pdftoppm', '-r', '95', '-png', str(result), str(qa / 'page')], check=True)
    return qa


def main():
    md_hash = sha(ROOT / '问题一_定位区域模型.md')
    q2_hash = sha(PROJECT / 'q2.tex')
    backup = ROOT / 'output/backups' / ('tex_before_section5_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.zip')
    backup.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in (OUT, PROJECT):
            for path in folder.rglob('*'):
                if path.is_file(): z.write(path, str(path.relative_to(ROOT)))

    original = (OUT / 'main.tex').read_text()
    preamble = original.split(r'\begin{document}', 1)[0]
    preamble = preamble.replace('% This file preserves the supplied text, mathematics, data and section numbering.',
        '% User-selected excerpt: only Section 5.1--5.3; original Markdown unchanged.')
    preamble = re.sub(r'\\hypersetup\{pdftitle=\{[^{}]*\}\}',
        lambda _: r'\hypersetup{pdftitle={问题一：模型的建立与求解}}', preamble)
    body = original[original.index(r'\section*{五、模型的建立与求解}'):]
    body = body.split(r'\subsection*{5.4 最小包围圆模型}', 1)[0].split(r'\end{document}', 1)[0].rstrip()
    body = body.replace(SENTENCE, '')
    body = body.replace('绿色虚线为第 5.4 节求得的最小包围圆。', '')
    tex = preamble + '\n' + r'\begin{document}\thispagestyle{plain}' + '\n' + body + '\n' + r'\end{document}' + '\n'
    assert re.findall(r'\\section\*\{([^}]+)\}', tex) == ['五、模型的建立与求解']
    assert len(re.findall(r'\\subsection\*\{', tex)) == 3
    assert SENTENCE not in tex and '5.4' not in tex and '5.5' not in tex
    assert re.findall(r'\\tag\{(\d+)\}', tex) == [str(n) for n in range(1, 7)]
    (OUT / 'main.tex').write_text(tex)

    figure = (OUT / 'figures/q1_case.tex').read_text()
    figure = re.sub(r'\s*\\draw\[qonecover,.*?;', '', figure, flags=re.S)
    figure = re.sub(r'\s*\\node\[[^\]]*qonecover[^\]]*\].*?;', '', figure, flags=re.S)
    figure = figure.replace('diameter circle, and enclosing circle.', 'diameter circle.')
    assert '最小包围圆' not in figure and '$C^*$' not in figure
    (OUT / 'figures/q1_case.tex').write_text(figure)

    q1 = (PROJECT / 'q1.tex').read_text()
    q1 = q1.split(r'\subsubsection{最小包围圆模型}', 1)[0].rstrip() + '\n'
    q1 = q1.replace(SENTENCE, '')
    q1 = q1.replace(r'绿色虚线为第\ref{sec:q1-4}节求得的最小包围圆。', '')
    assert 'sec:q1-4' not in q1 and 'sec:q1-5' not in q1
    (PROJECT / 'q1.tex').write_text(q1)
    shutil.copy2(OUT / 'main.tex', PROJECT / '问题一_当前版.tex')
    shutil.copy2(OUT / 'figures/q1_case.tex', PROJECT / 'figures/q1_case.tex')

    main_qa = compile_pdf(OUT, 'section5_only', OUT / 'main.pdf')
    project_qa = compile_pdf(PROJECT, 'section5_integrated', OUT / '整合主稿_本地预览.pdf')
    assert sha(ROOT / '问题一_定位区域模型.md') == md_hash
    assert sha(PROJECT / 'q2.tex') == q2_hash
    report = {'source_markdown_unchanged': True, 'retained_sections': ['5.1', '5.2', '5.3'],
        'removed_sections': ['摘要', '一', '二', '三', '四', '5.4', '5.5', '六'],
        'requested_sentence_removed': True, 'stale_figure_reference_removed': True,
        'numbered_equations': list(range(1, 7)), 'warnings': [], 'visual_review_passed': False,
        'tex_sha256': sha(OUT / 'main.tex'), 'pdf_sha256': sha(OUT / 'main.pdf'),
        'backup': str(backup), 'pushed': False}
    (OUT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    readme = '''# 问题一第五部分：模型的建立与求解

主文件 main.tex，仅保留第五部分的 5.1、5.2、5.3，包含算例图。
用户指定的“楔形张角仅 2 度……不需要另行排除”一句已删除。
原 Markdown 未改动；修改前的完整 TeX/PDF 已备份。
使用 XeLaTeX 编译两次。图源为 figures/q1_case.tex，无外部图片依赖。
以后若从 Markdown 重新生成完整文稿，需再运行 scripts/trim_current_tex.py 应用本次删减。
'''
    (OUT / 'README.md').write_text(readme)
    manifest_path = PROJECT / '同步清单.json'
    manifest = json.loads(manifest_path.read_text())
    for name in manifest['update_files']: manifest['update_files'][name] = sha(PROJECT / name)
    manifest.update(pushed=False, warnings=[], visual_review_passed=False,
        pdf_sha256=sha(OUT / '整合主稿_本地预览.pdf'), q1_retained_subsections=['5.1', '5.2', '5.3'])
    manifest.pop('pdf_pages', None)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    patch = ''
    for name in manifest['update_files']:
        old = (MCP / name).read_text() if (MCP / name).exists() else ''
        patch += ''.join(difflib.unified_diff(old.splitlines(True), (PROJECT / name).read_text().splitlines(True),
            fromfile='a/' + name, tofile='b/' + name))
    (PROJECT / '本地更新.diff').write_text(patch)
    (PROJECT / 'README.md').write_text('''# Overleaf 本地待同步版本

尚未推送。问题一独立文件“问题一_当前版.tex”仅保留第五部分的 5.1 至 5.3。
q1.tex 同步删除了最小包围圆、补充说明、小结及用户指定句子；图中也移除了对应的包围圆与章节引用。
项目总稿中的其他部分和 q2.tex 保留。图源为 figures/q1_case.tex，可用现有文本 MCP 同步。
同步前须通过 MCP 读取远端最新文件，与同步清单中的基线比较并合并后再推送。
''')
    archives = [(ROOT / 'output/overleaf/问题一_当前版.zip', OUT,
                 ['main.tex', 'figures/q1_case.tex', 'latexmkrc', 'README.md', 'source_original.md']),
                (ROOT / 'output/overleaf/Overleaf项目_本地待同步.zip', PROJECT,
                 [*manifest['update_files'], 'q2.tex', 'README.md', '同步清单.json', '本地更新.diff'])]
    for archive, directory, names in archives:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in names: z.write(directory / name, name)
        with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('QA:', main_qa, project_qa)


if __name__ == '__main__':
    main()
