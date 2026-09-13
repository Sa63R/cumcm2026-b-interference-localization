"""Split a verified Overleaf snapshot without changing the expanded TeX source."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = Path('/Users/zephyrr/.codex/mcp/overleaf-git-mcp/repos/6aa28083141219d8d2d1bdb1')
OUT = ROOT / 'output/overleaf/分章节项目'
QA = ROOT / 'tmp/pdfs/overleaf_modular'


def git(*args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(REPO), *args])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not git('status', '--porcelain').strip(), 'Existing checkout work must be preserved.'
    head = git('rev-parse', 'HEAD').decode().strip()
    source = git('show', f'{head}:main.tex').decode()
    assert (REPO / 'main.tex').read_text() == source
    assert not OUT.exists(), f'Output already exists: {OUT}'
    OUT.mkdir(parents=True)
    QA.mkdir(parents=True, exist_ok=True)
    (QA / 'original_main.tex').write_text(source)
    (QA / 'before_tree.txt').write_bytes(git('ls-tree', '-r', '-z', head))

    files = {}
    def extract(text, start, stop, name):
        a = text.index(start)
        b = text.index(stop, a)
        files[name] = text[a:b]
        return text[:a] + r'\input{' + name.removesuffix('.tex') + '}%\n' + text[b:]

    main_tex = source
    main_tex = extract(main_tex, r'\usepackage[a4paper', r'\begin{document}', 'config/preamble.tex')
    main_tex = extract(main_tex, r'\pagenumbering{Roman}', '\\clearpage\n\\pagenumbering{arabic}', 'sections/abstract.tex')
    main_tex = extract(main_tex, r'\section{问题重述}', r'\section{问题分析}', 'sections/01_problem.tex')
    main_tex = extract(main_tex, r'\section{问题分析}', r'\section{模型假设}', 'sections/02_analysis.tex')
    main_tex = extract(main_tex, r'\section{模型假设}', '\\clearpage\n% 各问正文', 'sections/03_assumptions.tex')
    main_tex = extract(main_tex, '\\clearpage\n% 各问正文', r'\section{模型的建立与求解}', 'sections/04_symbols.tex')
    main_tex = extract(main_tex, r'\section{模型的建立与求解}', '% 增删文献时', 'sections/05_models.tex')
    main_tex = extract(main_tex, '% 增删文献时', '% 第二、三问附录', 'references/references.tex')
    main_tex = extract(main_tex, '% 第二、三问附录', '\\clearpage\n\\section*{附录B', 'appendices/a_evidence.tex')
    main_tex = extract(main_tex, '\\clearpage\n\\section*{附录B', r'\end{document}', 'appendices/b_trajectories.tex')

    models = files['sections/05_models.tex']
    models = extract(models, '% ===== 问题一正文 =====', '% ===== 问题二正文 =====', 'sections/q1.tex')
    models = extract(models, '% ===== 问题二正文 =====', '% ===== 问题三正文 =====', 'sections/q2.tex')
    start = models.index('% ===== 问题三正文 =====')
    files['sections/q3.tex'] = models[start:]
    files['sections/05_models.tex'] = models[:start] + '\\input{sections/q3}%\n'

    q3 = files['sections/q3.tex']
    parts = [
        ('依据观测更新搜索状态并计算任务总耗时', '设置对照方法并建立强化学习调度模型', '01_model_and_algorithm'),
        ('设置对照方法并建立强化学习调度模型', '比较四种方法的清除耗时与计算效率', '02_baselines_and_setup'),
        ('比较四种方法的清除耗时与计算效率', '比较强化学习训练配置与追加训练效果', '03_results_and_ablations'),
        ('比较强化学习训练配置与追加训练效果', '结合相同场景的移动轨迹分析耗时差异', '04_training_and_practice'),
    ]
    for start, stop, name in parts:
        q3 = extract(q3, r'\subsubsection{' + start + '}', r'\subsubsection{' + stop + '}', f'sections/q3/{name}.tex')
    start = q3.index(r'\subsubsection{结合相同场景的移动轨迹分析耗时差异}')
    files['sections/q3/05_trajectories_and_discussion.tex'] = q3[start:]
    files['sections/q3.tex'] = q3[:start] + '\\input{sections/q3/05_trajectories_and_discussion}%\n'
    files['main.tex'] = main_tex

    def expand(text):
        return re.sub(r'\\input\{([^}]+)\}%\n', lambda m: expand(files[m[1] + '.tex']), text)

    assert expand(main_tex) == source, 'Expanded source differs before comment updates.'
    comment_edits = {
        '% 全文均在本文件中，可直接搜索标题并手工修改文字、公式或图表。':
            '% 主编译文件；正文按章节存放在 sections/，全局格式位于 config/preamble.tex。\n'
            '% 摘要：sections/abstract.tex；一、二问：sections/q1.tex、sections/q2.tex。\n'
            '% 第三问由 sections/q3.tex 汇总，细分内容见 sections/q3/。\n'
            '% 文献：references/references.tex；附录：appendices/。\n'
            '% 编译器使用 XeLaTeX，主文件保持 main.tex；子文件通过 input 引入。',
        '% 可直接在此文件手工修改问题一正文，公式和图表编号由主文件统一管理。':
            '% 在本子文件修改问题一正文，公式和图表编号由主文件统一管理。',
        '% 问题三正文；修改文字、公式与图表请直接编辑本文件。':
            '% 本文件汇总第三问；各部分正文、公式与图表请编辑 sections/q3/ 下的对应子文件。',
    }
    for name, text in files.items():
        for old, new in comment_edits.items():
            text = text.replace(old, new)
        files[name] = text
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    expected = source
    for old, new in comment_edits.items():
        expected = expected.replace(old, new)
    assert expand(files['main.tex']) == expected, 'Expanded source changed beyond navigation comments.'

    # Retrieve original figure objects, including binaries omitted by MCP sparse checkout.
    figures = git('ls-tree', '-r', '--name-only', '-z', head, '--', 'figures').decode().split('\0')
    figure_hashes = {}
    for name in filter(None, figures):
        data = git('show', f'{head}:{name}')
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        figure_hashes[name] = sha(data)

    readme = '''# 分章节 Overleaf 项目

主文件为 `main.tex`，编译器为 **XeLaTeX**。在 Overleaf 中始终编译 main.tex；
章节文件只有正文，不含 documentclass 或 document 环境，不单独编译。

| 修改内容 | 文件 |
| --- | --- |
| 宏包、字体、页边距、页眉、编号格式 | config/preamble.tex |
| 标题、摘要、关键词 | sections/abstract.tex |
| 问题重述 | sections/01_problem.tex |
| 问题分析 | sections/02_analysis.tex |
| 模型假设 | sections/03_assumptions.tex |
| 符号说明 | sections/04_symbols.tex |
| 各问排列顺序与“模型的建立与求解”标题 | sections/05_models.tex |
| 第一问、第二问 | sections/q1.tex、sections/q2.tex |
| 第三问标题、引言与各部分顺序 | sections/q3.tex |
| 第三问模型与算法 | sections/q3/01_model_and_algorithm.tex |
| 第三问对照方法与实验设置 | sections/q3/02_baselines_and_setup.tex |
| 第三问结果与消融 | sections/q3/03_results_and_ablations.tex |
| 第三问训练与官方演练 | sections/q3/04_training_and_practice.tex |
| 第三问轨迹与适用条件讨论 | sections/q3/05_trajectories_and_discussion.tex |
| 参考文献 | references/references.tex |
| 实验来源附录、轨迹附录 | appendices/a_evidence.tex、appendices/b_trajectories.tex |
| 图片及第一问 TikZ 算例 | figures/ |

使用 `\\input` 保留原稿的连续排版、分页命令、公式编号、标签和引用。
所有图片和 input 路径均相对于项目根目录书写。文献仍使用原有 thebibliography，
添加条目时沿用 bibitem 和 cite，不需要引入 BibTeX/Biber。

本次只整理文件结构。原稿仅有前三问模型，第四问仍只出现于重述和分析，未补写内容。
根目录外的旧版本未用作同步来源；本目录来自同步时的 Overleaf 最新稿。
Overleaf 内已有的参考论文 PDF 保持原位置，本地源码包仅包含编译本文所需文件。
'''
    (OUT / 'README.md').write_text(readme)
    report = {
        'project_id': '6aa28083141219d8d2d1bdb1', 'base_commit': head,
        'original_sha256': sha(source.encode()), 'original_lines': len(source.splitlines()),
        'main_lines': len(files['main.tex'].splitlines()),
        'expanded_source_identical_except_navigation_comments': True,
        'files_to_sync': {name: sha(text.encode()) for name, text in files.items()},
        'unchanged_figures': figure_hashes,
    }
    (QA / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
