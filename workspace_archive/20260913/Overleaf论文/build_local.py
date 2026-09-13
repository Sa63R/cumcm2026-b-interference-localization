#!/usr/bin/env python3
"""Build the manuscript using a disposable copy and the same TeX Live fonts."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
BUILD = ROOT / '.build'
FONT_NAMES = {
    r'\setmainfont{TeX Gyre Termes}':
        r'\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]',
    r'\setsansfont{TeX Gyre Heros}':
        r'\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]',
    r'\setmonofont{Latin Modern Mono}':
        r'\setmonofont{lmmono10-regular.otf}[ItalicFont=lmmono10-italic.otf]',
}


def main():
    search_path = os.environ.get('PATH', '') + os.pathsep + '/Library/TeX/texbin'
    latexmk = shutil.which('latexmk', path=search_path)
    xelatex = shutil.which('xelatex', path=search_path)
    if not latexmk or not xelatex:
        sys.exit('需要安装 MacTeX / TeX Live，确保 latexmk 和 xelatex 可用。')
    BUILD.mkdir(exist_ok=True)
    # A private staging copy prevents concurrent builds from editing each other's inputs.
    with tempfile.TemporaryDirectory(prefix='paper-', dir=BUILD) as temporary:
        source = Path(temporary)
        shutil.copytree(ROOT, source, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('.git', '.build', '__pycache__', '.DS_Store',
                                                     '25年国一论文用来参考格式'))
        for tex in source.rglob('*.tex'):
            original = tex.read_text(encoding='utf-8')
            adapted = original
            for name, filename in FONT_NAMES.items():
                adapted = adapted.replace(name, filename)
            if adapted != original:
                tex.write_text(adapted, encoding='utf-8')
        env = dict(os.environ, PATH=search_path)
        result = subprocess.run([latexmk, '-xelatex', '-interaction=nonstopmode',
                                 '-halt-on-error', '-file-line-error', 'main.tex'],
                                cwd=source, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        (BUILD / 'build.log').write_text(result.stdout, encoding='utf-8')
        for name in ('main.log', 'main.aux'):
            if (source / name).exists():
                shutil.copy2(source / name, BUILD / name)
        if result.returncode:
            print(result.stdout[-5000:])
            sys.exit('编译失败，详见 .build/build.log；已有 PDF 不代表本次编译成功。')
        shutil.copy2(source / 'main.pdf', BUILD / 'main.pdf')
        print('编译完成：' + str(BUILD / 'main.pdf'))
        print('编译日志：' + str(BUILD / 'main.log'))


if __name__ == '__main__':
    main()
