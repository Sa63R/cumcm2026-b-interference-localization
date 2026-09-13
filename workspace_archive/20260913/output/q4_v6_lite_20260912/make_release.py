"""Package verified code, reports and full evidence without changing old V6."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
EXCLUDE = {'__pycache__', '.matplotlib-cache', '.DS_Store', 'PACKAGE_SHA256.json'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def included(directory):
    return sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix != '.pyc'
                  and not any(part in EXCLUDE for part in p.relative_to(directory).parts))


def archive(directory, destination):
    files = included(directory)
    manifest = directory / 'PACKAGE_SHA256.json'
    manifest.write_text(json.dumps({p.relative_to(directory).as_posix(): digest(p) for p in files},
                                  ensure_ascii=False, indent=2) + '\n')
    files.append(manifest)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, arcname=(Path(directory.name) / p.relative_to(directory)).as_posix(),
                    compress_type=zipfile.ZIP_STORED if p.suffix in {'.gz', '.png', '.pdf'} else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None and len(z.namelist()) == len(files)
    return dict(path=str(destination), bytes=destination.stat().st_size,
                sha256=digest(destination), files=len(files), zip_crc_pass=True)


def main():
    import run_validation
    plan = json.loads((ROOT / 'plan.json').read_text())
    assert run_validation.verify_sources() == plan['source_hashes']
    assert json.loads((ROOT / 'replay_verification.json').read_text())['all_matched']
    assert json.loads((ROOT / 'serial/summary.json').read_text())['all_trace_hashes_equal']
    release = ROOT.parent / 'Q4_V6_Lite'
    release.mkdir()
    for name in ['source_v6_lite', 'reference/simulator', 'figures']:
        shutil.copytree(ROOT / name, release / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))
    for name in ['run_lite_local.py', 'local_device.py', 'V6_Lite_简化版实测报告.md', 'summary.json']:
        shutil.copy2(ROOT / name, release / name)
    (release / 'README.md').write_text('''# Q4 V6 Lite

简化版删除学习路线重排、后验共享评分，保留局部前瞻、沿途学习检测和前四站保护。具体收益与退步见同目录的实测报告；本地结果不等于官方演练或正式测试。

Python 3.10+，只需标准库。在本目录运行：

```bash
python3 run_lite_local.py --seed my-new-demo --out LOCAL_ONLY_lite_demo.json
```

已有输出会被保护，复跑请更换输出名。本命令在附带的复原模拟器生成新场景，不访问官方服务。

接入自己的设备时，把 `source_v6_lite/` 加入 Python 模块路径：

```python
from q4_v6_lite import solve_v6_lite, default_config
report = solve_v6_lite(device, default_config())
```

设备接口沿用原 V6 的 position、channel、move、detect、clear。运行目录为 18 个 Python 模块和 1 个沿途模型；局部前瞻仍使用后验推断。官方协议适配仍需单独验证。

完整对照代码、冻结地图、逐场会话和校验记录在另附的“第四问_V6_Lite_简化版_代码与完整实测数据.zip”。
''')
    results = [archive(release, ROOT.parent / '第四问_V6_Lite_可运行简化版.zip'),
               archive(ROOT, ROOT.parent / '第四问_V6_Lite_简化版_代码与完整实测数据.zip')]
    (ROOT.parent / 'q4_v6_lite_package_verification.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
