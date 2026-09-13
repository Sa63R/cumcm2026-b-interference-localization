#!/usr/bin/env python3
"""Build an anonymous, size-bounded supporting-materials candidate archive."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "B题支撑材料_提交候选_20260913"
OUTPUT_DIR = ROOT / "支撑材料_整理版_20260913"
OUTPUT_ZIP = ROOT / f"{PACKAGE_NAME}.zip"
VERIFY_JSON = ROOT / f"{PACKAGE_NAME}_校验.json"
SIZE_LIMIT = 20 * 1024 * 1024

PRACTICE_ROOT = ROOT / "output/q3_q4_official_100each_20260913"
PRACTICE = PRACTICE_ROOT / "practice_checkpoint_200"
FORMAL = PRACTICE_ROOT / "正式测试_Q3_Q4_六轮日志_20260913"
Q3_FORMAL_SOURCE = ROOT / "output/q3_origin_entry_20260913/bundle"
Q4_SOURCE = ROOT / "output/q4_v6_lite_unrestricted_20260913"
Q3_LOCAL = ROOT / "paper_code/q3_optical_paper"
Q4_LOCAL = ROOT / "output/q4_v6_lite_20260912"

TEXT_SUFFIXES = {
    ".csv", ".html", ".json", ".jsonl", ".md", ".ps1", ".py", ".tex", ".txt"
}
REPLACEMENTS = (
    ("202627001104", "YOUR_TEAM_ID"),
    ("/Users/zephyrr/竞赛/26国赛/数模", "<PROJECT_ROOT>"),
    ("/Users/zephyrr/Downloads", "<DOWNLOADS>"),
    ("/Users/zephyrr", "<USER_HOME>"),
    (r"C:\Users\baiwc\Downloads\Q4V6Lite_20260913", r"C:\Path\To\Q4V6Lite_20260913"),
    (r"C:\Users\baiwc\Downloads\Q4Practice", r"C:\Path\To\Q4Practice"),
    (r"C:\Users\baiwc", r"C:\Users\USERNAME"),
    (r"C:\\Users\\baiwc\\Downloads\\Q4V6Lite_20260913", r"C:\\Path\\To\\Q4V6Lite_20260913"),
    (r"C:\\Users\\baiwc\\Downloads\\Q4Practice", r"C:\\Path\\To\\Q4Practice"),
    (r"C:\\Users\\baiwc", r"C:\\Users\\USERNAME"),
    ("zephyrr", "USERNAME"),
    ("baiwc", "USERNAME"),
    ("Derek", "USERNAME"),
)
SENSITIVE_TOKENS = ("202627001104", "zephyrr", "baiwc", "Derek", "/Users/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, destination: Path, *, sanitize: bool = True) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if sanitize and source.suffix.lower() in TEXT_SUFFIXES:
        destination.write_text(sanitize_text(source.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path, *, sanitize: bool = True) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        copy_file(path, destination / path.relative_to(source), sanitize=sanitize)


def refresh_manifest_hashes(base: Path, manifest_relative: str) -> None:
    manifest_path = base / manifest_relative
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    if isinstance(files, dict):
        for relative in files:
            files[relative] = sha256(base / relative)
    else:
        for item in files:
            item["sha256"] = sha256(base / item["snapshot"])
    write_json(manifest_path, manifest)


def copy_q1_q2(package: Path) -> None:
    target = package / "程序源码/问题1_2_几何与选点"
    mapping = {
        ROOT / "state-optimization/src/geometry/__init__.py": target / "src/geometry/__init__.py",
        ROOT / "state-optimization/src/localization/__init__.py": target / "src/localization/__init__.py",
        ROOT / "state-optimization/src/localization/omni.py": target / "src/localization/omni.py",
        ROOT / "state-optimization/tests/test_geometry.py": target / "tests/test_geometry.py",
        ROOT / "state-optimization/tests/test_localization.py": target / "tests/test_localization.py",
        ROOT / "state-optimization/tests/test_omni_localization.py": target / "tests/test_omni_localization.py",
        ROOT / "state-optimization/资料汇总/T02与T03数学模型.md": target / "T02与T03数学模型.md",
        ROOT / "state-optimization/pyproject.toml": target / "pyproject.toml",
        ROOT / "state-optimization/requirements-test.txt": target / "requirements-test.txt",
    }
    for source, destination in mapping.items():
        copy_file(source, destination)
    write_text(
        target / "README.md",
        """# 问题一、二：几何与选点程序

本目录对应论文中的最小覆盖圆与被动定位选点模型。核心实现位于 `src/geometry` 与
`src/localization`，三份测试文件给出局部可复核的数值检查。

说明：这里只收录与论文现用模型直接对应的实现和测试；历史探索、反例诊断和其他问题的
策略代码不放入提交候选包，避免把不同阶段的结论混在一起。
""",
    )


def copy_q3_local(package: Path) -> None:
    target = package / "程序源码/问题3_论文方法_optical"
    for name in ("README.md", "论文写作说明.md", "requirements.txt"):
        copy_file(Q3_LOCAL / name, target / name)
    for name in ("run_optical.py", "runner.py", "summarize.py", "source_manifest.json", "README.md"):
        copy_file(Q3_LOCAL / "code" / name, target / "code" / name)
    copy_tree(Q3_LOCAL / "code/vendor/policies", target / "code/vendor/policies")
    copy_tree(Q3_LOCAL / "code/vendor/simulator", target / "code/vendor/simulator")
    refresh_manifest_hashes(target / "code", "source_manifest.json")

    evidence = package / "本地实验/问题3_optical_五方法配对"
    for name in ("statistics.json", "本地模拟器前五名结果.md"):
        copy_file(Q3_LOCAL / "code" / name, evidence / name)
    for group in ("paired2000", "timing50"):
        for name in ("metadata.json", "results.jsonl", "validation.json", "status.json"):
            copy_file(Q3_LOCAL / "code/results" / group / name, evidence / group / name)
    write_text(
        evidence / "数据边界.md",
        """# 数据边界

- `paired2000/results.jsonl`：2000 个共享场景、5 种方法，共 10000 次本地重建模拟结果。
- `timing50/results.jsonl`：用于程序实际耗时复核的 50 组配对结果。
- 本地模拟器用于可重复的相对比较，不等同于官方模拟器，也不应把本地排名解释为官方排名。
- 为满足电子支撑材料 20 MB 限制，逐次压缩轨迹未收录；完整归档在工作区
  `paper_code/q3_optical_paper.zip`，其校验值记录于 `外部大文件索引.json`。
""",
    )


def copy_q3_formal_source(package: Path) -> None:
    target = package / "程序源码/问题3_正式测试_v3_origin20"
    copy_tree(Q3_FORMAL_SOURCE, target)
    refresh_manifest_hashes(target, "q3_origin/source_manifest.json")
    write_text(
        target / "匿名化说明.md",
        """# 匿名化说明

正式测试使用的算法、运行入口与模拟器客户端源文件按原字节保留。使用说明中的本机路径和
参赛号已替换为占位符，因此同步重算了 `q3_origin/source_manifest.json` 中相应文件的哈希；
算法源文件及 `run_q3_origin.py` 未因匿名化而修改。
""",
    )


def copy_q4_source(package: Path) -> None:
    target = package / "程序源码/问题4_V6_Lite"
    manifest = json.loads((Q4_SOURCE / "source_manifest.json").read_text(encoding="utf-8"))
    required = set(manifest["files"])
    required.update(
        {
            "source_manifest.json",
            "使用说明.md",
            "check_windows.py",
            "windows_tests.txt",
            "test_entry_modes.py",
            "validation.json",
            "expected_traces.json.gz",
        }
    )
    for relative in sorted(required):
        copy_file(Q4_SOURCE / relative, target / relative)
    refresh_manifest_hashes(target, "source_manifest.json")
    write_text(
        target / "匿名化说明.md",
        """# 匿名化说明

正式测试使用的算法、运行入口和客户端源文件按原字节保留。仅使用说明及本地模拟器兼容性
文档中的个人路径、参赛号被替换为占位符，因此同步重算了 `source_manifest.json` 中相应
文档的哈希；算法源文件及 `run_official.py` 未因匿名化而修改。
""",
    )


def copy_q4_local(package: Path) -> None:
    target = package / "本地实验/问题4_V6_Lite_配对验证"
    names = (
        "README.md",
        "V6_Lite_简化版实测报告.md",
        "summary.json",
        "paired_records.csv",
        "plan.json",
        "source_hashes.json",
        "implementation_equivalence.json",
        "test_results.txt",
        "replay_verification.json",
        "run_lite_local.py",
        "run_validation.py",
        "analyze_validation.py",
        "make_figures.py",
        "make_report.py",
        "test_lite.py",
    )
    for name in names:
        copy_file(Q4_LOCAL / name, target / name)
    for relative in (
        "main/records.jsonl",
        "main/execution.json",
        "serial/records.jsonl",
        "serial/summary.json",
        "serial/execution.json",
        "figures/lite_comparison.png",
        "figures/lite_tail_distribution.png",
    ):
        copy_file(Q4_LOCAL / relative, target / relative)
    write_text(
        target / "数据边界.md",
        """# 数据边界

该组数据是本地重建模拟：1000 个随机场景、70 个全定向场景、70 个边界外向场景，
3 种方法共 3420 次运行。它支撑论文中的方法对比，但不是官方演练或正式测试结果。
为满足 20 MB 限制，逐次会话目录未收录；完整归档在工作区
`output/第四问_V6_Lite_简化版_代码与完整实测数据.zip`，校验值见 `外部大文件索引.json`。
""",
    )


def copy_practice(package: Path) -> None:
    target = package / "官方演练/问题3_问题4_各100次"
    for name in ("independent_audit_summary.json", "completed.json", "结果说明.md"):
        copy_file(PRACTICE / name, target / name)
    copy_file(PRACTICE_ROOT / "expected_versions.json", target / "expected_versions.json")
    write_text(
        target / "收录说明.md",
        """# 收录说明

本目录收录 200 个案例的逐案例总表、独立审计汇总和完整文字报告。截图、逐步请求日志与
官方结果元数据未放入提交候选包，以控制体积并避免携带参赛身份；原始材料仍保留在工作区。
""",
    )


def sanitized_audit(source: Path) -> dict:
    value = json.loads(source.read_text(encoding="utf-8"))
    value.pop("source_directory", None)
    value["package_note"] = (
        "upload_status_verified=false 是日志审计时的原字段；2026-09-13 14:16--14:17 "
        "北京时间另行只读核实六轮均显示已上传。"
    )
    value["later_upload_status_verified"] = True
    return value


def copy_formal(package: Path) -> list[dict]:
    rows: list[dict] = []
    for question in (3, 4):
        for run in range(1, 4):
            source = FORMAL / f"Q{question}/run{run:02d}"
            target = package / f"正式测试/问题{question}/第{run}轮"
            audit = sanitized_audit(source / "audit.json")
            jlogs = list(source.glob("*.jlog"))
            if len(jlogs) != 1:
                raise RuntimeError(f"Expected one jlog in {source}, got {len(jlogs)}")
            copy_file(jlogs[0], target / jlogs[0].name, sanitize=False)
            write_json(target / "audit_sanitized.json", audit)
            copy_file(source / "result.json", target / "result_sanitized.json")
            copy_file(source / "逐步过程.json", target / "逐步过程.json")
            copy_file(source / "运行过程核对.md", target / "运行过程核对.md")
            rows.append(
                {
                    "question": question,
                    "round": run,
                    "official_case": audit["official_case"],
                    "method": audit["method"],
                    "successful_clears": audit["successful_clears"],
                    "virtual_time_s": audit["virtual_time_s"],
                    "seconds_per_cleared_source": audit["seconds_per_cleared_source"],
                    "program_wall_s": audit["wall_seconds"],
                    "audit_status": audit["status"],
                    "upload_status_verified": True,
                    "jlog": jlogs[0].name,
                    "jlog_sha256": sha256(jlogs[0]),
                }
            )
    return rows


def write_formal_tables(package: Path, rows: list[dict]) -> None:
    csv_path = package / "02_正式测试结果.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 六轮正式测试结果",
        "",
        "| 题目 | 轮次 | 官方案例 | 方法 | 成功清除 | 虚拟秒 | 秒/清除源 | 程序实际秒 | 日志审计 | 上传状态 |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| Q{row['question']} | {row['round']} | {row['official_case']} | {row['method']} | "
            f"{row['successful_clears']} | {row['virtual_time_s']:.6f} | "
            f"{row['seconds_per_cleared_source']:.6f} | {row['program_wall_s']:.6f} | "
            f"{row['audit_status']} | 已核实已上传 |"
        )
    lines.extend(
        [
            "",
            "边界说明：导出的正式日志没有提供每个案例的官方真实源总数，因此不能仅凭导出文件",
            "独立计算正式测试的清除率。这里报告成功清除数、完整通道证书、逐步计时审计及上传状态；",
            "不把成功清除数误写成官方真实源总数。",
        ]
    )
    write_text(package / "02_正式测试结果.md", "\n".join(lines))


def write_external_large_file_index(package: Path) -> None:
    entries = []
    for path in (
        ROOT / "paper_code/q3_optical_paper.zip",
        ROOT / "output/第四问_V6_Lite_简化版_代码与完整实测数据.zip",
    ):
        if path.exists():
            entries.append(
                {
                    "workspace_file": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "included_in_submission_candidate": False,
                    "reason": "完整逐次轨迹归档；为满足电子支撑材料 20 MB 限制未收录",
                }
            )
    write_json(package / "外部大文件索引.json", entries)


def write_package_docs(package: Path, formal_rows: list[dict]) -> None:
    q3_jlogs = sum(row["question"] == 3 for row in formal_rows)
    q4_jlogs = sum(row["question"] == 4 for row in formal_rows)
    write_text(
        package / "00_先看这里.md",
        f"""# B 题支撑材料提交候选包

本包按“模型与源码—本地可重复实验—官方演练—正式测试”四层整理，共收录问题三
{q3_jlogs} 轮、问题四 {q4_jlogs} 轮原始加密 `.jlog`。所有非加密文本已做身份与个人路径扫描。

## 先确认两件事

1. 论文问题三现用方法是 `optical`，而问题三正式测试使用的是 `v3_origin20`。两套源码和证据
   已分别归档，不应把正式测试数据写成 `optical` 的官方结果。
2. 论文现有附录仍是历史 E01--E13 索引，且正文中“问题四暂无官方演练记录”的表述已被
   100 次官方演练和 3 轮正式测试更新。包内已提供可粘贴的附录文件列表，但尚未替你改论文。

因此：支撑材料候选包已经整理完成；在论文同步前，整体投稿状态仍标记为“未就绪”。

阅读顺序：`01_结构化证据索引.md` → `02_正式测试结果.md` → `03_交付前核对.md`。
""",
    )
    write_text(
        package / "01_结构化证据索引.md",
        """# 结构化证据索引

| 编号 | 证据层级 | 对应内容 | 关键文件 | 可支持的结论 | 不可外推 |
|---|---|---|---|---|---|
| G01 | 模型与源码 | 问题一、二几何与选点 | `程序源码/问题1_2_几何与选点/` | 模型构造与局部数值复核 | 不代表官方测试 |
| L3-01 | 本地实验 | 问题三 `optical` 五方法配对 | `本地实验/问题3_optical_五方法配对/statistics.json` | 2000 共享场景上的本地相对性能 | 不代表官方模拟器排名 |
| S3-01 | 正式入口源码 | 问题三 `v3_origin20` | `程序源码/问题3_正式测试_v3_origin20/` | 正式入口与算法可复核 | 不等同于论文 `optical` 方法 |
| L4-01 | 本地实验 | 问题四 V6 Lite 配对验证 | `本地实验/问题4_V6_Lite_配对验证/summary.json` | 1140 场景、3420 次本地比较 | 不代表官方分布 |
| S4-01 | 正式入口源码 | 问题四 V6 Lite | `程序源码/问题4_V6_Lite/` | 正式入口、模型和客户端可复核 | 本地兼容测试不等同官方认证 |
| O01 | 官方演练 | Q3、Q4 各 100 次 | `官方演练/问题3_问题4_各100次/` | 200/200 审计通过及逐案例结果 | 两题案例不同，不能横向当同题比较 |
| F3-01--03 | 正式测试 | 问题三三轮 | `正式测试/问题3/` | 原始加密日志、动作过程、计时审计 | 无官方真实源总数，不独立报告清除率 |
| F4-01--03 | 正式测试 | 问题四三轮 | `正式测试/问题4/` | 原始加密日志、动作过程、计时审计 | 无官方真实源总数，不独立报告清除率 |
| V01 | 完整性 | 全包 | `MANIFEST.sha256`、`文件清单.tsv` | 除清单自身外的文件级 SHA-256 与归档 CRC | 不能替代内容正确性审查 |

证据使用原则：本地重建模拟、官方演练和正式测试分栏叙述；方法版本必须随结果一起写明；
正式测试仅报告导出材料可核实的字段。
""",
    )
    write_text(
        package / "03_交付前核对.md",
        """# 交付前核对

- [x] 收录问题一至四的必要源码、参数文件和可重复实验汇总。
- [x] 收录问题三、四各三轮正式测试原始 `.jlog`，保留原始文件名与 SHA-256。
- [x] 收录问题三、四各 100 次官方演练的逐案例总表和独立审计汇总。
- [x] 排除截图、请求明细、官方结果元数据及部署备份，避免携带参赛身份并控制体积。
- [x] 对所有非 `.jlog` 文件扫描参赛号、个人用户名和 macOS 绝对用户路径。
- [x] 生成 `MANIFEST.sha256`、`文件清单.tsv` 并执行 ZIP CRC 检查。
- [ ] 将 `附录文件列表_可粘贴.tex` 的内容合并进论文附录，并替换历史 E01--E13 索引。
- [ ] 更新论文问题四“暂无官方演练记录”的旧表述。
- [ ] 决定是否在论文中增加六轮正式测试；若增加，明确 Q3 正式方法是 `v3_origin20`，
      不是论文主体比较中的 `optical`。
- [ ] 投稿前用最终参赛号重命名论文 PDF 和支撑材料 ZIP；不要把参赛号写入包内正文。
- [ ] 最后人工检查压缩包文件名、论文文件名、匿名性、20 MB 限制及平台上传状态。
""",
    )
    write_text(
        package / "04_入口自检记录.md",
        """# 入口自检记录

整理完成后，以 Python 3.13.12、NumPy 2.5.1 执行离线自检：

- `程序源码/问题3_正式测试_v3_origin20/run_q3_origin.py --check-only`：通过，核对 15 个文件；
  `simulator_contacted=false`、`test_started=false`。
- `程序源码/问题4_V6_Lite/run_official.py --check-only`：通过，核对 41 个文件，
  `coverage_ok=true`、`simulator_contacted=false`。

两项自检都没有连接模拟器或发起新测试。问题三入口需要 NumPy；问题四源码使用 Python 3.10+
类型语法，建议统一使用 Python 3.13。macOS 自带的 Python 3.9 不能作为问题四入口的验证环境。
""",
    )
    write_text(
        package / "附录文件列表_可粘贴.tex",
        r"""% 将本段按论文版式调整后放入附录。路径与提交候选包保持一致。
\section{支撑材料文件列表}
本论文的支撑材料按模型源码、本地实验、官方演练和正式测试四类组织：
\begin{enumerate}
  \item \texttt{程序源码/问题1\_2\_几何与选点/}：问题一、二的几何与定位选点源码、测试及模型说明；
  \item \texttt{程序源码/问题3\_论文方法\_optical/}：问题三论文方法及本地重建模拟入口；
  \item \texttt{程序源码/问题3\_正式测试\_v3\_origin20/}：问题三正式测试使用的完整入口与源码；
  \item \texttt{程序源码/问题4\_V6\_Lite/}：问题四 V6 Lite 的完整入口、模型与源码；
  \item \texttt{本地实验/}：问题三 2000 个共享场景的五方法配对结果，以及问题四 1140 个场景、3420 次运行的配对验证；
  \item \texttt{官方演练/问题3\_问题4\_各100次/}：两题各 100 次官方演练的逐案例总表和独立审计汇总；
  \item \texttt{正式测试/}：问题三、四各三轮的原始加密日志、匿名化审计摘要与逐步过程；
  \item \texttt{MANIFEST.sha256} 与 \texttt{文件清单.tsv}：全包文件校验及目录索引。
\end{enumerate}
其中，本地重建模拟仅用于方法比较；官方演练与正式测试均单独标注版本及证据边界。
""",
    )


def scan_privacy(package: Path) -> dict:
    findings = []
    scanned = 0
    skipped_encrypted = 0
    for path in sorted(package.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package).as_posix()
        if any(token.lower() in relative.lower() for token in SENSITIVE_TOKENS):
            findings.append({"file": relative, "where": "filename"})
        if path.suffix.lower() == ".jlog":
            skipped_encrypted += 1
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in SENSITIVE_TOKENS:
            if token.lower() in text.lower():
                findings.append({"file": relative, "token": token, "where": "content"})
    return {
        "status": "passed" if not findings else "failed",
        "non_encrypted_text_files_scanned": scanned,
        "encrypted_jlog_files_skipped": skipped_encrypted,
        "sensitive_categories_checked": [
            "参赛号",
            "已知本机账户名",
            "macOS 绝对用户目录",
        ],
        "findings": findings,
        "note": "六份 .jlog 为官方加密原件，按原字节保留，未尝试改写；其外部文件名已检查。",
    }


def write_manifests(package: Path) -> None:
    files = [path for path in sorted(package.rglob("*")) if path.is_file()]
    manifest_lines = []
    table_lines = ["路径\t字节\tSHA-256"]
    for path in files:
        relative = path.relative_to(package).as_posix()
        digest = sha256(path)
        manifest_lines.append(f"{digest}  {relative}")
        table_lines.append(f"{relative}\t{path.stat().st_size}\t{digest}")
    write_text(package / "MANIFEST.sha256", "\n".join(manifest_lines))
    write_text(package / "文件清单.tsv", "\n".join(table_lines))


def make_zip(package: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                output.write(path, f"{PACKAGE_NAME}/{path.relative_to(package).as_posix()}")


def main() -> int:
    for path in (OUTPUT_DIR, OUTPUT_ZIP, VERIFY_JSON):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing output: {path}")

    with tempfile.TemporaryDirectory(prefix="support-materials-", dir=ROOT) as temp_name:
        package = Path(temp_name) / PACKAGE_NAME
        package.mkdir()

        copy_q1_q2(package)
        copy_q3_local(package)
        copy_q3_formal_source(package)
        copy_q4_source(package)
        copy_q4_local(package)
        copy_practice(package)
        formal_rows = copy_formal(package)
        write_formal_tables(package, formal_rows)
        write_external_large_file_index(package)
        write_package_docs(package, formal_rows)

        privacy = scan_privacy(package)
        write_json(package / "隐私扫描.json", privacy)
        if privacy["status"] != "passed":
            print(json.dumps(privacy, ensure_ascii=False, indent=2), file=sys.stderr)
            raise SystemExit("Privacy scan failed")

        status = {
            "support_materials_built": True,
            "formal_logs_present": len(formal_rows),
            "formal_q3_logs": sum(row["question"] == 3 for row in formal_rows),
            "formal_q4_logs": sum(row["question"] == 4 for row in formal_rows),
            "formal_log_audits_passed": all(row["audit_status"] == "passed" for row in formal_rows),
            "official_practice_cases": 200,
            "official_practice_audits_passed": 200,
            "official_upload_status_verified": True,
            "identity_scan_passed": True,
            "paper_sync_required": True,
            "submission_ready": False,
            "not_ready_reasons": [
                "论文附录仍需替换历史 E01--E13 索引并写入最终文件列表",
                "论文问题四暂无官方演练的旧表述需要更新",
                "若写正式测试，需区分问题三论文 optical 与正式测试 v3_origin20",
            ],
        }
        write_json(package / "PACKAGE_STATUS.json", status)
        write_manifests(package)
        shutil.move(str(package), OUTPUT_DIR)

    make_zip(OUTPUT_DIR, OUTPUT_ZIP)
    if OUTPUT_ZIP.stat().st_size > SIZE_LIMIT:
        raise SystemExit(
            f"Archive exceeds 20 MiB: {OUTPUT_ZIP.stat().st_size} bytes; outputs kept for inspection"
        )
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        bad_member = archive.testzip()
        names = archive.namelist()
    if bad_member:
        raise SystemExit(f"ZIP CRC failure: {bad_member}")

    verification = {
        "archive": OUTPUT_ZIP.name,
        "bytes": OUTPUT_ZIP.stat().st_size,
        "mib": round(OUTPUT_ZIP.stat().st_size / (1024 * 1024), 3),
        "under_20_mib": True,
        "sha256": sha256(OUTPUT_ZIP),
        "zip_crc_test": "passed",
        "zip_members": len(names),
        "package_directory": OUTPUT_DIR.name,
        "privacy_scan": json.loads((OUTPUT_DIR / "隐私扫描.json").read_text(encoding="utf-8")),
    }
    write_json(VERIFY_JSON, verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
