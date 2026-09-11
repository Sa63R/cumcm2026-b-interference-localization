# 首版选择与最终评估的身份链

这是评估操作说明，不是论文正文。`experiments/research_v1_selection.py` 仅读取 JSON、gzip 和文件字节，以及只读 `git rev-parse`；不导入模拟器、不生成场景、不运行策略。开发测试全部为构造 JSON；另只读核验了已有 `validation-region-mean_point` 的 6000–6047 共 48 档，未读取扩展或最终封存集。

## 已补足的检查

| 阶段 | 工具检查 | 不由该工具保证 |
|---|---|---|
| `register` | 三方向各 1–3 个候选、固定顺序与集合内 fallback；真实 Git HEAD 与 freeze 一致；freeze 时间不晚于登记；重新计算完整源码/spec/实际 checkpoint 字节；同一 Linux 环境和模拟器/客户端/共同 harness 字节；规则摘要 | 操作者此前从未查看选择集；源码是否具有正确的数学含义 |
| `finalize` | 全部已登记候选和基线的完整选择集、逐案例 SHA 一致、gzip 中 row/spec 与汇总一致、结束后 truth SHA；固定门槛与未舍入均值选择 | 该选择集获胜者的区间是独立泛化证据 |
| `audit-final` | 再核全部选择原始档案及确定性选择；选中身份完整继承至 random/stress；两分区每档与真实结果、物理账字段一致；单独报告基线失败清除 | 逐动作物理重新计算、合法几何区域与全圆盘覆盖证明、下界、最终统计门槛 |

登记、选择与最终审计文件均拒绝覆盖，并含规范 JSON 内容摘要。摘要用于发现误改和追溯，不是数字签名，也不能证明外部没有查看过数据。`audit-final` 允许保留并报告真实失败案例：身份完整性通过不等于可靠性或性能通过。

选择使用 `P95(candidate penalty)/P95(baseline penalty)`，与共同报告的线性插值定义相同；不是逐局比值的 P95。基线须全部 `successful` 且零失败清除；候选还须该比值 ≤1.05。通过者按未舍入均值最小、完全相等按候选数组次序选择；没有通过者使用已登记 fallback 并明确未通过。不另加 CI 筛选。三方向无论通过与否都保留至最终比较。

## 冻结输入

所有路径在**实际评估的 Linux 主机**解析。登记 JSON 的相对路径以该 JSON 所在目录为准；spec 中 checkpoint/weights 的相对路径以对应 `source_dir` 为准，与在该目录运行 harness 的 cwd 相同。后续始终从原 source_dir 运行，保持原 spec、模型路径、模型字节和 Git HEAD。登记后即使只改 Git HEAD 而源码相同，也会被本工具的严格来源检查拒绝。

先在共用实际 Python 环境导出一次环境 JSON，例如：

```bash
"$PY" -c 'import importlib.metadata,json,platform; print(json.dumps({"system":platform.system(),"release":platform.release(),"machine":platform.machine(),"python_version":platform.python_version(),"python_implementation":platform.python_implementation(),"packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()}},sort_keys=True,indent=2))' > "$RUN/environment.json"
```

登记各项使用同一环境文件；工具严格比较整个对象，包含 Python、torch/numpy 等安装版本。环境文件是实际采集证据，不能自动证明之后启动了同一解释器，运行命令与环境隔离仍须保留。

便携 RL checkpoint 必须**先导出，再冻结实际导出的字节**。原文件与便携文件字节 SHA 不同是正常的；`research_rl.portable_checkpoint` 的单独张量摘要证明模型值保持。保留 `.pt.json` 中 original/output/tensor 三类摘要。导出包含时间戳，重复导出会改变字节；禁止选中后再次导出或替换。注册工具不加载 torch，也不重新证明张量等价。旧模型缺 action schema 时按照已冻结推理代码的 base 默认处理，不能在测试时换 schema。

每一项先在它的源码目录执行已有命令，输出 freeze 必须位于源码扫描范围以外，避免污染冻结工作区：

```bash
cd "$SOURCE"
"$PY" experiments/research_v1_eval.py freeze --spec "$SPEC" --output "$FREEZE"
```

`plan.json` 的最小示意如下；每方向实际可列 1–3 项，数组顺序就是固定 tie order：

```json
{
  "protocol":"/absolute/state/research/v1_protocol.json",
  "rule":"/absolute/v1_selection_rule.md",
  "baseline":{"id":"baseline","source_dir":"/absolute/baseline","spec":"/absolute/baseline-spec.json","freeze":"/absolute/baseline-freeze.json","environment":"/absolute/environment.json"},
  "directions":{
    "state":{"candidates":[{"id":"state-a","source_dir":"/absolute/state","spec":"/absolute/state-spec.json","freeze":"/absolute/state-freeze.json","environment":"/absolute/environment.json"}],"fallback":"state-a"},
    "rl":{"candidates":[{"id":"rl-a","source_dir":"/absolute/rl","spec":"/absolute/rl-spec.json","freeze":"/absolute/rl-freeze.json","environment":"/absolute/environment.json"}],"fallback":"rl-a"},
    "geo":{"candidates":[{"id":"geo-a","source_dir":"/absolute/geo","spec":"/absolute/geo-spec.json","freeze":"/absolute/geo-freeze.json","environment":"/absolute/environment.json"}],"fallback":"geo-a"}
  }
}
```

各项 `spec.name` 也须唯一，以兼容事后下界审计的重复键检查。基线 spec 必须精确匹配 protocol 的 `rollout_frozen` 配置，不是只改成相同名称。

```bash
"$PY" "$STATE/experiments/research_v1_selection.py" register --input "$RUN/plan.json" --output "$RUN/registry.json"
```

## 选择与最终命令模板

**以下是获准开启相应分区之后的模板；编写本说明时未执行。** 先完成所有登记，再逐方法在其固定源码目录运行完整分区。不建议人工拼接分片；共同报告和身份工具都要求一个目录覆盖完整声明种子列表。

```bash
cd "$SOURCE"
"$PY" experiments/research_v1_eval.py run --spec "$SPEC" --freeze-record "$FREEZE" --split validation_extended --output "$EXTENDED_OUTPUT"
```

`extended-index.json` 是 **所有** baseline+已登记候选 `id -> 完整输出目录` 的对象。不得仅保留看起来较优者。相对输出目录以 index JSON 所在目录解析。

```bash
"$PY" "$STATE/experiments/research_v1_selection.py" finalize --registry "$RUN/registry.json" --evaluations "$RUN/extended-index.json" --output "$RUN/selected.json"
```

按 `selected.json` 选中的 3 项和原 baseline，用**同一** source/spec/freeze/checkpoint 各跑两分区：

```bash
cd "$SOURCE"
"$PY" experiments/research_v1_eval.py run --spec "$SPEC" --freeze-record "$FREEZE" --split final_random --output "$RANDOM_OUTPUT"
"$PY" experiments/research_v1_eval.py run --spec "$SPEC" --freeze-record "$FREEZE" --split final_stress --output "$STRESS_OUTPUT"
```

`final-index.json` 格式为 `{"final_random":{"baseline":"...","state-a":"...","rl-a":"...","geo-a":"..."},"final_stress":{...}}`，键用实际选中 id；每分区只含已选三项和 baseline。

```bash
"$PY" "$STATE/experiments/research_v1_selection.py" audit-final --registry "$RUN/registry.json" --selection "$RUN/selected.json" --evaluations "$RUN/final-index.json" --output "$RUN/final-identity-audit.json"
"$PY" "$STATE/experiments/research_v1_report.py" --baseline "$BASE_RANDOM" --candidate "state=$STATE_RANDOM" --candidate "rl=$RL_RANDOM" --candidate "geo=$GEO_RANDOM" --output "$RUN/report-random"
"$PY" "$STATE/experiments/research_v1_report.py" --baseline "$BASE_STRESS" --candidate "state=$STATE_STRESS" --candidate "rl=$RL_STRESS" --candidate "geo=$GEO_STRESS" --output "$RUN/report-stress"
"$PY" "$STATE/experiments/research_v1_report.py" --final-random-report "$RUN/report-random/comparison.json" --final-stress-report "$RUN/report-stress/comparison.json" --output "$RUN/final-acceptance"
```

最终报告独立输出随机性能与压力可靠性；不能把两组混合后改善均值/区间，也不能据最终结果改选另一个模型。

## 深审、缓存与剩余边界

独立物理/几何审计应使用 `final-identity-audit.json` 中 `partitions.<split>.evaluations.<id>.directory` 与 `evidence.archives_sha256` 的**显式档案清单**，再核输入字节。现有 `research/theory_v1/audit_eval_bounds.py` 已能从 common gzip 重放每条物理动作、接收条件、清除距离、时间微秒账与场景 SHA，并计算事后下界；新增合法观测区域/每频道真实负反馈覆盖的深入审计由理论模块提供。

已有下界工具的命令模板（仅获准后使用 `--allow-heldout`）：

```bash
"$PY" "$STATE/research/theory_v1/audit_eval_bounds.py" "$BASE_RANDOM" "$STATE_RANDOM" "$RL_RANDOM" "$GEO_RANDOM" "$BASE_STRESS" "$STATE_STRESS" "$RL_STRESS" "$GEO_STRESS" --allow-heldout --cache "$RUN/bounds-cache.json" --output "$RUN/physical-bounds-audit.json"
```

不要给它宽泛的 artifacts 根目录，以免递归读到其它封存集。多个进程不能同时写同一缓存；单个批次串行复用按源几何键缓存，可避免四方法重复求同一个 ≤16 点动态规划下界。下界是明确假设下的事后乐观界，不能将差距直接解释成在线策略还能改进的百分比。

仍须人工/编排约束的事项：确保登记早于首次选择集访问；每项使用登记的实际解释器；评估期间保持源码/模型快照只读且不中途改写；保存真实程序超时与失败；记录全部方法的完整输入和 stdout。harness 只在启动时计算 identity，模型推理按路径/mtime 缓存，故“开始和结束字节一致”本身无法排除中途替换。并发运行的 wall time 只作为预算记录，不作算法计算速度优劣结论。

合法观测审计须覆盖 `certified_clear` 与近场清除、源数量达到 16 的终止条件、推断负反馈不得伪装物理扫描，以及每个仍未知频道的真实无信号圆盘联合覆盖整个搜索圆盘（含内部孔）。原有按报告 coverage_points 逐点查测量的审计不足以单独证明全域覆盖。
