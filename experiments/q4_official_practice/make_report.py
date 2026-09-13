"""Generate a descriptive, explicitly unpaired official-practice report."""
from pathlib import Path
import argparse
import json
import statistics

NAMES = {'v4':'V4', 'analytic':'解析概率', 'rollout':'一步前瞻', 'shared':'前瞻＋共享测向点', 'dynamic':'前瞻＋动态覆盖'}


def main(root):
    data = json.loads((root/'official_summary.json').read_text())
    runs = data['runs']
    assert data['audit'] == 'passed'
    assert len(runs) == 15 and all(g['runs'] == 3 for g in data['summary'])
    lines = [
        '# 第四问：官方模拟器 15 场演练实测', '',
        f"在 UTM 的 Windows 11 虚拟机中，五种方法各运行 3 场官方第四问演练，共 {len(runs)} 场，{sum(r['cleared'] for r in runs)} 个源全部清除。每场正常退出，清除成功反馈数量均与官方结束界面公布的源总数一致。没有启动正式测试。", '',
        '**这批数据证明了端到端运行和本批场景的全清除；不能据此证明某个版本提速。** 官方每次创建不同案例，本批没有相同案例的配对重放。每种只有 3 场，源数量、类型比例和位置均不同。下面仅作描述性统计，不计算“相对 V4 提速率”。', '',
        '| 方法 | 场次 | 全部清除源数 | 平均任务秒/源 ↓ | 单场范围（秒/源） | 平均完整任务秒 | 平均程序秒/场 ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for g in data['summary']:
        lines.append(f"| {NAMES[g['method']]} | {g['runs']} | {g['cleared']} | {g['mean_seconds_per_source']:.2f} | {g['min_seconds_per_source']:.2f}–{g['max_seconds_per_source']:.2f} | {g['mean_virtual_seconds']:.2f} | {g['mean_wall_seconds']:.3f} |")
    lines += ['',
        '任务秒/源先在每个案例内按 T/N 计算，再对案例等权平均。T 是官方 /exit 返回的虚拟任务总时间，包括移动、切换、检测、清除以及最后一个源清除后的补搜索；N 是官方公布总数，并与成功清除数核对。', '',
        '程序秒/场是 Windows Python 从连接模拟器到退出的实际墙钟时间，包含算法计算和本机 HTTP 通信，不包含人工操作、登录、服务器准备场景及一次性几何初始化。主机不是独占环境，这不是算法的纯 CPU 时间。虚拟任务的一次检测扣 5 秒，并不要求程序现实等待 5 秒。', '',
        '## 前瞻实际改变了什么', '',
        '| 方法 | 决策尝试 | 实际改变动作 | 模拟续跑次数 | 平均前瞻计算秒/场 | 超时/无效/后验回退 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for method in ['rollout','shared','dynamic']:
        rs = [r for r in runs if r['method'] == method]
        lines.append(f"| {NAMES[method]} | {sum(r['planning_attempts'] for r in rs)} | {sum(r['planning_accepted'] for r in rs)} | {sum(r['rollout_runs'] for r in rs)} | {statistics.mean(r['planning_seconds'] for r in rs):.3f} | {sum(r['planning_timeouts'] for r in rs)}/{sum(r['invalid_rollouts'] for r in rs)}/{sum(r['posterior_fallbacks'] for r in rs)} |")
    lines += ['',
        '唯一一次动作替换发生在一步前瞻第三场 H3TF-MXUU-39MF-6XRX：第 6 个策略动作由跟踪频道 14 改为扫描站 18。该选择通过了程序的独立场景复核，但没有同一个官方案例的 V4 对照，因此不能把预测收益当作真实节省时间。共享测向点和动态换站的额外动作本批均未执行，不能声称它们的特殊动作已在官方场景验证成功。', '',
        '## 演练发现的先验局限', '',
        'H3TF-MXUU-39MF-6XRX 的官方结束界面明确显示：全向 0 个、定向 12 个。当前前瞻采样器 posterior.py 的 Worlds.sample 却要求两种类型同时存在。这是规划先验与实测环境的支持范围不一致；该假设只参与排序，几何覆盖及真实清除反馈独立维护，所以这一场仍全清除。', '',
        '本批没有中途修改这一先验，保证五种方法沿用冻结配置。下一版应允许全向数或定向数为零，再用新数据验证；这项发现也意味着不能把当前前瞻结果解释为该方法在官方分布下的最优表现。', '',
        '## 逐场数据', '',
        '| 方法/轮次 | 官方案例编码 | 全向/定向 | 清除/总数 | 完整虚拟秒 | 秒/源 | 程序秒 | 未命中清除次数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in runs:
        label = 'v4-01' if r['method']=='v4' and r['label'].startswith('RH55') else r['label']
        lines.append(f"| {label} | {r['official_case_code']} | {r['omni']}/{r['directional']} | {r['cleared']}/{r['total_sources']} | {r['virtual_seconds']:.3f} | {r['seconds_per_source']:.2f} | {r['wall_seconds']:.3f} | {r['failed_clears']} |")
    movement = sum(r['time_breakdown']['movement_s'] for r in runs)
    total = sum(r['virtual_seconds'] for r in runs)
    lines += ['',
        f"本批累计虚拟任务时间 {total:.3f} 秒，其中移动 {movement:.3f} 秒，占 {movement/total:.1%}。累计 {sum(r['failed_clears'] for r in runs)} 次光学清除未命中，其成本已计入上表；未命中尝试不等于整场失败。所有场次最终实际扫描点均通过覆盖核验。", '',
        '## 环境与固定配置', '',
        '- UTM 虚拟机 Quartus9-Windows11-ARM；Windows 11 ARM；独立 Python 3.13.15 ARM64；官方模拟器 v1.1，来自用户下载的标准 Windows 64 位包。',
        '- 只通过 /enter、/measure、/clear、/exit 与官方模拟器通信。每轮发送 /enter 前，均在界面确认“问题4 演练 测试”；API 本身不能查询测试模式。',
        '- 前瞻粗评 8 场景、独立复核 8 场景；每场最多 8 次决策，间隔至少 5 个策略动作；每次预算 5 秒；规划随机种子 90210000。这个种子不控制官方案例。',
        '- 五组均使用同一适配器；测向误差界限为 1.005°，覆盖 ±1° 误差和两位小数舍入。可清除几何阈值为 19.5 m。',
        '- 本机与 Windows 的 3 项适配测试均通过，每套含 8 次完整合成任务，并检查掉线重试幂等性和角度舍入边界；这些自检不计入 15 场官方成绩。',
        '- 与最初部署包逐字节核对，46 个测试 Python 源码文件一致。新增统计和收集脚本另存，不影响算法。', '',
        '## 日志和审计', '',
        'evidence/practice_results 下每场有 result.json 和 requests.jsonl；evidence/Jammers-simulator 下保存 15 份原始官方 .jlog。官方日志原样保留，没有尝试解码未公开格式。official_summary.json 连接每场结果、界面记录、官方日志路径及 SHA-256。', '',
        '统计脚本逐场检查：正常退出、没有未决请求、成功 clear 响应去重数等于总数、已接受动作数一致、界面与接口虚拟时间的舍入一致、最后调用 /exit、最终覆盖证书通过。所有检查通过。', '',
        '首场操作标签曾把 HFWM 录为 HFMM；共享第三场视觉记录曾把 UR5F 录为 URSF。最终案例码以官方日志文件名核正，原标签及更正说明保留，不修改原始请求或运行结果。', '',
        '首场之后曾出现一次服务器准备场景超时，未发送 /enter；重启模拟器并重新登录后恢复。它是准备阶段连接失败，不计作已进入的算法场次。文件读取也曾遇到工具审批超时，重试后完成备份，没有遗失最终日志。', '',
        '## 与此前本地实验的关系', '',
        '此前本地合成模拟做过 60 个相同案例 × 5 方法的 300 次配对测试，全部完成。V4 为 478.74 秒/源，解析概率为 478.42，一步前瞻为 478.39，共享测向点为 478.07，动态覆盖为 478.38；复杂版本的配对收益区间跨零。详见随包提供的“此前本地配对实测报告.md”。', '',
        '目前建议继续把 V4 作为稳定基线。解析概率修正在本地配对实验中只有很小收益，官方本批不足以判断其优势；复杂前瞻带来额外计算，本批特殊动作采用率低，且暴露先验不匹配，尚不足以作为替换 V4 的依据。', '',
        '重算审计：在数据包根目录执行 `python3 code/experiments/q4_official_practice/summarize_results.py results`，再执行 `python3 code/experiments/q4_official_practice/make_report.py results`。继续演练的说明见 code/experiments/q4_official_practice/README.md。',
    ]
    (root/'第四问_官方演练15场实测报告.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    main(p.parse_args().root)
