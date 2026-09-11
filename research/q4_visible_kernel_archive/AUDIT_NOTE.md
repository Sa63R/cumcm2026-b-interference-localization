# 可见核：独立数学与数值审计

结论：顶点交核的命题成立；模拟器存在边界容差时，必须验证一个完整的内缩球。向某个方向移动一小段，或仅验证点在未内缩凸包内，都不够。本阶段没有接入策略，也没有运行新场景；27 项纯构造和篡改测试通过。

## 1. 为什么只检查 C 的顶点足够

记真实正观测位置的凸包为 A=conv(P)，源位置 s∈C=conv(v₁,…,vₘ)，定义

\[
K=\bigcap_{i=1}^{m}\operatorname{conv}(A\cup\{v_i\}).
\]

对固定 q∈K：若 q∈A，显然 q∈conv(A∪{s})。若 q∉A，则

\[
q\in\operatorname{conv}(A\cup\{s\})
\iff s=q+\mu(q-a),\quad a\in A,\ \mu\ge0.
\]

这来自 q=(1−λ)a+λs，取 μ=(1−λ)/λ；q∉A 排除了 λ=0。右侧集合是 q+cone(q−A)，因 A 凸而凸。K 的定义说明每个 vᵢ 均属于这个集合，故它包含所有 s∈conv(vᵢ)。因此 **K⊆conv(P∪{s}) 对每个可行 s 同时成立**。这里没有先按不同 s 选不同探点再平均。

理想精确模型中，正观测点和 s 同在接收圆盘 B(s,R) 与发射闭半平面中；两个集合都凸，因此 conv(P∪{s}) 中的点同时满足距离和朝向条件。全向源只需圆盘条件。

退化情况不能略去：只有一个正位置时，每个支持集可能是线段，交核可能只是原正位置；两正位置共线且某个 vᵢ 同线时，至少一个支持凸包没有二维内部。这些集合可能含合法点，却放不下正半径球。当前方法保守放弃，不声称“没有任何可接收位置”。同样，输出内核并不承诺等于最大 K。

## 2. engine 容差的真实反例

本地 `src/simulation/engine.py:_visible` 先检查距离，再用
dot≥−10⁻¹²max(1,d) 判断发射半平面，近距离反馈也要先通过可见性判断。

取 s=(0,0)、朝向 +x、R=1500，正点 p₁=(−10⁻⁹,−1000)、p₂=(−10⁻⁹,1000) 都可通过该判断；它们的中点 q=(−10⁻⁹,0) 却不通过，因为近处阈值仅约 −10⁻¹²。沿该线段挪动 10⁻⁵m 仍可能失败。测试直接使用这条判断公式，不构造或运行模拟场景。

如果证明 **B(q,δ)⊆K**，情况不同。令 n 为单位发射方向，所有正点的最坏外侧偏差为 ε。由于 q−δn∈conv(P∪{s})，得到

\[
n\cdot(q-s)\ge\delta-\varepsilon.
\]

精确几何中 ε≤1.5×10⁻⁹m（R≤1500）。δ=10⁻⁵m 留有严格正余量。距离也有余量：沿 s→q 的方向延长 δ 的点仍在 B(s,R)，故 ||q−s||≤R−δ。这避免了仅解决发射边界，却在圆盘边界被浮点舍入翻转的问题。

## 3. Fraction 证书与浮点范围

生产实现构造时对每条有向支持边 e=(dx,dy) 使用

\[
\operatorname{cross}(e,q-a)\ge 2\delta(|dx|+|dy|),
\]

输出为 float 后，再转为其精确 Fraction，验证至少 δ(|dx|+|dy|)。因为 L1 范数不小于欧氏长度，这足以证明每个输出点的 δ 球包含于所有支持凸包；输出点的凸组合仍满足这些线性约束。两倍生成余量避免把浮点输出正好压在声称的边界上。

独立检查器不导入生产 helper：它用精确 Jarvis 包裹法重建每一个 conv(P∪{vᵢ})，核对支持凸包并逐边验证

\[
\operatorname{cross}>0,\qquad
\operatorname{cross}^2\ge\delta^2(dx^2+dy^2),
\]

完整证书还检查更强的 L1 条件。输入坐标按二进制浮点的精确有理值处理，避免对生产日志中的“通过”字段自证。

传输与 engine 仍使用浮点。检查器限制坐标绝对值≤2×10⁶m，并单独列出宽松的 10⁻⁷m 算术包络，剩余设计裕量 9.9×10⁻⁶m。在普通 binary64 最近舍入下，有限次差、乘、加的误差为数个 10⁻⁹m 量级；加上接收阈值仍远小于该包络。Python 3.10 起 `hypot` 的文档误差界小于一个 ulp，但三角函数仍依赖平台数学库，故这里不把整个跨平台 engine 宣称为形式化区间算术证明。[Python 官方 math 文档](https://docs.python.org/3/library/math.html#math.hypot)

上述保证始终以“C 确实外包真实源、P 是该未清除源的真实正观测、接收模型满足题设”为前提。当前纯几何检查器不检查行为日志来源；后续接入策略仍需独立绑定实际前缀。

## 4. 检查入口与结果解释

```python
from experiments.audit_q4_visible_kernel import (
    audit_visible_kernel_certificate, audit_visible_kernel,
)

audit_visible_kernel_certificate(certificate)
audit_visible_kernel(C_vertices, positive_positions, [candidate], require_l1_margin=True)
```

异常为 `ValueError`；有效证书返回 `passed=True, usable=True`，无输出的保守回退返回 `usable=False, exact_emptiness_proved=False`。多个无序候选应逐个按 `[candidate]` 验证，不把无序列表当作凸多边形。

测试覆盖：顶点到内部源的有理重心构造、超出原正点线段的新核、engine 反例、点/线退化、±1,999,000m 平移、浮点边界下一格、真实生产证书以及漏支持凸包、改源顶点、改正点、改裕量、伪造输出等。测试不证明预测探测成本准确，不证明全局最优，也不说明整局提速；这些仍须后续同场景实验检验。
