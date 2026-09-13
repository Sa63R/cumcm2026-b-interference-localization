# 干扰源本地测试模拟器

这是根据本次 exe 静态分析和附件2接口说明实现的独立本地版本。支持第三问、第四问、固定种子地图、150米网格空间误差、微秒计时，以及四个机器狗HTTP接口。可在macOS直接运行，无需Windows、登录账号或安装第三方Python库。

## 启动与操作

需要Python 3.9及以上。在此电脑可双击本目录的 `start.command`，或在终端执行：

```bash
cd '<PROJECT_ROOT>/local-jammers-simulator'
python3 -m jammers_local serve --port 2027 --problem 4 --seed demo-20260912
```

浏览器打开 http://127.0.0.1:2027 。接口只监听本机回环地址。默认robot_id是 `local-test`，arena_id是 `default`。端口占用时改用例如 `--port 2028`，再在策略中修改地址。结束服务请在启动终端按Ctrl+C；网页的“停止演示”只停止示例策略。

网页可以：

- 选择第三问或第四问、输入种子后新建场景。同一种子生成相同场景。
- 运行一个观察反馈驱动的示例策略，查看轨迹、清除数、检测数及时间分解。
- 点击地图选坐标，再手动进入、检测、清除、退出。
- 打开“显示真值（调试）”，查看源的位置、类型、朝向和清除状态。
- 导出完整场景和已接受动作记录，以后重放核对。示例策略结束、新建下一局及正常停止服务时也会保存记录到results/server。

默认新场景立即开放接口，省去官方准备和5秒倒计时。25分钟窗口从新场景创建开始，进入后还有最长20分钟程序时间；到期可点“新建场景”开始下一局。各动作的5秒、3秒只累加虚拟时间，不在现实中等待。网页示例为了可视化每个动作额外暂停约6毫秒，这不改变虚拟时钟；批量测试没有这个暂停。

## 接入已有策略

四个POST路径保持 `/enter`、`/measure`、`/clear`、`/exit`，返回字段使用附件2的 `accepted`、`virtual_time_s`、`measure_result`、`svd_deg`、`clear_result`、`exit_reason` 等。将现有程序的base_url改为 `http://127.0.0.1:2027`，robot_id改为 `local-test` 即可。

本工作区已有的客户端可这样接入（该客户端需要Python 3.10及以上；每局创建一个新客户端，先在网页新建场景）：

```python
import sys
sys.path.insert(0, '<PROJECT_ROOT>/code/src')
from simulator_client import SimulatorClient

with SimulatorClient('local-test', base_url='http://127.0.0.1:2027') as client:
    print(client.enter())
    print(client.measure((300, 400), 1))
    print(client.clear((300, 400), 1))
    print(client.exit())
```

只有合法measure会切换检测频道，clear不切频。无信号也消耗检测和切频时间。拒绝响应virtual_time_s为0，不能据此将客户端时钟清零。

每个新动作使用新的request_id；同ID、同内容返回第一次的完整响应，不重复执行。同ID改路径、位置或频道返回409。未知字段、结构错误、队号不匹配不会占用ID。`1`和`1.0`、正零和负零按规范化后的数值比较。

四个机器狗接口不返回真值。本地专用的 `/local/state?truth=1` 和导出文件供调试及评测器查看；策略测试请只使用四个机器狗接口，不读这些调试入口。这里是接口职责区分，并非防止恶意Python代码访问内部对象的安全沙箱。

## 固定种子、批量评测与重放

以下命令在本目录执行。文本seed先经SHA-256变为32字节，是本地提供的便捷输入方式。若已有确切的生成器输入，使用 `--seed-hex` 传64位十六进制文本。

```bash
python3 -m jammers_local generate --problem 4 --seed 42 --output results/scene-42.json
python3 -m jammers_local serve --scenario results/scene-42.json --port 2028
python3 -m jammers_local batch --problem 4 --count 100 --start-seed 1 --output results/q4-first100
python3 -m jammers_local replay results/q4-first100/case-0001.json
```

batch使用与HTTP完全相同的Session和Engine，省去HTTP开销。输出每局完整记录及summary.json，包括全清比例、虚拟时间、移动距离、检测/切频/失败清除次数。统计用的真值由评测器在结束后读取，传给策略的client只提供动作和反馈。

默认baseline是演示用搜索策略，使用500米网格、循着测向前进、失去信号后回退缩步，以及近距离清除。它没有利用真实源位置，也没有做速度优化。`examples/policy.py`展示自定义入口：

```bash
python3 -m jammers_local batch --problem 3 --count 10 \
  --strategy examples/policy.py:run --output results/custom-q3
```

自定义函数签名为 `run(client, problem)`。client提供enter、measure((x,y),channel)、clear((x,y),channel)、exit；measure/clear返回与HTTP一致的字典。函数负责进入和退出。示例模板只扫描原点，不是完整搜索策略。

重放核对已接受动作的业务返回值和微秒时间，忽略现实时间戳与进入时剩余现实秒数；它不重现现实时间等待、网络断开或服务端鉴权。输出目录非空时batch拒绝覆盖，便于保留对照实验。

## 规则与实现范围

- 源在1770米圆盘内按面积生成；源数10至16，频道从1至20无放回选择。
- 第四问定向数量从1至N均匀抽取，允许全定向；第三问全部全向。
- 接收半径为1000至1500米，按整数微米生成；场景位置与朝向也保留整数单位。
- 计数随机源使用HMAC-SHA256与独立标签计数器；测向噪声使用BLAKE2b八字节摘要和150米网格平滑插值。
- 角度先量化再钳制到真方位±1°，最后归一化；同点同频道测向误差固定。
- near需要距离≤5米且通过覆盖判定；清除只要求指定未清除源在20米内，与朝向无关。
- 初始位置(0,0)、频道1。移动按5米/秒，每段取整到微秒；检测5秒、切频1秒、清除成功5秒/失败3秒。
- 输入坐标为有限浮点数，单坐标绝对值≤2000000米，可在目标圆外。

完整差异和证据来源见 `COMPATIBILITY.md`。这不是官方程序的完整克隆，没有实现登录、正式地图下载、加密上传或服务端计分；不声称演练分布就是正式分布，也不声称跨语言浮点结果逐位一致。

`profile=fixture`允许1至20个源，专门测试边界，界面会标识fixture；它不属于10至16源的演练分布。例子 `examples/backside.json`只有一个向东发射的定向源，可用来验证背面near与clear的区别。

## 验证和文件说明

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖生成约束、方向边界、接收距离边界、近距离/清除边界、噪声确定性、微秒计时、请求幂等、并发冲突、JSON错误、超时及记录重放。本工作区还包含现有simulator_client真实回环HTTP集成测试；拷贝到其他目录后没有该客户端时会跳过这一项。

核心文件：`jammers_local/algorithms.py`为上一轮静态重写算法；`core.py`为场景/引擎/请求状态机；`server.py`为本地HTTP服务；`dashboard.html`为无需CDN的网页；`baseline.py`为演示策略；`__main__.py`为命令行入口。
