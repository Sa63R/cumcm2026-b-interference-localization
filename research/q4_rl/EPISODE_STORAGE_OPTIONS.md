# G1/G3 GAE轨迹存储方案：只改无损表示，暂不实施

建议独立审核后优先试行**仅对新episode使用XZ/LZMA2 preset3**，保留完整原始JSON字节，旧文件不改。四个预定本地TRAIN样本压缩到现有gzip6文件的8.7%–12.1%。这个分母是**当前磁盘上的gzip6大小，不是未压缩JSON大小**。这是有限样本的存储证据，不能当未来产物大小上限，也不足以证明恢复四个训练job的空间安全。继续训练已在当前任务授权范围内；尚需完成的是容量与恢复事务验证。

## 当前写入与恢复依赖

- `src/q4_rl/train.py:338–345`：原生 `json.dumps(..., ensure_ascii=False, allow_nan=False).encode('utf-8')`，gzip level6，临时文件后原子替换。当前已经压缩；不能把启用gzip当新改进。
- `src/q4_rl/training_journal.py:23–34`：每个episode独立只写一次，再把实际压缩文件SHA、seed和文件名加入小型gzip批次索引。不能改成反复重写整个批次的大数组。
- 同文件 `:37–58`：旧索引/旧列表批次与episode reader都明确使用gzip。XZ需要显式兼容扩展，不能把XZ数据偷偷装进 `.gz` 名称。
- G1 `micro_train.py:278–318`、G3 `memory_train.py:279–319`：先保存包含pending seeds/action_seeds的 `latest.pt`，再按 `executor.map` 原始任务顺序接收结果；GAE actor advantage在append之前附加，全部字段进入journal，然后训练直接使用内存里的 `batch/records`。
- G1 `:339–353`、G3 `:340–354`：更新完成、计数增加并清空pending后，提交包含 `last_progress/raw_attempt` 的模型事务，再写编号checkpoint及进度。读取压缩文件不参与这次PPO更新。
- G1 `restore_checkpoint:139`、G3 `:141`：恢复依赖latest中的模型、Adam、Python/Torch RNG、训练config、计数和pending预留；不读取旧journal作为训练回放。行政中断恢复相同pending seeds/action_seeds，用新的attempt编号重做整个未提交批次，旧尝试仍留档。压缩改动不能改变这套事务、任务顺序或将残留轨迹偷偷用于补齐批次。
- `scripts/q4_cpu_supervisor.py:242–243` 的白名单目前只有 `.json/.jsonl/.log/.pt/.json.gz/.zip`，新 `.json.xz` 若不加入会**漏传**。`q4_fetch_results.py`本身按对象列表取文件，与codec无关，但它会重复下载已有文件，应使用先前提出的SHA缓存/流量门控。
- `experiments/q4_bc_fit_diagnostic.py:25–59` 还有硬编码v1/gzip reader；相关审核器必须明确支持新格式或清楚拒绝，不能报告“旧reader通过”来证明新档案可恢复。

## 四个预定样本的实测

数据源是已完整回读的 `server-memory-v4-training-001`，G1/G3 h64各取首个BC episode（8012000）和256BC之后首个PPO episode（8012256）。先按已有readback及小索引核对原文件SHA，再进行压缩。没有按压缩效果、策略结果或文件大小选择样本；没有访问验证数据库、新下载或新仿真。

| 样本 | 现有gzip6字节 | XZ3字节 | XZ/现有gzip | gzip6压缩CPU秒 | XZ3压缩/解压CPU秒 | XZ进程峰值MiB |
|---|---:|---:|---:|---:|---:|---:|
| G1 BC | 2,277,304 | 206,536 | 9.07% | 0.344 | 0.250 / 0.015625 | 120.98 |
| G3 BC | 2,864,822 | 249,356 | 8.70% | 0.406 | 0.297 / 0.031250 | 144.14 |
| G1 PPO | 2,578,922 | 275,828 | 10.70% | 0.297 | 0.344 / 0.031250 | 88.86 |
| G3 PPO | 2,584,897 | 311,968 | 12.07% | 0.313 | 0.453 / 0.000000* | 87.59 |

每种codec各用一个新进程、固定一核、单次压缩/解压；计时仅覆盖codec，不含初始读取、解压源文件或JSON编码。`*` Windows CPU时钟粒度约15.625ms，零读数不代表零计算；原始wall计时同时保存在JSON。峰值是进程工作集高水位，含Python、原始和往返JSON缓冲区，不是编码器独占内存，也不是训练进程总内存；压缩阶段峰值约74.3–106.0MiB。单次时间有噪声，不保证比gzip快；两条PPO样本XZ更费CPU，但绝对增加约0.05–0.14秒。

原始JSON约21.3–41.2MB，四例XZ往返后**所有原始字节严格相等**，包括浮点字符串、字段、重复数据、数组顺序及空白。不做舍入、字段裁剪、dtype变换、轨迹摘要或候选删减。新的gzip基准比既有文件少53字节文件名header，已披露，压缩率仍除以既有文件大小。

机器可读结果：`episode_storage_probe/measurements.json`；复现脚本：`episode_storage_probe/run_probe.py`。初探XZ0/3/6后，完整四例固定比较gzip6与XZ3；没有继续扩样追求有利比率。既有SCST单例level9基准只省约4.1%但CPU约2.1倍，不足以解决当前空间限制；见其他工作树 `q4-rl-feature-cache/research/q4_rl/journal_benchmark/timing.json`。

这些是旧G1/G3同schema的BC/PPO，而非新GAE日志。GAE新增actor advantage标量预计影响很小，但这只是结构推断，不是已经测得的新GAE压缩率。四例20–41MB解压大小不构成压力场景或最大候选数的上界。

## 最小兼容改动提案

1. 新增独立存储codec函数，输入仍为原来JSON编码得到的**完全相同字节**；只对新episode调用标准库 `lzma` 的XZ preset3，保留临时文件→原子发布，不产生临时未压缩JSON文件。旧gzip writer和模型/GAE代码保留。
2. 新episode文件使用 `.json.xz`；小批次索引仍可用gzip。新索引明确版本及entry codec，记录实际压缩SHA/bytes，建议再记解压JSON SHA/bytes。reader同时支持旧列表gzip、旧v1索引及新版本，按明确codec解码，验证压缩SHA、解压长度/SHA和seed。未知codec、路径逃逸、截断均拒绝。
3. 存储版本/参数写独立run sidecar和发布manifest；不把它伪装成policy feature或训练算法参数。旧checkpoint的训练config严格相等校验保留，resume继续使用原config和pending语义。新源代码版本、codec sidecar及恢复起点SHA应单独冻结，避免静默换实现。
4. 新增 `.json.xz` 同步白名单，archive/payload先于其完整索引和完成标记发布；所有S3操作保留 `--s3-no-check-bucket`。兼容更新所有需要读取新轨迹的审核入口。旧冻结评估源及已生成结果不覆写。
5. 只需小型合约验证：混合新旧格式完整字节往返；缺损/错SHA拒绝；中断在payload写完但索引未提交、以及checkpoint提交前后的事务恢复；相同fabricated batch的记录顺序、returns/GAE和模型更新等价。无需为存储验证启动真实场景或大训练。

无损压缩并不保证固定墙钟截止时完成同样数量的批次；它保证逻辑样本顺序和训练函数不变，存储时间差可能影响截止前完成的前缀长度。必须保留行政截止计数，不能把存储变快宣称为策略收益。

## 有界空间与对象归档

约21GB空闲不能直接视为可给训练使用的额度。应先以实际字节扣除18GiB保护线、C评估预留1GB、终局checkpoint/日志和共享盘增长余量，再给**四job合计**确定一个新增本地字节预算；不能四组各自认为有同一份余量。四例XZ只有0.21–0.31MB/episode是估计，不能据此无门控恢复32批次。

保守第一阶段是不删除任何已有轨迹，只让未来新文件使用无损编码，并在提交下批任务前进行空间/预算预留；不足则在事务边界暂停。字节门槛必须给当前批次返回结果和临时发布留足余量，不能在写完样本之后才发现硬限额，造成实际执行过的样本无证据。新格式只能降低增长率，不能在无限期不删除的前提下提供永久空间上界。

如果需要长期固定磁盘占用，须另行明确授权并审核**未来新产物的对象归档与本地副本回收**；旧5184文件清理授权不能扩大为自动清理新训练数据。可执行的归档事务为：

- 只有整个批次及其模型事务明确完成后，按SHA不可变对象名上传该批所有episode、索引、配置/源引用与对应checkpoint；pending、失败和行政尝试分别完整归档并标注不参与学习。
- 使用单上传器与共享100G流量账本，逐个已知SHA对象跳过已有成功上传，保存对象名、SHA、大小和上传完成回执。必要的完整校验GET必须计入预算；避免本地再取整批作为默认审核。
- archive完整性确认后写不可变“已归档”提交标记，本地常驻latest、最后已提交checkpoint、pending预留、全部轻量索引/回执。原格式离线审核需要时按明确hash hydrate；普通resume无需拉取历史训练轨迹。
- 在未取得未来副本回收授权前，上传不会释放磁盘；不能删除，也不能因上传成功就声称当前磁盘问题解决。另一种无本地大副本方案是新episode压缩后直接流式写对象、再发布本地小索引；但这改变写入故障路径，需要单独审核网络失败下的事务保全，暂不作为最小改动。

每批出站/校验流量、archive状态、当前磁盘占用应独立记录。持续上传失败或流量不足时在批次边界暂停，不能静默丢轨迹。以上均为提案，本轮仅新增这份说明、probe脚本和测量JSON；没有生产改动、远程操作、文件清理、训练或仿真。
