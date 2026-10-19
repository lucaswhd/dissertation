# Benchmark设计与现有证据边界

本文件前述构建流程主要依据2026-10-19（Asia/Singapore）对`/root/hrc_pipeline`旧采集主树的只读审计，给出可写进论文的实际设计及后续实验规范。这里没有新增训练或仿真结果；所有建议项必须标为设计，不能填成已完成实验。

## 1. 已实现的benchmark构建流程

源BEHAVIOR示教HDF5 → 在base场景补齐外部任务所需对象 → human asset/运动学actor → 按轮次指令拼接脚本接近、源示教状态回放、返回/面向、迁移 → 双视角视频、轨迹、命令JSON、阶段标注HDF5。

具体例子：`configs/hrc_combo_D.yaml:97–112`配置第一轮external task-0045 cook_hot_dogs，第二轮base task-0046 cook_bacon且replay_all_segments=true。它提供了任务序列和场景整合的实验条件，不等同于策略自主完成两项烹饪。raw reward spike把base轨迹分片（utils:66–85），含正奖励后的尾部也可成为新片段；不能一概把每片段称为独立完整任务。

已有采集器支持按共同源episode编号配对（run_collection_AB:311–331），以及固定base、变化external episode的采样（267–309）。固定base会使base物体布局和示教重复；这些派生episode之间具有相关性，不能仅按导出目录随机划分来证明独立泛化。

## 2. 应拆开的评估层次

| 层次 | 可以衡量的目标 | 所需输入/结果 | 当前核心可支持到什么程度 |
|---|---|---|---|
| 构建完整性 | 文件/字段/长度、状态转移、轮次命令、相机元数据 | 导出HDF5、视频、commands及manifest | 已有导出，可只读统计；batch默认校验很弱 |
| 几何与视觉质量 | 人物出生点有效、平面距离、路径、画面可见率 | 位置、相机、RGB/分割、必要场景几何 | 位置和启发式visible已记；遮挡可见性需独立验证 |
| 行为克隆离线 | 学习动作与监督目标误差 | 对齐的obs/action、按来源分离的train/val/test | 需额外数据转换与模型训练证据；脚本导航阶段action为零会误导 |
| 阶段识别/切换 | phase accuracy、macro-F1、边界延迟 | 固定FSM标签、模型预测 | 标签可用于训练；标签本身是oracle，不能作为测试时观测 |
| 在线闭环 | 完整任务谓词成功、返回、人机安全距离、长程成功 | 冻结模型、seed、可执行控制、无逐帧源state恢复的rollout | 核心构建入口不提供此证据 |
| 跨任务泛化 | 新源episode/场景/组合下成功率 | 严格去重来源、hold-out和逐rollout结果 | 应单独设计，不能由导出数据量推断 |

## 3. 防止监督与划分泄漏

1. 以**源文件+源episode+片段区间**为最小分组，所有复用同一源示教的合成副本归入同一split；固定base的所有复用至少在base任务评估中同组。将训练、人类轨迹来源、视觉变体、采集重试的重复关联入manifest。
2. 建议manifest记录：组合ID、base/external源标识与内容hash、两个episode、起止帧、kind、人物资产版本（不含受限内容）、spawn seed、版本、输出hash和split。对同一源的重叠区间做分组去重。
3. 将FSM ID、reward、done、task_success、未来轨迹及动作后果从“非oracle”策略输入中排除；可以单列显式oracle-FSM基线，但不能与纯RGB/语言方法混淆。
4. 逐demo语言先用commands.json核对；当前write_hdf5可能将末轮指令写给所有demo（minimal:345）。没有修复/证实前不要把HDF5 instruction当可靠逐轮真值。
5. 先验证回放state/action帧对齐，再谈action监督；对脚本导航、面向、迁移阶段，action为零而base位姿直接被修改，应改用明确定义的delta-pose目标，或在报告中限制动作学习评估为可信示教操纵段。
6. 视频与HDF5需逐episode检查帧数、轮次偏移、FPS/timebase；合并HDF5不含RGB，不能只检查HDF5就认为视觉输入完备。

## 4. 建议指标与定义

- **构建产物成功率**＝通过schema与文件完整性检查的采集次数/总尝试次数，与task success分开；分别报重试、超时、失败原因。
- **轮次任务成功率**：在线环境最终目标谓词整体成立的比例；不要用“任意一帧reward>0”替代。报告每项任务、每场景与整体。
- **整段HRC成功率**：全部配置轮次任务成功、机器人返回到指定stand_off容差、末态朝向误差合格；任一失败即整段失败。阈值在评估前固定，不从结果倒推。
- **返回指标**：最终人机距离误差、朝向角误差、返回时长、路径长度；运动学脚本结果只能描述生成质量。
- **可见性**：分别报告现有base-yaw几何proxy、相机frustum、分割像素占比/遮挡后可见率；不得混称。横纵FOV由真实相机内参确定。
- **安全距离/碰撞**：xy最小距离、人体代理/机器人接触次数、场景穿插率。现有include_contacts=False且kinematic路径不能用于物理安全证明。
- **离线动作误差**：每动作维度MAE/RMSE、训练集归一化参数、chunk长度与预测时域；分阶段/任务报告并注明是否包含脚本零动作。
- **不确定性**：独立源episode或独立rollout单位上的置信区间；多seed分别列出。源回放重复帧不能冒充独立样本。

## 5. 可实施但尚未完成的消融

| 消融 | 固定项 | 检验假设 |
|---|---|---|
| 固定相机 vs phase pitch | 相同源示教、人物轨迹、模型与split | 下视任务与抬头交互是否受益；需先解决像素-外参时序 |
| 静止人物 vs waypoint/记录人轨迹 | 相同任务、场景与seed组 | 人体运动是否改变识别/返回/可见性 |
| 单任务 vs 两轮组合 | 匹配每项任务的源示教分布 | 多轮记忆与返回衔接代价 |
| 无人输入 vs 人物pose/RGB输入 | 相同模型容量、训练预算和测试环境 | 人物信息的额外价值 |
| oracle FSM vs 预测FSM | 同一低层控制器和场景 | 阶段识别误差与任务执行误差拆解 |
| 仅base布局 vs hold-out源/场景 | 固定训练预算和方法 | 数据重复与真实泛化的区别 |

## 6. 论文叙述约束与真实可画图

可以画：核心组件依赖图；PROJECT_AUDIT.md中的10状态调度图；双视角相机与人机几何示意；按真实HDF5统计的phase timeline、round_id和距离曲线；数据构建与未来闭环评估分开的流程图。

不能画成已验证事实：感知自动触发成功转移、学习策略自主操作所有任务、语音实时理解、多策略在线对比结果、严格无碰撞保证、500完整episode已交付。架构文档的5×100=500（ARCHITECTURE.md:10–13）是目标；实际规模应由主代理只读统计填入。TaskSegmentDone和AllTasksDone代表调度完成，不是已验证任务谓词成功。

## 7. 当前研究分支的补充边界

`/root/autodl-tmp/hrc_pipeline_traintest/scripts/collection/hrc_multitask_demo.py`已经提供online-nav和policy-rollout，不能用旧树缺接口来否认其实现。policy_step:812–868实际向服务端请求动作并env.step；policy Execute:5359–5389/5510起与源回放分支分开。独立结果报告需以主代理收集的checkpoint、rollout日志与metrics为准，本报告不新增成功率。

该策略评估仍接受scripted oracle FSM phase：`scripts/evaluation/pi05_policy_client.py:191`把Task和Phase组成prompt；轮末谓词在multitask:5562–5571评估，无论结果如何均会转入后续返回阶段。论文宜称“scripted FSM编排下的策略闭环执行”，避免称模型自主决定全部任务阶段。导航采集可启用真实物理动作（multitask:2293–2338、4531–4543、5058–5061）；不应把旧版零action导航当作全部最终数据的描述。实际每批数据使用何种模式需从命令/attrs逐批核实。
