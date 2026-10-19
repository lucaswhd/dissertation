# 论文实验可追溯性审计

审计日期：2026-10-18（UTC）。对照文本：`/root/dissertation_local_audit/thesis_paragraphs.txt`。此次仅读取本地代码、日志、JSON/CSV、Parquet 与保存的预测张量；未运行训练、模型推理、仿真或网络操作，未修改原始文件。

## 结论与核验范围

当前论文主线是 HRC 数据上的 π0.5 A20/B20/C20 frozen audit，不是 ACT 或 RoboBenchMart/Octo 原子任务比较。论文核心数字可以追溯到实际预测张量和数据导出。本次另写独立评分代码，读取全部 15 份 `predictions.npz` 与对应 Parquet 真值，共重计分 **9,780 个 32×23 action chunks**。逐 anchor 的三组 MAE、null MAE、夹爪事件计数，以及逐 seed 的 context 汇总与原档一致，所比较字段的最大绝对差为 **0.0**。另将重算结果聚合回`comparison.csv`的156行可比较指标，mean/std最大差也为0。这证实的是保存结果的数值一致性。

**训练过程、模型重新推理、仿真行为与完整端到端实验：Not independently verified。** 旧 `qa_report.json` 中的 repeated-inference、归一化重新拟合与“independent spot check”仅为历史记录，本次未将其当成独立复现。

当前存在重大恢复缺口：A20 的 `20000` checkpoint 目录尚存；B20、C20 在 frozen summary 中指定的 checkpoint 目录目前均不存在。B/C 原始训练日志记录了 20,000 步保存成功，预测张量也保存完整，但这些不能代替权重。`act_runs/` 当前为空。发布资料应区分“可重算保存预测的分数”和“可从权重重新推理”。

## 样本与随机性

| 项目 | 可核对的实际定义 |
| --- | --- |
| 数据集 | 45 source trajectories；两个导出的 `meta/info.json` 均给出 train/val/test=37/4/4 episodes |
| 总帧数 | train 341,269，val 31,084，test 37,317；合计 409,670 |
| validation source IDs | `[0,2,15,18]` |
| test source IDs | `[10,17,44,47]`；不用于主要 policy 比较 |
| teacher-forced manifest | 652 anchors：wait 52，pickup/follow/radio 各 200；每个非wait context每source50个 |
| 模型数 | A20/B20/C20 各一个训练checkpoint；每个20,000 optimizer updates |
| training seed | 42；未测训练seed不确定性 |
| flow-noise seeds | `[0,1,2,3,4]`，同anchor在三模型间匹配噪声；不是5次训练 |
| protocol seed | 20260907 |
| conditioning | 每模型324 anchors=validation108（pickup96、radio12）+train coverage216（pickup192、radio24） |
| conditioning sources | validation4个；coverage8个 `[4,6,12,14,23,35,46,52]`，不与test重叠 |

`anchor_manifest.json` 的 SHA-256 本次重算为 `d5553026605163a74820a865db80e90e81249d3fcceba9cb19655a43fa2bd6a8`。本次核对所有保存预测均有限、形状为 `[652,32,23]`、anchor顺序完全一致；所有选中anchor及其 `[t,t+32]` 边界保持source/episode/coarse context。A与B/C在匹配anchor处的物理state/action完全相同。

Anchor是来自4条validation source的相关窗口，652不能解释为652条独立轨迹。事件计数在窗口中累计，不能直接当成去重后的整条轨迹事件数量。

## 配置与训练日志

实现入口：`/root/autodl-tmp/openpi_hrc/scripts/b1k/train_b1k.py`；配置：`src/openpi/training/config.py:_make_hrc_train_config`、`_get_hrc_configs`；优化器：`src/openpi/training/optimizer.py`。

| 项目 | A20/B20/C20 共用配置 |
| --- | --- |
| 模型 | π0.5；`gemma_2b_lora` + `gemma_300m_lora`；额外冻结视觉塔；EMA=None |
| 物理state/action | 23维；模型action32维，horizon32；token budget200；discrete state开启 |
| 输入图像 | head经padding resize到224×224；wrist slots为masked placeholders |
| batch | physical microbatch2，梯度累积4，有效batch8 |
| optimizer | AdamW，β1=.9、β2=.95、ε=1e-8、weight_decay=1e-10、global clip1.0 |
| LR | warmup1000，peak2.5e-5，cosine schedule30000，末端2.5e-6；训练在20000停止 |
| objective | normalized flow-matching loss；A/B/C不设physical-dimension loss mask，32维含9个padding均参与 |
| 训练验证 | 每1000步，10批×2；使用training normalization；不等于frozen physical-action audit |
| 保存 | 每1000步，长期保留间隔5000 |

| 标签 | 配置ID / dataset | 干预 |
| --- | --- | --- |
| A20 | `pi05_hrc_e6_rerun_20k` / `hrc_pi05_continuous_rerun_v1` | 长指令、自然anchor采样、z-score |
| B20 | `pi05_hrc_b_subinstr_20k` / `hrc_subinstr_rerun_v1` | 局部sub-instructions与decision采样同时改变；task_weights=(.8,.2,0)，z-score |
| C20 | `pi05_hrc_c_qnorm_20k` / 同B | B的整套recipe改为state和action共同quantile scaling |

A→B不是单独语言消融；B→C不是单独state-tokenization消融。代码中若有早期“only labels change”“state repaired”注释，应以当前配置与论文的限定表述为准。

三份原始日志分别是 `logs/hrc/train_e6_rerun_20k.log`、`train_b_subinstr_20k.log`、`train_c_qnorm_20k.log`。本次各解析出200个训练点（100..20000）和20个验证点（1000..20000），结果另存本目录 `A20_training_curve.csv`、`B20_training_curve.csv`、`C20_training_curve.csv`。终点train/val分别为 `.1041/.0071`、`.1935/.0082`、`.0417/.0041`。日志是训练留下的记录，未重新运行训练；B/C归一化不同，不能用loss纵轴直接比较物理控制准确性。

## 指标核对

物理分组：base=0..2，grippers=14/22，arms=其余18个torso/arm维度。预测与真值均为23维物理actions；null保持anchor关节pose、零base速度、保持前一gripper命令。Skill=`1 - pooled MAE(model)/pooled MAE(null)`，null分母0时未定义。

夹爪事件为相对前一命令的sign变化，按gripper、方向、±8帧贪心一对一匹配；context内先pool事件计数，再算P/R/F1。存在expert事件却无TP时F1=0；无predicted事件时precision未定义；无expert事件时F1未定义。不能将空值填成0并视作有效样本。模型汇总的std为5个inference seeds的样本标准差；comparison表的`n=5`不是source数，dispersion行的`n`另指anchor数。

| pickup指标 | A20 | B20 | C20 |
| --- | ---: | ---: | ---: |
| arm MAE | .020449934 | .019986582 | .019941248 |
| arm skill | -.010727462 | .012173453 | .014414084 |
| arm skill sample std | .019401267 | .023083684 | .027448440 |
| event F1 | .111995626 | .335790336 | .382495305 |
| event precision | .131349206 | .283295195 | .316334314 |
| event recall | .100000000 | .414285714 | .485714286 |

以上数值由本次独立脚本从预测与真值重算，和论文摘要及正文第5章的四位小数吻合。它们不支持可靠抓取或准确arm控制的强主张。

lookup baseline分数可在`comparison.csv`与`baselines/summary.json`找到，代码显示train-only拟合。本次没有重新拟合lookup表，**lookup重新拟合：Not independently verified**。保存表中的base persistence skill为pickup .540214536、radio .578696694、follow .509309511，均高于对应learned checkpoints。

## Conditioning与frustum

本次直接从每模型`per_anchor.csv`按validation筛选，逐anchor重算 `arms_D_image/arms_D_noise` 再取平均，得到pickup B=.358100543、C=.441394964；radio B=.285087432、C=.437800440，与论文吻合。只重算了保存的输出差，未重新生成perturbation预测。10,000次anchor bootstrap的实现可见`dissertation_protocol.py:bootstrap_summary`；本次未重跑bootstrap。比值小于1是相对噪声的局部响应尺度，不是vision无用或不被依赖的证据。

本次从`dissertation_target_visibility/per_anchor.csv`独立计数：全45source pickup valid anchors为154,560/179,353=.861764230（86.18%），decision anchors为10,010/11,115=.900584795（90.06%）；validation分别为10,911/12,302=.886928955与903/988=.913967611。原始数值与论文一致。**几何投影重新计算：Not independently verified**；这里只重新计数已有布尔结果。

该变量是3D target center处于几何camera frustum，不含遮挡检测、像素mask或可见性人工标签。罐目标身份通过后续deposit段运动量的一对一assignment推断；radio使用配置中的静态位置。45source中的test几何只作描述性分析。旧代码的human_visible_flag不是独立camera calibration。

## Closed-loop与早期ACT

本次确认`pi05_rerun_v1_closed_loop`下有5份完成的`policy_rollout_metrics.json`，全部pickup/radio task contact=0。其source0015与source0000是validation，source0038为training；execution horizon、时长和阈值存在不同设置，不能当成完整matched benchmark。source0038标准设置collision=2879，horizon1设置=0，contact仍=0，与论文相符。source0000_seed1目录存在但无完成metric，不能计为完成run。

另有`pi05_b_closed_loop/b20_TRAIN_source0038_seed0_h8/policy_rollout_metrics.json`，为额外单例，pickup/radio contact均0；最近hand-to-can距离分别1.1429/2.3158/1.0949m，radio1.8509m。本例不属于论文“五份A20历史metric”的计数。未发现可用于A/B/C统一成功率排名的完整closed-loop矩阵；未进行视频独立复判或仿真复跑。

ACT早期实现位于`hrc_pipeline_traintest/scripts/{training,evaluation,preprocess}`；示例训练recipe是seed1000、batch8、20k steps、`lerobot-train --policy.type=act`。早期adapter丢dim6，物理动作23→22维，state44维；不能与论文π0.5的23→32 padded模型接口混用。当前`/root/autodl-tmp/act_runs`为空，因此ACT完整权重与训练复现不可核验。

## 资产与版本边界

仓库norm资产的当前SHA与历史QA一致：A=`038b3453843efdb1dcab05cb679dbe58b1ab94f506e834a2f81428a6051f685f`；B=`b0328714f8415e586ffe63dc9eba6c65c3ff0446baa8e79c57f581f7d93da793`；C=`ef9567d4aed6c3d3fc8963173a445157b9804403bb0c2353b8983aeb4d4aab2b`。A当前checkpoint中的norm与仓库资产字节一致；B/C checkpoint已缺失，无法做当前checkpoint内核对。本次没有从37source重新拟合norm。

Frozen README记载完成时commit `b8e247845cb03cd721f2032a13fa2429863f4b13`且worktree dirty；论文附录给出归档commit `cfd097d6ca0f957b68d05616a3b71b86dabdf2cf`。这可能是后续归档提交，但单凭两份文字不能证明历史执行对应精确源码；应同时发布实际脚本、manifest、hash与变更说明，不能把单个commit号等同于完整训练环境。论文RTX4090硬件等描述本次未独立核验。

RoboBenchMart目录确有2026-09/10的Octo与π0.5 42-config评测结果，属于另一条任务线。未将其成功率、1260回合或图片用于当前HRC论文主张。

## 复核产物

- `independent_archive_checks.json`：逐模型/seed核验、重算pickup数值、conditioning分层、当前checkpoint清单、geometry重计数与closed-loop清单。
- `audit_saved_predictions.py`：本次独立评分脚本。仅执行读取与写staging，不加载项目评分实现、不加载模型。需要OpenPI环境的NumPy/PyArrow；脚本记录的scope为重计分而非实验独立复现。
- `A20_training_curve.csv`、`B20_training_curve.csv`、`C20_training_curve.csv`：直接解析既有训练日志，均保留source_log与metric定义。

历史`_invalid_prompt_stable_protocol_20260907`与`_invalid_f1_null_protocol_20260907`必须排除在主表和主图之外。A20 corrected结果明确由旧保存预测重新评分而来，这段历史应保留，不能把旧无效summary混入最终比较。
