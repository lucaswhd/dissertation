# v8新增证据图：Figure 3.6与Figure 5.4

日期：2026-10-21。仅新增两组图、专用生成脚本及`v8_`数值表；未覆盖v7原图、既有audit数据或原始实验记录，未运行训练、推理、仿真或v7全量audit。本报告的caption、lead和traceability片段供主代理整合正文。

## Figure 3.6：设计与数据

标题：**Temporal and Action Characteristics of the HRC Demonstration Dataset**。

文件：

- `figures/dataset/fig_3_6_temporal_action_characteristics.png`（2280×2540，400dpi）。
- 同stem `.pdf`、`.svg`，画布5.7×6.35in；建议整图按5.7in宽插入，完整图及图注留在同页。

采用三个纵向panel，避免A4正文宽度下横排文字拥挤。数据总体现有45 retained sources，train/validation/test=37/4/4，全部作为描述性数据集分析；不以test几何或动作统计选择模型。

| panel | 变量与单位 | 实际population / 输入 | 解释边界 |
| --- | --- | --- | --- |
| (a) | 非standby局部任务段duration，秒，`(stop−start)/30` | 实际保存的495个segment：Acquire bin90、Pick can135、Deposit135、Put down90、Radio45，来自45source的真实subinstruction plan | 每个点的单位是segment，不是独立source。重复can cycles与两round重复语义均保留；不从min/median/p90合成数据点；不把不同segment中位数相加当总时长中位数 |
| (b) | 每source/context平均18维`abs(next-action target−current state)`，rad | 每context45个source均值，共180个统计量；从最终B/C numeric Parquet读取409,670行，去除每source每context最后一个无同context下一帧的raw行，实际409,490行参与 | 指command target相对当前joint position的幅度，不是观测到的实际关节位移、模型误差或32-step audit metric。先source内均值，再画45source分布；不按帧数加权混合不同source |
| (c) | raw frames与B/C valid anchor counts，千个；条末百分比=valid/raw | 实际Parquet boolean mask与context ID，全45source，387,544 valid anchors | 两种计数并排，不相加；“valid anchors”包括各自split的eligible rows，不表示test加入训练；未使用B manifest的stale anchor counts |

(a,b)箱体为真实样本的median/IQR，whiskers使用10th–90th percentile convention，未绘whisker外极值；不是CI。两轴使用log scale以保留跨量级结构，图注与轴标签明确标识。

### 动作单位核实

为避免混合米和弧度，专门读取本机R1Pro URDF查证18个关节类型：`torso_joint1..4`、`left_arm_joint1..7`、`right_arm_joint1..7`全部为`revolute`。因此图中18维使用rad，有直接本机asset依据。生成脚本在提取时断言18/18 revolute；`tables/v8_evidence_figure_statistics.json`仅保留joint/type/unit列表与URDF SHA-256及本地路径，**未将受限URDF或场景资产复制到仓库**。physical action/state索引为`3..13`与`15..21`，不包含base/grippers/padding。

### 可直接用于正文的结果

真实segment中位数：Acquire trash can37.8667s、Pick soda can22.7667s、Deposit6.3667s、Put down9.8167s、Radio interaction4.8667s。该语义聚合与附录按round/segment位置分组不同，不能用它们直接替换附录逐段中位数。

| coarse context | raw frames | B/C valid anchors | decision valid anchors | source内平均arm delta的跨source median（rad） |
| --- | ---: | ---: | ---: | ---: |
| wait | 2,025 | 585 | 0 | 1.4917354e-5 |
| pickup | 239,162 | 223,515 | 11,251 | 1.4854678e-3 |
| follow | 18,234 | 16,794 | 0 | 9.1627007e-6 |
| radio | 150,249 | 146,650 | 4,251 | 8.8291068e-4 |

Decision列保存在统计表以便追溯，不挤进panel(c)的图例。此处15,502/387,544是全split描述性口径；论文3.955%特指training有效anchors的12,785/323,255，二者不得互换。20%仍只是B/C配置的sampler target，本图未声称观察到该比例的真实training draws。

### Recommended caption（English）

**Figure 3.6: Temporal and action characteristics of the 45 retained HRC sources (all splits, descriptive).** (a) Durations of 495 measured non-standby sub-instruction segments; n counts segment instances, including repeated cycles. (b) Across-source distributions of the mean absolute difference between next-action targets and current states over 18 revolute torso/arm joints, with 45 sources per context. Boxes show medians and interquartile ranges; whiskers use the 10th–90th percentiles, with more extreme values omitted. (c) Recorded frames and eligible B/C anchors from the actual masks; labels show valid/raw ratios. Panels (a,b) use logarithmic axes.

### Recommended lead / interpretation（English）

Figure 3.6 shows why total frame count alone does not describe the supervision available to the learner. Measured acquisition and soda-can pickup segments last longer than deposit or radio-interaction segments, while the physical next-action target changes much less in wait and follow than in the two manipulation contexts. Eligibility for a complete action horizon further changes the available anchor population, especially in the short wait context. These descriptive differences motivate RQ1: chunk construction and sampling determine which parts of the recorded sources contribute to learning. The segment distributions include repeated instances within each source and should not be read as independent trajectory-level uncertainty estimates.

建议插入Chapter3、数据表示/anchor construction附近；本图直接补充RQ1的duration、physical action structure与anchor eligibility，不重复Figure3.5的QC/total-size漏斗。

## Figure 5.4：设计与数据

标题：**Gripper Event Prediction Performance across A20, B20 and C20**。

文件：

- `figures/results/fig_5_4_gripper_event_performance.png`（2280×1180，400dpi）。
- 同stem `.pdf`、`.svg`，画布5.7×2.95in。

Pickup与radio两个panel共享纵轴。F1用实心圆/实线，precision用空心菱形/虚线；不同marker与line style使灰度打印仍可辨。点和error bars同时展示event profile与inference variability，避免只重复一排无误差柱状图。线仅为读图引导，A/B/C仍是分类配置。

直接读取既有15份`per_seed/seed_0..4/summary.json`，每seed在context内pool sparse event counts，再对5个seed重算mean与sample SD（`ddof=1`）。与仓库`evidence/frozen_audit/{A20,B20,C20}/summary.json`中未四舍五入数据逐项对照，F1/precision/recall mean和SD在1e-12 tolerance内一致；六个F1值亦通过论文四位小数sanity check。无需重新评分预测、加载权重或运行推理。

| context | model | F1 mean ± sample SD | Precision mean ± sample SD |
| --- | --- | --- | --- |
| pickup | A20 | .111996 ± .044452 | .131349 ± .058304 |
| pickup | B20 | .335790 ± .081027 | .283295 ± .068275 |
| pickup | C20 | .382495 ± .046073 | .316334 ± .042734 |
| radio | A20 | .023529 ± .052613 | .025000 ± .055902 |
| radio | B20 | .176848 ± .102468 | .120903 ± .072696 |
| radio | C20 | .218085 ± .129260 | .156026 ± .101204 |

每context每seed使用同一200个validation anchors（完整manifest652，4source）。每个model仅一次20k-step training，training seed42；5个点的随机性是inference flow noise。误差条**不是训练seed标准差或95%CI**。SD未截断，因此A20 radio下端可越过0；它不是负的实测score，也不是物理有界置信区间。

### Recommended caption（English）

**Figure 5.4: Gripper-event prediction performance across A20, B20 and C20 for (a) pickup and (b) radio.** Points show mean F1 and precision; bars show sample standard deviations over five inference flow-noise seeds. Each seed pools event counts over 200 matched validation anchors per context. Each checkpoint comes from one training run, so the bars do not measure training-run uncertainty or confidence intervals. Lines guide comparison between configurations. SD bars are untruncated and may extend below zero. Higher event scores do not establish successful grasping.

### Recommended lead / interpretation（English）

Figure 5.4 compares the event-detection profiles in the two manipulation contexts. B20 and C20 have higher mean F1 than A20 on the frozen validation anchors, but precision remains limited: even the highest pickup precision is 0.3163. The radio results also show substantial variation across inference noise. These are narrow event-prediction differences rather than evidence of successful manipulation. A20 to B20 changes prompts and sampling together, while B20 to C20 changes state and action normalisation together; the figure therefore does not identify a single-factor causal effect or establish statistical significance.

建议Chapter5 Section5.3，紧随Table5.3之后的结果解释。完整precision/recall/F1仍保留native table；新图只突出F1/precision profile与inference SD，不改变研究结论或编号体系。

## 变量与追溯表

| 新资料 | 内容 / 单位 / 来源 |
| --- | --- |
| `tables/v8_segment_durations.csv` | 495真实segments；source/split/round/segment index/start/exclusive stop/frames/seconds；源`hrc_subinstructions_rerun_v1.json` |
| `tables/v8_segment_duration_summary.csv` | 五语义标签真实样本quantiles，unit=seconds，source_count与segment n分列 |
| `tables/v8_source_context_characteristics.csv` | 180 source×context统计；raw/valid/decision row counts，eligible single-next-action rows，source mean18维delta rad |
| `tables/v8_context_characteristic_summary.csv` | 四context总体counts及45source均值的跨sourcequantiles |
| `tables/v8_gripper_event_per_seed.csv` | 90个model×context×metric×seed行；F1/precision/recall及context-pooled事件counts；score无量纲 |
| `tables/v8_gripper_event_summary.csv` | 18个model×context×metric汇总；mean、sample_sd、inference n=5、training runs=1、source n=4、context anchors=200 |
| `tables/v8_evidence_figure_metadata.json` | Word整合用编号键、PNG路径、英文lead/caption、population、variables和尺寸；由专用脚本重建 |
| `tables/v8_evidence_figure_statistics.json` | 定义、population、输入路径/SHA-256、URDF joint/type/unit证据、生成器SHA、checks与figure paths |
| `tools/plotting/generate_v8_evidence.py` | 仅生成本次两图；先提取轻量数值，再从CSV作图；不会调用既有全量audit |

最终manifest只提供source身份/split/原始路径；**所有B/C anchor/decision计数都来自实际Parquet masks**。源文件为本机最终B/C train/val/test numeric Parquet（投影读取state/action及必要索引，无RGB）与已保存subinstruction/per-seed摘要。统计CSV均为小型派生数值表，没有上传raw HDF5、视频、模型权重、URDF或场景资产。

### 可粘贴至总traceability的简表

| figure | panel / evidence | population / unit | verification / limitation |
| --- | --- | --- | --- |
| 3.6a | Actual subinstruction `start/stop` annotations; `v8_segment_durations.csv` | 495 non-standby segments from45 sources; seconds | Actual instance quantiles; repeated segments within source are correlated; no synthetic quartiles or median sums |
| 3.6b | Final B/C Parquet physical state/action; `v8_source_context_characteristics.csv` |45 source means per context,409490 within-context-next-action rows;18 revolute joints,rad | Locally checked URDF joint types; descriptive commanded target changes, not realised motion/policy error |
| 3.6c | Actual B/C `valid_anchor` masks by coarse context |409670 frames versus387544 eligible anchors across45 sources | Counts match existing v7 split summary; excludes stale B manifest counts; all-split description is not test-set model evaluation |
| 5.4 |15 archived per-seed summaries cross-checked with included frozen summaries |200 matched anchors/context,4validation sources,5 inference seeds;1 training run/model | Recomputed sampleSD; no retraining/inference; no causal or statistical-significance claim; event prediction is not grasp success |

## 生成命令与QA

完整提取与作图：

```bash
/root/autodl-tmp/openpi_hrc/.venv/bin/python \
  tools/plotting/generate_v8_evidence.py --data-root /root/autodl-tmp
```

仅从仓库内新增CSV重绘（不需要原始大数据或URDF）：

```bash
python tools/plotting/generate_v8_evidence.py --from-tables
```

已执行完整提取与table-only重绘。两图均真实打开PNG检查；初稿发现的Figure3.6顶注/标题、legend/标题、底注/xlabel重叠已修复。生成器对文字canvas边界以及title/legend/xlabel/figure-note bounding-box交叉进行断言；最终两图通过。图宽固定5.7in，字号和画布不靠`bbox_inches=tight`事后改变实际尺寸；PDF/SVG保留矢量图形与可编辑text。

**图源QA：Pass。正文嵌入后的分页与Microsoft Word/LibreOffice实际渲染由主代理继续检查，本文不将源PNG检查等同于DOCX分页验证。** 原始数据采集/训练/推理/仿真：Not independently verified。本次没有新增实验测量，只提取既有记录和重算保存摘要的统计量。
