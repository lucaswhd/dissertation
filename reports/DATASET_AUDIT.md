# 数据集与导出对齐审计

审计日期：2026-10-18（UTC）。本次新增独立脚本`reports/audit_export_alignment.py`，使用OpenPI虚拟环境中的NumPy、PyArrow和h5py，直接读取最终45个source HDF5与A、B/C的train/val/test Parquet。脚本**未导入或调用项目alignment构建函数**，一次处理一个source，仅读取numeric columns；未解码视频、未读取完整图像集合、未训练或运行仿真，也未修改原始数据。

结论：最终两个π0.5 continuous导出的物理行对齐在**全量范围通过**；存在B manifest有效anchor统计继承A而未更新的元数据问题。完整采集过程、训练、模型推理、视频逐帧视觉一致性与物理时钟同步：**Not independently verified**。

## 检查范围与结果

| 全量项目 | 数量 / 结果 |
| --- | ---: |
| 最终source HDF5 | 45个，每个含demo_0、demo_1 |
| 唯一raw observation行 | 409,670 |
| A与B/C导出合计行 | 819,340；每个导出409,670 |
| 两导出state23/action23数值 | 37,689,640个，非有限值0 |
| state23对raw重建最大绝对差 | 0.0 |
| shifted action对raw重建最大绝对差 | 0.0 |
| A与B/C state最大绝对差 | 0.0 |
| A与B/C action最大绝对差 | 0.0 |
| source首尾及demo边界 | 每导出45个source，全部通过 |
| invalid repeat-last-action sentinel | 每导出45行，共90行，全部通过 |
| source/raw global frame/demo-local frame/FSM/coarse-context映射 | 全819,340导出行通过 |
| valid H=32 anchor跨source/coarse-context违规 | A=0，B/C=0 |
| nominal timestamp=`float32(frame_index/30)` | 所有行逐值完全一致，每source严格递增 |
| 独立重建失败条目 | 0 |

每source、每导出、每split的实际计数、Parquet SHA-256、manifest SHA-256与当前检查结果保存在`reports/export_alignment_checks.json`。结果状态为`passed_with_metadata_warnings`，不是忽略了39条错误；这39条是已单独列出的B manifest计数漂移，物理alignment检查本身没有失败。

## 最终数据路径与物理定义

原始来源：`/root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_rerun_v1/ep_XXXX/hrc_dataset.hdf5`。

A导出：`/root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1/`。

B/C导出：`/root/autodl-tmp/lerobot_datasets/hrc_subinstr_rerun_v1/`。

source先将`demo_0`与`demo_1`沿时间拼接。令拼接后的raw action为`u[t]`，joint_positions为`q[t]`，gripper_state为`g[t]`，独立重建的state为：

```text
state[0:3]   = u[t,0:3]；仅该source的t=0强制为0
state[3:7]   = q[t,6:10]
state[7:14]  = q[t,[10,12,14,16,18,20,22]]
state[14]    = sum(g[t,0:2])
state[15:22] = q[t,[11,13,15,17,19,21,23]]
state[22]    = sum(g[t,2:4])
```

目标行`action[t]=u[t+1]`；source最后一行重复`u[-1]`作为sentinel，`valid_anchor=False`。`demo_0→demo_1`边界处base state沿连续时间线保留，不重新清零。这一continuous公式不同于早期segment代码中泛称的`previous_action`接口，不能据名称误写成`raw_action[t-1]`。

coarse context根据`annotations/commands.json`的全局issued_frame及raw demo长度独立重建：wait到round0指令，pickup到demo0结尾，follow到round1指令，radio到source结尾。未使用raw HDF5的instruction attrs作为语言真值；抽查中demo0 attr仍可见错误radio指令，符合项目既有的历史标签问题，实际指令应来自commands和subinstruction plan。

H32安全性按`[t,t+32]`全部33个位置检查：观测及32个shift1目标对应的raw时间必须保留同source/coarse context，目标不得到达无未来动作的sentinel。检查不是只比较区间端点。

## Anchor数量与采样口径

| split | source数 | 每导出帧数 | A valid anchors | B/C valid anchors | B/C decision valid anchors |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 37 | 341,269 | 336,533 | 323,255 | 12,785 |
| val | 4 | 31,084 | 30,572 | 29,387 | 1,367 |
| test | 4 | 37,317 | 36,805 | 34,902 | 1,350 |
| 合计 | 45 | 409,670 | 403,910 | 387,544 | 15,502 |

A的valid mask由四个context各自的完整未来窗口确定；B/C保留相同物理行，再按保存的subinstruction plan将idle tails的anchors置invalid。独立脚本逐行重建两种mask，与实际Parquet完全一致。B/C的decision flags也按plan内flip_frames的±16帧范围逐行核对，全部一致。该检查证明导出遵循保存的plan，**没有重新人工标注或独立判定plan的语义正确性**。

论文约3.955%的自然decision占比是**training valid anchors中的12,785/323,255**；它不是全45source比例，也不等于训练weighted sampler实际抽样占比。B/C配置目标20%的decision draws，是训练采样分布，不能与自然数据比例互换。frozen teacher-forced audit仅使用其中652个匹配validation anchors，不能用全量export计数替换其评估样本量。

## 发现的元数据问题

B/C exporter的`manifest.json`由A manifest浅复制而来，`num_valid_anchors`和`context_anchor_totals`仍保留A的统计。实际B/C已mask idle tails，因此**39个source的manifest num_valid_anchors与Parquet不一致**。例如source4记为11,000，实际10,577；source6记为8,489，实际8,156。

这不影响本次验证的逐行state/action或真实valid mask，但会误导只读取manifest的分析。发布和作图应采用B/C Parquet、B/C `export_report.json`或本次实际计数，并明确A/B不同的anchor资格；原始manifest不得悄悄改写，修正应另留派生索引和说明。`tables/source_statistics.csv`中的单列`valid_anchors`对应A口径，不能未经标注当成B/C统计。

## 与原始文件及视频元数据检查的关系

主代理已另行生成`tables/dataset_checks.json`，记录全部45个保留HDF5按2048行分块扫描，检查50,799,080个numeric values、issue列表为空；45个head视频的ffprobe容器元数据与对应HDF5长度一致。该记录同时保留QC分布CLEAN45、REJECT19、TRIM1，以及37/4/4 split。

本次审计在这份记录之外增加了raw HDF5→两个Parquet的独立公式重建和行身份核验；没有把文件可读、视频长度相同或代码中已有assert当成alignment证明。两份检查的numeric计数范围不同，不能相加后解释成独立数据量，也不能将容器元数据匹配称为所有视频帧的内容验证。

## 时间同步与未验证部分

本次已验证导出timestamp与名义30Hz帧序号一致、严格递增、source内raw frame不缺不重，以及明确的shift1算术关系。但这些均是在现有文件约定下的**索引与数值检查**。

未测相机曝光时刻、渲染帧生成时刻、joint sensor采样时刻、action apply时刻与simulator step实际时钟的因果顺序或延迟；未逐帧解码图像并对应关节/动作，也未检测视频内部丢帧、重复画面或可变编码时间误差。即使frame count完全一致，仍不能由此证明真实多模态物理时间同步。此部分应标为**Not independently verified**，不宜把本次结果写成“已验证sensor/action实时同步”。

本次也未重新判定45source的任务成功、所有raw reward/terminated标签的语义可信度、目标身份推断或人工视频质量；这些不属于导出alignment检查。

## Canonical历史支线

只读取metadata的旧文件：`/root/autodl-tmp/hrc_canonical/hrc_multitask_v01_step10/canonical.hdf5`，4,210,663字节，包含10,239行、23维action、28维joint_positions、4维gripper_state、FSM/camera/human字段及deprecated reward/task_success/terminated/truncated。

这是早期ACT preprocessing支线；其后ACT adapter可以删除dim6，形成22维action与44维state。最终论文45source π0.5 continuous路径由rerun HDF5直接构建当前LeRobot导出，**不要求先经过这份10,239行旧canonical**。本次未把旧canonical数据范围、字段或历史成功标签混入最终45source结果，也未全量核验该旧分支。

## 复查入口

```bash
/root/autodl-tmp/openpi_hrc/.venv/bin/python \
  /root/dissertation_release_staging/reports/audit_export_alignment.py
```

脚本默认仅写`reports/export_alignment_checks.json`，原数据只读。它保留可配置dataset base/output路径，便于迁移到另一个本地副本；raw HDF5位置沿manifest取值，迁移时需先使manifest路径在新环境有效。
