# 论文结果追溯表

审计日期2026-10-18。下表路径`P`=`/root/autodl-tmp/openpi_hrc`，`T`=`/root/autodl-tmp/hrc_pipeline_traintest`。论文段落号来自`thesis_paragraphs.txt`中方括号。**完整训练、推理与仿真实验均 Not independently verified；“重算”只表示本次从已有张量/记录重新计算。**

| 论文主张 / 位置 | 直接证据 | 本次核验与结论 |
| --- | --- | --- |
| 45source，37/4/4，409670frames；[52],[302] | `/root/autodl-tmp/lerobot_datasets/{hrc_pi05_continuous_rerun_v1,hrc_subinstr_rerun_v1}/{manifest.json,train/meta/info.json,val/meta/info.json,test/meta/info.json}` | 两导出元数据一致；341269+31084+37317=409670；原始采集质量另属数据审计 |
| A/B/C各一次训练，seed42，20k；[461] | `P/src/openpi/training/config.py`；`P/logs/hrc/train_{e6_rerun,b_subinstr,c_qnorm}_20k.log` | 配置及日志存在；各200训练点/20验证点；训练未复跑；B/C实际checkpoint缺失 |
| microbatch2×accum4=effective8；[463] | `P/scripts/b1k/train_b1k.py`、`config.py`，日志`Physical batch=2...effective batch=8` | 源码/运行日志吻合；160000为sample draws，非独立anchor数 |
| optimizer/LR/frozen vision/all32loss；[456]–[469], A.3 | `P/src/openpi/training/{config.py,optimizer.py}`、`P/src/openpi/models/pi0.py` | 配置支持；C action/state一起quantile，训练loss不可跨尺度解释为物理误差 |
| 652同manifest：52+200+200+200；[500]–[507] | `P/outputs/dissertation_final_audit/anchor_manifest.json`与15份`predictions.npz` | 本次核对所有保存张量形状、顺序、有限性、数据边界；3×5×652=9780chunks |
| flow seed0..4，非training uncertainty；[54],[140],[505] | manifest、每model/seed summary、`dissertation_protocol.py:flow_noise` | 文本与代码一致；未重新推理，fixed-noise determinism旧记录仅为历史证据 |
| pickup skill -.0107/.0122/.0144；[55],[557]–[563] | `P/outputs/dissertation_final_audit/{A20,B20,C20}/per_seed/seed_{0..4}/predictions.npz` + validation Parquet | 本次独立重算0误差；无充分准确arm control证据 |
| pickup F1 .1120/.3358/.3825；P最高.3163；[55],[571]–[576] | 同上；对应per_anchor.csv与summary | 本次独立复算event greedy matching与seed聚合，吻合；+/-8frames，P/R/F1按context counts pooled；不是成功率 |
| radio arm C=-.3099±.0394；[566] | `comparison.csv`；C20每seed summary/npz | 每seed arm MAE/skill本次重算吻合 |
| persistence pickup/radio/follow=.5402/.5787/.5093；[584] | `P/outputs/dissertation_final_audit/{comparison.csv,baselines/summary.json}` | 直接数表吻合；本次未重新拟合lookup，baseline生成过程Not independently verified |
| pickup image/noise B=.3581 C=.4414；[590] | `P/outputs/dissertation_conditioning_probe/{B20,C20}/per_anchor.csv` | 本次validation96pickup anchors逐ratio重算吻合；radio12；不能把324当validation |
| bootstrap95%CI与10000resamples；[594], A.3 | `P/scripts/hrc/dissertation_protocol.py:bootstrap_summary`、probe summary/by_context | 实现与参数可追踪；本次未重抽bootstrap；anchor级而非source population CI |
| 全sourcepickup frustum86.18%/decision90.06%；[548] | `P/outputs/dissertation_target_visibility/per_anchor.csv` | 本次重计154560/179353、10010/11115；未重新投影；不代表无遮挡像素可见 |
| 5份历史closed-loop零contact；[611]–[615], A.7 | `T/outputs/pi05_rerun_v1_closed_loop/*/policy_rollout_metrics.json` | 现有5份均零contact；source0000_seed1目录无完成metric不计入；历史非统一benchmark |
| horizon8→1 collision2879→0；[613] | 上述`e6_rerun_20k_TRAIN_source0038_seed0`与`e6_rerun_20k_TRAIN_s0038_horizon1` | JSON吻合，contact均0；单一development comparison |
| repeated inference0、独立spot check≈1.93e-7、norm重算0；[618]–[620] | `P/outputs/dissertation_quality_control/qa_report.json` | 存档有此结果；本次未执行这三项实验，只核norm文件hash且A checkpoint asset字节一致；Not independently verified |
| D被排除；[497]–[499] | `config.py`与`P/logs/hrc/train_d_vision_40k.log` | 多变更混合recipe；当前D31k目录存在不代表纳入主结果；不得混图 |

## 推荐的真实图数据入口

- 主比较：`P/outputs/dissertation_final_audit/comparison.csv`，筛model A20/B20/C20、context/action_group/metric。mean/std是5个flow seeds，`n`语义随metric变化。主表只使用该目录当前结果。
- conditioning：`P/outputs/dissertation_conditioning_probe/{B20,C20}/by_context.csv`，筛`split=val`；也可从per_anchor.csv重算。若画early/middle/late，使用`by_position.csv`并保持train_coverage和val分开。
- geometry：`P/outputs/dissertation_target_visibility/summary.csv`，保持`split/context/segment/position/population`筛选明确；不是policy success。
- training：本目录三个`*_training_curve.csv`，不同norm用独立panel并标“normalized flow-matching loss”；这些是日志曲线而非独立训练复现。
- closed-loop：JSON中的task_contact_frames、collision_frames、hand_object_min_distance_m；只标“历史diagnostic”，不汇总为A/B/C matched success排名。

## 禁止混入的记录

`P/outputs/dissertation_final_audit/_invalid_*`，早期`hrc_*_audit_train`和`phase0_trend`，D探索结果，单source overfit结果，ACT早期检查，以及`RoboBenchMart_dissertation/results/`的42-config atomic task结果，均不能替代当前论文frozen validation主表。它们可以另列development history，但须标协议、source split及证据边界。
