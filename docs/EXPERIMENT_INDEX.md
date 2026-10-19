# 实验与代码索引

本索引为本地可追溯性入口，审计日期2026-10-18。路径是审计时的绝对路径，不表示这些大文件已被公开发布。完整端到端复现状态：**Not independently verified**。本次仅独立重计分保存预测，并生成审计报告和日志曲线CSV。

## 论文frozen主结果

根：`/root/autodl-tmp/openpi_hrc`。

| 用途 | 相对上述根的路径 |
| --- | --- |
| A/B/C final结果表 | `outputs/dissertation_final_audit/{comparison.csv,comparison.json}` |
| frozen questions、sources、prompts、row joins | `outputs/dissertation_final_audit/anchor_manifest.json` |
| 逐模型保存预测/评分 | `outputs/dissertation_final_audit/{A20,B20,C20}/per_seed/seed_{0..4}/{predictions.npz,per_anchor.csv,summary.json}` |
| 模型汇总/状态 | `outputs/dissertation_final_audit/{A20,B20,C20}/{summary.json,run_status.json}` |
| 参考baseline | `outputs/dissertation_final_audit/baselines/summary.json` |
| Conditioning | `outputs/dissertation_conditioning_probe/{anchor_manifest.json,summary.json,by_position.csv}`与`{B20,C20}/{per_anchor.csv,by_context.csv,by_source.csv,by_segment.csv,by_position.csv,summary.json}` |
| 几何frustum | `outputs/dissertation_target_visibility/{summary.json,summary.csv,per_source.csv,per_anchor.csv,per_frame.csv}` |
| 历史QA | `outputs/dissertation_quality_control/qa_report.json`；是归档记录，不等于本次复验 |
| 无效历史协议 | `outputs/dissertation_final_audit/_invalid_prompt_stable_protocol_20260907`、`_invalid_f1_null_protocol_20260907`；排除于主图 |

## 论文训练与评估实现

| 层 | 原始文件 |
| --- | --- |
| 训练入口 | `/root/autodl-tmp/openpi_hrc/scripts/b1k/train_b1k.py` |
| training recipes | `/root/autodl-tmp/openpi_hrc/src/openpi/training/config.py` |
| optimizer | `/root/autodl-tmp/openpi_hrc/src/openpi/training/optimizer.py` |
| policy/model实现 | `/root/autodl-tmp/openpi_hrc/src/openpi/models/pi0.py`、`pi0_config.py`；属于基于OpenPI的模型栈，不能整体称作项目原创 |
| HRC数据构建 | `/root/autodl-tmp/openpi_hrc/src/openpi/training/hrc_data.py`、`hrc_export.py`，以及`src/openpi/policies/hrc_policy.py` |
| HRC LeRobot export/validation | `/root/autodl-tmp/openpi_hrc/scripts/hrc/export_hrc_lerobot.py`、`validate_hrc_lerobot.py` |
| frozen纯协议与metrics | `/root/autodl-tmp/openpi_hrc/scripts/hrc/dissertation_protocol.py` |
| frozen audit | `/root/autodl-tmp/openpi_hrc/scripts/hrc/run_dissertation_final_audit.py` |
| perturbation probe | `/root/autodl-tmp/openpi_hrc/scripts/hrc/run_dissertation_conditioning_probe.py` |
| geometric target audit | `/root/autodl-tmp/openpi_hrc/scripts/hrc/run_dissertation_target_visibility.py`、`compute_target_visibility.py` |
| 历史QA与作图 | `/root/autodl-tmp/openpi_hrc/scripts/hrc/verify_dissertation_results.py`、`plot_dissertation_results.py` |
| 早期audit/probe | `/root/autodl-tmp/openpi_hrc/scripts/hrc/audit_hrc_teacher_forced.py`、`audit_hrc_lookup_baseline.py`、`probe_hrc_visual_conditioning.py` |
| policy server | `/root/autodl-tmp/openpi_hrc/scripts/hrc/serve_hrc_pi05_rollout.py` |

上述HRC专用脚本是优先整理的项目实现候选，原创性和上游license仍需按实际文件来源核对，不应将整个OpenPI或third_party目录称为原创。

训练日志：`/root/autodl-tmp/openpi_hrc/logs/hrc/train_e6_rerun_20k.log`（A）、`train_b_subinstr_20k.log`（B）、`train_c_qnorm_20k.log`（C）。对应20000-step路径在各model summary中；当前只有A的`checkpoints/pi05_hrc_e6_rerun_20k/hrc_e6_rerun_seed42/20000`尚存，B/C指定checkpoint缺失。Norm资产均在`assets/hrc/{main_continuous_rerun_v1_natural,subinstr_rerun_v1,subinstr_rerun_v1_qnorm}/norm_stats.json`。

## 数据导出与采集链

- 论文rerun原始输出：`/root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_rerun_v1/`。
- 论文A导出：`/root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1/{train,val,test}`。
- 论文B/C导出：`/root/autodl-tmp/lerobot_datasets/hrc_subinstr_rerun_v1/{train,val,test}`。
- frozen split：`/root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1_split.json`；三split source身份须保留。
- 核心较早采集仓库：`/root/hrc_pipeline/{scripts,configs,docs}`；训练审计实际扩展在`/root/autodl-tmp/hrc_pipeline_traintest`和`openpi_hrc`。只公开最早仓库不足以重现论文。
- 早期canonical链：`/root/autodl-tmp/hrc_snapshots/hrc_multitask_v01_step10`→`/root/autodl-tmp/hrc_canonical/hrc_multitask_v01_step10`，由`hrc_pipeline_traintest/scripts/preprocess/export_canonical.py`生成。早期snapshot不是45source final population。

## Closed-loop与ACT历史

`/root/autodl-tmp/hrc_pipeline_traintest/scripts/evaluation/`中的`hrc_rollout_core.py`、`pi05_policy_client.py`、`prepare_pi05_closed_loop_runs.py`、`prepare_pi05_continuous_runs.py`、`summarize_pi05_closed_loop.py`、`act_rollout_demo.py`、`act_policy_server.py`、`eval_act_overfit.py`提供项目集成实现。

历史π0.5输出：`outputs/pi05_closed_loop/{gate0*,gate2,gate3}`、`outputs/pi05_continuous_clean/closed_loop`、`outputs/pi05_rerun_v1_closed_loop`与`outputs/pi05_b_closed_loop`。它们的协议不同，不能拼成统一A/B/C benchmark。

ACT adapter：`scripts/preprocess/export_lerobot_act.py`（23→22维，丢dim6、state44维、action_shift参数）；早期recipes在`scripts/training/run_train*.sh`。本地`/root/autodl-tmp/act_runs`当前为空。上游模型候选`/root/autodl-tmp/act-plus-plus`、LeRobot安装包均不能替代项目训练checkpoint。

## 当前论文外的相邻项目

`/root/autodl-tmp/RoboBenchMart_dissertation/results/{octo-2026-09-28,pi05-2026-10-02}`和`/root/autodl-tmp/robobenchmart-dissertation/data/reproduced-2026-10-02`是另一个benchmark任务的42-config评测。存在真实CSV、episode index和日志清单，但本次论文正文的实验population、任务与主张均不采用这套数据；不可混入HRC主结果。

## 本次新增的复核资料

在发布staging的`reports/`中：`EXPERIMENT_AUDIT.md`、`RESULTS_TRACEABILITY.md`、`independent_archive_checks.json`、`audit_saved_predictions.py`和三个`*_training_curve.csv`。独立脚本仅从既有数据重新评分，不重新训练或推理。SHA、路径清单与缺失checkpoint状态均是此次审计时的观察。
