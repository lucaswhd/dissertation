# 图表证据追踪

所有新增统计图来自真实记录；未运行仿真或训练，未使用图像生成模型补造场景。

| Figure | Output | Evidence | Regeneration |
|---|---|---|---|
|3.3|`figures/methodology/fig_3_3_task_storyboard.png`|45-source final manifest; retained source0 head_rgb.mp4; subinstruction segment boundaries|`tools/plotting/generate_figures.py --data-root <LOCAL_DATA_ROOT>`|
|3.4|`figures/methodology/fig_3_4_controller_data_flow.png`|hrc_pipeline_traintest/scripts/collection/hrc_multitask_demo.py:812,5206,5359,5573; scripts/evaluation/pi05_policy_client.py:191|`tools/plotting/generate_figures.py --data-root <LOCAL_DATA_ROOT>`|
|3.5|`figures/dataset/fig_3_5_dataset_construction.png`|rescan_qc.json; 45 HDF5 sources; final export manifests/reports; parquet alignment audit|`tools/plotting/generate_figures.py --data-root <LOCAL_DATA_ROOT>`|
|A.1|`figures/experiments/fig_A_1_training_diagnostics.png`|openpi_hrc/logs/hrc/train_{e6_rerun,b_subinstr,c_qnorm}_20k.log|`tools/plotting/generate_figures.py --data-root <LOCAL_DATA_ROOT>`|

每张新增图有PNG/PDF/SVG；PNG为300dpi导出，但3.3底图来自原始352×256模拟视频，导出dpi不会增加原始像素细节。SVG中嵌入相同真实帧。截图准确source/round/global-frame在`tables/storyboard_frames.json`。

方法图对应最终研究分支，旧`hrc_pipeline`另存历史快照。FSM失败仍可能继续返回，图中状态顺序不表示任务成功。

源数据哈希/尺寸、数值检查和记录范围分别见LOCAL_RESOURCE_MANIFEST.csv（仅本机）、tables/dataset_checks.json、reports/export_alignment_checks.json、reports/independent_archive_checks.json。

训练图来自三个run的原始loss日志；每个run200 train点+20 monitored-validation点。不同normalisation不能据loss高低作物理性能排名。

原有5幅图在Word中保留；其对应frozen records主数字已重计分，figure样式未重绘。18张表格由`build_tables.py`从frozen records和原稿配置重建，表内sample数、SEED SD和anchor-level bootstrap语义保留。