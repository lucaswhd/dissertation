# 论文图表补强与插入方案

日期：2026-10-19（Asia/Singapore）。原始稿保留；成稿另存 v7。

主文保留5幅原有Figure，新增3幅主文Figure；附录新增1幅训练图。共8幅主文图、1幅附录图。18张原图片表格恢复为原生Word表格。第三方渲染素材与受限源码均仅在本地准备，未发布。

## 已落稿与候选

### Figure 3.3: Representative head-camera observations from retained validation source 0

- Status: **Generated**; Priority: P0
- 章节：3.2.1 Simulation Task Structure；原稿paragraph锚点：282
- 学术目的：使读者看到benchmark真实场景和四类语境
- 展示内容：Figure 3.3 provides representative recorded observations for the four coarse contexts. The images show the visual setting and task variation underlying the benchmark. They are examples of dataset execution rather than a qualitative evaluation of the learnt controllers.
- 证据：45-source final manifest; retained source0 head_rgb.mp4; subinstruction segment boundaries；数据：Yes
- 文件：`figures/methodology/fig_3_3_task_storyboard.png`
- Proposed caption: Representative head-camera observations from retained validation source 0. Panels show waiting, acquiring the trash can, picking up and depositing a soda can, following the virtual human, and interacting with the radio. Panel identifiers give the round and raw global frame (30 Hz). Human-facing phases use scripted control; manipulation uses recorded BEHAVIOR replay. These are dataset examples, not successful fine-tuned policy rollouts.
- 正文修改：Figure 3.3 provides representative recorded observations for the four coarse contexts. The images show the visual setting and task variation underlying the benchmark. They are examples of dataset execution rather than a qualitative evaluation of the learnt controllers.

### Figure 3.4: Implemented controller scope and data flow

- Status: **Generated**; Priority: P0
- 章节：3.2.3 HRC System Scope；原稿paragraph锚点：294
- 学术目的：区分原创编排、upstream replay与策略作用范围
- 展示内容：Figure 3.4 locates the learnt controller within the implemented HRC system. The policy replaces replay only inside ExecuteBehaviorTask, while the surrounding interaction and transition logic remains scripted. Source identity is retained from recording through export to the diagnostic benchmark.
- 证据：hrc_pipeline_traintest/scripts/collection/hrc_multitask_demo.py:812,5206,5359,5573; scripts/evaluation/pi05_policy_client.py:191；数据：Yes
- 文件：`figures/methodology/fig_3_4_controller_data_flow.png`
- Proposed caption: Implemented controller scope and data flow. (a) The recorded finite-state schedule; non-final rounds continue through WaitNextCommand and Relocate, while the final round exits from FaceHuman to AllTasksDone. (b) ExecuteBehaviorTask uses upstream state/action replay for dataset construction and the predicted-action client for policy rollout. (c) Source-level quality control and splitting precede diagnostic use of the exported data. High-level phases are supplied by the scripted scheduler.
- 正文修改：Figure 3.4 locates the learnt controller within the implemented HRC system. The policy replaces replay only inside ExecuteBehaviorTask, while the surrounding interaction and transition logic remains scripted. Source identity is retained from recording through export to the diagnostic benchmark.

### Figure 3.5: Dataset construction and effective supervision

- Status: **Generated**; Priority: P0
- 章节：3.2.4 Population and Quality Control；原稿paragraph锚点：311
- 学术目的：说明源级QC、时长与稀疏事件监督；不把采样目标当实测频率
- 展示内容：Figure 3.5 summarises dataset construction and the supervision available to the learner. Quality control and splitting operate at source level. Chunk validity and weighted sampling operate on derived anchors, so the natural decision-frame population differs from the configured training-sampler target.
- 证据：rescan_qc.json; 45 HDF5 sources; final export manifests/reports; parquet alignment audit；数据：Yes
- 文件：`figures/dataset/fig_3_5_dataset_construction.png`
- Proposed caption: Dataset construction and effective supervision. (a) Quality-control outcomes for 65 collected sources and the frozen split of the 45 retained sources. (b) Complete recorded source durations at 30 Hz, including scripted transition phases. (c) Decision frames account for 3.955% of valid B/C training anchors; the 20% value is the configured sampling target, rather than an observed frequency of realised training draws.
- 正文修改：Figure 3.5 summarises dataset construction and the supervision available to the learner. Quality control and splitting operate at source level. Chunk validity and weighted sampling operate on derived anchors, so the natural decision-frame population differs from the configured training-sampler target.

### Figure A.1: Training diagnostics for the three 20,000-update runs (one training seed, 42, per configuration)

- Status: **Generated**; Priority: P1
- 章节：A.2 Main Training Hyperparameters；原稿paragraph锚点：802
- 学术目的：保留真实训练记录，同时把训练loss与物理性能分开
- 展示内容：Figure A.1 records the optimisation history of the archived runs. It documents training progress and the fixed 20,000-update endpoint; the paper-facing comparison is instead defined by the frozen physical-action audit.
- 证据：openpi_hrc/logs/hrc/train_{e6_rerun,b_subinstr,c_qnorm}_20k.log；数据：Yes
- 文件：`figures/experiments/fig_A_1_training_diagnostics.png`
- Proposed caption: Training diagnostics for the three 20,000-update runs (one training seed, 42, per configuration). The training objective was logged every 100 updates and the monitored validation objective every 1,000 updates. All 32 model output dimensions, including padding, contribute to the normalised flow objective. Different supervision and normalisation pipelines prevent interpreting these magnitudes as directly comparable physical-action errors or task-success measures.
- 正文修改：Figure A.1 records the optimisation history of the archived runs. It documents training progress and the fixed 20,000-update endpoint; the paper-facing comparison is instead defined by the frozen physical-action audit.

### Supplementary candidate: FSM recorded-frame frequency

- Status: **Generated**; Priority: P2
- 章节：Appendix or supplementary repository；原稿paragraph锚点：
- 学术目的：检查长任务中的状态失衡
- 展示内容：Recorded-frame frequencies, not policy success.
- 证据：All45 retained HDF5 obs/fsm_state_id；数据：Yes
- 文件：`figures/dataset/supp_fsm_state_frequency.png`
- Proposed caption: Recorded frame counts by implemented FSM state over 45 retained sources.
- 正文修改：Optional; not inserted to avoid repeating the task-order discussion.

### Candidate: Gripper precision/recall/F1 plot

- Status: **Ready to generate**; Priority: P2
- 章节：5.3 or A.5；原稿paragraph锚点：
- 学术目的：避免重复和无证据结论
- 展示内容：dissertation_final_audit/comparison.csv; already provided as native Table5.3
- 证据：dissertation_final_audit/comparison.csv; already provided as native Table5.3；数据：Yes
- 文件：``
- Proposed caption: Not inserted.
- 正文修改：See experiment audit limitations.

### Candidate: Matched A20/B20/C20 closed-loop success comparison

- Status: **Unavailable**; Priority: P2
- 章节：5.7；原稿paragraph锚点：
- 学术目的：避免重复和无证据结论
- 展示内容：Matched final rollout experiment is absent; no success-rate chart generated.
- 证据：Matched final rollout experiment is absent; no success-rate chart generated.；数据：Partial
- 文件：``
- Proposed caption: Not inserted.
- 正文修改：See experiment audit limitations.

### Candidate: B20/C20 fresh inference reproduction

- Status: **Requires verification**; Priority: P2
- 章节：A.8；原稿paragraph锚点：
- 学术目的：避免重复和无证据结论
- 展示内容：Archived predictions can be rescored, but current checkpoint directories are missing.
- 证据：Archived predictions can be rescored, but current checkpoint directories are missing.；数据：Partial
- 文件：``
- Proposed caption: Not inserted.
- 正文修改：See experiment audit limitations.

## 原有图表修复

Figure3.1/3.2/5.1/5.2/5.3保持原编号。移除image8中重复的Figure5.1，仅保留原image9单图。Table3.2、5.2、5.5移出句子中间；TableA.2移回A.1节；TableA.1/A.3表头与散落正文重组成完整表。全部表格的来源见tables/native_tables.json。

源级split 37/4/4与frame数409670已全量核对；valid B/C training decision fraction来自实际export与parquet，20%只标target。

Word改稿中的目录、图目录和表目录已改为可更新域。服务器无Word/LibreOffice排版引擎，不能声称分页已视觉核验；需在Word更新域后检查分页并导出PDF。