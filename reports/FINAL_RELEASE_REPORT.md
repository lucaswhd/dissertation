# NTU论文项目审计、图表与发布报告

日期：2026-10-19（Asia/Singapore）。本次不运行大规模训练、不生成完整新数据集、不改变原研究目录和v6论文。用户已明确确认代码权属并授权本论文与指定私有仓库发布。

## A. 研究项目结构

1. `hrc_pipeline`：用户指定的历史核心树，保留HRC FSM、人动画、相机、任务回放与采集代码。其本地commit为`94e89df6c8db41d23aa4d1a3f3be2907663ba478`，审计开始已存在`docs/notion.md`未提交修改，本次保留。
2. `hrc_pipeline_traintest`：当前研究分支。`scripts/collection/hrc_multitask_demo.py`、`nav_online_driver.py`及`evaluation/pi05_policy_client.py`实现在线导航、FSM编排和策略注入。不能用旧树概括最终论文实现。
3. `openpi_hrc`：π0.5适配、continuous LeRobot导出、子指令/decision labels、训练配置、最终teacher-forced与conditioning/geometry诊断；本地实验代码commit为`cfd097d6ca0f957b68d05616a3b71b86dabdf2cf`。
4. 最终研究数据：`hrc_batch_rerun_v1`及A/B连续导出。旧`hrc_canonical`/ACT路径为历史支线，未冒充本论文主实验。
5. 独立发布目录保留三个source snapshot原相对结构。共223个源码/配置/许可快照：217份原样，6个YAML仅暂存副本脱敏历史私人路径18处。OpenPI差异patch覆盖82个新增/变更文件，已通过dry-run恢复检查。

详见PROJECT_INVENTORY.md、FILE_CLASSIFICATION.csv、PROJECT_AUDIT.md、BENCHMARK_DESIGN.md、docs/SOURCE_LAYOUT.md。完整本地资源清单单独保存在`/root/dissertation_local_audit/LOCAL_RESOURCE_MANIFEST.csv`，权限0600，不进入Git。

## B. 项目自检结果

### 本次核查通过

- 元数据清单24,045条；对环境、VCS、cache等剪枝项明确记录，不声称遍历其内部。
- 65条采集源的QC为45 CLEAN、19 REJECT、1 TRIM。最终只保留45条；37/4/4 train/validation/test无source交叉。
- 45个最终HDF5共409,670帧；分块检查50,799,080个数值，没有发现非有限值或字段长度问题。45段head视频的容器帧数均匹配HDF5；没有把metadata检查说成全视频时钟验证。
- 两套导出共819,340行；37,689,640个state/action数值与独立重建完全一致。shift-1、source首帧base、source末行sentinel、demo边界及H32 coarse-context边界检查通过。
- 独立重计分15份预测归档、9,780个chunks；已核对主比较指标与记录一致，差异为0。没有重新调用模型来获得这些预测。
- 既有轻量协议、评分验证、绘图及continuous-data测试24项通过；源码语法检查通过。没有运行仿真、训练或推理。
- 本次提交前执行精确Git index的secret/private-path、symlink、文件大小、禁止资产类型与.gitignore检查。结果见SECURITY_CHECKS.json，文件hash见RELEASE_FILE_MANIFEST.json（避免自引用，两个审查回执自身不列hash）。

### 实际发现和未验证项

- B/C manifest继承部分A的anchor统计，39个source的`num_valid_anchors`与实际parquet有效mask不一致。已记录，未改动原档。应使用已核验的parquet/导出统计：B有效anchors为387,544；train/val/test为323,255/29,387/34,902。训练decision为12,785/323,255，即3.955%。
- B20/C20最终权重当前缺失；A20尚在。归档预测可重计分，当前不能独立重新推理B/C。
- 当前安装版本为Python3.10.12、OG3.7.2、Isaac4.5.0.0，与历史README的较新版本不一致；不是一份已重装验证的完整环境锁。
- 旧树有源HDF5写入、instruction元数据、task_success语义、human trajectory路径/进度与heuristic visibility等问题，详见PROJECT_AUDIT.md。已区分历史树与最终研究实现，未把旧问题未经核对强加给最终导出。
- 名义timestamp等于frame/30且递增；真实相机曝光、render与action apply时钟同步未独立验证。几何frustum coverage不等于无遮挡像素可见性。
- 完整训练、重新推理和仿真均为 **Not independently verified**。

## C. 论文可追溯性

实际稿件主题是分层诊断评估框架，基于HRC数据和π0.5案例，不是新foundation model或当前ACT主实验。benchmark贡献包括构建、实现、源级QC、监督/指标/基线/扰动协议和评估；第三方提供的模拟器、机器人、原始示教和预训练模型均保留归属。

FSM编排中的`ExecuteBehaviorTask`：数据构建使用上游state/action replay，policy rollout才执行预测动作；外围FSM与在线导航保持脚本控制。虚拟人是脚本人物，不能表述为新招募的人类参与者或端到端HRC自主规划。末轮返回/结束不自动证明任务成功。

主teacher-forced比较为4条validation sources上的652个共同anchors、5个inference flow seeds。5 seeds不是5次训练。pickup arm skill A/B/C约−0.0107/0.0122/0.0144，F1约0.1120/0.3358/0.3825；这些归档指标与论文相符。conditioning validation为108 anchors（96 pickup、12 radio），另216为training coverage，不混成324个validation样本。历史closed-loop失败单列，不生成没有匹配实验支持的A/B/C成功率排名。

详见RESULTS_TRACEABILITY.md、EXPERIMENT_AUDIT.md和DATASET_AUDIT.md。

## D. 图表补强结果

原稿24个嵌入图像实际为5幅编号Figure、18张表格图像和1个校徽。原Word原生表格数为0。此处不是再盲加24张图。

| 图 | 章节/位置 | 结果与价值 |
|---|---|---|
| Figure3.1 | §3.1 | 保留分层诊断框架，明确最终audit与历史rollout边界。 |
| Figure3.2 | §3.2.1 | 保留两轮任务顺序、decision window与future chunk示意。 |
| **Figure3.3** | §3.2.1末 | **新增**真实source0视频六格图：等待、取桶、拾罐、投罐、跟随、收音机。每格给round/global frame/time；仅本地成稿，截图不冒充学习策略成功。 |
| **Figure3.4** | §3.2.3 Table3.1后 | **新增**真实FSM、replay/policy控制接口和源级数据流。 |
| **Figure3.5** | §3.2.4末 | **新增**65→45源QC、45条时长分布、3.955%自然decision与20%配置采样目标。 |
| Figure5.1 | §5.1 | 保留单图，去除原混合截图中的重复图；恢复其独立原生Table5.1。 |
| Figure5.2 | §5.2 | 保留真实arm skill图及seed误差棒。 |
| Figure5.3 | §5.5 | 保留conditioning图及anchor-level区间语义。 |
| **FigureA.1** | §A.2 | **新增**真实A/B/C训练及monitored-validation曲线；不同归一化不作为共同物理性能尺度。 |

另生成FSM帧频率补充图，未强行塞入正文。Gripper事件图标为Ready to generate（相同数字已有Table5.3，避免重复）；匹配closed-loop成功率图为Unavailable；B/C重新推理为Requires verification。图表脚本、状态、英文caption、正文引文和精确原稿锚点见FIGURE_IMPROVEMENT_PLAN.md、FIGURE_TRACEABILITY.md及tables/figure_to_thesis_mapping.json。

**可编辑成稿**：`/root/dissertation_figures_delivery/NTU_Dissertation_Reviewed_v7_figures_editable.docx`。

- 新增3张主文图和1张附录图，共8张主文图+1张附录图；18张表格全部恢复为原生可编辑表。
- 修复TableA.1/A.3仅表头、正文数字散落；TableA.2移回A.1节；Table3.2/5.2/5.5移出句子中间。
- 所有新增图有PNG/PDF/SVG。截图原始分辨率352×256，300dpi导出不会增加原生细节。
- 原稿为US Letter；v7已设A4，原v6字节hash不变。v7的目录/图目录/表目录改为可更新域。
- 无Word/LibreOffice排版引擎，**没有声称分页已视觉验证**。必须在Word更新域、检查分页后再导出最终PDF。

## E. GitHub上传结果与许可范围

目标：[lucaswhd/dissertation](https://github.com/lucaswhd/dissertation)。认证账号SaladMike；在用户添加协作者后，本次已接受该仓库邀请。API确认private=true、push=true、默认main；git ls-remote确认远程最初为空。使用独立staging，不在原研究目录初始化发布仓库，不force push、不覆盖既有提交。

本报告随研究快照提交；**实际commit SHA、推送状态、远程ref与最终文件数量/字节以本机`/root/dissertation_local_audit/RELEASE_RECEIPT.json`及后续推送回执为准**。这避免把尚未发生的push写成成功；最终聊天会报告已实际核验的SHA与链接。

发布包括经检查源码、必要原许可/补丁、numeric evidence、非视频图表、全部图表源脚本、审计报告与文档。排除原始视频/HDF5/LeRobot、checkpoint、环境/cache、人物/场景/纹理/运动派生素材、私人绝对资源清单、Word全文和含第三方渲染素材的storyboard成图。图表/Word成品仍在本地交付目录完整保留。未启用Git LFS。

用户确认已解决原创源码权属问题；原All Rights Reserved与上游MIT/Apache/Gemma声明均保留，不擅自更改许可。用户授权不自动授予第三方资产再分发权。

## F. Remaining Actions

1. 在Word打开v7，更新全部域（Ctrl+A→F9），检查图注、跨页表头、标题孤行、分页后导出PDF。服务器结构检查不能替代该最终排版检查。
2. 找回B20/C20 checkpoint后，才可补做重新推理验证；需要匹配closed-loop实验后才可新增多模型成功率图，当前不要补造。
3. 后续维护导出器时修复B manifest inherited anchor counts，并验证后再改训练输入；本次审计保留原档，问题和正确数值均已记录。
4. 如要将真实视频截图/模型/数据向外共享，核对其原提供方许可范围。现有本地论文插图已生成，不要求用户手动逐一选文件。
5. 导师/作者需最终确认的是稿件叙事与结论范围，尤其replay/policy、teacher-forced/closed-loop、inference seed/training run以及benchmark与模型贡献的区别。

## 验收记录

已完成真实项目发现、三版本边界识别、数据/训练/评估链路审计、清单、真实图表与可编辑Word成稿、按章图表方案、干净发布目录与低成本验证。远程push及验证由随后独立回执记录；未复现实验与未验证分页保持明确限制。
