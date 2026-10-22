# v8最终交付报告

日期：2026-10-22（Asia/Singapore）。本轮只修订v7，没有重新训练、仿真、数据集生成或重复完整研究审计。事件图原请求为Figure5.4；为保持首次出现顺序，正文用Figure5.3，原conditioning图顺延5.4。对应脚本stem保留原请求名，映射明确。

## A. What Changed
修复四张图继承exact行高导致的裁切；重绘3.1/3.2/3.4及5.1（修复旧图纵轴标签裁切），重排同六帧3.3标签；新增3.6和事件profile（正文5.3，原请求5.4），原conditioning改5.4。保留原章节、题目、WUHAODONG、18表数据、References、Appendix和签字占位。恢复12式为原生Word数学，补全可更新目录与页码，修复缩写和表格分页。

## B. Figures
| Figure | Purpose | Change | PDF page | File | Source |
|---|---|---|---|
|3.1|Hierarchy|Revised|27|`figures/methodology/v8_fig_3_1_diagnostic_hierarchy.png`|`V8_METHOD_FIGURES.md`|
|3.2|Task and shift-1 supervision|Revised|29|`figures/methodology/v8_fig_3_2_task_supervision.png`|`V8_METHOD_FIGURES.md`|
|3.3|Six actual simulation frames|Revised layout; local only|30|`figures/methodology/v8_fig_3_3_task_storyboard.png`|`tables/storyboard_frames.json`|
|3.4|Scripted scheduler and controller boundary|Revised|32|`figures/methodology/v8_fig_3_4_controller_data_flow.png`|`V8_METHOD_FIGURES.md`|
|3.5|QC and sampling target|Retained; layout repaired|34|`figures/dataset/fig_3_5_dataset_construction.png`|`tables/dataset_checks.json`|
|3.6|Temporal/action characteristics|New|38|`figures/dataset/fig_3_6_temporal_action_characteristics.png`|`V8_EVIDENCE_FIGURES.md`|
|5.1|Validation frustum coverage|Revised; source axis clipping repaired|54|`figures/results/v8_fig_5_1_frustum.png`|`V8_FRUSTUM_FIGURE.md`|
|5.2|Arm skill relative to no motion|Retained|55|`figures/results/v8_fig_5_2_arm_skill.png`|`evidence/frozen_audit/comparison.csv`|
|5.3|Gripper event profile|New; requested as 5.4|57|`figures/results/fig_5_4_gripper_event_performance.png`|`V8_EVIDENCE_FIGURES.md`|
|5.4|Observation conditioning|Retained; renamed from 5.3|59|`figures/results/v8_fig_5_4_conditioning.png`|`evidence/conditioning/summary.json`|
|A.1|Training diagnostics|Retained; layout repaired|75|`figures/experiments/fig_A_1_training_diagnostics.png`|`tables/training_curves.csv`|

11图均在实际PDF中完整显示并与caption同页。新增/重绘图提供PNG/PDF/SVG和源脚本。

## C. Dataset and Benchmark Improvements
3.6用真实分段时长、18维旋转关节next-action偏差和raw/valid总体区别解释长时程监督失衡，支撑RQ1；495个segments不是495个独立sources。事件图将pickup/radio的F1、precision及inference variability并置，支撑RQ2的窄结论；没有把事件命中写成抓取成功或模型总体能力。

## D. DOCX/PDF QA
实际LibreOffice Writer+Math渲染78页，全部页完成几何扫描和contact概览，重点图/数学/表格单页放大核查。零越界/异常空白/图注无同页图；全部11图和18表号严格核对，A3续页表头完整、5.5整表同页、A8表头不再断词。Word有12原生OMML与18原生表，原v7哈希保持。详见V8_DOCUMENT_VALIDATION.md、V8_VISUAL_QA.md。

DOCX：`/root/dissertation_v8_delivery/NTU_Dissertation_Reviewed_v8_figures_editable.docx`，3,729,808bytes，SHA256`4a8be224061693504453aed26ef8e86e89b0d4a8ddbfbab290eb0b30e20c280a`。

PDF：`/root/dissertation_v8_delivery/NTU_Dissertation_Reviewed_v8_review.pdf`，1,884,499bytes，SHA256`617b1c80a95c9650857949235a8b0ddbd1603b7c89ae2a872138d390146f8273`。

## E. Academic Integrity
全部数值来自已有记录或本轮真实派生统计；没有生成仿真证据、训练或成功率。保留scripted FSM、upstream replay、Execute内低层policy的贡献边界。五seed是推理噪声，A/B与B/C均非单因素实验；teacher forcing和历史closed-loop分开。学校具体声明措辞未擅造，未代签。

## F. GitHub Sync
目标私有仓库https://github.com/lucaswhd/dissertation，开始时已fetch且main与9c2cf27一致。上传非视频图、代码、CSV/JSON、README、映射与报告；完整Word/PDF、真实帧和逐页预览仅本地（第三方渲染再分发许可未明确）。**Pass**：branch `main`；内容提交 `16a68ad5ff2a112920f26882d9d8b85918815e9c`。GitHub API确认Private，fresh fetch后的远程tree与本地一致，全部373个文件逐一SHA256一致；无受禁资产。该回执验证内容提交；最终补记报告的HEAD另见本地V8_RELEASE_RECEIPT.json。更新文件的完整列表及hash见V8_PUSH_VERIFICATION.json。

## G. Remaining NTU Submission Actions
1. **Pending official template verification**：已取得真实25Nov2025公开NTU模板，尚未查实July2026后EEE适用版本；由学校/作者核对，保留真实AI披露并补确知的工具版本。
2. 作者在Microsoft Word做最终跨引擎预览、更新域；Linux实际PDF已审核，但字体替代可能令Word分页略变。
3. 作者/导师本人签字和日期；本轮未代填。
4. 完整稿/帧向仓库或外部再分发前核对上游场景/人物渲染许可；本轮已隔离，未阻止安全材料发布。
