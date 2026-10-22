# V8 visual QA

日期：2026-10-22（Asia/Singapore）。本轮只修订v7，没有重新训练、仿真、数据集生成或重复完整研究审计。事件图原请求为Figure5.4；为保持首次出现顺序，正文用Figure5.3，原conditioning图顺延5.4。对应脚本stem保留原请求名，映射明确。

**Final PDF: 78 pages, SHA256 `617b1c80a95c9650857949235a8b0ddbd1603b7c89ae2a872138d390146f8273`.**

Pass: every page was rasterized and scanned for geometry, near-edge text, image bounds and blank pages; every page was visually reviewed on contact sheets. Figure pages and table/math problem pages were opened at individual-page scale; separate reviewer reports record inspected ranges and intermediate candidate hashes. Only the final hash above governs this report.

Resolved: fixed picture-line-height clipping (v7 p26 narrow-strip storyboard versus auto-spacing probe p27 full storyboard); congested method diagrams; incomplete private-TC indexes; cross-chapter SEQ accumulation (explicit per-caption resets and exact PDF label checks); run-on acronyms; short table5.5 splitting; TableA8 Median header wrapping; source-image5.1 y-axis clipping; inherited flattened math; formula renderer's initially missing Math component. The latter was caught before publication and verified with a12-equation probe plus actual final pages.

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

TableA3 continues across pages with repeated header and all31 data rows; no fixed row clipping. Short table5.5 is kept together. No unresolved Fail remains in the checked output. A small section opening near the bottom of p32 has two following lines and is not a detached heading; a stylistic author review may still prefer a different break.

Pending: Microsoft Word cross-engine preview and exact NTU declaration version/applicability; no generated signature or claim of official approval. Source frame redistribution remains unconfirmed, so full manuscript/contact images stay local.
