# V8 figure traceability

日期：2026-10-22（Asia/Singapore）。本轮只修订v7，没有重新训练、仿真、数据集生成或重复完整研究审计。事件图原请求为Figure5.4；为保持首次出现顺序，正文用Figure5.3，原conditioning图顺延5.4。对应脚本stem保留原请求名，映射明确。

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

Method diagrams: `generate_v8_methodology.py` and V8_METHOD_FIGURES.md. Storyboard: same six video-frame files/identifiers from tables/storyboard_frames.json, relaid at5.7in by generate_v8_storyboard.py; no invented pixels or policy-success claim. Frustum5.1: generate_v8_frustum.py plus v8_frustum_coverage.csv and V8_FRUSTUM_FIGURE.md. Dataset3.6 and event profile: generate_v8_evidence.py plus v8_* CSV/JSON and V8_EVIDENCE_FIGURES.md. All new/revised graphics havePNG/PDF/SVG and editable plotting sources. The two unchanged result figures retain their raster originals without artificial vector conversion.

3.6 populations:45 sources,495 actual non-standby segments; per-source/context mean next-action deviation over18 verified revolute joints (rad); raw frames versus actual B/C valid masks (409670 versus387544). Segments within source are correlated; all splits are descriptive. Raw/anchor units remain separate.

Event profile:15 per-seed frozen summaries,652 total anchors; pickup/radio each200,4 validation sources. Mean±sample SD over five inference noise seeds; one20k training run per configuration. F1 and precision agree with native Table5.3; no grasp-success, single-factor causal, or significance claim.

Commands: `python tools/plotting/generate_v8_methodology.py`; `python tools/plotting/generate_v8_evidence.py --from-tables`; local storyboard requires `--frames /path/to/verified/frames`. Neither full dataset nor weights are needed for from-tables plotting.
