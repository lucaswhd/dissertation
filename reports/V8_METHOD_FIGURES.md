# V8 methodology figures

三幅方法图均为概念示意，由已审计协议、v7文字和现有源码引用绘制，不重新审计源数据、不引入新的统计或运行结果。PNG按300dpi输出，PDF/SVG保留矢量；画布固定5.7inch宽，图中文字主要为8–10pt。旧图和旧脚本未覆盖。

## Figure 3.1

**Files:** `figures/methodology/v8_fig_3_1_diagnostic_hierarchy.{png,pdf,svg}`

**Lead:** Figure 3.1 separates the controlled final policy audit from supporting historical behavioural evidence.

**Caption:** Hierarchical diagnostic evaluation framework. Levels 1–4 form the controlled final audit of supervision, behavioural baselines, stochastic inference and observation conditioning. Level 5 summarises historical closed-loop diagnostics under a separate supporting protocol; it is not a matched A20/B20/C20 rollout comparison.

**Variables/source:** The level numbers denote evidential layers, not quantitative scores. Five inference seeds refer to inference flow noise, not independent training runs. Source: v7 Figure 3.1 and reports/THESIS_REVIEW.md (Figure 3.1 section); no numerical data are plotted.

## Figure 3.2

**Files:** `figures/methodology/v8_fig_3_2_task_supervision.{png,pdf,svg}`

**Lead:** Figure 3.2 distinguishes the fixed task order from the frame-level rules used to construct action supervision.

**Caption:** Task structure and supervision annotation. (a) Fixed task order within each retained source: cleanup in round 0, followed by radio interaction in round 1. Cleanup contains three soda-can pickup–deposit cycles. Box widths do not encode recorded duration. (b) An illustrative expert gripper transition at frame f marks decision frames within ±16 frames. At anchor t, shift-1 alignment predicts the 32 future actions from t+1 to t+32. A valid anchor preserves source, episode and coarse context over [t,t+32]; the B/C chunk may cross a local-prompt boundary while that coarse context remains unchanged. The dashed orange line marks such an allowed local boundary, whereas “Next” lies beyond the valid coarse-context interval.

**Variables/source:** f = illustrative gripper-transition frame; t = observation/state anchor frame; future targets = a_(t+1),…,a_(t+32); decision annotation = |t−f|≤16; “Next” = another coarse context. The relative positions of f, t and the local boundary are schematic, not measured source timestamps. Task order follows v7 Figure 3.2. Supervision rules and stable [t,t+32] boundaries: reports/EXPERIMENT_AUDIT.md (“样本与随机性”) and reports/THESIS_REVIEW.md (Figure 3.2 section and population notes). No raw demonstrations were loaded.

## Figure 3.4

**Files:** `figures/methodology/v8_fig_3_4_controller_data_flow.{png,pdf,svg}`

**Lead:** Figure 3.4 locates the policy diagnostic within the scripted scheduler and links the recorded sources to the benchmark.

**Caption:** Controller scope and data flow for the evaluated HRC protocol. (a) The scripted finite-state scheduler returns through WaitNextCommand and Relocate to WaitNearHuman for a non-final round; after the final FaceHuman phase it enters AllTasksDone. (b) Dataset construction uses upstream BEHAVIOR state–action replay inside ExecuteBehaviorTask; the policy diagnostic substitutes π0.5 action prediction only in that execution scope. Scheduling and surrounding navigation remain scripted. Replay is not autonomous learned manipulation. (c) Source-level quality control and splitting precede supervision export and diagnostic evaluation, with source identity retained throughout.

**Variables/source:** FSM labels are categorical scheduler states, not success predicates. ExecuteBehaviorTask is the control-substitution boundary, not a claim that every policy rollout succeeds. State names/order: source_snapshots/hrc_pipeline_traintest/scripts/collection/hrc_utils.py:52–63 and hrc_multitask_demo.py:5206–5222,5573–5625. Replay/policy interface evidence: same hrc_multitask_demo.py:812–868,5359–5398,5510–5552, plus reports/FIGURE_TRACEABILITY.md and reports/THESIS_REVIEW.md (Figure 3.4). The current branch includes broader optional rollout hooks; the drawing follows the dissertation's narrower evaluated policy scope, as required by the v8 brief. Source-level QC/export population is documented in reports/DATASET_AUDIT.md. No old core branch is used as final-implementation evidence.

## Reproduce and QA

Run from the repository root:

```bash
python3 tools/plotting/generate_v8_methodology.py
```

The script does not use network access, git, source-data reads, simulation or training. It writes only these three v8 figure families and this report. Before export it checks text bounding boxes for text–text overlap and text escaping node boxes. The initial v8 exports were additionally inspected with view_image; all three standalone figures passed visual inspection with no label overlap or cropping. Visual checks concern the standalone figures; final DOCX pagination is the main document workflow's responsibility.


| Figure | PNG pixels | Physical size | Standalone visual QA | PDF / SVG |
|---|---|---|---|---|
| 3.1 | 1710 × 1200 | 5.7 × 4.00 in | Pass | Single page / editable text |
| 3.2 | 1710 × 1635 | 5.7 × 5.45 in | Pass | Single page / editable text |
| 3.4 | 1710 × 1845 | 5.7 × 6.15 in | Pass | Single page / editable text |

All PNGs carry 300 dpi metadata (rounded from 299.9994). PDF page widths are 410.4 pt = 5.7 in. Geometry checks passed for every node label and all text pairs. The figures contain no scene screenshots, third-party meshes, or raw data. The main agent should still inspect the inserted figures and captions in the final rendered document.
