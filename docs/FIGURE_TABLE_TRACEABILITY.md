# Figure and table traceability

This maps the 19 numbered figures and 18 numbered tables in the controlled final LaTeX source to the relevant released evidence or restricted source. A path beginning with `controlled` is intentionally absent from this repository pending asset/data rights review. Figure numbers are manuscript numbers, not asset filename stems.

## Figures

| No. | LaTeX label | Controlled source asset | Evidence or restriction |
|---|---|---|---|
| 1.1 | `fig:hrc-scene-overview` | `assets/phase2/scene_overview.pdf` | controlled Phase 1 scene record and source 0 frames |
| 1.2 | `fig:dissertation-contribution-pipeline` | `assets/phase2/contribution_pipeline.pdf` | controlled project architecture record |
| 2.1 | `fig:related-work-positioning` | `assets/phase2/related_work_positioning.pdf` | cited literature in manuscript references |
| 3.1 | `fig:diagnostic-framework` | `assets/v8_fig_3_1.png` | reports/BENCHMARK_DESIGN.md |
| 3.2 | `fig:task-structure` | `assets/v8_fig_3_2.png` | docs/DATASET_SCHEMA.md |
| 3.3 | `fig:can-demonstration-sequence` | `assets/phase2/can_storyboard.pdf` | controlled source 0 frames; no public image redistribution |
| 3.4 | `fig:radio-demonstration-sequence` | `assets/phase2/radio_storyboard.pdf` | controlled source 0 frames; no public image redistribution |
| 3.5 | `fig:recorded-scene-map` | `assets/phase2/scene_map_upright.pdf` | controlled cached layout and source 0 HDF5/logged anchors |
| 3.6 | `fig:controller-flow` | `assets/v8_fig_3_4.png` | source_snapshots/hrc_pipeline_traintest/ |
| 3.7 | `fig:scripted-human-relocation` | `assets/phase2/human_relocation.pdf` | controlled source 0 recorded positions |
| 3.8 | `fig:dataset-construction` | `assets/v8_fig_3_5.png` | tables/split_summary.csv; tables/decision_anchor_summary.json |
| 3.9 | `fig:actual-supervision-sample` | `assets/phase2/observation_annotation.pdf` | controlled B/C export sample; docs/DATASET_SCHEMA.md |
| 3.10 | `fig:temporal-characteristics` | `assets/v8_fig_3_6.png` | tables/v8_segment_duration_summary.csv; tables/v8_source_context_characteristics.csv |
| 5.1 | `fig:visibility-segment` | `assets/v8_fig_5_1.png` | evidence/visibility/summary.csv |
| 5.2 | `fig:teacher-skill` | `assets/teacher_forced_skill_recreated.pdf` | evidence/frozen_audit/comparison.csv |
| 5.3 | `fig:gripper-event-profile` | `assets/v8_fig_5_3.png` | tables/v8_gripper_event_summary.csv |
| 5.4 | `fig:conditioning` | `assets/v8_fig_5_4.png` | evidence/conditioning/summary.json |
| A.1 | `fig:training-diagnostics` | `assets/v8_fig_A_1.png` | tables/training_curves.csv |
| A.2 | `fig:historical-basket-auxiliary` | `assets/phase2/basket_auxiliary_clean.pdf` | controlled historical auxiliary basket frames; excluded from final 45 sources |

Figure 3.5 was visually checked on the rendered dissertation page: both panels are upright, world axes and labels read horizontally, and the caption is below. Its cached background is spatial context, not a complete runtime collision map. Figure A.2 is a historical basket illustration, not a member of the 45-source training/evaluation population.

## Tables

| No. | LaTeX label | Supporting evidence |
|---|---|---|
| 3.1 | `tab:hrc-scope` | controlled study scope |
| 3.2 | `tab:dataset-split` | tables/split_summary.csv |
| 4.1 | `tab:configs` | source_snapshots/openpi_hrc/; evidence/checkpoints/verification_summary.json |
| 4.2 | `tab:protocol-populations` | evidence/frozen_audit/anchor_manifest.json |
| 5.1 | `tab:visibility-segment` | evidence/visibility/summary.csv |
| 5.2 | `tab:main-audit` | evidence/frozen_audit/comparison.csv |
| 5.3 | `tab:gripper-events` | tables/v8_gripper_event_summary.csv |
| 5.4 | `tab:baselines` | evidence/frozen_audit/comparison.csv |
| 5.5 | `tab:conditioning` | evidence/conditioning/summary.json |
| A.1 | `tab:round0-duration` | tables/v8_segment_duration_summary.csv |
| A.2 | `tab:round1-duration` | tables/v8_segment_duration_summary.csv |
| A.3 | `tab:hyperparameters` | source_snapshots/openpi_hrc/ |
| A.4 | `tab:checkpoint-identifiers` | evidence/checkpoints/verification_summary.json |
| A.5 | `tab:manifest` | evidence/frozen_audit/anchor_manifest.json |
| A.6 | `tab:all-context-arm-mae` | evidence/frozen_audit/comparison.csv |
| A.7 | `tab:event-counts` | tables/v8_gripper_event_summary.csv |
| A.8 | `tab:conditioning-full` | evidence/conditioning/summary.json |
| A.9 | `tab:extra-conditioning` | evidence/conditioning/summary.json |

The static mapping does not re-run collection or inference. Use `python3 tools/verify_released_evidence.py` to check released split, provenance, anchor, prediction-manifest, and checkpoint consistency. Compare any future manuscript edit against the actual compiled List of Figures and List of Tables.
