# Dataset and resource manifest

The private, absolute-path inventory is stored only on the research server at `/root/dissertation_local_audit/LOCAL_RESOURCE_MANIFEST.csv` (mode 0600). `reports/FILE_CLASSIFICATION.csv` uses root aliases and relative paths; directory pruning and hashing coverage are explicit. Large data/checkpoint files are represented by metadata rather than uploaded contents or invented full-file hashes.

| Local resource (relative to research data root) | Purpose | Release treatment |
|---|---|---|
| `hrc_pipeline_traintest/outputs/hrc_batch_rerun_v1/ep_*/hrc_dataset.hdf5` | Raw composed HRC recordings, 45 retained sources / 409,670 frames | Local only; all retained numeric fields checked in chunks. |
| Same sources' `videos/head_rgb.mp4` | 352×256 head RGB at 30 Hz | Local only; frame-count metadata matches all 45 HDF5 sources; six exact frames extracted for local thesis storyboard. |
| `lerobot_datasets/hrc_pi05_continuous_rerun_v1/` | A continuous export | Local only; state/action shift-1 checks cover all rows. |
| `lerobot_datasets/hrc_subinstr_rerun_v1/` | B/C instruction/event-labelled export | Local only; actual valid masks verified; inherited manifest counts flagged. |
| `hrc_canonical/hrc_multitask_v01_step10/canonical.hdf5` | Historical ACT pathway, 10,239 rows | Local only; not required by final continuous π0.5 pipeline. |
| `datasets/2025-challenge-rawdata`, scene/robot assets | Upstream manipulation and simulator inputs | Excluded; acquire from original providers under their terms. |
| `openpi_hrc/checkpoints/...` | Policy weights | Excluded; A20 remains, B20/C20 final checkpoint paths absent at audit. |
| `openpi_hrc/outputs/dissertation_final_audit` | Frozen predictions and scores | Only selected small numeric summaries copied under `evidence/`; saved predictions remain local. |

Source separation is 37 / 4 / 4 train/validation/test; no source overlaps. The four test sources are excluded from training/model selection/main policy-performance evaluation. Later descriptive geometry use of test sources is distinguished from policy evaluation.

`tables/source_statistics.csv`, `split_summary.csv`, `dataset_checks.json`, `decision_anchor_summary.json`, `reports/export_alignment_checks.json` and `reports/independent_archive_checks.json` preserve the checked numeric evidence. Original pixels, raw trajectories, latent model weights, scenes, textures and human-motion files are not necessary to read these summaries and are not redistributed here.
