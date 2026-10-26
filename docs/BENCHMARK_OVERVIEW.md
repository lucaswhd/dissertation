# Benchmark overview

The dissertation uses a five-level diagnostic framework for long-horizon human–robot collaboration (HRC). It tests the construction of supervision, simple behavioural references, matched stochastic inference, observation conditioning, and a separate historical behavioural audit. The fifth level is supporting closed-loop evidence, not a matched A20/B20/C20 rollout comparison.

## Frozen evaluation population

The final teacher-forced audit selects 652 matched validation anchors from four held-out validation sources. Each of A20, B20 and C20 is scored at the same anchors with inference flow-noise seeds 0–4. The five seeds vary stochastic inference for **one trained checkpoint per configuration**. They do not estimate variability across independent training runs. The anchor manifest is `evidence/frozen_audit/anchor_manifest.json`; its SHA-256 is `d5553026605163a74820a865db80e90e81249d3fcceba9cb19655a43fa2bd6a8`.

| Level | Population and purpose | Main boundary |
|---|---|---|
| 1. Supervision | 45 retained sources, chunk validity, source splits and sampling | Recorded rows and eligible anchors are different populations. |
| 2. Behavioural references | No motion, prompt-conditioned lookup and base persistence | Simple references can outperform learned components. |
| 3. Matched inference | 652 anchors × five inference seeds × three models | One training run per model; no across-run uncertainty. |
| 4. Conditioning | Separate validation probe of 108 anchors plus training-coverage diagnostics | Local sensitivity is not causal proof that an input is unnecessary. |
| 5. Historical behaviour | Archived closed-loop diagnostics | Different protocol; no matched A/B/C task-success ranking. |

A20→B20 changes prompt granularity and sampling together. B20→C20 changes state and action normalisation together. These comparisons are diagnostic associations, not single-factor causal ablations.

Teacher-forced arm error/skill scores saved action chunks against observed targets. Gripper-event F1 scores sparse event detection. Closed-loop task success requires a separate rollout with feedback from earlier actions. Neither teacher-forced quality nor event F1 alone establishes successful autonomous manipulation.

The manuscript's numerical tables are supported by `evidence/frozen_audit/`, `evidence/conditioning/`, and `tables/`. The archived per-seed prediction binaries are withheld pending a derivative-data rights review; their 15 hashes, sizes and array schemas are in `evidence/predictions/manifest.json`.
