# A Hierarchical Diagnostic Evaluation Framework for Long-Horizon Human–Robot Collaboration Policy Learning

This private supplementary research repository accompanies the MSc dissertation by **WUHAODONG** at Nanyang Technological University. The dissertation studies structured human–robot collaboration (HRC) dataset construction, a diagnostic benchmark, a policy adaptation case study, and the evidence needed to interpret their limits. It does not propose a new foundation model.

The current branch contains the curated supplementary research material for the final dissertation; the immutable release commit is cited in the dissertation Appendix A.8. The earlier `msc-dissertation-submission-2026` tag is retained as a historical pre-cleanup snapshot. Access to this private repository must be granted to authorised examiners. The formal PDF and LaTeX source are held in the controlled dissertation delivery, pending a separate third-party asset redistribution review.

## Research objective and contributions

The objective is to diagnose where a long-horizon simulated HRC pipeline gains or loses reliable supervision and prediction quality. The three contributions are distinct:

1. **Dataset construction:** 45 quality-controlled HRC source trajectories combining scripted collaboration phases with replayed upstream manipulation demonstrations, plus source-level provenance and frozen splits.
2. **Diagnostic benchmark:** fixed validation anchors, behavioural baselines, matched inference noise, observation-conditioning probes, and clearly separated historical closed-loop evidence.
3. **Policy case study:** A20, B20, and C20 configurations of an existing π0.5 model, each represented by one historical 20,000-step training run.

The scripted controller handles human-facing phases, navigation and transitions. The learned policy supplies actions only within `ExecuteBehaviorTask` in the rollout interface. The dataset's manipulation phases replay upstream actions. These recorded trajectories are not evidence of a newly trained policy completing tasks autonomously.

## Dataset and provenance

Quality control retained 45 of 65 collected sources: 45 CLEAN, 19 REJECT and one excluded TRIM. The frozen source split is **37 train / 4 validation / 4 test**, with **409,670 recorded frames** at 30 Hz. Each retained HRC source contains two task rounds. A targeted verification traced both rounds to **90 distinct upstream raw demonstrations**, hashed the full files and action/state arrays, and matched all 90 recorded task-execution action sequences to their corresponding upstream actions. No exact complete-demonstration reuse was found within or across the frozen splits. This does not prove generalisation to new scenes, objects or tasks, nor exclude partial behavioural overlap.

The [sanitised source mapping](evidence/provenance/source_mapping_sanitized.csv), [verification summary](evidence/provenance/verification_summary.json), [dataset schema](docs/DATASET_SCHEMA.md), and [data-access policy](docs/DATA_ACCESS.md) document the releasable evidence. Raw BEHAVIOR HDF5 files and restricted simulation assets are not in this repository.

## Benchmark and policy case study

The frozen teacher-forced comparison uses **652 matched validation anchors** and inference flow-noise seeds **0–4**. These are five stochastic inference seeds applied to one trained checkpoint per configuration, not five training runs. A20→B20 changes prompt granularity and sampling together; B20→C20 changes state and action normalisation together. Neither transition is a single-factor causal ablation.

Pickup arm skill relative to a no-motion baseline is approximately −0.0107 / 0.0122 / 0.0144 for A20 / B20 / C20. Pickup gripper-event F1 is approximately 0.1120 / 0.3358 / 0.3825. A simple base-persistence rule exceeds the learned base predictors in the relevant contexts. Teacher-forced action prediction, sparse gripper-event scoring, and closed-loop task success are different outcomes; historical rollouts do not support a matched A20/B20/C20 task-success ranking. See [benchmark overview](docs/BENCHMARK_OVERVIEW.md) and [reproduction guide](docs/REPRODUCING_RESULTS.md).

## Checkpoints and reproducibility

The October 2026 verification restored **71 A20 inference parameter leaves** and bound them to the archived model architecture. The A20 training state was absent, and no matched fresh inference was run. B20/C20 save records and frozen predictions exist, but recoverable weights were not located in the documented project and backup search scope. The [checkpoint summary](evidence/checkpoints/verification_summary.json) records these boundaries. The 15 frozen prediction archives are represented by a [size, SHA-256 and array-schema manifest](evidence/predictions/manifest.json); the binary archives are withheld pending a derivative-data rights review.

The included evidence supports inspection of source splits and provenance, cross-checking frozen numerical summaries, and regenerating selected unrestricted charts where analysis dependencies are available. It does not establish complete end-to-end training, simulator, or model-inference reproduction. See [reproducibility limitations](docs/REPRODUCIBILITY_LIMITATIONS.md).

## Repository contents

| Path | Purpose |
|---|---|
| `source_snapshots/` | Audited research source trees and restoration notes; retain upstream notices. |
| `evidence/` | Frozen numeric summaries, provenance mapping, checkpoint status, and prediction manifest. |
| `tables/` | Released dataset, metric, and figure-source tables. |
| `figures/` | Previously reviewed non-video diagrams and plots; manuscript assets are separately controlled. |
| `tools/` | Release consistency checker and historical plotting/audit utilities. |
| `docs/` | Protocol, schema, access, traceability, and reproduction guidance. |
| `manuscript/` | LaTeX build requirements and correspondence to the submitted manuscript. |
| `reports/` | Core methodology, dataset, experiment and numerical-traceability audits; superseded Word-layout reports were removed from the current tree. |

## Checked reproduction entry point

From the repository root, run:

```bash
python3 tools/verify_released_evidence.py
```

This standard-library check validates the published mapping, splits, frozen anchor/seed protocol, summary identities, prediction manifest, and bounded checkpoint claims. It does **not** re-read restricted raw datasets. See [reproducing results](docs/REPRODUCING_RESULTS.md) for prerequisites and additional commands.

## Attribution, access, and citation

BEHAVIOR/OmniGibson assets, upstream teleoperated demonstrations, OpenPI and pretrained π0.5 components remain third-party contributions under their own terms. Source notices are retained; this repository does not grant a blanket licence to those assets. The earlier basket recording is a historical auxiliary illustration and is excluded from the final 45-source fine-tuning dataset and quantitative policy evaluation. [Release scope](docs/RELEASE_SCOPE.md), [third-party notices](docs/THIRD_PARTY_SOFTWARE.md), and [data-access restrictions](docs/DATA_ACCESS.md) provide the release boundary.

Suggested dissertation citation (replace with the institution's final bibliographic record after acceptance):

> WUHAODONG. *A Hierarchical Diagnostic Evaluation Framework for Long-Horizon Human–Robot Collaboration Policy Learning*. MSc dissertation submission candidate, Nanyang Technological University, 2026.

The [submission correspondence](manuscript/SUBMISSION_CORRESPONDENCE.md) identifies the fixed repository tag and explains what the controlled final manuscript can and cannot reproduce. Institutional originality declarations, supervisor approval, signatures, and Turnitin remain separate submission steps.
