# Reproducibility limitations

The released mapping, frozen summaries and selected figure tables permit inspection and internal-consistency checks. They do not provide complete end-to-end reproduction of collection, training, inference or closed-loop policy evaluation.

| Boundary | Verified evidence | What remains unverified or unavailable |
|---|---|---|
| Dataset provenance | 45 retained sources map to 90 distinct upstream full-file hashes; all 90 recorded task-execution action sequences matched. | Statistical independence, unseen-scene generalisation, and all possible partial overlap or leakage mechanisms. |
| Dataset split | 37/4/4 source-level split and 409,670 exported rows. | A new random split or unseen-domain test. |
| Frozen audit | 652 common validation anchors, five matched inference flow-noise seeds and archived numerical summaries. | Independent training-run uncertainty; there is one historical training run per configuration. |
| A20 checkpoint | All 71 inference parameter leaves restored and bound to the archived model architecture. | Full training-state resumption and matched fresh inference. |
| B20/C20 checkpoints | Historical final-save logs and saved predictions remain. | Recoverable weight files were not located within the documented 20-root project/backup search; absence elsewhere is not proved. |
| Saved prediction archives | 15 file hashes, sizes and schemas are documented. | Binaries are withheld pending derivative-data rights review; repository-only metric re-scoring is unavailable. |
| Behaviour | Historical closed-loop diagnostics are retained separately. | A matched A20/B20/C20 closed-loop success comparison. |
| Third-party materials | Source notices and aggregate evidence are retained. | Redistribution of raw demonstrations, rendered scene/human imagery and model weights without rights review. |

The five seeds are inference-noise seeds, not independent training seeds. A20→B20 and B20→C20 each change more than one factor. Teacher-forced prediction quality and sparse gripper-event F1 must not be described as closed-loop task success. The historical basket example is outside the final 45-source fine-tuning population.
