# Executed provenance and checkpoint audit scripts

The four Python files in this directory are byte-for-byte copies of the October 2026 scripts executed on the controlled research server. Their source hashes were checked when preparing this repository. They contain server-specific paths and, where applicable, deliberately refuse writes outside the controlled audit directory. They are retained for method inspection; they are **not** a portable one-command reproduction and do not include the restricted HDF5 files, model weights, or saved prediction arrays.

| Script | SHA-256 of executed source copy | Role |
|---|---|---|
| `verify_provenance.py` | `242428d21565212c098796994ca80e6f4e34ba2cf92f60cb59c9ecf0e7427ff4` | Full upstream file and action/state identity audit. |
| `verify_checkpoints.py` | `295b88938cf61130f63fa2fe9848a0303781652ec8f320a9f05c15ad4f3a2bef` | Bounded checkpoint search, metadata and hashes. |
| `checkpoint_context_readonly.py` | `731c0e308384bd35673347c7cafb2ea421cf54353687fc259069ee7c1a0180a3` | Configuration, normalisation and archive context. |
| `checkpoint_restore_readonly.py` | `40ff2706d120d64b060242cf979b3a2bccb23681fbe35acc378310d521ec1ea9` | Read-only A20 parameter restoration and architecture binding; no prediction inference. |

The controlled server report records the executed commands below. These are **historical command records**, not commands a reader can run without the licensed datasets, checkpoint environment and server directory layout. Re-running the provenance or checkpoint inventory scripts rewrites derived audit JSON and can take substantial I/O; it does not retrain a model.

```bash
/root/autodl-tmp/openpi_hrc/.venv/bin/python \
  /root/dissertation_evidence_verification/scripts/verify_provenance.py \
  --batch-root /root/autodl-tmp/hrc_pipeline_traintest/outputs/hrc_batch_rerun_v1 \
  --dataset-root /root/autodl-tmp/lerobot_datasets/hrc_pi05_continuous_rerun_v1 \
  --pipeline-root /root/autodl-tmp/hrc_pipeline_traintest \
  --output-root /root/dissertation_evidence_verification
python /root/dissertation_evidence_verification/scripts/verify_checkpoints.py --hash-payload
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/openpi_hrc/.venv/bin/python \
  /root/dissertation_evidence_verification/scripts/checkpoint_context_readonly.py
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/openpi_hrc/.venv/bin/python \
  /root/dissertation_evidence_verification/scripts/checkpoint_restore_readonly.py
```

The corresponding published conclusions are in `evidence/provenance/verification_summary.json` and `evidence/checkpoints/verification_summary.json`; full raw-path inventories remain in the controlled server evidence directory.

For a safe repository-only check, run `python3 tools/verify_released_evidence.py` from the repository root. That command was actually smoke-tested and does not access or modify restricted originals.
