# Reproducing the released checks

## Checked command with no external datasets

From the repository root, run:

```bash
python3 tools/verify_released_evidence.py
```

This standard-library command was executed against the release tree. It checks 45 source rows, unique full-file upstream hashes, 37/4/4 splits, 409,670 frame totals, the frozen 652-anchor manifest hash, common A20/B20/C20 seed protocols, 15 prediction archive descriptions, and bounded checkpoint claims. It checks consistency of **published evidence**, not the original restricted raw files.

## Checked non-video figure smoke test

With `numpy`, `matplotlib`, and `Pillow` installed in a separate analysis environment, run from a clean copy of the repository:

```bash
python3 tools/plotting/generate_figures.py --from-release-tables
```

This command was smoke-tested on the research server using the same script and released input-table hashes as this repository. It regenerated four non-video figure families. It writes into `figures/` and rewrites `tables/training_curves.csv`; run it in a disposable copy if you need to preserve a pristine checkout. It does not redraw restricted scene maps, camera storyboards, or other frame-based manuscript assets.

## Evidence layers

| Layer | What a reviewer can do with this repository | Additional prerequisite |
|---|---|---|
| Published statistics | Inspect tables, context summaries and documented formulae. | None beyond a text/CSV/JSON reader. |
| Existing frozen predictions | Validate file identity and schema against the manifest. | Authorised access to withheld NPZ files. |
| Re-score saved predictions | Re-run the historical scoring scripts and compare outputs. | Withheld NPZ files, matching exported truth and the original analysis environment. |
| Verify provenance from raw files | Re-hash 90 upstream files and compare full action sequences. | Licensed raw HDF5, final batch, LeRobot export and research dependencies. |
| Fresh model inference | Regenerate the 652-anchor outputs. | Correct historical execution environment and model weights; B20/C20 weights were not located in the documented search scope. |
| Train or simulate | Re-create training and closed-loop trajectories. | Licensed simulator/assets, complete run configuration and substantially greater compute. Not attempted in this release. |

The executed provenance and checkpoint audit sources, with their SHA-256 identities and server-specific limitations, are documented in `tools/verification/README.md`. The historical plot scripts are in `tools/plotting/`. They are not a one-command reproduction of the paper. Each plot's inputs and manuscript placement are listed in `docs/FIGURE_TABLE_TRACEABILITY.md`; dataset- or frame-dependent plots require licensed local data. The report and source snapshots retain historical implementation details. The two checked commands above have no simulator or GPU dependency.
