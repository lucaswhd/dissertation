# Current supplementary research evidence

This branch contains the current evidence bundle corresponding to the final NTU MSc dissertation: selected HRC and OpenPI source snapshots, vetted numerical evidence, evaluation/verification scripts, figure-source tables and research documentation.

## Scope and cleanup
The current tree was curated by removing 49 superseded or duplicate files, including old Word/PDF delivery QA, review receipts, version-specific document-assembly scripts, obsolete release instructions and a redundant controller figure. Original Git commits and an earlier submission tag remain in history; neither has been rewritten. No training data, frozen numerical results or model weights were modified.

## Research evidence retained
- **Dataset:** 45 retained HRC sources, 409,670 frames; 37 training, 4 validation, 4 test. Ninety unique complete upstream demonstrations were provenance-checked; no exact complete-demo reuse was found within or across splits.
- **Offline policy audit:** 652 matched validation anchors, five inference-noise seeds applied to one checkpoint per A20/B20/C20 training run. Teacher-forced metrics and historical closed-loop runs remain explicitly separate.
- **Checkpoints:** A20 inference parameters were restored and architecture-bound without matched fresh inference. B20/C20 weights were not located within the documented search scope.
- **Reproducibility:** the released code and numeric summaries permit selected checks, but third-party raw HDF5, scene/human assets, full archived predictions and checkpoints are not redistributed.

Start with `README.md`, `docs/REPRODUCING_RESULTS.md`, `docs/FIGURE_TABLE_TRACEABILITY.md`, and `docs/RELEASE_SCOPE.md`. The dissertation Appendix A.8 cites the immutable SHA of this curated release.
