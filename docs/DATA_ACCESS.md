# Data access and redistribution boundary

This private repository is available only to authorised collaborators and reviewers. Access to the GitHub repository and access to third-party raw data are separate permissions. Repository privacy does not itself permit redistribution of BEHAVIOR raw HDF5, OmniGibson scene or human assets, simulation videos, unreviewed rendered frames, or model checkpoints.

The release contains selected research code, aggregate numerical tables, audit metadata, a sanitised 45-source mapping, and hashes of withheld evidence. The 15 frozen `predictions.npz` files total 26,957,010 bytes and are omitted pending a specific review of derivative-data rights and embedded content. Their file hashes, sizes, and array schemas are in `evidence/predictions/manifest.json`; each file contains `predictions` with shape `[652, 32, 23]` and `anchor_ids` with shape `[652]`. This manifest allows an authorised holder to check file identity, but not to recompute metrics from the absent arrays.

The final dissertation PDF and source archive contain third-party rendered imagery. They remain in controlled delivery on the research server while redistribution rights are reviewed separately. No missing B20/C20 checkpoint has been substituted or represented as available. A20 parameter restoration is documented, but its weights are not distributed here.

For authorised access, contact the dissertation author through the institution's normal review process. A recipient must obtain required upstream data and simulator rights independently, and must not infer a general publication licence from this private repository. The source-tree notices in `source_snapshots/` and `docs/THIRD_PARTY_SOFTWARE.md` remain applicable.
