# V8 Figure 5.1 repair

Pass: redrawn from evidence/visibility/summary.csv, filtered to val / pickup / valid_anchors. Early/middle/late counts are summed before division and checked exactly against native Table5.1. Four validation sources; this is an anchor-weighted descriptive geometric metric, not an independent-source estimate. No error bars or new inference.

Variables: n = valid anchors in segment; numerator = anchors with target centre in the camera frustum; ratio = numerator/n. Counts:2916/3569,6054/6792,1560/1560,381/381. The metric does not model occlusion or rendered-pixel visibility. Segment labels are shortened only in the axis.

Caption: Validation pickup target-centre geometric frustum coverage by task segment. Counts are aggregated over early/middle/late positions before division; n denotes valid anchors. Geometric coverage does not establish rendered visibility or policy success.

The legacy raster clipped the left edge of its y-axis label. This fixed-width5.7in redraw checks every label against canvas bounds and preserves the counts. PNG400dpi, editable text SVG and vector PDF are saved under figures/results/v8_fig_5_1_frustum.*; derived CSV:tables/v8_frustum_coverage.csv. Reproduce: python tools/plotting/generate_v8_frustum.py.
