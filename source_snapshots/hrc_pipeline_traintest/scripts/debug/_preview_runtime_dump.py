#!/usr/bin/env python3
"""Validate the REAL _dump_trav_map_debug() from hrc_multitask_demo.py without
launching OmniGibson. Extracts the function source via AST and execs it with
stub globals + a stub trav_map built from the on-disk floor_trav_0.png downsized
to runtime resolution. Produces a preview of the runtime overlay."""
import ast
import os
import traceback
from typing import Any, List, Optional, Tuple  # noqa: F401 (used by exec'd fn)

import numpy as np
from PIL import Image

MOD = "/root/autodl-tmp/hrc_pipeline_traintest/scripts/collection/hrc_multitask_demo.py"
LAYOUT = "/root/autodl-tmp/datasets/behavior-1k-assets/scenes/house_double_floor_lower/layout"
OUT = "/root/autodl-tmp/hrc_pipeline_traintest/outputs/trav_map_viz"


# --- stub the module-level names the function closes over ---
def to_numpy(x):
    return np.asarray(x)


def ensure_parent(p):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


_ROBOT_INFLATE_M = 0.55

# --- extract and exec the EXACT function from the module source ---
src = open(MOD).read()
tree = ast.parse(src)
fn_src = next(ast.get_source_segment(src, n) for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "_dump_trav_map_debug")
ns = dict(globals())
exec(fn_src, ns)
_dump_trav_map_debug = ns["_dump_trav_map_debug"]


# --- stub trav_map / human_actor exactly as the runtime exposes them ---
RUNTIME_PX = 593  # int(5932 * 0.01 / 0.1)
raw = Image.open(os.path.join(LAYOUT, "floor_trav_0.png")).convert("L")
floor_map_arr = np.asarray(raw.resize((RUNTIME_PX, RUNTIME_PX), Image.BILINEAR))


class _StubTravMap:
    floor_map = [floor_map_arr]
    map_resolution = 0.1


class _StubHumanActor:
    def _try_get_trav_map(self):
        return _StubTravMap()


# Sample avoid_objects rects (world frame, already inflated) + actor markers, to
# exercise the overlay path. Real values come from the live scene at run time.
sample_rects = [
    (-9.6, 6.0, -8.0, 7.6),     # a chair near the big sofa zone
    (-1.2, 0.0, 0.4, 1.6),      # chair by the table cluster
    (3.0, -3.6, 4.6, -2.0),     # object in the lower object field
    (8.4, 0.2, 10.0, 1.8),      # chair near the right doorway
]
sample_points = [
    ("human", (2.0, 3.0), "magenta"),
    ("robot", (0.0, 1.0), "lime"),
]

_dump_trav_map_debug(_StubHumanActor(), OUT, floor=0,
                     tag="PREVIEW_with_avoid_rects",
                     rects=sample_rects, points=sample_points)
print("preview done")
