# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

"""Offline per-pair Relocate ENDPOINT search (2026-07-08).

Why: the first predefined endpoint (7,3) was found by eyeballing the trav_map and
running one episode to check — and the first guess (-3,3.5) was WRONG (it sits in
the 0.65m slot between the mirror and the sofa; the human squeezed through, the
0.6m-wide robot jammed). That process is neither reproducible nor generalizable.

This module picks, for a GIVEN human spawn, the best relocation endpoint using only
the scene's traversability image — no simulator, ~1-2s per call:

  1. Load `floor_trav_0.png` (1cm/px, white=free, includes walls + furniture + stairs).
     Verified transform (known points 6/6): col = 100*x + W/2, row = 100*y + H/2.
  2. Downsample to 10cm cells (a cell is free if ANY 1cm pixel is — see calibration).
  3. clearance = distance_transform(free) -> metres from each cell to nearest obstacle.
  4. Passable-for-the-ROBOT = cells with clearance >= robot_clear. Planning on THIS map
     is what makes the search generalize: gaps the human fits through but the robot
     doesn't (the mirror slot) are simply not traversable, so they can never be chosen
     — exactly the failure that (-3,3.5) hit.
  5. ONE Dijkstra from the spawn over that map -> geodesic length L to every cell.
  6. Keep cells that are: reachable, endpoint-open (clearance >= end_clear), within the
     distance band, and NOT a detour (L / straight-line <= max_detour).
  7. Score by straightness, then openness. Return the best (and optionally a spread-out
     top-K for a global endpoint pool).

ROOM CONSTRAINT (2026-07-08): steps 4-6 run on `passable` masked to the human's own room
(floor_insseg_0.png, same 1cm transform), so the PATH — not merely the endpoint — cannot
leave through a doorway and break the robot's line of sight. Pure geometry has no notion
of walls' openings and picked a y~9-11m endpoint out the north door for 3 of 8 spawns.
Tiers, tried in order: in-room [3,6]m -> in-room [2,6] -> in-room [1.2,6] -> any-room
[3,6]. Staying in view beats walking far: a boxed-in spawn (ep0005 — the robot's own
0.75m mask seals the neck of its room, leaving 1.34m of reach) gets a short in-room walk
rather than a 5.6m trek into the next room. The any-room tier exists only so that a room
too small even for 1.2m still yields SOME endpoint; skipping Relocate is a QC failure.

CALIBRATION (2026-07-08) — the geometry must agree with the SIMULATOR, not just look
reasonable. Two real-run ground truths were used:
    (7.0, 3.0)   58h: clean follow, 5.16m near-straight   -> must be REACHABLE
    (-3.0, 3.5)  58g: robot jammed in the 0.65m mirror/sofa slot -> must be UNREACHABLE
Grid-searching (downsample x robot_clear) against both:
    min-downsample  : any robot_clear -> (7,3) unreachable   [too conservative, kills good pts]
    max-downsample + robot_clear 0.45 -> (7,3) 5.3m/1.07x, (-3,3.5) unreachable   << CHOSEN
    max-downsample + robot_clear 0.55 -> (7,3) unreachable   [too strict]
With the chosen values the offline path length (5.3m, detour 1.07x) matches 58h's measured
5.16m near-straight walk, and (7,3)'s clearance reads 0.50m — consistent with the spawn
logic finding a 0.50m-tier point next to it. Re-run this calibration when the scene or the
robot footprint changes.

Generalizes to a new scene by swapping the png; to a new task pairing by passing that
pair's spawn. Geometry only — a candidate still MUST be confirmed by one real episode
(runtime trav_map is 10cm + baked furniture rects, and the leash follow is reactive).
"""
from typing import Any, Dict, List, Optional, Tuple

import heapq
import numpy as np

DEFAULT_TRAV_PNG = ("/root/autodl-tmp/datasets/behavior-1k-assets/scenes/"
                    "house_double_floor_lower/layout/floor_trav_0.png")

NATIVE_RES_M = 0.01     # floor_trav_0.png: 1 cm per pixel
CELL_M = 0.10           # downsampled planning cell
_SCENE_CACHE: Dict[str, Any] = {}


def _load_scene(png_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """-> (clearance_m[H,W] on 10cm cells, free[H,W] bool, origin_cell)
    origin_cell = cell index of world (0,0) along each axis (square map)."""
    if png_path in _SCENE_CACHE:
        return _SCENE_CACHE[png_path]
    from PIL import Image
    from scipy.ndimage import distance_transform_edt

    occ = np.asarray(Image.open(png_path).convert("L"))
    H, W = occ.shape
    k = int(round(CELL_M / NATIVE_RES_M))            # 10 px per cell
    h, w = H // k, W // k
    # MAX downsample: a 10cm cell is FREE if ANY 1cm pixel is. The conservative MIN
    # version inflates obstacles by up to 10cm and made even the verified-good (7,3)
    # unreachable at every robot_clear (see CALIBRATION in the module docstring).
    ds = occ[:h * k, :w * k].reshape(h, k, w, k).max(axis=(1, 3))
    free = ds > 0
    clearance = distance_transform_edt(free) * CELL_M
    origin = (W / 2.0) / k                            # world (0,0) -> this cell index
    _SCENE_CACHE[png_path] = (clearance, free, origin)
    return _SCENE_CACHE[png_path]


def _load_rooms(png_path: str) -> Optional[np.ndarray]:
    """-> room-instance id per 10cm cell (0 = wall/doorway/void), or None if absent.

    `floor_insseg_0.png` sits next to floor_trav_0.png, is the same 5932² @1cm image, and
    OmniGibson reads it with the SAME transform (BaseMap.world_to_map: row = y/res + N/2,
    col = x/res + N/2), so the trav-map cell indices address it directly. Downsampled by
    point-sampling the cell centre (OG itself uses cv2 INTER_NEAREST)."""
    key = png_path + "#rooms"
    if key in _SCENE_CACHE:
        return _SCENE_CACHE[key]
    from PIL import Image
    ins_path = png_path.replace("floor_trav_0.png", "floor_insseg_0.png")
    try:
        ins = np.asarray(Image.open(ins_path).convert("L"))
    except Exception:
        _SCENE_CACHE[key] = None
        return None
    H, W = ins.shape
    k = int(round(CELL_M / NATIVE_RES_M))
    h, w = H // k, W // k
    ds = ins[:h * k, :w * k][k // 2::k, k // 2::k]
    _SCENE_CACHE[key] = ds
    return ds


def _room_id_at(rooms: np.ndarray, rc: Tuple[int, int], search_cells: int = 6) -> int:
    """Room id at cell rc; if it lands on a boundary (0), take the nearest labelled cell.
    A spawn standing in a doorway would otherwise report 'no room' and disable the mask."""
    H, W = rooms.shape
    r, c = rc
    if 0 <= r < H and 0 <= c < W and rooms[r, c] != 0:
        return int(rooms[r, c])
    r0, r1 = max(r - search_cells, 0), min(r + search_cells + 1, H)
    c0, c1 = max(c - search_cells, 0), min(c + search_cells + 1, W)
    win = rooms[r0:r1, c0:c1]
    rr, cc = np.nonzero(win)
    if rr.size == 0:
        return 0
    i = int(np.argmin((rr + r0 - r) ** 2 + (cc + c0 - c) ** 2))
    return int(win[rr[i], cc[i]])


def _world_to_cell(xy, origin: float) -> Tuple[int, int]:
    # col = 100x + W/2 (1cm px)  ->  /10 cells  ->  10x + origin
    return (int(round(10.0 * float(xy[1]) + origin)),    # row  <- y
            int(round(10.0 * float(xy[0]) + origin)))    # col  <- x


def _cell_to_world(rc: Tuple[int, int], origin: float) -> Tuple[float, float]:
    return ((rc[1] - origin) / 10.0, (rc[0] - origin) / 10.0)


_NBR8 = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, 1.41421356), (-1, 1, 1.41421356),
         (1, -1, 1.41421356), (1, 1, 1.41421356))


def _dijkstra(passable: np.ndarray, start_rc: Tuple[int, int],
              max_cells: Optional[float] = None) -> np.ndarray:
    """Geodesic distance (in CELLS) from start over 8-connected passable cells.

    Uses scipy's C Dijkstra on a sparse 8-neighbour graph. A pure-Python heapq version
    measured 42s for a mere 2014-cell reachable region (~21ms/cell — all interpreter and
    numpy-scalar-index overhead, not search breadth); this runs in well under a second.
    ``max_cells`` bounds the geodesic radius (scipy's ``limit``): an endpoint can only
    qualify within band_max * max_detour metres, so the rest of the floor is irrelevant."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra as _sp_dijkstra

    H, W = passable.shape
    out = np.full((H, W), np.inf, dtype=np.float32)
    sr, sc = start_rc
    if not (0 <= sr < H and 0 <= sc < W and passable[sr, sc]):
        return out

    rr, cc = np.nonzero(passable)
    n = int(rr.size)
    idx = np.full((H, W), -1, dtype=np.int64)
    idx[rr, cc] = np.arange(n)

    src_list, dst_list, w_list = [], [], []
    base = np.arange(n)
    for dr, dc, wgt in _NBR8:
        nr, nc = rr + dr, cc + dc
        ok = (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        j = np.full(n, -1, dtype=np.int64)
        j[ok] = idx[nr[ok], nc[ok]]
        good = j >= 0
        src_list.append(base[good])
        dst_list.append(j[good])
        w_list.append(np.full(int(good.sum()), wgt, dtype=np.float64))

    graph = coo_matrix((np.concatenate(w_list),
                        (np.concatenate(src_list), np.concatenate(dst_list))),
                       shape=(n, n)).tocsr()
    d = _sp_dijkstra(graph, directed=False, indices=int(idx[sr, sc]),
                     limit=(np.inf if max_cells is None else float(max_cells)))
    out[rr, cc] = d.astype(np.float32)
    return out


def search_endpoints(spawn_xy,
                     robot_xy=None,
                     png_path: str = DEFAULT_TRAV_PNG,
                     robot_clear: float = 0.45,   # calibrated (see module docstring)
                     end_clear: float = 0.45,     # (7,3) reads 0.50m; score prefers more
                     band: Tuple[float, float] = (3.0, 6.0),
                     max_detour: float = 1.30,
                     robot_pad: float = 0.75,
                     top_k: int = 5,
                     spread_m: float = 2.5,
                     same_room: bool = True) -> List[Dict[str, float]]:
    """Rank relocation endpoints for a human spawning at *spawn_xy*.

    ``robot_xy`` (the robot's pose when Relocate begins) is masked out as impassable
    with radius ``robot_pad``, so the planned human path can never run THROUGH the
    robot. That is the planning-stage cure for the knock-over: the kinematic human
    walks its path at constant speed and shoves the dynamic robot flat (ep0004 tilt
    180°). Human half-width (~0.25) + robot radius (~0.30) = 0.55m of real contact, so
    0.75m leaves margin.

    Returns up to top_k dicts {x, y, dist, path_len, detour, clear}, best first,
    mutually >= spread_m apart. Empty list if nothing qualifies."""
    clearance, free, origin = _load_scene(png_path)
    base_passable = (clearance >= float(robot_clear)).copy()   # robot-body-width map
    if robot_xy is not None:
        rr, cc = _world_to_cell(robot_xy, origin)
        rad = max(int(round(float(robot_pad) / CELL_M)), 1)
        r0, r1 = max(rr - rad, 0), min(rr + rad + 1, base_passable.shape[0])
        c0, c1 = max(cc - rad, 0), min(cc + rad + 1, base_passable.shape[1])
        if r1 > r0 and c1 > c0:
            yy = np.arange(r0, r1)[:, None] - rr
            xx = np.arange(c0, c1)[None, :] - cc
            base_passable[r0:r1, c0:c1] &= (yy * yy + xx * xx) > rad * rad
    start_cell = _world_to_cell(spawn_xy, origin)

    # ROOM CONSTRAINT. Masking `passable` (not just the endpoint) means the geodesic
    # path itself cannot leave the human's room — so the human can never walk out of a
    # doorway and break the robot's line of sight (ep0003: "人走到门外去了"). Filtering
    # only the endpoint would still allow a path that ducks through an adjacent room.
    # Pure geometry ranks by straightness and has no idea where the walls' openings are,
    # so without this the top pick for 3 of 8 spawns was y~9-11m, through the north door.
    rooms = _load_rooms(png_path) if same_room else None
    home_id = _room_id_at(rooms, start_cell) if rooms is not None else 0

    # Attempts, in order of decreasing desirability. Staying in the room matters more
    # than walk length: a short in-room relocate is good training data, while a long
    # one that exits a doorway costs the robot line-of-sight. So relax the band's LOWER
    # bound in-room (a 2m walk still clears RELOC_MIN_WALK_M=0.5 by 4x) before ever
    # considering a cross-room endpoint. Several rooms of this scene have no 3m straight
    # span at all (ep0000/ep0005 spawn in one), which is why the strict band alone fell
    # through to a walk out the north door.
    # 1.2m is the floor: it still clears the QC's RELOC_MIN_WALK_M=0.5 by ~2.4x, and a
    # spawn boxed that tightly (ep0005: the robot's own 0.75m mask seals the neck of a
    # 23.7 m^2 room, leaving 1.34m of in-room reach) has nowhere better to go.
    tiers = [b for b in (float(band[0]), 2.0, 1.2) if b < float(band[0])]
    attempts: List[Tuple[np.ndarray, Tuple[float, float], str]] = []
    if home_id != 0:
        in_room = base_passable & (rooms == home_id)
        attempts.append((in_room, band, "same"))
        for lo in tiers:
            attempts.append((in_room, (lo, float(band[1])), f"same-{lo:g}m"))
    # Last resort: a room too small even for `relaxed`. A cross-room endpoint is still
    # robot-navigable and robot-masked; better than skipping Relocate entirely, which
    # would be a `no_reloc` QC failure (an episode with no Relocate segment at all).
    attempts.append((base_passable, band, "any"))

    for passable, att_band, room_tag in attempts:
        out = _rank_endpoints(passable, clearance, origin, spawn_xy, start_cell,
                              end_clear, att_band, max_detour, top_k, spread_m, room_tag)
        if out:
            return out
    return []


def _rank_endpoints(passable, clearance, origin, spawn_xy, start,
                    end_clear, band, max_detour, top_k, spread_m,
                    room_tag: str) -> List[Dict[str, float]]:
    """One Dijkstra over *passable* + band/detour/openness filtering + spread-out top-K.
    Returns [] when the mask admits no qualifying endpoint, which is how the caller
    detects that the same-room attempt failed and the "any" fallback is needed."""
    # The spawn itself may sit slightly tighter than robot_clear (the human is slimmer
    # than the robot, and the room mask can exclude the spawn cell outright). Snap the
    # planning start to the nearest passable cell so Dijkstra can leave.
    # Measure `straight` from the START CELL CENTRE, not from the exact spawn_xy: Dijkstra
    # departs from the cell centre, so a sub-cell (<=7cm) offset otherwise makes a nearby
    # endpoint report detour < 1.0 — geometrically impossible (ep0005 read 0.99x). With
    # both measured from the same grid point the triangle inequality guarantees >= 1.0.
    src_xy = _cell_to_world(start, origin)

    snap_m = 0.0
    if not (0 <= start[0] < passable.shape[0] and 0 <= start[1] < passable.shape[1]
            and passable[start]):
        rr, cc = np.nonzero(passable)
        if rr.size == 0:
            return []
        d2 = (rr - start[0]) ** 2 + (cc - start[1]) ** 2
        i = int(np.argmin(d2))
        if d2[i] > (30 ** 2):                          # >3m away -> spawn is walled in
            return []
        start = (int(rr[i]), int(cc[i]))
        # The human still starts at spawn_xy, so the real walk is snap + geodesic. Charge
        # the offset to every path length: otherwise `straight` (measured from spawn_xy)
        # is compared against a geodesic measured from the SNAPPED cell, and an endpoint
        # the snap moved toward reports detour < 1.0 — geometrically impossible, and it
        # sails through the max_detour filter (ep0000 read 0.88x).
        snap_m = float(np.sqrt(float(d2[i]))) * CELL_M

    # An endpoint can only qualify within band_max * max_detour metres of geodesic
    # distance, so stop the wavefront there instead of flooding the whole floor.
    _max_cells = (float(band[1]) * float(max_detour)) / CELL_M
    geo = _dijkstra(passable, start, max_cells=_max_cells) * CELL_M   # metres, robot-safe
    geo += snap_m                                     # geo stays +inf where unreachable

    rows, cols = np.nonzero(np.isfinite(geo) & (clearance >= float(end_clear)))
    if rows.size == 0:
        return []
    xs = (cols - origin) / 10.0
    ys = (rows - origin) / 10.0
    straight = np.hypot(xs - src_xy[0], ys - src_xy[1])
    L = geo[rows, cols]

    ok = (straight >= band[0]) & (straight <= band[1]) & (straight > 1e-6)
    ok &= (L / np.maximum(straight, 1e-6)) <= float(max_detour)
    if not np.any(ok):
        return []

    xs, ys, L, straight = xs[ok], ys[ok], L[ok], straight[ok]
    clr = clearance[rows[ok], cols[ok]]
    detour = L / straight
    # STRAIGHTNESS dominates, openness is a tie-breaker (capped so a cavernous but
    # winding endpoint can't win). A straight path is what keeps the reactive leash
    # follow clean; extra clearance beyond ~1.2m buys nothing. Ranking by openness
    # first picked (0.2, 8.7) — clearance 1.36m but detour 1.24x and almost certainly
    # through a doorway into another room — over the verified-good (7.4, 2.7) at 1.04x.
    score = (float(max_detour) - detour) + 0.3 * np.minimum(clr, 1.2)

    out: List[Dict[str, float]] = []
    for i in np.argsort(-score):
        x, y = float(xs[i]), float(ys[i])
        if any(np.hypot(x - o["x"], y - o["y"]) < spread_m for o in out):
            continue
        out.append({"x": round(x, 2), "y": round(y, 2), "dist": round(float(straight[i]), 2),
                    "path_len": round(float(L[i]), 2), "detour": round(float(detour[i]), 3),
                    "clear": round(float(clr[i]), 2), "room": room_tag})
        if len(out) >= top_k:
            break
    return out


def pick_endpoint(spawn_xy, **kw) -> Optional[Tuple[float, float]]:
    """Best single endpoint for this spawn, or None if the scene offers none."""
    r = search_endpoints(spawn_xy, top_k=1, **kw)
    return (r[0]["x"], r[0]["y"]) if r else None
