# Copyright (c) 2026 Mike Xu (wuhaodong808@gmail.com). All Rights Reserved.
# Unauthorized copying, modification, distribution, or use of this file,
# via any medium, including academic submissions, is strictly prohibited.
# See LICENSE in the project root for full terms.

import argparse
import hashlib
import json
import pickle
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


HABITAT_SMPLX_MOTION_JOINT_ORDER = [
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
]

_NUM_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_IMAGE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}
_HABITAT_HUMANOID_BRIDGE_VERSION = 15
_HABITAT_CHARACTER_PACKAGE_LAYOUT_VERSION = 6
_HABITAT_PACKAGE_AUTHORED_UP_AXIS = "Y"
_HABITAT_TARGET_STAGE_UP_AXIS = "Z"
_HABITAT_RUNTIME_VISUAL_CORRECTION_QUAT_XYZW = np.array(
    [0.70710677, 0.0, 0.0, 0.70710677],
    dtype=np.float32,
)

_SKINNING_ZUP_TO_YUP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
_SKINNING_ZUP_TO_YUP_4X4 = np.eye(4, dtype=np.float32)
_SKINNING_ZUP_TO_YUP_4X4[:3, :3] = _SKINNING_ZUP_TO_YUP


def is_habitat_humanoid_asset(asset_path: str) -> bool:
    glb_path = Path(asset_path).expanduser().resolve()
    if glb_path.suffix.lower() != ".glb" or not glb_path.exists():
        return False
    motion_path = glb_path.with_name(f"{glb_path.stem}_motion_data_smplx.pkl")
    return motion_path.exists()


def convert_habitat_humanoid_to_usda(
    asset_path: str,
    cache_root: Optional[str] = None,
) -> str:
    glb_path = Path(asset_path).expanduser().resolve()
    if not is_habitat_humanoid_asset(str(glb_path)):
        raise ValueError(f"Not a Habitat humanoid asset: {glb_path}")

    motion_path = glb_path.with_name(f"{glb_path.stem}_motion_data_smplx.pkl")
    default_cache_root = Path(__file__).resolve().parents[1] / "outputs" / "external" / "converted_humans"
    cache_root_path = Path(cache_root).expanduser().resolve() if cache_root else default_cache_root

    cache_key = _make_cache_key(glb_path, motion_path)
    cache_dir = cache_root_path / cache_key
    usda_path = cache_dir / "human_stop_pose.usda"
    character_usda_path = cache_dir / "character.usda"
    skinned_character_usda_path = cache_dir / "character_skinned.usda"
    rig_usda_path = cache_dir / "rig.usda"
    walk_clip_usda_path = cache_dir / "walk.usda"
    metadata_path = cache_dir / "source_asset.json"
    walk_motion_path = cache_dir / "walk_motion_metadata.npz"

    if (
        usda_path.exists()
        and character_usda_path.exists()
        and skinned_character_usda_path.exists()
        and rig_usda_path.exists()
        and walk_clip_usda_path.exists()
        and metadata_path.exists()
        and walk_motion_path.exists()
    ):
        print(
            f"[habitat_bridge] Cache hit for {glb_path.name} "
            f"(bridge v{_HABITAT_HUMANOID_BRIDGE_VERSION}, key={cache_key}) -> {character_usda_path}",
            flush=True,
        )
        return str(character_usda_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[habitat_bridge] Generating package for {glb_path.name} "
        f"(bridge v{_HABITAT_HUMANOID_BRIDGE_VERSION}, key={cache_key}) in {cache_dir}",
        flush=True,
    )

    gltf_doc, bin_blob = _load_glb(glb_path)
    primitive, mesh_node_index, skin_index = _find_primary_mesh_primitive(gltf_doc)
    mesh = _extract_mesh_data(gltf_doc, bin_blob, primitive, skin_index=skin_index)
    skin_data = _extract_skin_data(gltf_doc, bin_blob, mesh["skin_index"])
    texture_suffix, texture_bytes = _extract_base_color_texture(gltf_doc, bin_blob, primitive, glb_path)
    motion_data = _load_motion_data(motion_path)

    motion_joint_quats = np.asarray(motion_data["stop_pose"]["joints"], dtype=np.float32)
    if motion_joint_quats.shape != (len(HABITAT_SMPLX_MOTION_JOINT_ORDER), 4):
        raise RuntimeError(
            f"Unexpected stop_pose joint shape {motion_joint_quats.shape} for {motion_path}"
        )

    urdf_path = glb_path.with_suffix(".urdf")
    candidate_joint_orders = [("smplx", HABITAT_SMPLX_MOTION_JOINT_ORDER)]
    if urdf_path.exists():
        urdf_joint_order = _load_urdf_joint_order(urdf_path)
        if urdf_joint_order and urdf_joint_order != HABITAT_SMPLX_MOTION_JOINT_ORDER:
            candidate_joint_orders.append(("urdf", urdf_joint_order))

    posed_points = None
    chosen_joint_order = HABITAT_SMPLX_MOTION_JOINT_ORDER
    chosen_joint_order_source = "smplx"
    best_score = None
    for joint_order_source, joint_order in candidate_joint_orders:
        candidate_points = _pose_points_for_joint_order(
            nodes=gltf_doc["nodes"],
            mesh=mesh,
            mesh_node_index=mesh_node_index,
            skin_data=skin_data,
            motion_joint_quats=motion_joint_quats,
            motion_joint_order=joint_order,
        )
        score = _score_humanoid_extent(candidate_points)
        if best_score is None or score < best_score:
            posed_points = candidate_points
            chosen_joint_order = joint_order
            chosen_joint_order_source = joint_order_source
            best_score = score

    if posed_points is None:
        raise RuntimeError(f"Failed to pose Habitat humanoid mesh for {glb_path}")

    posed_local_mats = _build_local_transforms(
        gltf_doc["nodes"],
        motion_joint_quats=motion_joint_quats,
        motion_joint_order=chosen_joint_order,
    )
    posed_world_mats = _build_world_transforms(gltf_doc["nodes"], posed_local_mats)

    rest_points = mesh["positions"].astype(np.float32)
    rest_height = float(rest_points[:, 1].max() - rest_points[:, 1].min())
    posed_height = float(posed_points[:, 1].max() - posed_points[:, 1].min())
    normalization_scale = 1.0 if posed_height <= 1e-8 else rest_height / posed_height
    posed_points *= np.float32(normalization_scale)
    floor_offset_y = float(posed_points[:, 1].min())
    posed_points[:, 1] -= np.float32(floor_offset_y)

    bounds_min = posed_points.min(axis=0)
    bounds_max = posed_points.max(axis=0)

    final_texture_path = cache_dir / f"albedo{texture_suffix}"
    final_texture_path.write_bytes(texture_bytes)
    _write_usda(
        usda_path=usda_path,
        points=posed_points,
        texcoords=mesh["texcoords"],
        face_indices=mesh["indices"],
        texture_name=final_texture_path.name,
        extent_min=bounds_min,
        extent_max=bounds_max,
    )
    rig_data = _build_rig_data(
        nodes=gltf_doc["nodes"],
        skin_data=skin_data,
        posed_local_mats=posed_local_mats,
        posed_world_mats=posed_world_mats,
        normalization_scale=normalization_scale,
        floor_offset_y=floor_offset_y,
    )
    _write_rig_usda(
        rig_usda_path=rig_usda_path,
        glb_path=glb_path,
        rig_data=rig_data,
        normalization_scale=normalization_scale,
    )
    _write_skinned_character_usda(
        skinned_character_usda_path=skinned_character_usda_path,
        rig_usda_name=rig_usda_path.name,
        glb_path=glb_path,
        motion_path=motion_path,
        rig_data=rig_data,
        points=posed_points,
        texcoords=mesh["texcoords"],
        joint_indices=mesh["joint_indices"],
        joint_weights=mesh["joint_weights"],
        face_indices=mesh["indices"],
        texture_name=final_texture_path.name,
        extent_min=bounds_min,
        extent_max=bounds_max,
    )

    walk_motion = motion_data["walk_motion"]
    walk_joint_quats = np.asarray(walk_motion["joints_array"], dtype=np.float32)
    walk_root_transforms = np.asarray(walk_motion["transform_array"], dtype=np.float32)
    walk_displacement = np.asarray(walk_motion["displacement"], dtype=np.float32)
    walk_fps = float(walk_motion["fps"])
    walk_root_motion_preview = _build_relative_walk_root_motion_preview(
        walk_root_transforms=walk_root_transforms,
        normalization_scale=normalization_scale,
    )
    walk_skel_animation_data = _build_walk_skel_animation_data(
        nodes=gltf_doc["nodes"],
        rig_data=rig_data,
        walk_joint_quats=walk_joint_quats,
        motion_joint_order=chosen_joint_order,
        normalization_scale=normalization_scale,
        floor_offset_y=floor_offset_y,
    )
    stop_skel_pose_data = _build_stop_skel_pose_data(rig_data=rig_data)
    np.savez_compressed(
        walk_motion_path,
        walk_joint_quats=walk_joint_quats,
        walk_root_transforms=walk_root_transforms,
        walk_root_motion_preview=walk_root_motion_preview,
        walk_displacement=walk_displacement,
        joint_order=np.asarray(chosen_joint_order),
        skel_joint_paths=np.asarray(walk_skel_animation_data["joints"]),
        skel_translations=walk_skel_animation_data["translations"],
        skel_scales=walk_skel_animation_data["scales"],
        skel_rotations_default=walk_skel_animation_data["rotations"],
        skel_rotations_time_samples=walk_skel_animation_data["rotation_time_samples"],
        stop_translations=stop_skel_pose_data["translations"],
        stop_scales=stop_skel_pose_data["scales"],
        stop_rotations=stop_skel_pose_data["rotations"],
        fps=np.asarray([walk_fps], dtype=np.float32),
    )
    _write_character_entry_usda(
        character_usda_path=character_usda_path,
        stop_pose_usda_name=usda_path.name,
        skinned_character_usda_name=skinned_character_usda_path.name,
        rig_usda_name=rig_usda_path.name,
        glb_path=glb_path,
        motion_path=motion_path,
        walk_motion_metadata_name=walk_motion_path.name,
        walk_clip_usda_name=walk_clip_usda_path.name,
        posed_height=posed_height,
        rest_height=rest_height,
    )
    _write_walk_clip_usda(
        walk_clip_usda_path=walk_clip_usda_path,
        skinned_character_usda_name=skinned_character_usda_path.name,
        walk_motion_metadata_name=walk_motion_path.name,
        walk_root_motion_preview=walk_root_motion_preview,
        walk_skel_animation_data=walk_skel_animation_data,
        walk_frame_count=int(walk_joint_quats.shape[0]),
        walk_fps=walk_fps,
    )

    metadata_path.write_text(
        json.dumps(
            {
                "package_layout_version": _HABITAT_CHARACTER_PACKAGE_LAYOUT_VERSION,
                "bridge_version": _HABITAT_HUMANOID_BRIDGE_VERSION,
                "cache_key": cache_key,
                "cache_dir": str(cache_dir),
                "source_glb": str(glb_path),
                "source_urdf": str(urdf_path) if urdf_path.exists() else None,
                "motion_data_path": str(motion_path),
                "generated_usda": str(character_usda_path),
                "package_files": {
                    "entry_usda": str(character_usda_path),
                    "stop_pose_usda": str(usda_path),
                    "skinned_character_usda": str(skinned_character_usda_path),
                    "rig_usda": str(rig_usda_path),
                    "walk_clip_usda": str(walk_clip_usda_path),
                    "walk_motion_metadata": str(walk_motion_path),
                    "texture_path": str(final_texture_path),
                },
                "texture_path": str(final_texture_path),
                "joint_order": chosen_joint_order,
                "joint_order_source": chosen_joint_order_source,
                "posed_height_m": posed_height,
                "rest_height_m": rest_height,
                "normalization_scale": normalization_scale,
                "mesh_bounds_min": bounds_min.astype(float).tolist(),
                "mesh_bounds_max": bounds_max.astype(float).tolist(),
                "authored_up_axis": _HABITAT_PACKAGE_AUTHORED_UP_AXIS,
                "target_stage_up_axis": _HABITAT_TARGET_STAGE_UP_AXIS,
                "runtime_visual_correction_quat_xyzw": _HABITAT_RUNTIME_VISUAL_CORRECTION_QUAT_XYZW.astype(float).tolist(),
                "skinned_character_authored_in_usd": True,
                "skinned_character_binds_to_rig": True,
                "skinned_character_geom_bind_transform": "identity",
                "skinned_character_bind_pose": "stop_pose",
                "rig_authored_in_usd": True,
                "rig_joint_count": int(len(rig_data["joint_paths"])),
                "rig_root_joint": rig_data["joint_paths"][0] if rig_data["joint_paths"] else None,
                "rig_joint_path_source": "glb_skin_hierarchy",
                "rig_rest_pose_source": "stop_pose_local_transforms",
                "rig_bind_pose_source": "stop_pose_world_transforms",
                "walk_frame_count": int(walk_joint_quats.shape[0]),
                "walk_fps": walk_fps,
                "walk_clip_authored_in_usd": True,
                "walk_clip_contains_skel_animation": True,
                "walk_clip_contains_root_motion": True,
                "walk_root_motion_source": "transform_array",
                "walk_root_motion_relative_to_first_frame": True,
                "walk_root_motion_preview_path_length_m": _estimate_path_length_m(walk_root_motion_preview),
                "walk_clip_skeleton_path": "/Character/Rig/Skel",
                "walk_clip_animation_prim": "/Character/Rig/Skel/WalkAnimation",
                "walk_clip_base_character_usda": str(skinned_character_usda_path),
                "walk_clip_animation_joint_count": int(len(walk_skel_animation_data["joints"])),
                "walk_clip_runtime_data_in_npz": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[habitat_bridge] Generated package for {glb_path.name} -> {character_usda_path}",
        flush=True,
    )

    return str(character_usda_path)


def _make_cache_key(glb_path: Path, motion_path: Path) -> str:
    payload = {
        "bridge_version": _HABITAT_HUMANOID_BRIDGE_VERSION,
        "glb_path": str(glb_path),
        "glb_mtime_ns": glb_path.stat().st_mtime_ns,
        "glb_size": glb_path.stat().st_size,
        "motion_path": str(motion_path),
        "motion_mtime_ns": motion_path.stat().st_mtime_ns,
        "motion_size": motion_path.stat().st_size,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _load_motion_data(motion_path: Path) -> Dict[str, Any]:
    with motion_path.open("rb") as f:
        motion_data = pickle.load(f)
    if not isinstance(motion_data, dict):
        raise RuntimeError(f"Unexpected motion data payload in {motion_path}")
    return motion_data


def _load_urdf_joint_order(urdf_path: Path) -> List[str]:
    root = ET.parse(urdf_path).getroot()
    return [joint.attrib["name"] for joint in root.findall("joint") if "name" in joint.attrib]


def _load_glb(glb_path: Path) -> Tuple[Dict[str, Any], bytes]:
    raw = glb_path.read_bytes()
    magic, version, total_length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2 or total_length != len(raw):
        raise RuntimeError(f"Invalid GLB file: {glb_path}")

    offset = 12
    json_chunk = None
    bin_chunk = None
    while offset < len(raw):
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            json_chunk = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk
    if json_chunk is None or bin_chunk is None:
        raise RuntimeError(f"Incomplete GLB file: {glb_path}")
    return json_chunk, bin_chunk


def _find_primary_mesh_primitive(gltf_doc: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    scene_index = int(gltf_doc.get("scene", 0))
    scene_nodes = list(gltf_doc.get("scenes", [{}])[scene_index].get("nodes", []))
    queue = list(scene_nodes)
    visited = set()
    while queue:
        node_index = queue.pop(0)
        if node_index in visited:
            continue
        visited.add(node_index)
        node = gltf_doc["nodes"][node_index]
        if "mesh" in node:
            mesh = gltf_doc["meshes"][node["mesh"]]
            if len(mesh.get("primitives", [])) != 1:
                raise RuntimeError("Only single-primitive Habitat humanoid meshes are supported.")
            if "skin" not in node:
                raise RuntimeError("Skinned mesh node is missing a skin reference.")
            return mesh["primitives"][0], node_index, int(node["skin"])
        queue.extend(node.get("children", []))
    raise RuntimeError("Could not locate skinned mesh node in GLB scene.")


def _extract_mesh_data(
    gltf_doc: Dict[str, Any],
    bin_blob: bytes,
    primitive: Dict[str, Any],
    skin_index: int,
) -> Dict[str, Any]:
    attrs = primitive["attributes"]
    positions = _read_accessor(gltf_doc, bin_blob, attrs["POSITION"]).astype(np.float32)
    texcoords = _read_accessor(gltf_doc, bin_blob, attrs["TEXCOORD_0"]).astype(np.float32)
    joint_indices = _read_accessor(gltf_doc, bin_blob, attrs["JOINTS_0"]).astype(np.int32)
    joint_weights = _read_accessor(gltf_doc, bin_blob, attrs["WEIGHTS_0"]).astype(np.float32)
    indices = _read_accessor(gltf_doc, bin_blob, primitive["indices"]).astype(np.int32).reshape(-1)

    joint_weight_sums = np.sum(joint_weights, axis=1, keepdims=True)
    joint_weight_sums[joint_weight_sums <= 1e-8] = 1.0
    joint_weights = joint_weights / joint_weight_sums

    return {
        "positions": positions.reshape(-1, 3),
        "texcoords": texcoords.reshape(-1, 2),
        "joint_indices": joint_indices.reshape(-1, 4),
        "joint_weights": joint_weights.reshape(-1, 4),
        "indices": indices,
        "material_index": int(primitive.get("material", 0)),
        "skin_index": int(skin_index),
    }


def _extract_skin_data(
    gltf_doc: Dict[str, Any],
    bin_blob: bytes,
    skin_index: int,
) -> Dict[str, Any]:
    skin = gltf_doc["skins"][skin_index]
    joint_nodes = np.asarray(skin["joints"], dtype=np.int32)
    inverse_bind = _read_accessor(gltf_doc, bin_blob, skin["inverseBindMatrices"]).astype(np.float32)
    inverse_bind = inverse_bind.reshape(-1, 4, 4).transpose(0, 2, 1)
    return {
        "joint_nodes": joint_nodes,
        "inverse_bind_mats": inverse_bind,
    }


def _pose_points_for_joint_order(
    nodes: List[Dict[str, Any]],
    mesh: Dict[str, Any],
    mesh_node_index: int,
    skin_data: Dict[str, Any],
    motion_joint_quats: np.ndarray,
    motion_joint_order: List[str],
) -> np.ndarray:
    local_mats = _build_local_transforms(
        nodes,
        motion_joint_quats=motion_joint_quats,
        motion_joint_order=motion_joint_order,
    )
    world_mats = _build_world_transforms(nodes, local_mats)

    mesh_world = world_mats[mesh_node_index]
    skinned_points = _apply_linear_blend_skinning(
        points=mesh["positions"],
        joint_indices=mesh["joint_indices"],
        joint_weights=mesh["joint_weights"],
        mesh_world=mesh_world,
        joint_world_mats=np.stack([world_mats[j] for j in skin_data["joint_nodes"]], axis=0),
        inverse_bind_mats=skin_data["inverse_bind_mats"],
    )
    return _convert_skinning_output_to_y_up(skinned_points)


def _convert_skinning_output_to_y_up(points: np.ndarray) -> np.ndarray:
    return points.astype(np.float32) @ _SKINNING_ZUP_TO_YUP.T


def _score_humanoid_extent(points: np.ndarray) -> float:
    extent = points.max(axis=0) - points.min(axis=0)
    height = max(float(extent[1]), 1e-6)
    width_ratio = float(extent[0]) / height
    depth_ratio = float(extent[2]) / height
    return abs(width_ratio - 0.38) + abs(depth_ratio - 0.22)


def _extract_base_color_texture(
    gltf_doc: Dict[str, Any],
    bin_blob: bytes,
    primitive: Dict[str, Any],
    glb_path: Path,
) -> Tuple[str, bytes]:
    material_index = int(primitive.get("material", 0))
    material = gltf_doc["materials"][material_index]
    pbr = material.get("pbrMetallicRoughness", {})
    texture_index = pbr.get("baseColorTexture", {}).get("index")
    if texture_index is None:
        raise RuntimeError("Habitat humanoid material is missing a baseColorTexture.")
    texture = gltf_doc["textures"][texture_index]
    image = gltf_doc["images"][texture["source"]]
    if "bufferView" in image:
        buffer_view = gltf_doc["bufferViews"][image["bufferView"]]
        start = int(buffer_view.get("byteOffset", 0))
        end = start + int(buffer_view["byteLength"])
        mime_type = image.get("mimeType", "image/jpeg")
        return _IMAGE_SUFFIXES.get(mime_type, ".bin"), bin_blob[start:end]
    if "uri" in image:
        image_path = (glb_path.parent / image["uri"]).resolve()
        return image_path.suffix.lower(), image_path.read_bytes()
    raise RuntimeError("Unsupported GLB image payload.")


def _read_accessor(gltf_doc: Dict[str, Any], bin_blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = gltf_doc["accessors"][accessor_index]
    if "sparse" in accessor:
        raise RuntimeError("Sparse GLB accessors are not supported.")

    buffer_view = gltf_doc["bufferViews"][accessor["bufferView"]]
    count = int(accessor["count"])
    num_components = _NUM_COMPONENTS[accessor["type"]]
    dtype = np.dtype(_COMPONENT_DTYPES[accessor["componentType"]])
    component_size = dtype.itemsize
    item_size = num_components * component_size
    byte_offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    byte_stride = int(buffer_view.get("byteStride", item_size))

    if byte_stride == item_size:
        array = np.frombuffer(
            bin_blob,
            dtype=dtype,
            count=count * num_components,
            offset=byte_offset,
        )
        return array.reshape(count, num_components)

    raw = memoryview(bin_blob)
    out = np.empty((count, num_components), dtype=dtype)
    for i in range(count):
        start = byte_offset + i * byte_stride
        end = start + item_size
        out[i] = np.frombuffer(raw[start:end], dtype=dtype, count=num_components)
    return out


def _build_local_transforms(
    nodes: List[Dict[str, Any]],
    motion_joint_quats: np.ndarray,
    motion_joint_order: List[str],
) -> List[np.ndarray]:
    motion_map = {name: motion_joint_quats[idx] for idx, name in enumerate(motion_joint_order)}
    local_mats: List[np.ndarray] = []
    for node in nodes:
        node_name = node.get("name")
        rotation = motion_map.get(node_name, node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
        local_mats.append(
            _compose_trs_matrix(
                translation=node.get("translation", [0.0, 0.0, 0.0]),
                rotation=rotation,
                scale=node.get("scale", [1.0, 1.0, 1.0]),
                matrix=node.get("matrix"),
            )
        )
    return local_mats


def _compose_trs_matrix(
    translation: Any,
    rotation: Any,
    scale: Any,
    matrix: Optional[List[float]] = None,
) -> np.ndarray:
    if matrix is not None:
        return np.asarray(matrix, dtype=np.float32).reshape(4, 4).T

    t = np.asarray(translation, dtype=np.float32)
    q = np.asarray(rotation, dtype=np.float32)
    s = np.asarray(scale, dtype=np.float32)

    x, y, z, w = q.tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    rot = rot @ np.diag(s.astype(np.float32))

    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    mat[:3, 3] = t
    return mat


def _build_parent_map(nodes: List[Dict[str, Any]]) -> Dict[int, int]:
    parent_map: Dict[int, int] = {}
    for parent_idx, node in enumerate(nodes):
        for child_idx in node.get("children", []):
            parent_map[int(child_idx)] = parent_idx
    return parent_map


def _build_world_transforms(nodes: List[Dict[str, Any]], local_mats: List[np.ndarray]) -> List[np.ndarray]:
    parent_map = _build_parent_map(nodes)
    cache: Dict[int, np.ndarray] = {}

    def compute_world(node_index: int) -> np.ndarray:
        if node_index in cache:
            return cache[node_index]
        parent_idx = parent_map.get(node_index)
        parent_world = np.eye(4, dtype=np.float32) if parent_idx is None else compute_world(parent_idx)
        cache[node_index] = parent_world @ local_mats[node_index]
        return cache[node_index]

    return [compute_world(i) for i in range(len(nodes))]


def _apply_linear_blend_skinning(
    points: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    mesh_world: np.ndarray,
    joint_world_mats: np.ndarray,
    inverse_bind_mats: np.ndarray,
) -> np.ndarray:
    joint_mats = joint_world_mats @ inverse_bind_mats
    mesh_world_inv = np.linalg.inv(mesh_world).astype(np.float32)
    joint_mats = mesh_world_inv[None, ...] @ joint_mats

    hom_points = np.concatenate(
        [points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    posed = np.zeros((points.shape[0], 3), dtype=np.float32)
    for influence_idx in range(4):
        ids = joint_indices[:, influence_idx]
        weights = joint_weights[:, influence_idx : influence_idx + 1]
        transforms = joint_mats[ids]
        transformed = np.einsum("nij,nj->ni", transforms, hom_points, optimize=True)
        posed += transformed[:, :3] * weights
    return posed


def _format_vec_list(values: np.ndarray, precision: int = 6) -> str:
    fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join("(" + ", ".join(fmt.format(float(v)) for v in row) + ")" for row in values) + "]"


def _format_int_list(values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    return "[" + ", ".join(str(int(v)) for v in flat) + "]"


def _format_float_list(values: np.ndarray, precision: int = 6) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    fmt = f"{{:.{precision}f}}"
    return "[" + ", ".join(fmt.format(float(v)) for v in flat) + "]"


def _format_quatf_list(values_xyzw: np.ndarray, precision: int = 6) -> str:
    values = np.asarray(values_xyzw, dtype=np.float32).reshape(-1, 4)
    fmt = f"{{:.{precision}f}}"
    items = []
    for x, y, z, w in values:
        items.append(
            "("
            + ", ".join(
                [
                    fmt.format(float(w)),
                    fmt.format(float(x)),
                    fmt.format(float(y)),
                    fmt.format(float(z)),
                ]
            )
            + ")"
        )
    return "[" + ", ".join(items) + "]"


def _format_token_list(values: List[str]) -> str:
    return "[" + ", ".join(json.dumps(str(v)) for v in values) + "]"


def _format_matrix4d(matrix: np.ndarray, precision: int = 6) -> str:
    # USD matrix4d uses row-vector convention (translation in row 3),
    # our NumPy matrices use column-vector convention (translation in column 3).
    # Transpose so USD reads the translation correctly.
    mat = np.asarray(matrix, dtype=np.float32).reshape(4, 4).T
    fmt = f"{{:.{precision}f}}"
    rows = []
    for row in mat:
        rows.append("(" + ", ".join(fmt.format(float(v)) for v in row) + ")")
    return "(" + ", ".join(rows) + ")"


def _matrix3_to_quat_xyzw(matrix3: np.ndarray) -> np.ndarray:
    rot = np.asarray(matrix3, dtype=np.float32).reshape(3, 3)
    u, _, vh = np.linalg.svd(rot.astype(np.float64), full_matrices=False)
    ortho = (u @ vh).astype(np.float32)
    if np.linalg.det(ortho.astype(np.float64)) < 0.0:
        u[:, -1] *= -1.0
        ortho = (u @ vh).astype(np.float32)

    m00, m01, m02 = ortho[0]
    m10, m11, m12 = ortho[1]
    m20, m21, m22 = ortho[2]
    trace = float(m00 + m11 + m22)

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(max(1.0 + m00 - m11 - m22, 1e-12))
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(max(1.0 + m11 - m00 - m22, 1e-12))
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + m22 - m00 - m11, 1e-12))
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    quat_xyzw = np.asarray([x, y, z, w], dtype=np.float32)
    norm = float(np.linalg.norm(quat_xyzw))
    if norm <= 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat_xyzw /= np.float32(norm)
    if quat_xyzw[3] < 0.0:
        quat_xyzw *= -1.0
    return quat_xyzw


def _decompose_matrix_to_trs(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    translation = mat[:3, 3].astype(np.float32)
    basis = mat[:3, :3].astype(np.float32)
    scale = np.linalg.norm(basis, axis=0).astype(np.float32)

    safe_scale = scale.copy()
    safe_scale[safe_scale <= 1e-8] = 1.0
    rotation_matrix = (basis / safe_scale[np.newaxis, :]).astype(np.float32)
    rotation_quat_xyzw = _matrix3_to_quat_xyzw(rotation_matrix)
    return translation, rotation_quat_xyzw, scale


def _estimate_path_length_m(root_transforms: np.ndarray) -> float:
    transforms = np.asarray(root_transforms, dtype=np.float32)
    if transforms.shape[0] <= 1:
        return 0.0
    positions = transforms[:, :3, 3]
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _convert_transform_matrix_to_y_up(
    matrix: np.ndarray,
    translation_scale: float = 1.0,
    floor_offset_y: float = 0.0,
) -> np.ndarray:
    converted = (
        _SKINNING_ZUP_TO_YUP_4X4
        @ np.asarray(matrix, dtype=np.float32)
        @ _SKINNING_ZUP_TO_YUP_4X4.T
    ).astype(np.float32)
    converted[:3, 3] *= np.float32(translation_scale)
    converted[1, 3] -= np.float32(floor_offset_y)
    converted[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return converted


def _rescale_transform_matrix(
    matrix: np.ndarray,
    translation_scale: float = 1.0,
    floor_offset_y: float = 0.0,
) -> np.ndarray:
    """
    glTF joint local / world transforms are already authored in a Y-up basis.

    The skinned mesh points we export are also kept in that same basis, so the
    skeleton and animation data must only be rescaled / floor-aligned here.
    Applying an additional Z-up -> Y-up basis conversion to the joint matrices
    rotates the whole rig onto its back.
    """

    converted = np.asarray(matrix, dtype=np.float32).copy()
    converted[:3, 3] *= np.float32(translation_scale)
    converted[1, 3] -= np.float32(floor_offset_y)
    converted[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return converted


def _build_rig_data(
    nodes: List[Dict[str, Any]],
    skin_data: Dict[str, Any],
    posed_local_mats: List[np.ndarray],
    posed_world_mats: List[np.ndarray],
    normalization_scale: float,
    floor_offset_y: float,
) -> Dict[str, Any]:
    joint_nodes = [int(node_index) for node_index in np.asarray(skin_data["joint_nodes"], dtype=np.int32).tolist()]
    joint_node_set = set(joint_nodes)
    parent_map = _build_parent_map(nodes)

    def node_name(node_index: int) -> str:
        return str(nodes[node_index].get("name") or f"joint_{node_index}")

    path_cache: Dict[int, str] = {}

    def joint_path(node_index: int) -> str:
        if node_index in path_cache:
            return path_cache[node_index]
        parent_index = parent_map.get(node_index)
        current_name = node_name(node_index)
        if parent_index in joint_node_set:
            path_cache[node_index] = f"{joint_path(parent_index)}/{current_name}"
        else:
            path_cache[node_index] = current_name
        return path_cache[node_index]

    joint_names: List[str] = []
    joint_paths: List[str] = []
    rest_transforms: List[np.ndarray] = []
    bind_transforms: List[np.ndarray] = []

    for joint_node in joint_nodes:
        joint_names.append(node_name(joint_node))
        joint_paths.append(joint_path(joint_node))
        parent_index = parent_map.get(joint_node)
        is_root_joint = parent_index not in joint_node_set
        local_floor_offset = floor_offset_y if is_root_joint else 0.0
        rest_source_matrix = posed_world_mats[joint_node] if is_root_joint else posed_local_mats[joint_node]
        rest_transforms.append(
            _rescale_transform_matrix(
                rest_source_matrix,
                translation_scale=normalization_scale,
                floor_offset_y=local_floor_offset,
            )
        )
        bind_transforms.append(
            _rescale_transform_matrix(
                posed_world_mats[joint_node],
                translation_scale=normalization_scale,
                floor_offset_y=floor_offset_y,
            )
        )

    return {
        "joint_node_indices": joint_nodes,
        "joint_names": joint_names,
        "joint_paths": joint_paths,
        "rest_transforms": np.asarray(rest_transforms, dtype=np.float32),
        "bind_transforms": np.asarray(bind_transforms, dtype=np.float32),
    }


def _build_relative_walk_root_motion_preview(
    walk_root_transforms: np.ndarray,
    normalization_scale: float,
) -> np.ndarray:
    transforms = np.asarray(walk_root_transforms, dtype=np.float32)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise RuntimeError(f"Unexpected walk_root_transforms shape: {transforms.shape}")

    base_inv = np.linalg.inv(transforms[0]).astype(np.float32)
    relative = np.einsum("nij,jk->nik", transforms, base_inv, optimize=True).astype(np.float32)

    # Match the static stop-pose asset scale and keep the preview on the floor.
    relative[:, :3, 3] *= np.float32(normalization_scale)
    relative[:, 1, 3] = np.maximum(relative[:, 1, 3], 0.0)
    relative[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return relative


def _build_walk_skel_animation_data(
    nodes: List[Dict[str, Any]],
    rig_data: Dict[str, Any],
    walk_joint_quats: np.ndarray,
    motion_joint_order: List[str],
    normalization_scale: float,
    floor_offset_y: float,
) -> Dict[str, Any]:
    walk_joint_quats = np.asarray(walk_joint_quats, dtype=np.float32)
    if walk_joint_quats.ndim != 3 or walk_joint_quats.shape[2] != 4:
        raise RuntimeError(f"Unexpected walk_joint_quats shape: {walk_joint_quats.shape}")

    joint_nodes = [int(node_index) for node_index in rig_data["joint_node_indices"]]
    joint_node_set = set(joint_nodes)
    parent_map = _build_parent_map(nodes)

    rotation_frames: List[np.ndarray] = []
    translation_defaults: Optional[np.ndarray] = None
    scale_defaults: Optional[np.ndarray] = None

    for frame_quats in walk_joint_quats:
        local_mats = _build_local_transforms(
            nodes=nodes,
            motion_joint_quats=frame_quats,
            motion_joint_order=motion_joint_order,
        )
        world_mats = _build_world_transforms(nodes, local_mats)
        frame_rotations: List[np.ndarray] = []
        frame_translations: List[np.ndarray] = []
        frame_scales: List[np.ndarray] = []

        for joint_node in joint_nodes:
            parent_index = parent_map.get(joint_node)
            is_root_joint = parent_index not in joint_node_set
            local_floor_offset = floor_offset_y if is_root_joint else 0.0
            source_matrix = world_mats[joint_node] if is_root_joint else local_mats[joint_node]
            converted_local = _rescale_transform_matrix(
                source_matrix,
                translation_scale=normalization_scale,
                floor_offset_y=local_floor_offset,
            )
            translation, rotation_quat_xyzw, scale = _decompose_matrix_to_trs(converted_local)
            frame_translations.append(translation)
            frame_rotations.append(rotation_quat_xyzw)
            frame_scales.append(scale)

        frame_rotations_array = np.asarray(frame_rotations, dtype=np.float32)
        rotation_frames.append(frame_rotations_array)
        if translation_defaults is None:
            translation_defaults = np.asarray(frame_translations, dtype=np.float32)
        if scale_defaults is None:
            scale_defaults = np.asarray(frame_scales, dtype=np.float32)

    if translation_defaults is None or scale_defaults is None:
        raise RuntimeError("Failed to build walk skeletal animation defaults.")

    return {
        "joints": list(rig_data["joint_paths"]),
        "translations": translation_defaults,
        "rotations": rotation_frames[0],
        "scales": scale_defaults,
        "rotation_time_samples": np.asarray(rotation_frames, dtype=np.float32),
    }


def _build_stop_skel_pose_data(rig_data: Dict[str, Any]) -> Dict[str, Any]:
    stop_translations: List[np.ndarray] = []
    stop_rotations: List[np.ndarray] = []
    stop_scales: List[np.ndarray] = []
    for local_matrix in np.asarray(rig_data["rest_transforms"], dtype=np.float32):
        translation, rotation_quat_xyzw, scale = _decompose_matrix_to_trs(local_matrix)
        stop_translations.append(translation)
        stop_rotations.append(rotation_quat_xyzw)
        stop_scales.append(scale)
    return {
        "translations": np.asarray(stop_translations, dtype=np.float32),
        "rotations": np.asarray(stop_rotations, dtype=np.float32),
        "scales": np.asarray(stop_scales, dtype=np.float32),
    }


def _write_character_entry_usda(
    character_usda_path: Path,
    stop_pose_usda_name: str,
    skinned_character_usda_name: str,
    rig_usda_name: str,
    glb_path: Path,
    motion_path: Path,
    walk_motion_metadata_name: str,
    walk_clip_usda_name: str,
    posed_height: float,
    rest_height: float,
) -> None:
    contents = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Y"',
        "    subLayers = [",
        f"        @./{stop_pose_usda_name}@,",
        f"        @./{rig_usda_name}@,",
        "    ]",
        ")",
        "",
        'over "World" (',
        '    kind = "component"',
        "    assetInfo = {",
        f'        string name = "{glb_path.stem}"',
        f'        string identifier = "{character_usda_path.name}"',
        "    }",
        "    customData = {",
        f'        string bridge_step = "step1_static_character_package"',
        f'        string source_glb = "{str(glb_path)}"',
        f'        string motion_data_path = "{str(motion_path)}"',
        f'        string stop_pose_usda = "{stop_pose_usda_name}"',
        f'        string skinned_character_usda = "{skinned_character_usda_name}"',
        f'        string rig_usda = "{rig_usda_name}"',
        f'        string walk_motion_metadata = "{walk_motion_metadata_name}"',
        f'        string walk_clip_usda = "{walk_clip_usda_name}"',
        f"        double posed_height_m = {float(posed_height):.6f}",
        f"        double rest_height_m = {float(rest_height):.6f}",
        "    }",
        ")",
        "{",
        "}",
        "",
    ]
    character_usda_path.write_text("\n".join(contents), encoding="utf-8")


def _omnipbr_material_lines(root_prim: str, texture_name: str, texture_absolute_path: Optional[str] = None) -> list:
    """Generate USDA lines for an OmniPBR (MDL) material with an albedo texture.

    OmniGibson / Isaac Sim's RTX renderer uses OmniPBR natively.
    UsdPreviewSurface textures are not reliably rendered (results in default brown).

    Uses the **Material interface pattern** required by Kit / RTX:
      - Texture value lives on the Material prim as ``inputs:diffuse_texture``.
      - The Shader prim connects to the Material input via ``.connect``.
    Setting the texture directly on the Shader (without the interface) causes
    Isaac Sim to show a pink fallback colour because the MDL resolver cannot
    find the texture.

    When *texture_absolute_path* is provided it is used instead of the relative
    *texture_name*.  This avoids MDL resolver failures when the material lives
    inside a sublayered file that is later referenced into the main stage.
    """
    tex_ref = texture_absolute_path if texture_absolute_path else texture_name
    mat_path = f"/{root_prim}/Looks/HumanMaterial"
    return [
        '        def Material "HumanMaterial"',
        "        {",
        f'            token outputs:mdl:surface.connect = <{mat_path}/Shader.outputs:out>',
        "",
        f'            asset inputs:diffuse_texture = @{tex_ref}@ (',
        '                colorSpace = "auto"',
        "            )",
        "            float inputs:metallic_constant = 0.0",
        "            float inputs:reflection_roughness_constant = 0.6",
        "",
        '            def Shader "Shader"',
        "            {",
        '                uniform token info:implementationSource = "sourceAsset"',
        '                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@',
        '                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"',
        f'                asset inputs:diffuse_texture.connect = <{mat_path}.inputs:diffuse_texture>',
        f'                float inputs:metallic_constant.connect = <{mat_path}.inputs:metallic_constant>',
        f'                float inputs:reflection_roughness_constant.connect = <{mat_path}.inputs:reflection_roughness_constant>',
        "                token outputs:out",
        "            }",
        "        }",
    ]


def _write_skinned_character_usda(
    skinned_character_usda_path: Path,
    rig_usda_name: str,
    glb_path: Path,
    motion_path: Path,
    rig_data: Dict[str, Any],
    points: np.ndarray,
    texcoords: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    face_indices: np.ndarray,
    texture_name: str,
    extent_min: np.ndarray,
    extent_max: np.ndarray,
) -> None:
    face_vertex_counts = np.full((face_indices.shape[0] // 3,), 3, dtype=np.int32)
    joint_index_values = np.asarray(joint_indices, dtype=np.int32).reshape(-1)
    joint_weight_values = np.asarray(joint_weights, dtype=np.float32).reshape(-1)

    texture_abs = str((skinned_character_usda_path.parent / texture_name).resolve())
    material_lines = _omnipbr_material_lines("Character", texture_name, texture_absolute_path=texture_abs)

    contents = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Character"',
        "    metersPerUnit = 1",
        '    upAxis = "Y"',
        ")",
        "",
        'def SkelRoot "Character" (',
        '    prepend apiSchemas = ["SkelBindingAPI"]',
        '    kind = "component"',
        "    assetInfo = {",
        f'        string name = "{glb_path.stem}_skinned"',
        f'        string identifier = "{skinned_character_usda_path.name}"',
        "    }",
        "    customData = {",
        '        string bridge_step = "step1_skinned_character_package"',
        f'        string source_glb = "{str(glb_path)}"',
        f'        string motion_data_path = "{str(motion_path)}"',
        f'        string rig_usda = "{rig_usda_name}"',
        '        string bind_pose = "stop_pose"',
        f"        int joint_count = {int(len(rig_data['joint_paths']))}",
        "        bool contains_skinning_binding = true",
        "    }",
        ")",
        "{",
        f"    float3[] extent = [{_format_vec_list(np.asarray([extent_min, extent_max], dtype=np.float32))[1:-1]}]",
        f"    rel skel:skeleton = </Character/Rig/Skel>",
        "    matrix4d xformOp:transform = ((1.000000, 0.000000, 0.000000, 0.000000), (0.000000, 1.000000, 0.000000, 0.000000), (0.000000, 0.000000, 1.000000, 0.000000), (0.000000, 0.000000, 0.000000, 1.000000))",
        '    uniform token[] xformOpOrder = ["xformOp:transform"]',
        "",
        '    def Scope "Rig" (',
        f"        prepend references = @./{rig_usda_name}@",
        "    )",
        "    {",
        "    }",
        "",
        '    def Scope "Looks"',
        "    {",
        *material_lines,
        "    }",
        "",
        '    def Mesh "Human" (',
        '        prepend apiSchemas = ["MaterialBindingAPI", "SkelBindingAPI"]',
        "    )",
        "    {",
        "        uniform bool doubleSided = 1",
        f"        float3[] extent = [{_format_vec_list(np.asarray([extent_min, extent_max], dtype=np.float32))[1:-1]}]",
        f"        int[] faceVertexCounts = {_format_int_list(face_vertex_counts)}",
        f"        int[] faceVertexIndices = {_format_int_list(face_indices)}",
        f"        point3f[] points = {_format_vec_list(points)}",
        "        matrix4d primvars:skel:geomBindTransform = ((1.000000, 0.000000, 0.000000, 0.000000), (0.000000, 1.000000, 0.000000, 0.000000), (0.000000, 0.000000, 1.000000, 0.000000), (0.000000, 0.000000, 0.000000, 1.000000))",
        f"        int[] primvars:skel:jointIndices = {_format_int_list(joint_index_values)} (",
        "            elementSize = 4",
        '            interpolation = "vertex"',
        "        )",
        f"        float[] primvars:skel:jointWeights = {_format_float_list(joint_weight_values)} (",
        "            elementSize = 4",
        '            interpolation = "vertex"',
        "        )",
        f"        texCoord2f[] primvars:st = {_format_vec_list(texcoords)}",
        '        uniform token primvars:st:interpolation = "vertex"',
        '        rel material:binding = </Character/Looks/HumanMaterial>',
        '        rel skel:skeleton = </Character/Rig/Skel>',
        '        uniform token subdivisionScheme = "none"',
        "    }",
        "}",
        "",
    ]
    skinned_character_usda_path.write_text("\n".join(contents), encoding="utf-8")


def _write_rig_usda(
    rig_usda_path: Path,
    glb_path: Path,
    rig_data: Dict[str, Any],
    normalization_scale: float,
) -> None:
    bind_transform_values = np.asarray(rig_data["bind_transforms"], dtype=np.float32)
    rest_transform_values = np.asarray(rig_data["rest_transforms"], dtype=np.float32)
    bind_transform_lines = [
        f"            {_format_matrix4d(matrix)}{',' if idx < len(bind_transform_values) - 1 else ''}"
        for idx, matrix in enumerate(bind_transform_values)
    ]
    rest_transform_lines = [
        f"            {_format_matrix4d(matrix)}{',' if idx < len(rest_transform_values) - 1 else ''}"
        for idx, matrix in enumerate(rest_transform_values)
    ]
    contents = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Rig"',
        "    metersPerUnit = 1",
        '    upAxis = "Y"',
        ")",
        "",
        'def Scope "Rig"',
        "{",
        '    def Skeleton "Skel" (',
        '        prepend apiSchemas = ["SkelBindingAPI"]',
        "        customData = {",
        '            string bridge_step = "step1_character_package_rig_layer"',
        f'            string source_glb = "{str(glb_path)}"',
        f"            int joint_count = {int(len(rig_data['joint_paths']))}",
        f"            double normalization_scale = {float(normalization_scale):.6f}",
        '            string bind_pose_source = "stop_pose_world_transforms"',
        '            string rest_pose_source = "stop_pose_local_transforms"',
        "        }",
        "    )",
        "    {",
        f"        uniform token[] jointNames = {_format_token_list(rig_data['joint_names'])}",
        f"        uniform token[] joints = {_format_token_list(rig_data['joint_paths'])}",
        "        uniform matrix4d[] bindTransforms = [",
        *bind_transform_lines,
        "        ]",
        "        uniform matrix4d[] restTransforms = [",
        *rest_transform_lines,
        "        ]",
        "    }",
        "}",
        "",
    ]
    rig_usda_path.write_text("\n".join(contents), encoding="utf-8")


def _write_walk_clip_usda(
    walk_clip_usda_path: Path,
    skinned_character_usda_name: str,
    walk_motion_metadata_name: str,
    walk_root_motion_preview: np.ndarray,
    walk_skel_animation_data: Dict[str, Any],
    walk_frame_count: int,
    walk_fps: float,
) -> None:
    end_time = max(int(walk_frame_count) - 1, 0)
    root_motion_values = np.asarray(walk_root_motion_preview, dtype=np.float32)
    rotation_values = np.asarray(walk_skel_animation_data["rotation_time_samples"], dtype=np.float32)
    translation_values = np.asarray(walk_skel_animation_data["translations"], dtype=np.float32)
    default_rotation_values = np.asarray(walk_skel_animation_data["rotations"], dtype=np.float32)
    scale_values = np.asarray(walk_skel_animation_data["scales"], dtype=np.float32)

    time_samples = [
        f"            {frame_idx}: {_format_matrix4d(matrix)}{',' if frame_idx < len(root_motion_values) - 1 else ''}"
        for frame_idx, matrix in enumerate(root_motion_values)
    ]
    rotation_time_samples = [
        f"                    {frame_idx}: {_format_quatf_list(frame_rotations)}{',' if frame_idx < len(rotation_values) - 1 else ''}"
        for frame_idx, frame_rotations in enumerate(rotation_values)
    ]
    contents = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "Character"',
        "    metersPerUnit = 1",
        '    upAxis = "Y"',
        f"    timeCodesPerSecond = {float(walk_fps):.6f}",
        f"    framesPerSecond = {float(walk_fps):.6f}",
        "    startTimeCode = 0",
        f"    endTimeCode = {end_time}",
        "    subLayers = [",
        f"        @./{skinned_character_usda_name}@,",
        "    ]",
        ")",
        "",
        'over "Character" (',
        "    customData = {",
        f'        string bridge_step = "step2_playable_walk_layer"',
        f'        string walk_motion_metadata = "{walk_motion_metadata_name}"',
        f"        int walk_frame_count = {int(walk_frame_count)}",
        f"        double walk_fps = {float(walk_fps):.6f}",
        f"        int skel_animation_joint_count = {int(len(walk_skel_animation_data['joints']))}",
        "        bool contains_authored_skel_animation = true",
        "        bool contains_authored_root_motion = true",
        '        string root_motion_source = "transform_array"',
        '        string animation_source = "/Character/Rig/Skel/WalkAnimation"',
        "    }",
        ")",
        "{",
        "    matrix4d xformOp:transform.timeSamples = {",
        *time_samples,
        "    }",
        '    uniform token[] xformOpOrder = ["xformOp:transform"]',
        "",
        '    over "Rig"',
        "    {",
        '        over "Skel"',
        "        {",
        '            prepend rel skel:animationSource = </Character/Rig/Skel/WalkAnimation>',
        "",
        '            def SkelAnimation "WalkAnimation"',
        "            {",
        f"                uniform token[] joints = {_format_token_list(list(walk_skel_animation_data['joints']))}",
        f"                quatf[] rotations = {_format_quatf_list(default_rotation_values)}",
        "                quatf[] rotations.timeSamples = {",
        *rotation_time_samples,
        "                }",
        f"                half3[] scales = {_format_vec_list(scale_values)}",
        f"                float3[] translations = {_format_vec_list(translation_values)}",
        "            }",
        "        }",
        "    }",
        "}",
        "",
    ]
    walk_clip_usda_path.write_text("\n".join(contents), encoding="utf-8")


def _write_usda(
    usda_path: Path,
    points: np.ndarray,
    texcoords: np.ndarray,
    face_indices: np.ndarray,
    texture_name: str,
    extent_min: np.ndarray,
    extent_max: np.ndarray,
) -> None:
    face_vertex_counts = np.full((face_indices.shape[0] // 3,), 3, dtype=np.int32)

    texture_abs = str((usda_path.parent / texture_name).resolve())
    material_lines = _omnipbr_material_lines("World", texture_name, texture_absolute_path=texture_abs)

    contents = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Y"',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Scope "Looks"',
        "    {",
        *material_lines,
        "    }",
        "",
        '    def Mesh "Human" (',
        '        prepend apiSchemas = ["MaterialBindingAPI"]',
        "    )",
        "    {",
        "        uniform bool doubleSided = 1",
        f"        float3[] extent = [{_format_vec_list(np.asarray([extent_min, extent_max], dtype=np.float32))[1:-1]}]",
        f"        int[] faceVertexCounts = {_format_int_list(face_vertex_counts)}",
        f"        int[] faceVertexIndices = {_format_int_list(face_indices)}",
        f"        point3f[] points = {_format_vec_list(points)}",
        f"        texCoord2f[] primvars:st = {_format_vec_list(texcoords)}",
        '        uniform token primvars:st:interpolation = "vertex"',
        '        rel material:binding = </World/Looks/HumanMaterial>',
        '        uniform token subdivisionScheme = "none"',
        "    }",
        "}",
        "",
    ]
    usda_path.write_text("\n".join(contents), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb_path")
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()
    print(convert_habitat_humanoid_to_usda(args.glb_path, cache_root=args.cache_root))


if __name__ == "__main__":
    main()
