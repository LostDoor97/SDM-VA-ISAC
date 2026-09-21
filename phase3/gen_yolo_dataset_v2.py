"""
Phase 3 (new scene) - Tier 2 fix, v2: auto-labeled synthetic dataset generator
for the dual-node deployment in setup_new_scene.py.

Unlike the previous scene's dataset generator, camera POSE is NOT randomized
here -- TX1, TX2, and the two pedestrian sensor nodes are meant to be fixed,
deployed infrastructure (a real camera doesn't re-aim itself), so each
rendered frame uses one of these 4 exact, validated node poses. What varies
per frame is the ACTORS (small position jitter, simulating where the car/
pedestrians actually are at a given moment) and lighting exposure. Bounding
boxes are computed exactly via pinhole projection of each actor's known 3D
AABB corners through whichever node's real pose produced that frame -- no
manual annotation needed, same principle as the previous scene's generator.

Two classes only (car=0, pedestrian=1): pedestrian_a and pedestrian_b both
map to "pedestrian" -- fine-grained identity doesn't matter for Module 2/3,
only car-vs-pedestrian does (matches the previous scene's convention).

Usage:
    .venv\\Scripts\\python.exe phase3\\gen_yolo_dataset_v2.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import mitsuba as mi

try:
    mi.set_variant("cuda_ad_mono_polarized")
    print(f"Variante GPU activee : {mi.variant()}")
except Exception as e:
    print(f"CUDA indisponible ({e}), repli sur LLVM CPU")
    mi.set_variant("llvm_ad_mono_polarized")

from sionna.rt import load_scene, Camera

WORKSPACE = Path(r"C:\Users\AdminFix\Desktop\PFA")
PHASE3_DIR = WORKSPACE / "phase3"
SCENE_XML = PHASE3_DIR / "new.xml"
DATASET_DIR = PHASE3_DIR / "yolo_dataset_v2"

FRAMES_PER_NODE_TRAIN = 60
FRAMES_PER_NODE_VAL = 12
IMG_W, IMG_H = 1280, 960
SEED = 42

CLASS_NAMES = ["car", "pedestrian"]
ACTOR_TO_CLASS = {"car": 0, "pedestrian_a": 1, "pedestrian_b": 1}
ACTOR_GROUPS = ["car", "pedestrian_a", "pedestrian_b"]
ACTOR_MATERIALS = {
    "car": {"itu_metal", "itu_glass", "itu_car_wood"},
    "pedestrian_a": {"itu_ped_a_wood"},
    "pedestrian_b": {"itu_ped_B_wood"},
}
JITTER_RADIUS = {"car": 4.0, "pedestrian_a": 3.0, "pedestrian_b": 3.0}

rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)

# -- Load scene + group by material (much simpler than the previous scene's
#    filename-substring matching -- every actor now has dedicated materials) --
print("Chargement de la scene...")
scene = load_scene(str(SCENE_XML), merge_shapes=False, remove_duplicate_vertices=False)
print(f"Scene chargee ({len(scene.objects)} objets)")

group_objects = {g: [] for g in ACTOR_GROUPS}
for name, obj in scene.objects.items():
    matname = obj.radio_material.name
    for g, matset in ACTOR_MATERIALS.items():
        if matname in matset:
            group_objects[g].append(obj)

def _group_corners_and_positions(objs):
    mins, maxs, init_pos = [], [], []
    for obj in objs:
        bb = obj.mi_mesh.bbox()
        mins.append(np.array(bb.min).flatten())
        maxs.append(np.array(bb.max).flatten())
        init_pos.append(np.array(obj.position).flatten())
    mn = np.min(mins, axis=0)
    mx = np.max(maxs, axis=0)
    corners = np.array([[x, y, z]
                         for x in (mn[0], mx[0])
                         for y in (mn[1], mx[1])
                         for z in (mn[2], mx[2])])
    return corners, np.array(init_pos)

actor_init_corners = {}
actor_init_centroid = {}
actor_init_positions = {}
for g in ACTOR_GROUPS:
    corners, positions = _group_corners_and_positions(group_objects[g])
    actor_init_corners[g] = corners
    actor_init_centroid[g] = corners.mean(axis=0)
    actor_init_positions[g] = positions
    print(f"  acteur '{g}': {len(group_objects[g])} shapes, centroide {actor_init_centroid[g].round(2)}")

actors_centroid = np.mean(list(actor_init_centroid.values()), axis=0)


def set_actor_delta(group, delta):
    for obj, p0 in zip(group_objects[group], actor_init_positions[group]):
        new_pos = p0 + delta
        obj.position = mi.Point3f(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))


# -- The 4 fixed, validated deployment nodes (exact poses from setup_new_scene.py) --
TX1_POS = np.array([-119.55, 54.933, 20.79 + 1.5])
TX2_POS = np.array([47.9, 110.0, 18.0 + 1.5])
NODE_PED_A_POS = actor_init_centroid["pedestrian_a"] + np.array([-6.0, -7.0, 3.5])
NODE_PED_B_POS = actor_init_centroid["pedestrian_b"] + np.array([-6.0, 7.0, 3.5])

NODES = {
    "tx1": {"pos": TX1_POS, "look_at": actors_centroid, "fov": 25.0},
    "tx2": {"pos": TX2_POS, "look_at": actors_centroid, "fov": 25.0},
    "ped_a": {"pos": NODE_PED_A_POS, "look_at": actor_init_centroid["pedestrian_a"], "fov": 45.0},
    "ped_b": {"pos": NODE_PED_B_POS, "look_at": actor_init_centroid["pedestrian_b"], "fov": 45.0},
}


# -- Pinhole projection, parameterized per node (position/look_at/fov each differ) --
def look_at_rotation(cam_pos, target, up=np.array([0.0, 0.0, 1.0])):
    cam_pos = np.array(cam_pos, dtype=float)
    target = np.array(target, dtype=float)
    forward = target - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, forward)
    return np.stack([right, true_up, -forward], axis=0)


def project_bbox(corners_world, cam_pos, R_wc, f_px, margin=40.0):
    pts = []
    for c in corners_world:
        p_cam = R_wc @ (c - cam_pos)
        if p_cam[2] >= 0:
            continue
        x, y, z = p_cam
        u = f_px * (x / -z) + IMG_W / 2
        v = -f_px * (y / -z) + IMG_H / 2
        pts.append((u, v))
    if len(pts) < 3:
        return None
    pts = np.array(pts)
    inside = ((pts[:, 0] > -margin) & (pts[:, 0] < IMG_W + margin) &
              (pts[:, 1] > -margin) & (pts[:, 1] < IMG_H + margin))
    if inside.sum() < 3:
        return None
    u0, v0 = pts[:, 0].min(), pts[:, 1].min()
    u1, v1 = pts[:, 0].max(), pts[:, 1].max()
    u0c, v0c = max(0.0, u0), max(0.0, v0)
    u1c, v1c = min(float(IMG_W), u1), min(float(IMG_H), v1)
    if u1c - u0c < 8 or v1c - v0c < 8:
        return None
    full_area = max(u1 - u0, 1e-6) * max(v1 - v0, 1e-6)
    clip_area = (u1c - u0c) * (v1c - v0c)
    if clip_area / full_area < 0.15:
        return None
    return u0c, v0c, u1c, v1c


# -- Frame generation loop --------------------------------------------------
def gen_split(split_name, n_frames_per_node, img_dir, lbl_dir):
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_boxes = 0
    i = 0
    for node_name, node in NODES.items():
        cam_pos = node["pos"]
        look_at = node["look_at"]
        fov = node["fov"]
        R_wc = look_at_rotation(cam_pos, look_at)
        f_px = (IMG_W / 2) / np.tan(np.radians(fov) / 2)

        for k in range(n_frames_per_node):
            deltas = {}
            for g in ACTOR_GROUPS:
                r = JITTER_RADIUS[g]
                dx, dy = np_rng.uniform(-r, r, size=2)
                deltas[g] = np.array([dx, dy, 0.0])
                set_actor_delta(g, deltas[g])

            lighting_scale = rng.uniform(0.8, 1.3)
            img_path = img_dir / f"{split_name}_{node_name}_{k:03d}.png"
            try:
                scene.render_to_file(
                    camera=Camera(position=cam_pos.tolist(), look_at=look_at.tolist()),
                    filename=str(img_path),
                    resolution=[IMG_W, IMG_H],
                    num_samples=48,
                    fov=fov,
                    lighting_scale=lighting_scale,
                    show_devices=False,   # camera co-located with a node -- would render from inside the marker otherwise
                )
            except Exception as e:
                print(f"  [{node_name} {k}] render echoue: {e}")
                continue

            lines = []
            for g in ACTOR_GROUPS:
                cur_corners = actor_init_corners[g] + deltas[g]
                box = project_bbox(cur_corners, cam_pos, R_wc, f_px)
                if box is None:
                    continue
                u0, v0, u1, v1 = box
                cx = (u0 + u1) / 2 / IMG_W
                cy = (v0 + v1) / 2 / IMG_H
                bw = (u1 - u0) / IMG_W
                bh = (v1 - v0) / IMG_H
                cls = ACTOR_TO_CLASS[g]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_boxes += 1

            lbl_path = lbl_dir / f"{split_name}_{node_name}_{k:03d}.txt"
            lbl_path.write_text("\n".join(lines), encoding="utf-8")
            n_written += 1
            i += 1
            if i % 40 == 0:
                print(f"  [{split_name}] {i} frames ({n_boxes} boites au total)")

    print(f"{split_name}: {n_written} images, {n_boxes} boites au total")


print(f"\nGeneration du dataset synthetique -> {DATASET_DIR}")
gen_split("train", FRAMES_PER_NODE_TRAIN, DATASET_DIR / "images/train", DATASET_DIR / "labels/train")
gen_split("val", FRAMES_PER_NODE_VAL, DATASET_DIR / "images/val", DATASET_DIR / "labels/val")

data_yaml = f"""path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
names:
  0: car
  1: pedestrian
"""
(DATASET_DIR / "data.yaml").write_text(data_yaml, encoding="utf-8")
print(f"\ndata.yaml ecrit -> {DATASET_DIR / 'data.yaml'}")
print("Termine.")
