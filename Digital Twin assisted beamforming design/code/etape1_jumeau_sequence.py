"""Etape 1 -- sequence longue (100 frames) et sortie du jumeau de la phase 5 a chaque frame (M1).

Reprend les modules de la phase 5 (meme bilan de liaison, memes mesures radar apparentes, meme
YOLOv8n fine-tune, meme fusion et memes filtres), appliques au monde "reel" du projet :
scene_reelle/real.xml, acteurs animes le long des trajectoires dessinees dans Blender.

Changements par rapport a la phase 5, et pourquoi :
  1. Cameras fixes planifiees (et non plus pointees a chaque frame sur le barycentre vrai des
     acteurs, qui s'etalent maintenant sur une centaine de metres) : chaque noeud recoit jusqu'a
     4 cameras (FOV 25 deg, celui d'entrainement de YOLO) qui couvrent les rues parcourues, comme
     une installation reelle. Un noeud reste UN point de vue, quel que soit son nombre de cameras.
  2. Confirmation temporelle causale : un objet est confirme s'il existait deja un candidat a moins
     de 3 m a la frame precedente. La phase 5 regardait aussi t+1, ce qu'un beamforming a
     l'instant t ne peut pas faire.
  3. N acteurs quelconques (voitures, bus, pietons) au lieu des trois de la phase 5.

Etages mis en cache (relancer le script reprend ou il s'est arrete) :
  etape1/cache/radar_FR*_tXXX.npz, etape1/frames/*.png, etape1/cache/detections.json

Usage : .venv\\Scripts\\python.exe "Digital Twin assisted beamforming design\\code\\etape1_jumeau_sequence.py"
        [--scene DOSSIER] [--out DOSSIER] [--frames N] [--trials K] [--samples S]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from PIL import Image as PILImage

import dt_common as dc
import mitsuba as mi
from sionna.rt import Transmitter, Receiver, PlanarArray, PathSolver, Camera

# ======================= 1. Bandes et bilan de liaison (identiques a la phase 5) =================
CPI_S, SNR_MIN_DB, MAX_DEPTH = 10e-3, 13.0, 4
IMG_W, IMG_H = 1280, 960
ROI_MIN, ROI_MAX = np.array([-176.0, -186.0, 0.0]), np.array([202.0, 170.0, 26.0])
VOXEL = 2.0
GROUND_Z_MAX, MIN_EVIDENCE, MIN_VIEWS, CLUSTER_RADIUS, MATCH_RADIUS = 0.8, 2.0, 2, 3.0, 3.0
DEPTH_TOL_M, DEPTH_TOL_REL = 4.0, 0.10
INVALID = np.uint32(0xFFFFFFFF)


def make_band(freq, bw, scs, n_array, tx_dbm, nf_db):
    n_sc = int(bw / scs * 0.95) // 4
    slots = int(CPI_S / (1e-3 / (scs / 15e3)))
    g_proc = 10 * np.log10(n_sc * 2 * slots)
    g_array = 2 * 10 * np.log10(n_array ** 2)
    noise = -174.0 + 10 * np.log10(bw) + nf_db
    return {"freq_hz": freq, "bw_hz": bw, "n_array": n_array, "tx_dbm": tx_dbm, "g_proc_db": g_proc,
            "g_array_db": g_array, "noise_dbm": noise, "range_res_m": dc.C0 / (2 * bw),
            "beamwidth_rad": 1.772 / n_array, "doppler_res_hz": 1.0 / CPI_S}


BANDS = {"FR1": make_band(3.5e9, 100e6, 30e3, 8, 43.0, 5.0),
         "FR2": make_band(28e9, 400e6, 120e3, 16, 30.0, 7.0)}
CONFIGS = {"C1_FR1": {"bands": ["FR1"], "vision_objects": False},
           "C2_FR2": {"bands": ["FR2"], "vision_objects": False},
           "C3_FR1+FR2": {"bands": ["FR1", "FR2"], "vision_objects": False},
           "C4_FR1+FR2+vision": {"bands": ["FR1", "FR2"], "vision_objects": True}}
YOLO_CLASS_OF = {"car": "car", "pedestrian": "pedestrian", "bus": None}   # le bus n'est pas une classe YOLO


# ======================= 2. Module 1a : mesures radar apparentes ================================
class Radar:
    def __init__(self, world: dc.RealWorld):
        self.w = world
        sc = world.scene
        for n in dc.BS_NAMES:
            sc.add(Transmitter(name=f"tx_{n}", position=dc.BS_POS[n].tolist()))
            sc.add(Receiver(name=f"rx_{n}", position=(dc.BS_POS[n] + dc.RX_OFFSET).tolist()))
        self.tx_pos = np.array([dc.BS_POS[n] for n in dc.BS_NAMES])
        self.rx_pos = self.tx_pos + dc.RX_OFFSET
        self.solver = PathSolver()
        self.actor_ids = np.array(sorted(world.objid_actor), dtype=np.uint32)
        self.aoa_sign = None

    def measure(self, band_name, frame):
        band = BANDS[band_name]
        sc = self.w.scene
        self.w.set_band(band["freq_hz"], band["bw_hz"])
        sc.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        paths = self.solver(sc, max_depth=MAX_DEPTH, los=True, specular_reflection=True,
                            diffuse_reflection=True, refraction=False, diffraction=True,
                            edge_diffraction=False, seed=42 + frame)
        valid = np.array(paths.valid).astype(bool)
        tau, doppler = np.array(paths.tau), np.array(paths.doppler)
        theta_r, phi_r = np.array(paths.theta_r), np.array(paths.phi_r)
        objs, verts = np.array(paths.objects), np.array(paths.vertices)
        a, _ = paths.cir(sampling_frequency=band["bw_hz"], num_time_steps=1, normalize_delays=False,
                         out_type="numpy")
        gain_db = 10 * np.log10(np.abs(a[:, 0, :, 0, :, 0]) ** 2 + 1e-30)
        r_i, t_i, p_i = np.nonzero(valid & (objs[0] != INVALID))
        order = np.lexsort((tau[r_i, t_i, p_i], t_i, r_i))
        r_i, t_i, p_i = r_i[order], t_i[order], p_i[order]
        depth = (objs[:, r_i, t_i, p_i] != INVALID).sum(axis=0)
        last_vertex = verts[depth - 1, r_i, t_i, p_i]
        u = np.stack([np.sin(theta_r) * np.cos(phi_r), np.sin(theta_r) * np.sin(phi_r),
                      np.cos(theta_r)], -1)[r_i, t_i, p_i]
        rxp, txp = self.rx_pos[r_i], self.tx_pos[t_i]
        if self.aoa_sign is None:
            one = depth == 1
            to_v = last_vertex[one] - rxp[one]
            to_v /= np.linalg.norm(to_v, axis=1, keepdims=True)
            self.aoa_sign = float(np.sign(np.median((to_v * u[one]).sum(1))))
        u = self.aoa_sign * u
        L = dc.C0 * tau[r_i, t_i, p_i]
        d = txp - rxp
        dist = (L ** 2 - (d ** 2).sum(1)) / (2 * (L - (u * d).sum(1)))
        snr = band["tx_dbm"] + gain_db[r_i, t_i, p_i] + band["g_array_db"] + band["g_proc_db"] - band["noise_dbm"]
        snr_lin = 10 ** (snr / 10)
        sig_r = np.maximum(band["range_res_m"] / 10, band["range_res_m"] / np.sqrt(2 * snr_lin))
        sig_a = np.maximum(band["beamwidth_rad"] / 10, band["beamwidth_rad"] / np.sqrt(2 * snr_lin))
        return {"tx": t_i, "rx": r_i, "depth": depth, "snr_db": snr, "doppler_hz": doppler[r_i, t_i, p_i],
                "u": u, "dist": dist, "rxp": rxp, "sig_r": sig_r, "sig_a": sig_a,
                "point_clean": rxp + dist[:, None] * u, "last_vertex": last_vertex,
                "mover_gt": np.isin(objs[:, r_i, t_i, p_i], self.actor_ids).any(axis=0)}


def noisy_points(m, gen):
    u = m["u"]
    e1 = np.cross(u, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(u, e1)
    n = gen.standard_normal((len(u), 3))
    u_n = u + (m["sig_a"] * n[:, 1])[:, None] * e1 + (m["sig_a"] * n[:, 2])[:, None] * e2
    u_n /= np.linalg.norm(u_n, axis=1, keepdims=True)
    return m["rxp"] + (m["dist"] + m["sig_r"] * n[:, 0])[:, None] * u_n


# ======================= 3. Module 1b : cameras fixes planifiees + YOLO ==========================
def look_at_rotation(cam_pos, target, up=np.array([0.0, 0.0, 1.0])):
    f = np.asarray(target, float) - np.asarray(cam_pos, float)
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    return np.stack([r, np.cross(r, f), -f], axis=0)


def project_point(p, cam_pos, R, f_px):
    pc = R @ (np.asarray(p) - cam_pos)
    if pc[2] >= 0:
        return None
    u, v = f_px * (pc[0] / -pc[2]) + IMG_W / 2, -f_px * (pc[1] / -pc[2]) + IMG_H / 2
    return (float(u), float(v)) if (0 <= u < IMG_W and 0 <= v < IMG_H) else None


def plan_cameras(world: dc.RealWorld, T: int):
    """Couverture gloutonne des points de trajectoire visibles depuis chaque noeud."""
    targets = []
    for t in range(0, T, 2):
        for a, s in world.actor_state(t).items():
            targets.append(np.r_[s["pos"][:2], 0.9])
    targets = np.array(targets)
    c0 = world.c0
    nodes = {**{n.lower(): (dc.BS_POS[n], 25.0, 4, 260.0) for n in dc.BS_NAMES},
             "ped_a": (c0["ped_a"] + np.array([-6.0, -7.0, 3.5]), 45.0, 2, 70.0),
             "ped_b": (c0["ped_b"] + np.array([-6.0, 7.0, 3.5]), 45.0, 2, 70.0)}
    cams = []
    for node, (pos, fov, max_cams, rng) in nodes.items():
        d = targets - pos
        dist = np.linalg.norm(d, axis=1)
        blocked, _, oid = world.first_hit(np.repeat(pos[None], len(targets), 0), targets)
        vis = (~blocked | np.isin(oid, list(world.objid_actor))) & (dist < rng)
        dirs = d / dist[:, None]
        half = np.radians(0.8 * fov / 2 * 0.75)            # demi-champ vertical (4:3), marge 20 %
        unc = vis.copy()
        for k in range(max_cams):
            idx = np.nonzero(unc)[0]
            if len(idx) < max(5, 0.03 * vis.sum()):
                break
            cand = idx[np.linspace(0, len(idx) - 1, min(300, len(idx))).astype(int)]
            cover = (dirs[cand] @ dirs[idx].T) > np.cos(half)
            best = cand[int(cover.sum(1).argmax())]
            got = idx[(dirs[idx] @ dirs[best]) > np.cos(half)]
            look = targets[got].mean(0)
            cams.append({"name": f"{node}_{k}", "node": node, "pos": pos, "look_at": look, "fov": fov,
                         "R": look_at_rotation(pos, look), "f_px": (IMG_W / 2) / np.tan(np.radians(fov) / 2)})
            unc[got] = False
        print(f"  noeud {node:6s}: {sum(c['node'] == node for c in cams)} camera(s), "
              f"{vis.sum()} / {len(targets)} points de trajectoire visibles, {unc.sum()} non couverts")
    return cams


def render_and_detect(world, cams, t, frames_dir, samples, yolo, cache, imgsz=640):
    key = str(t)
    if key in cache and all(c["name"] in cache[key] for c in cams):
        return cache[key]
    out = {}
    for c in cams:
        img = frames_dir / f"frame_{t:03d}_{c['name']}.png"
        if img.exists():                        # image tronquee par un arret brutal -> on la refait
            try:
                with PILImage.open(img) as im:
                    im.load()
            except Exception:
                img.unlink()
        if not img.exists():
            world.scene.render_to_file(camera=Camera(position=c["pos"].tolist(), look_at=c["look_at"].tolist()),
                                       filename=str(img), resolution=[IMG_W, IMG_H], num_samples=samples,
                                       fov=c["fov"], show_devices=False)
        res = yolo.predict(source=str(img), verbose=False, imgsz=imgsz)[0]
        dets = []
        if res.boxes is not None:
            for box, cls_id, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                         res.boxes.conf.cpu().numpy()):
                dets.append({"bbox_xyxy": box.tolist(), "cls": yolo.names[int(cls_id)], "conf": float(conf)})
        out[c["name"]] = dets
    cache[key] = out
    return out


# ======================= 4. Module 2 : fusion radar-vision (phase 5) =============================
def backproject_to_ground(u, v, cam_pos, R, f_px, z_ground=0.0):
    d = R.T @ np.array([(u - IMG_W / 2) / f_px, -(v - IMG_H / 2) / f_px, -1.0])
    if d[2] >= -1e-6:
        return None
    return cam_pos + (z_ground - cam_pos[2]) / d[2] * d


def refine(p, cams, dets):
    best, best_d, best_key = None, np.inf, None
    for c in cams:
        D = dets[c["name"]]
        if not D:
            continue
        uv = project_point(p, c["pos"], c["R"], c["f_px"])
        if uv is None:
            continue
        cent = np.array([[(d["bbox_xyxy"][0] + d["bbox_xyxy"][2]) / 2, (d["bbox_xyxy"][1] + d["bbox_xyxy"][3]) / 2]
                         for d in D])
        dist, j = cKDTree(cent).query(uv)
        x0, y0, x1, y1 = D[j]["bbox_xyxy"]
        rel = dist / (0.5 * np.hypot(x1 - x0, y1 - y0))
        if rel > 1.0 or rel >= best_d:
            continue
        g = backproject_to_ground((x0 + x1) / 2, y1, c["pos"], c["R"], c["f_px"])
        if g is not None:
            r_det, r_echo = np.linalg.norm(g - c["pos"]), np.linalg.norm(np.asarray(p) - c["pos"])
            if abs(r_echo - r_det) > max(DEPTH_TOL_M, DEPTH_TOL_REL * r_det):
                continue
        best, best_d, best_key = D[j], rel, (c["name"], j)
    return best, best_key


def vision_objects(cams, dets, used):
    out = []
    for c in cams:
        for j, d in enumerate(dets[c["name"]]):
            if (c["name"], j) in used:
                continue
            x0, y0, x1, y1 = d["bbox_xyxy"]
            g = backproject_to_ground((x0 + x1) / 2, y1, c["pos"], c["R"], c["f_px"])
            if g is not None and np.all((g >= ROI_MIN) & (g < ROI_MAX)):
                out.append((g, d["cls"], c["node"]))
    return out


# ======================= 5. Module 3 : jumeau a deux couches, causal =============================
def in_roi(p):
    return np.all((p >= ROI_MIN) & (p < ROI_MAX), axis=1)


def clusters(xy, radius):
    parent = np.arange(len(xy))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    if len(xy):
        for a, b in cKDTree(xy).query_pairs(radius, output_type="ndarray"):
            parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(len(xy))])
    return [np.nonzero(roots == r)[0] for r in np.unique(roots)]


def build_twin(cfg, meas, points, cams, dets, T):
    """Couche dynamique par frame (causale) + nuage statique cumule."""
    candidates, confirmed, stat = [], [], []
    for t in range(T):
        pts, labs, views, used = [], [], [], set()
        for b in cfg["bands"]:
            m, p = meas[(b, t)], points[(b, t)]
            keep = (m["snr_db"] >= SNR_MIN_DB) & (p[:, 2] > -0.5) & in_roi(np.c_[p[:, :2], np.clip(p[:, 2], 0, None)])
            moving = np.abs(m["doppler_hz"]) >= BANDS[b]["doppler_res_hz"]
            s = keep & ~moving
            stat.append((np.c_[p[s, :2], np.clip(p[s, 2], 0, None)], m["rx"][s]))
            for q, rx in zip(p[keep & moving & (p[:, 2] < 3.5)], m["rx"][keep & moving & (p[:, 2] < 3.5)]):
                det, key = refine(q, cams, dets[t])
                if det is not None:
                    used.add(key)
                pts.append(q)
                labs.append(det["cls"] if det is not None else "target")
                views.append({f"rx{rx}"})
        if cfg["vision_objects"]:
            for g, cls, node in vision_objects(cams, dets[t], used):
                pts.append(g)
                labs.append(cls)
                views.append({f"cam_{node}"})
        pts = np.array(pts).reshape(-1, 3)
        cand = []
        for idx in clusters(pts[:, :2], CLUSTER_RADIUS):
            named = [labs[i] for i in idx if labs[i] != "target"]
            vs = sorted(set().union(*(views[i] for i in idx)))
            if len(vs) < MIN_VIEWS:
                continue
            cand.append({"xy": np.median(pts[idx, :2], axis=0),
                         "cls": Counter(named).most_common(1)[0][0] if named else "target",
                         "views": vs, "n_radar": int(sum(any(v.startswith("rx") for v in views[i]) for i in idx)),
                         "n_vision": int(sum(not any(v.startswith("rx") for v in views[i]) for i in idx))})
        candidates.append(cand)
        prev = candidates[t - 1] if t > 0 else []
        confirmed.append([o for o in cand if any(np.linalg.norm(o["xy"] - q["xy"]) <= MATCH_RADIUS for q in prev)])
    return {"dynamic": confirmed, "candidates": candidates, "static": stat}


def evaluate_dynamic(twin, gt, T):
    """Par acteur : part des frames detectees (< 3 m), bien classees, erreur ; faux objets / frame."""
    res = {a: {"detecte": 0, "bien_classe": 0, "err": []} for a in gt[0]}
    false_obj = []
    for t in range(1, T):                       # la frame 0 n'a pas d'historique (confirmation causale)
        objs = twin["dynamic"][t]
        matched = set()
        for a, (xy, cls) in gt[t].items():
            if not objs:
                continue
            d = [np.linalg.norm(o["xy"] - xy) for o in objs]
            k = int(np.argmin(d))
            if d[k] <= MATCH_RADIUS:
                matched.add(k)
                res[a]["detecte"] += 1
                res[a]["err"].append(d[k])
                res[a]["bien_classe"] += int(YOLO_CLASS_OF[cls] is not None and objs[k]["cls"] == YOLO_CLASS_OF[cls])
        false_obj.append(len(objs) - len(matched))
    n = T - 1
    return {"acteurs": {a: {"detecte": v["detecte"] / n, "bien_classe": v["bien_classe"] / n,
                            "erreur_moyenne_m": float(np.mean(v["err"])) if v["err"] else None}
                        for a, v in res.items()},
            "faux_objets_par_frame": float(np.mean(false_obj))}


# ======================= 6. Programme principal ==================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(dc.SCENE_DIR))
    ap.add_argument("--out", default=str(dc.ROOT / "etape1"))
    ap.add_argument("--frames", type=int, default=dc.N_FRAMES)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--yolo", default=str(dc.PHASE3_DIR / "yolo_finetune_v2" / "weights" / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640, help="resolution d'inference YOLO")
    ap.add_argument("--tag", default="", help="suffixe des fichiers de sortie (ex. _yolo_v3)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    T = args.frames
    t_start = time.time()

    world = dc.RealWorld(Path(args.scene))
    radar = Radar(world)
    print("Cameras :")
    cams = plan_cameras(world, T)

    if "mlflow" not in sys.modules:
        sys.modules["mlflow"] = types.ModuleType("mlflow")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    if "bus" in yolo.names.values():
        YOLO_CLASS_OF["bus"] = "bus"
    tag = args.tag
    det_file = out / "cache" / f"detections{tag}.json"
    det_cache = json.load(open(det_file)) if det_file.exists() else {}

    meas, dets, gt, ue_pos = {}, [], [], []
    for t in range(T):
        st = world.apply_frame(t)
        gt.append({a: (st[a]["pos"][:2].copy(), dc.actor_class(a)) for a in world.actors})
        ue_pos.append(world.ue_positions())
        for b in BANDS:
            f = out / "cache" / f"radar_{b}_t{t:03d}.npz"
            if f.exists():                      # fichier tronque par un arret brutal -> on recalcule
                try:
                    meas[(b, t)] = dict(np.load(f))
                except Exception:
                    f.unlink()
            if f.exists():
                if radar.aoa_sign is None:
                    radar.aoa_sign = float(meas[(b, t)].pop("aoa_sign"))
                else:
                    meas[(b, t)].pop("aoa_sign")
            else:
                meas[(b, t)] = radar.measure(b, t)
                np.savez(f, aoa_sign=radar.aoa_sign, **meas[(b, t)])
        dets.append(render_and_detect(world, cams, t, out / "frames", args.samples, yolo, det_cache, args.imgsz))
        if t % 5 == 0 or t == T - 1:
            json.dump(det_cache, open(det_file, "w"))
            el = time.time() - t_start
            nd = sum(len(v) for v in dets[-1].values())
            print(f"frame {t:3d}/{T}: echos FR1 {int((meas[('FR1', t)]['snr_db'] >= SNR_MIN_DB).sum()):5d}, "
                  f"FR2 {int((meas[('FR2', t)]['snr_db'] >= SNR_MIN_DB).sum()):5d}, detections {nd:3d} | "
                  f"{el:6.0f} s, ~{el / (t + 1) * (T - t - 1):6.0f} s restantes", flush=True)

    # ---- jumeaux C1..C4, plusieurs tirages du bruit --------------------------------------------
    results, twins_out = {}, {}
    for name, cfg in CONFIGS.items():
        runs = []
        for i in range(args.trials):
            gen = np.random.default_rng(1000 + i)
            pts = {k: noisy_points(m, gen) for k, m in meas.items()}
            tw = build_twin(cfg, meas, pts, cams, dets, T)
            runs.append(evaluate_dynamic(tw, gt, T))
            twins_out.setdefault(name, []).append(
                [[{"xy": o["xy"].round(3).tolist(), "cls": o["cls"], "views": o["views"],
                   "n_radar": o["n_radar"], "n_vision": o["n_vision"]} for o in fr] for fr in tw["dynamic"]])
        agg = {}
        for a in gt[0]:
            for k in ("detecte", "bien_classe", "erreur_moyenne_m"):
                v = [r["acteurs"][a][k] for r in runs if r["acteurs"][a][k] is not None]
                agg.setdefault(a, {})[k] = (float(np.mean(v)), float(np.std(v))) if v else (None, None)
        fo = [r["faux_objets_par_frame"] for r in runs]
        results[name] = {"acteurs": agg, "faux_objets_par_frame": (float(np.mean(fo)), float(np.std(fo)))}
        print(f"{name:18s} faux objets {np.mean(fo):.2f} +- {np.std(fo):.2f} | " + ", ".join(
            f"{a} {agg[a]['detecte'][0]:.0%}/{agg[a]['bien_classe'][0]:.0%}" for a in gt[0]))

    # ---- sauvegardes -----------------------------------------------------------------------------
    json.dump({"configs": list(CONFIGS), "n_tirages": args.trials, "frames": T,
               "note": "jumeau[config][tirage][frame] = objets confirmes (causal) ; frame 0 vide par construction",
               "jumeau": twins_out}, open(out / f"jumeau_sequence{tag}.json", "w"))
    actors = world.actors
    np.savez_compressed(out / "verite_terrain.npz", actor_names=np.array(actors),
                        actor_xy=np.array([[gt[t][a][0] for a in actors] for t in range(T)]),
                        actor_class=np.array([dc.actor_class(a) for a in actors]),
                        ue_names=np.array(world.ue_actors),
                        ue_pos=np.array([[ue_pos[t][a] for a in world.ue_actors] for t in range(T)]))
    summary = {"etape": "1 - sequence longue + jumeau phase 5 par frame (M1)", "frames": T, "frame_s": dc.FRAME_S,
               "scene": str(Path(args.scene) / "real.xml"), "monde_reel": world.summary(),
               "cameras": [{"nom": c["name"], "noeud": c["node"], "fov": c["fov"],
                            "vise": c["look_at"].round(1).tolist()} for c in cams],
               "n_tirages_bruit": args.trials, "rendu_echantillons": args.samples,
               "yolo": {"poids": args.yolo, "imgsz": args.imgsz, "classes": list(yolo.names.values())},
               "confirmation_temporelle": "causale (frame precedente)",
               "resultats_couche_dynamique": results, "temps_calcul_s": round(time.time() - t_start, 1)}
    json.dump(summary, open(out / f"etape1_summary{tag}.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    figure_sequence(out, world, gt, twins_out, cams, T, tag)
    print(f"termine en {time.time() - t_start:.0f} s")


def figure_sequence(out, world, gt, twins_out, cams, T, tag=""):
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab10(np.arange(10))
    for k, a in enumerate(world.actors):
        xy = np.array([gt[t][a][0] for t in range(T)])
        ax.plot(xy[:, 0], xy[:, 1], "-", color=colors[k % 10], lw=2.5, label=f"{a} (verite)")
        ax.plot(*xy[0], "o", color=colors[k % 10], ms=8)
    tw = twins_out["C4_FR1+FR2+vision"][0]
    obj = np.array([o["xy"] for fr in tw for o in fr]).reshape(-1, 2)
    cls = [o["cls"] for fr in tw for o in fr]
    for c, mk in (("car", "s"), ("pedestrian", "^"), ("bus", "D"), ("target", "x")):
        m = np.array([q == c for q in cls], bool)
        if m.any():
            ax.scatter(obj[m, 0], obj[m, 1], s=10, marker=mk, color="k", alpha=0.35, label=f"jumeau C4 : {c}")
    for n, p in dc.BS_POS.items():
        ax.plot(*p[:2], "^", color="crimson", ms=12, mec="k")
        ax.annotate(n, p[:2], xytext=(5, 5), textcoords="offset points", color="crimson", weight="bold")
    for c in cams:
        ax.plot([c["pos"][0], c["look_at"][0]], [c["pos"][1], c["look_at"][1]], ":", color="0.5", lw=0.8)
    ax.set_aspect("equal")
    ax.set(xlabel="x [m]", ylabel="y [m]", title="Sequence de 100 frames : trajectoires vraies et objets du jumeau C4 (tirage 0)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"sequence_trajectoires_jumeau{tag}.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
