"""Jeu de donnees YOLO v3 : rendus de la scene REELLE (mobilier inclus), 3 classes (car, pedestrian, bus).

Protocole (pour ne pas entrainer sur le jeu de test) :
  - les 100 frames de la sequence de l'etape 1 ne sont jamais utilisees ;
  - les acteurs sont places AU HASARD dans les rues (et non sur leurs trajectoires de test), avec 35 %
    des pietons poses pres du mobilier (abribus, lampadaires, poubelle) ;
  - aucun pieton a moins de 2 m des positions de test des pietons immobiles (ped_b, ped_c) ;
  - cameras : memes points de montage que le deploiement (toits des BS, poteaux) + poteaux aleatoires,
    mais visees, champs et contenus differents des 22 cameras de test.
Etiquettes automatiques : projection des sommets de chaque acteur, en ne gardant que les sommets vus
depuis la camera (lancer de rayons) ; boite = enveloppe des sommets visibles.

Usage : .venv\\Scripts\\python.exe "Digital Twin assisted beamforming design\\code\\yolo_v3_donnees.py" [--n-train 1100] [--n-val 200]
Sortie : yolo_v3/dataset/{images,labels}/{train,val} + data.yaml   (reprise possible : images deja faites sautees)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import dt_common as dc
import mitsuba as mi
from sionna.rt import Camera

IMG_W, IMG_H = 1280, 960
CLASSES = ["car", "pedestrian", "bus"]
CLS_OF = {"car": 0, "pedestrian": 1, "bus": 2}
OUT = dc.ROOT / "yolo_v3" / "dataset"
HOLDOUT_STATIC = {"ped_b": (-51.80, 106.96), "ped_c": (-29.0, 116.3)}
DEPLOY_CENTRE = np.array([-27.4, 51.7])  # barycentre des UE de la sequence (cible des panneaux)
SPP = 256                               # meme bruit de rendu que les images de test


def look_at_rotation(cam_pos, target, up=np.array([0.0, 0.0, 1.0])):
    f = np.asarray(target, float) - np.asarray(cam_pos, float)
    f /= np.linalg.norm(f) + 1e-12
    r = np.cross(f, up)
    r /= np.linalg.norm(r) + 1e-12
    return np.stack([r, np.cross(r, f), -f], axis=0)


class Placer:
    """Carte des cellules libres (lancer vertical) et pose rigide des acteurs."""

    def __init__(self, world: dc.RealWorld, rng):
        self.w, self.rng = world, rng
        x0, x1, y0, y1, step = -176.0, 202.0, -186.0, 170.0, 1.0
        gx, gy = np.meshgrid(np.arange(x0, x1, step) + 0.5, np.arange(y0, y1, step) + 0.5)
        self.ext = (x0, y0, step, gx.shape)
        # cacher les acteurs pour la carte (sinon leurs positions de depart sont "occupees")
        self.hide_all()
        o = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 200.0)]
        si = world.scene.mi_scene.ray_intersect(mi.Ray3f(o=mi.Point3f(*[mi.Float(o[:, i].astype(np.float32)) for i in range(3)]),
                                                         d=mi.Vector3f(0.0, 0.0, -1.0)))
        self.hz = np.where(np.array(si.is_valid()), np.array(si.p.z), -1.0).reshape(gx.shape)
        self.free = (self.hz > -0.5) & (self.hz < 0.6)          # sol, trottoir, route (pas de batiment ni de mobilier)
        self.road = self.free & (self.hz > 0.1)
        self.cells = np.c_[gx[self.free], gy[self.free]]
        self.road_cells = np.c_[gx[self.road], gy[self.road]]
        self.clutter_xy = np.array([np.array(c.mi_mesh.bbox().center())[:2].ravel()
                                    for c in world.clutter_objs]).reshape(-1, 2)
        self.ground0 = {a: self.height(*world.c0[a][:2]) for a in world.actors}
        self.half = {}
        for a in world.actors:                                   # demi-empreinte dans le repere propre
            xy = np.concatenate([world._v0[o.mi_mesh.id()][:, :2] for o in world.actor_objs[a]]) - world.c0[a][:2]
            loc = xy @ dc.rotz(-world.yaw0.get(a, 0.0))[:2, :2].T
            self.half[a] = np.abs(loc).max(0)

    def idx(self, x, y):
        x0, y0, s, sh = self.ext
        i, j = int((y - y0) // s), int((x - x0) // s)
        return (i, j) if 0 <= i < sh[0] and 0 <= j < sh[1] else None

    def height(self, x, y):
        k = self.idx(x, y)
        return float(self.hz[k]) if k is not None and self.hz[k] > -0.5 else 0.0

    def footprint_ok(self, a, x, y, yaw, mask):
        hl, hw = self.half[a] + 0.2
        c, s = np.cos(yaw), np.sin(yaw)
        for u in np.linspace(-hl, hl, max(2, int(2 * hl) + 1)):
            for v in np.linspace(-hw, hw, max(2, int(2 * hw) + 1)):
                k = self.idx(x + c * u - s * v, y + s * u + c * v)
                if k is None or not mask[k]:
                    return False
        return True

    def road_yaw(self, x, y):
        near = self.road_cells[np.linalg.norm(self.road_cells - [x, y], axis=1) < 8]
        if len(near) < 5:
            return self.rng.uniform(-np.pi, np.pi)
        ev, evec = np.linalg.eigh(np.cov((near - near.mean(0)).T))
        d = evec[:, -1]
        return float(np.arctan2(d[1], d[0]) + (np.pi if self.rng.random() < 0.5 else 0.0))

    def pose(self, a, x, y, yaw):
        w = self.w
        dz = self.height(x, y) - self.ground0[a]
        R = dc.rotz(yaw - w.yaw0.get(a, 0.0))
        shift = np.array([x, y, w.c0[a][2] + dz])
        for o in w.actor_objs[a]:
            key = o.mi_mesh.id()
            v = (w._v0[key] - w.c0[a]) @ R.T + shift
            w.params[key + ".vertex_positions"] = mi.Float(v.astype(np.float32).ravel())

    def hide(self, a):
        for o in self.w.actor_objs[a]:
            key = o.mi_mesh.id()
            v = self.w._v0[key] + np.array([5000.0, 5000.0, -500.0])
            self.w.params[key + ".vertex_positions"] = mi.Float(v.astype(np.float32).ravel())

    def hide_all(self):
        for a in self.w.actors:
            self.hide(a)
        self.commit()

    def commit(self):
        self.w.params.update()
        self.w.scene.scene_geometry_updated()

    def random_layout(self):
        """Place un sous-ensemble aleatoire d'acteurs ; renvoie {acteur: (x, y)}."""
        rng, placed = self.rng, {}
        actors = list(self.w.actors)
        rng.shuffle(actors)
        n_keep = rng.integers(3, len(actors) + 1)
        # Groupement : tous les acteurs dans un rayon de ~35 m autour d'un centre tire (mobilier, zone du
        # deploiement ou rue quelconque), pour avoir plusieurs acteurs et des masquages dans chaque image.
        u = rng.random()
        if u < 0.3 and len(self.clutter_xy):
            centre = self.clutter_xy[rng.integers(len(self.clutter_xy))]
        elif u < 0.65:
            centre = DEPLOY_CENTRE + rng.uniform(-60, 60, 2)
        else:
            centre = self.road_cells[rng.integers(len(self.road_cells))]
        cells = self.cells[np.linalg.norm(self.cells - centre, axis=1) < 35]
        roads = self.road_cells[np.linalg.norm(self.road_cells - centre, axis=1) < 40]
        cells = cells if len(cells) > 50 else self.cells
        roads = roads if len(roads) > 50 else self.road_cells
        clut = self.clutter_xy[np.linalg.norm(self.clutter_xy - centre, axis=1) < 40] if len(self.clutter_xy) else []
        self.centre = centre
        for a in actors[:n_keep]:
            cls = dc.actor_class(a)
            for _ in range(60):
                if cls == "pedestrian" and rng.random() < 0.35 and len(clut):
                    c = clut[rng.integers(len(clut))]
                    r, th = rng.uniform(0.8, 4.0), rng.uniform(-np.pi, np.pi)
                    x, y = c + r * np.array([np.cos(th), np.sin(th)])
                    yaw = rng.uniform(-np.pi, np.pi)
                    mask = self.free
                elif cls == "pedestrian":
                    x, y = cells[rng.integers(len(cells))] + rng.uniform(-0.5, 0.5, 2)
                    yaw, mask = rng.uniform(-np.pi, np.pi), self.free
                else:
                    x, y = roads[rng.integers(len(roads))] + rng.uniform(-0.5, 0.5, 2)
                    yaw, mask = self.road_yaw(x, y), self.road
                if cls == "pedestrian" and any(np.hypot(x - hx, y - hy) < 2.0 for hx, hy in HOLDOUT_STATIC.values()):
                    continue
                if any(np.hypot(x - px, y - py) < (self.half[a].max() + self.half[b].max() + 0.5)
                       for b, (px, py) in placed.items()):
                    continue
                if not self.footprint_ok(a, x, y, yaw, mask):
                    continue
                self.pose(a, x, y, yaw)
                placed[a] = (x, y)
                break
        for a in self.w.actors:
            if a not in placed:
                self.hide(a)
        self.commit()
        return placed


def random_camera(world, placer, placed, rng):
    """Visee d'un acteur present depuis un point de montage plausible."""
    if rng.random() < 0.5:
        xy = np.mean(list(placed.values()), axis=0)
    else:
        xy = np.array(placed[list(placed)[rng.integers(len(placed))]])
    tgt = np.r_[xy, 1.0] + np.r_[rng.normal(0, 4.0, 2), 0.0]
    mode = rng.random()
    if mode < 0.45:                                       # toit d'une BS (montage du deploiement)
        bs = [b for b in dc.BS_NAMES if 25 < np.linalg.norm(dc.BS_POS[b][:2] - tgt[:2]) < 190]
        if bs:
            return dc.BS_POS[bs[rng.integers(len(bs))]].copy(), tgt, float(rng.choice([25.0, 25.0, 35.0]))
    for _ in range(50):                                   # poteau aleatoire dans la rue, 8 a 60 m de la cible
        p = placer.cells[rng.integers(len(placer.cells))]
        d = np.linalg.norm(p - tgt[:2])
        if 8 < d < 60:
            cam = np.r_[p, placer.height(*p) + rng.uniform(3.0, 8.0)]
            return cam, tgt, float(rng.uniform(35.0, 55.0))
    return dc.BS_POS["TX2"].copy(), tgt, 25.0


def labels_for(world, placed, cam, R, f_px, rng):
    out = []
    for a in placed:
        V = np.concatenate([np.array(world.params[o.mi_mesh.id() + ".vertex_positions"]).reshape(-1, 3)
                            for o in world.actor_objs[a]])
        V = V[rng.choice(len(V), min(500, len(V)), replace=False)]
        pc = (V - cam) @ R.T
        front = pc[:, 2] < -0.1
        if front.sum() < 10:
            continue
        uv = np.c_[f_px * pc[:, 0] / -pc[:, 2] + IMG_W / 2, -f_px * pc[:, 1] / -pc[:, 2] + IMG_H / 2]
        inimg = front & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
        if inimg.sum() < 10:
            continue
        P = V[inimg]
        blocked, _, oid = world.first_hit(np.repeat(cam[None], len(P), 0).astype(np.float32), P.astype(np.float32))
        own = np.isin(oid, [int(o.object_id) for o in world.actor_objs[a]])
        vis = (~np.asarray(blocked)) | own
        if vis.sum() < max(10, 0.15 * inimg.sum()):
            continue
        q = uv[inimg][vis]
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        if (y1 - y0) < 8 or (x1 - x0) < 4:
            continue
        out.append((CLS_OF[dc.actor_class(a)], (x0 + x1) / 2 / IMG_W, (y0 + y1) / 2 / IMG_H,
                    (x1 - x0) / IMG_W, (y1 - y0) / IMG_H, a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1100)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    world = dc.RealWorld(dc.SCENE_DIR)
    world.set_band(28e9, 400e6)               # memes couleurs de pietons que les rendus de l'etape 1
    rng = np.random.default_rng(args.seed)
    placer = Placer(world, rng)
    print(f"cellules libres {len(placer.cells)}, routes {len(placer.road_cells)}, mobilier {len(placer.clutter_xy)}")
    out.mkdir(parents=True, exist_ok=True)
    open(out / "data.yaml", "w").write(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
                                       + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    stats = {"images": 0, "boites": {c: 0 for c in CLASSES}, "boites_pres_mobilier": 0}
    t0 = time.time()
    jobs = [("train", i) for i in range(args.n_train)] + [("val", i) for i in range(args.n_val)]
    for n, (split, i) in enumerate(jobs):
        img = out / "images" / split / f"{split}_{i:05d}.png"
        lbl = out / "labels" / split / f"{split}_{i:05d}.txt"
        img.parent.mkdir(parents=True, exist_ok=True)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        placed = placer.random_layout()           # tirage fait meme si l'image existe : reprise reproductible
        cam, tgt, fov = random_camera(world, placer, placed, rng)
        light = float(rng.uniform(0.85, 1.2))
        if img.exists() and lbl.exists():
            continue
        R = look_at_rotation(cam, tgt)
        f_px = (IMG_W / 2) / np.tan(np.radians(fov) / 2)
        labs = labels_for(world, placed, cam, R, f_px, np.random.default_rng(100000 + n))
        world.scene.render_to_file(camera=Camera(position=cam.tolist(), look_at=tgt.tolist()), filename=str(img),
                                   resolution=[IMG_W, IMG_H], num_samples=SPP, fov=fov, lighting_scale=light,
                                   show_devices=False)
        with open(lbl, "w") as f:
            for c, xc, yc, w_, h_, _ in labs:
                f.write(f"{c} {xc:.6f} {yc:.6f} {w_:.6f} {h_:.6f}\n")
        stats["images"] += 1
        for c, *_, a in labs:
            stats["boites"][CLASSES[c]] += 1
            if c == 1 and len(placer.clutter_xy) and \
                    np.min(np.linalg.norm(placer.clutter_xy - np.array(placed[a]), axis=1)) < 4.5:
                stats["boites_pres_mobilier"] += 1
        if n % 50 == 0:
            el = time.time() - t0
            print(f"{n:5d}/{len(jobs)} {split}  {el:6.0f} s  ~{el / max(stats['images'], 1) * (len(jobs) - n - 1):6.0f} s restantes  "
                  f"boites {stats['boites']}", flush=True)
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print("termine :", stats)


if __name__ == "__main__":
    main()
