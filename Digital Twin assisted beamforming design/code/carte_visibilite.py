"""Carte de visibilite des 4 BS au niveau de la rue (z = 1,3 m), pour guider le trace des
trajectoires dans Blender.

Usage : .venv\\Scripts\\python.exe "Digital Twin assisted beamforming design\\code\\carte_visibilite.py"
Sortie : etape1/carte_visibilite_BS.png et etape1/carte_visibilite_par_BS.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import mitsuba as mi

mi.set_variant("cuda_ad_mono_polarized")
from sionna.rt import load_scene  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "etape1"
OUT.mkdir(exist_ok=True)
SCENE_XML = ROOT.parent / "phase3" / "new.xml"

import json  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dt_common import BS_POS  # noqa: E402

NB = len(BS_POS)
ACTORS = {"car": (-34.1, 92.7), "ped_a": (-46.4, 60.8), "ped_b": (-51.8, 107.0)}
# Trajectoires exportees de Blender (si disponibles) : partie parcourue en 10 s en trait plein
MANIFEST = ROOT / "scene_reelle" / "trajectoires_blender.json"
TRAJS = {}
if MANIFEST.exists():
    _man = json.load(open(MANIFEST, encoding="utf-8"))
    for _o in _man["objets"]:
        if _o["role"] == "acteur":
            ACTORS[_o["objet"]] = ((_o["bbox_min"][0] + _o["bbox_max"][0]) / 2,
                                   (_o["bbox_min"][1] + _o["bbox_max"][1]) / 2)
    TRAJS = {a: (np.array(t["points"])[:, :2], t["vitesse_m_s"] * _man["n_frames"] * _man["frame_s"])
             for a, t in _man["trajectoires"].items()}
UE_Z, STEP = 1.3, 1.0
X0, X1, Y0, Y1 = -176.0, 202.0, -186.0, 170.0

scene = load_scene(str(SCENE_XML), merge_shapes=False)
gx, gy = np.meshgrid(np.arange(X0, X1, STEP) + STEP / 2, np.arange(Y0, Y1, STEP) + STEP / 2)
shape = gx.shape
fx, fy = gx.ravel().astype(np.float32), gy.ravel().astype(np.float32)
n = fx.size

# Hauteur de la premiere surface vue du ciel
si = scene.mi_scene.ray_intersect(mi.Ray3f(o=mi.Point3f(mi.Float(fx), mi.Float(fy), mi.Float(np.full(n, 200.0, np.float32))),
                                           d=mi.Vector3f(0.0, 0.0, -1.0)))
hz = np.where(np.array(si.is_valid()), np.array(si.p.z), 0.0).reshape(shape)
outdoor = hz < 2.5                      # sol, route, trottoir (les acteurs font < 2,5 m)

# Ligne de vue UE -> BS
vis = {}
for name, p in BS_POS.items():
    d = np.c_[p[0] - fx, p[1] - fy, np.full(n, p[2] - UE_Z)].astype(np.float32)
    dist = np.linalg.norm(d, axis=1)
    d /= dist[:, None]
    o = np.c_[fx, fy, np.full(n, UE_Z, np.float32)] + 0.05 * d
    ray = mi.Ray3f(o=mi.Point3f(*[mi.Float(o[:, i].copy()) for i in range(3)]),
                   d=mi.Vector3f(*[mi.Float(d[:, i].copy()) for i in range(3)]))
    s = scene.mi_scene.ray_intersect(ray)
    blocked = np.array(s.is_valid()) & (np.array(s.t) < dist - 0.5)
    vis[name] = (~blocked).reshape(shape) & outdoor
count = sum(v.astype(int) for v in vis.values())

ext = [X0, X1, Y0, Y1]


def base(ax, title):
    bld = np.ma.masked_where(outdoor, hz)
    ax.imshow(bld, origin="lower", extent=ext, cmap="Greys", vmin=-20, vmax=60, interpolation="nearest")
    road = np.ma.masked_where(~(outdoor & (hz > 0.1)), np.ones(shape))
    ax.imshow(road, origin="lower", extent=ext, cmap=ListedColormap(["#000000"]), alpha=0.18,
              interpolation="nearest")
    for nme, p in BS_POS.items():
        ax.plot(*p[:2], marker="^", ms=13, color="crimson", mec="k")
        ax.annotate(f"{nme}\n({p[0]:.0f}, {p[1]:.0f})", p[:2], xytext=(6, 6), textcoords="offset points",
                    fontsize=10, weight="bold", color="crimson")
    for nme, (P, run) in TRAJS.items():
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        k = min(len(P), max(2, int(np.searchsorted(seg, run)) + 1))
        ax.plot(P[:, 0], P[:, 1], "-", color="royalblue", lw=0.8, alpha=0.4)
        ax.plot(P[:k, 0], P[:k, 1], "-", color="royalblue", lw=2.2)
        ax.plot(*P[k - 1], "x", color="royalblue", ms=9, mew=2)
    for nme, (x, y) in ACTORS.items():
        ax.plot(x, y, "o", ms=7, color="gold", mec="k")
        ax.annotate(nme, (x, y), xytext=(5, -12), textcoords="offset points", fontsize=8)
    ax.set_xticks(np.arange(-160, 201, 20))
    ax.set_yticks(np.arange(-180, 171, 20))
    ax.tick_params(labelsize=8)
    ax.grid(color="k", alpha=0.15, lw=0.6)
    ax.set_xlabel("x [m] (= X Blender)")
    ax.set_ylabel("y [m] (= Y Blender)")
    ax.set_title(title)


cmap = ListedColormap(["#d7301f", "#fdae61", "#fee08b", "#a6d96a", "#1a9641", "#006837"][:NB + 1])
fig, ax = plt.subplots(figsize=(16, 15))
base(ax, "BS en ligne de vue a 1,3 m (gris = batiments) ; bleu = trajectoire, trait plein = parcouru en 10 s, "
         "x = position finale")
im = ax.imshow(np.ma.masked_where(~outdoor, count), origin="lower", extent=ext, cmap=cmap, vmin=-0.5,
               vmax=NB + 0.5, alpha=0.55, interpolation="nearest")
cb = fig.colorbar(im, ax=ax, ticks=range(NB + 1), fraction=0.035, pad=0.01)
cb.set_label("BS visibles")
fig.tight_layout()
fig.savefig(OUT / "carte_visibilite_BS.png", dpi=130)

fig, axes = plt.subplots(2, 3, figsize=(30, 19))
for ax in axes.ravel()[NB:]:
    ax.axis("off")
for ax, (nme, v) in zip(axes.ravel(), vis.items()):
    base(ax, f"Ligne de vue vers {nme} (vert)")
    ax.imshow(np.ma.masked_where(~outdoor, v), origin="lower", extent=ext,
              cmap=ListedColormap(["#d7301f", "#1a9641"]), vmin=0, vmax=1, alpha=0.5, interpolation="nearest")
fig.tight_layout()
fig.savefig(OUT / "carte_visibilite_par_BS.png", dpi=100)

print("part de la rue vue par k BS :", {k: f"{(count[outdoor] == k).mean():.0%}" for k in range(NB + 1)})
print("ecrit :", OUT / "carte_visibilite_BS.png", OUT / "carte_visibilite_par_BS.png")
