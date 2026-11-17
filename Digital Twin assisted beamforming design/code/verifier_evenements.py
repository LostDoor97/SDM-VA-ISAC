"""Controle rapide du scenario (sans canal ni rendu) : chronologie des evenements, collisions entre
acteurs, BS en ligne de vue par UE et obstacles, frame par frame.

Usage : .venv\\Scripts\\python.exe "Digital Twin assisted beamforming design\\code\\verifier_evenements.py"
Sortie : console + etape1/verification_scenario.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dt_common as dc  # noqa: E402

w = dc.RealWorld(dc.SCENE_DIR)
T = dc.N_FRAMES
A = w.actors

# Empreinte au sol de chaque acteur (demi-longueur / demi-largeur dans son repere propre)
half = {}
for a in A:
    xy = np.concatenate([w._v0[o.mi_mesh.id()][:, :2] for o in w.actor_objs[a]]) - w.c0[a][:2]
    yaw = w.yaw0.get(a, 0.0)
    R = dc.rotz(-yaw)[:2, :2]
    loc = xy @ R.T
    half[a] = np.abs(loc).max(0)        # (demi-longueur selon le cap, demi-largeur)

pos = np.zeros((T, len(A), 2))
hdg = np.zeros((T, len(A), 2))
for t in range(T):
    st = w.actor_state(t)
    for i, a in enumerate(A):
        pos[t, i] = st[a]["pos"][:2]
        hdg[t, i] = st[a]["heading"]


def gap(t, i, j):
    """Distance approx. entre les empreintes rectangulaires de i et j (0 = contact/chevauchement)."""
    d = pos[t, j] - pos[t, i]
    best = np.inf
    for (p, q) in ((i, j), (j, i)):
        h = hdg[t, p] / (np.linalg.norm(hdg[t, p]) + 1e-9)
        n = np.array([-h[1], h[0]])
        dd = (pos[t, q] - pos[t, p])
        loc = np.abs([dd @ h, dd @ n]) - half[A[p]] - half[A[q]].min()
        best = min(best, np.linalg.norm(np.maximum(loc, 0)))
    return best


print("\n=== Acteurs ===")
for i, a in enumerate(A):
    tr = w.traj.get(a)
    moved = np.linalg.norm(pos[-1, i] - pos[0, i])
    print(f"  {a:6s} {dc.actor_class(a):10s} empreinte {2*half[a][0]:.1f} x {2*half[a][1]:.1f} m  "
          f"{'v = %.1f m/s, ' % tr.speed if tr else 'immobile, '}deplacement net {moved:5.1f} m, "
          f"arrive au bout a {'%.1f s' % (tr.length / tr.speed) if tr else '-'}")

print("\n=== Rapprochements entre acteurs (ecart minimal entre empreintes) ===")
close = []
for i in range(len(A)):
    for j in range(i + 1, len(A)):
        g = np.array([gap(t, i, j) for t in range(T)])
        k = int(g.argmin())
        if g[k] < 6.0:
            n_contact = int((g < 0.05).sum())
            close.append({"paire": [A[i], A[j]], "ecart_min_m": round(float(g[k]), 2), "t_s": round(k * dc.FRAME_S, 1),
                          "frames_en_contact": n_contact})
            flag = "  <- CONTACT/CHEVAUCHEMENT" if n_contact else ""
            print(f"  {A[i]:6s} - {A[j]:6s} ecart min {g[k]:5.2f} m a t = {k*dc.FRAME_S:4.1f} s{flag}")

print("\n=== Ligne de vue UE -> BS (geometrique) ===")
ue = w.ue_actors
bs = dc.BS_NAMES
los = np.zeros((T, len(ue), len(bs)), bool)
blk = {}
for t in range(T):
    w.apply_frame(t)
    up = w.ue_positions(t)
    O = np.array([up[u] for u in ue for _ in bs], dtype=np.float32)
    D = np.array([dc.BS_POS[b] for _ in ue for b in bs], dtype=np.float32)
    blocked, _, ids = w.first_hit(O, D)
    los[t] = ~np.asarray(blocked).reshape(len(ue), len(bs))
    for k, (u, b) in enumerate([(u, b) for u in ue for b in bs]):
        if blocked[k]:
            blk[(t, u, b)] = w.blocker_class(int(ids[k]))
events = {}
for j, u in enumerate(ue):
    seen = [tuple(b for m, b in enumerate(bs) if los[t, j, m]) for t in range(T)]
    print(f"  {u}: t=0 voit {list(seen[0]) or 'aucune'}")
    ev = []
    for t in range(1, T):
        if seen[t] != seen[t - 1]:
            lost = sorted(set(seen[t - 1]) - set(seen[t]))
            got = sorted(set(seen[t]) - set(seen[t - 1]))
            why = {b: blk.get((t, u, b), "?") for b in lost}
            ev.append({"t_s": round(t * dc.FRAME_S, 1), "perd": lost, "gagne": got, "obstacle": why})
            print(f"     t={t*dc.FRAME_S:4.1f} s  perd {lost} {why if lost else ''}  gagne {got}")
    actor_blocks = sorted({(b, blk[(t, u, b)]) for t in range(T) for b in bs
                           if (t, u, b) in blk and blk[(t, u, b)].startswith("acteur")})
    if actor_blocks:
        print(f"     bloque par un acteur : {actor_blocks}")
    events[u] = {"bs_visibles_t0": list(seen[0]), "changements": ev,
                 "frames_sans_los": int((~los[:, j].any(1)).sum()),
                 "blocages_par_acteur": [list(x) for x in actor_blocks]}

out = dc.ROOT / "etape1" / "verification_scenario.json"
json.dump({"acteurs": w.summary()["acteurs"], "rapprochements": close, "los": events}, open(out, "w"), indent=1)
print(f"\n-> {out}")
