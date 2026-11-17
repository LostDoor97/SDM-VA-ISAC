"""Export Blender -> projet DT beamforming (a lancer DANS Blender 3.6, onglet Scripting > Run Script).

Exporte, a cote du .blend (dossier scene_reelle/) :
  - meshes_dt/<objet>-<materiau>.ply : chaque objet de la collection "DT_nouveaux", en coordonnees
    monde, decoupe par materiau ITU (PLY binaire, triangles) ;
  - trajectoires_blender.json : les trajectoires (points dans l'ordre de parcours, vitesse), la liste
    des objets exportes et la position des acteurs existants.

Conventions (souples) :
  - un objet de DT_nouveaux est un ACTEUR si son nom (ou celui de son parent racine) est ped_<x>,
    car<n> ou bus<n> ; tout le reste (lantern1, bus_stop, trashcan, clutter_*...) est du MOBILIER ;
  - les objets enfants (parent) sont regroupes avec leur parent racine ;
  - trajectoire = courbe "traj_<acteur>" OU, a defaut, l'animation par images-cles de l'acteur
    (seule la forme du chemin est gardee ; la vitesse reste celle par defaut ou la propriete
    personnalisee "vitesse" de l'objet) ;
  - les maillages sont exportes a la premiere image de l'animation (position de depart) ;
  - un materiau non ITU est converti automatiquement (nom contenant glass/verre/vitre/window ->
    itu_glass, sinon le materiau par defaut de la classe) et signale dans les NOTES.

La scene de base (batiments, sol, routes, voiture et pietons A/B) n'est PAS reexportee : elle
reste celle de phase3/new.xml, a l'identique, pour garder la continuite avec les phases 3 a 5.
"""
import json
import os
import re

import bpy
import numpy as np
from mathutils import Vector
from mathutils.geometry import interpolate_bezier

NEW_COLLECTION = "DT_nouveaux"
ITU_TYPES = {"concrete", "brick", "plasterboard", "wood", "glass", "ceiling_board", "chipboard",
             "plywood", "marble", "floorboard", "metal", "very_dry_ground", "medium_dry_ground",
             "wet_ground"}
DEFAULT_SPEED = {"car": 8.0, "bus": 6.0, "ped": 1.4}           # m/s
N_FRAMES, FRAME_S = 100, 0.1
# Acteurs deja presents dans phase3/new.xml (centre XY de leur boite englobante)
EXISTING = {"car": (-34.09, 92.72), "ped_a": (-46.36, 60.76), "ped_b": (-51.80, 106.96)}
EXISTING_BLEND_OBJ = {"car": "car_body", "ped_a": "man_in_suit", "ped_b": "man_in_suit.001"}
ACTOR_RE = re.compile(r"^(ped_[A-Za-z0-9]+|car\d*|bus\d*)$")
GLASS_RE = re.compile(r"glass|verre|vitre|window|fenetre|windshield", re.I)
FALLBACK = {"ped": "wood", "car": "metal", "bus": "metal", "decor": "metal"}

assert bpy.data.filepath, "Enregistre d'abord le .blend dans 'Digital Twin assisted beamforming design/scene_reelle/'"
if bpy.context.object and bpy.context.object.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")
OUT = bpy.path.abspath("//")
MESH_DIR = os.path.join(OUT, "meshes_dt")
os.makedirs(MESH_DIR, exist_ok=True)
problems, report, notes = [], [], []


def loc_keys(obj):
    ad = obj.animation_data
    if not (ad and ad.action):
        return []
    return sorted({k.co[0] for fc in ad.action.fcurves if fc.data_path == "location"
                   for k in fc.keyframe_points})


scn = bpy.context.scene
coll = bpy.data.collections.get(NEW_COLLECTION)
_cands = (list(coll.all_objects) if coll else []) + [bpy.data.objects.get(n) for n in EXISTING_BLEND_OBJ.values()]
_starts = [loc_keys(o)[0] for o in _cands if o is not None and loc_keys(o)]
FRAME0 = int(min([scn.frame_start] + _starts))
FRAME_USER = scn.frame_current
scn.frame_set(FRAME0)                      # maillages exportes a la position de depart
deps = bpy.context.evaluated_depsgraph_get()


def itu_type(mat):
    if mat is None:
        return None
    base = mat.name.split(".")[0]
    t = base[4:] if base.startswith("itu_") else None
    return t if t in ITU_TYPES else None


def world_bbox(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return [min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]


def write_ply(path, V, F):
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        f.write(np.asarray(V, "<f4").tobytes())
        rec = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)])
        rec["n"], rec["i"] = 3, F
        f.write(rec.tobytes())


def export_object(obj, kind):
    ev = obj.evaluated_get(deps)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    mw = ev.matrix_world
    V = np.array([tuple(mw @ v.co) for v in me.vertices], dtype=np.float64).reshape(-1, 3)
    tris = np.array([t.vertices[:] for t in me.loop_triangles], dtype=np.int64).reshape(-1, 3)
    mats = np.array([t.material_index for t in me.loop_triangles], dtype=np.int64)
    slots = [s.material for s in obj.material_slots]
    by_type = {}
    for k in np.unique(mats):
        mat = slots[k] if k < len(slots) else None
        t = itu_type(mat)
        if t is None:
            name = mat.name if mat else "(aucun)"
            t = "glass" if mat and GLASS_RE.search(name) else FALLBACK[kind]
            notes.append(f"{obj.name}: materiau '{name}' -> itu_{t} (auto)")
        by_type.setdefault(t, []).append(tris[mats == k])
    out = []
    for t, parts in by_type.items():
        T = np.concatenate(parts)
        used, inv = np.unique(T.ravel(), return_inverse=True)
        fname = f"{obj.name.replace('.', '_')}-itu_{t}.ply"
        write_ply(os.path.join(MESH_DIR, fname), V[used], inv.reshape(-1, 3))
        out.append({"materiau": t, "ply": f"meshes_dt/{fname}", "triangles": int(len(T))})
    ev.to_mesh_clear()
    return out


def curve_points(obj):
    mw = obj.matrix_world
    if obj.type == "CURVE":
        sp = obj.data.splines[0]
        if len(obj.data.splines) > 1:
            problems.append(f"{obj.name}: {len(obj.data.splines)} splines, seule la premiere est utilisee")
        if sp.type == "BEZIER":
            pts = []
            for a, b in zip(sp.bezier_points[:-1], sp.bezier_points[1:]):
                seg = interpolate_bezier(a.co, a.handle_right, b.handle_left, b.co, 16)
                pts.extend(seg if not pts else seg[1:])
        else:
            pts = [Vector(p.co[:3]) for p in sp.points]
    else:                                       # maillage : chaine de sommets reliee par des aretes
        me = obj.data
        adj = {v.index: [] for v in me.vertices}
        for e in me.edges:
            a, b = e.vertices
            adj[a].append(b)
            adj[b].append(a)
        ends = [i for i, n in adj.items() if len(n) == 1]
        if len(ends) != 2:
            problems.append(f"{obj.name}: le maillage n'est pas une simple ligne (extremites : {len(ends)})")
            return []
        order, prev, cur = [ends[0]], None, ends[0]
        while True:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            order.append(cur)
        pts = [me.vertices[i].co for i in order]
    return [list(mw @ Vector(p)) for p in pts]


def kind_of(actor):
    return "car" if actor.startswith("car") else "bus" if actor.startswith("bus") else "ped"


# ---- 1. objets nouveaux (regroupes par parent racine) -------------------------------------------
objects = []
if coll is None:
    problems.append(f"collection '{NEW_COLLECTION}' introuvable (as-tu enregistre le .blend ? Ctrl+S)")
else:
    members = set(coll.all_objects)

    def root_of(o):
        while o.parent is not None and o.parent in members:
            o = o.parent
        return o

    groups = {}
    for obj in coll.all_objects:
        if obj.name.startswith("traj_"):
            continue
        groups.setdefault(root_of(obj), []).append(obj)
    for root, objs in sorted(groups.items(), key=lambda kv: kv[0].name):
        role = "acteur" if ACTOR_RE.match(root.name) else "decor"
        kind = kind_of(root.name) if role == "acteur" else "decor"
        meshes = [o for o in objs if o.type == "MESH"]
        files, boxes = [], []
        for o in meshes:
            fs = export_object(o, kind)
            if fs:
                files += fs
                boxes.append(world_bbox(o))
        if not files:
            problems.append(f"{root.name}: aucune face exportable (objet vide, ligne ou courbe ?) - ignore ; "
                            f"si c'est une trajectoire, renomme-le traj_<acteur>, sinon supprime-le")
            continue
        lo = [min(b[0][i] for b in boxes) for i in range(3)]
        hi = [max(b[1][i] for b in boxes) for i in range(3)]
        for o in objs:
            if o is not root and loc_keys(o):
                problems.append(f"{o.name}: anime alors que c'est un enfant de {root.name} - anime le parent "
                                f"(les enfants suivent)")
        objects.append({"objet": root.name, "role": role, "fichiers": files, "bbox_min": lo, "bbox_max": hi})
        ntri = sum(f["triangles"] for f in files)
        report.append(f"  objet {root.name:20s} {role:7s} {len(meshes):2d} maillage(s) {ntri:7d} tri "
                      f"{sorted({f['materiau'] for f in files})} centre ({(lo[0]+hi[0])/2:7.1f}, "
                      f"{(lo[1]+hi[1])/2:7.1f}) z {lo[2]:.2f}..{hi[2]:.2f} m")
        if lo[2] > 0.8 or lo[2] < -0.5:
            problems.append(f"{root.name}: le bas de l'objet est a z = {lo[2]:.2f} m (flotte ou enterre ?)")
        if ntri > 200000:
            notes.append(f"{root.name}: {ntri} triangles - lourd pour le lancer de rayons ; un modificateur "
                         f"Decimate (ratio 0.2) suffit")

# ---- 2. trajectoires ----------------------------------------------------------------------------
actor_centres = dict(EXISTING)
for o in objects:
    if o["role"] == "acteur":
        actor_centres[o["objet"]] = ((o["bbox_min"][0] + o["bbox_max"][0]) / 2,
                                     (o["bbox_min"][1] + o["bbox_max"][1]) / 2)
trajs = {}


def add_traj(actor, pts, speed, source):
    P = np.array(pts)
    length = float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum())
    trajs[actor] = {"points": pts, "longueur_m": length, "vitesse_m_s": speed, "source": source}
    need = speed * N_FRAMES * FRAME_S
    msg = f"  traj  {actor:10s} {len(pts):4d} pts, {length:6.1f} m, {speed:.1f} m/s (besoin {need:.0f} m) [{source}]"
    if actor in actor_centres:
        d0 = float(np.hypot(*(P[0, :2] - np.array(actor_centres[actor]))))
        msg += f", depart a {d0:.1f} m du centre"
        if d0 > 2.0 and source.startswith("courbe"):
            problems.append(f"traj_{actor}: le premier point est a {d0:.1f} m de l'acteur (dessine la courbe "
                            f"a partir de l'acteur, dans le sens du deplacement)")
    else:
        problems.append(f"trajectoire de '{actor}': aucun acteur de ce nom (attendus : {sorted(actor_centres)})")
    if length < need:
        msg += "  <- trop courte : l'acteur s'arretera au bout"
    report.append(msg)


# 2a. courbes traj_<acteur>
for obj in bpy.data.objects:
    if not obj.name.startswith("traj_") or obj.type not in {"CURVE", "MESH"}:
        continue
    actor = obj.name[5:].split(".")[0]
    pts = curve_points(obj)
    if len(pts) < 2:
        problems.append(f"{obj.name}: moins de 2 points")
        continue
    add_traj(actor, pts, float(obj.get("vitesse", DEFAULT_SPEED[kind_of(actor)])), "courbe")

# 2b. animation par images-cles (acteurs sans courbe)
anim_src = {o["objet"]: bpy.data.objects[o["objet"]] for o in objects if o["role"] == "acteur"}
anim_src.update({a: bpy.data.objects.get(n) for a, n in EXISTING_BLEND_OBJ.items()})
for actor, ob in anim_src.items():
    keys = loc_keys(ob) if ob else []
    if not keys:
        continue
    if actor in trajs:
        notes.append(f"{actor}: a une courbe traj_ ET des images-cles - la courbe est prioritaire")
        continue
    pts = []
    for fr in range(int(keys[0]), int(keys[-1]) + 1):
        scn.frame_set(fr)
        p = list(ob.matrix_world.translation)
        if not pts or np.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 0.05:
            pts.append(p)
    if len(pts) < 2:
        notes.append(f"{actor}: images-cles mais aucun deplacement - considere immobile")
        continue
    add_traj(actor, pts, float(ob.get("vitesse", DEFAULT_SPEED[kind_of(actor)])),
             f"images-cles {int(keys[0])}-{int(keys[-1])}")
scn.frame_set(FRAME_USER)

static_actors = [a for a in actor_centres if a not in trajs]

# ---- 3. acteurs existants (controle de non-deplacement) -----------------------------------------
existing = {}
for cname in ("Car", "Pedestrians"):
    c = bpy.data.collections.get(cname)
    if c:
        for obj in c.all_objects:
            if obj.type == "MESH" and (coll is None or obj.name not in coll.all_objects):
                lo, hi = world_bbox(obj)
                existing[obj.name] = {"bbox_min": lo, "bbox_max": hi}

manifest = {"blend": bpy.data.filepath, "n_frames": N_FRAMES, "frame_s": FRAME_S, "frame_export": FRAME0,
            "trajectoires": trajs, "acteurs_immobiles": static_actors, "objets": objects,
            "acteurs_existants_blend": existing}
with open(os.path.join(OUT, "trajectoires_blender.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)

print("\n=== Export DT beamforming ===")
print(f"  maillages exportes a l'image {FRAME0}")
print("\n".join(report))
print(f"  acteurs immobiles : {static_actors}")
print(f"  -> {os.path.join(OUT, 'trajectoires_blender.json')}")
if notes:
    print("\nNOTES (a verifier, pas bloquant) :")
    for n in notes:
        print("  - " + n)
print("\nPROBLEMES :" if problems else "\nAucun probleme detecte.")
for p in problems:
    print("  - " + p)
