"""Module commun du projet Digital Twin assisted beamforming.

Contient le "monde reel" simule : scene Sionna scene_reelle/real.xml, acteurs animes le long des
trajectoires dessinees dans Blender, UE portes par les acteurs, BS, materiaux perturbes, cameras.

Conventions
-----------
- 100 frames a 10 frames/s (FRAME_S = 0,1 s), t = 0..N_FRAMES-1.
- Les acteurs sont des corps rigides : a chaque frame, tous les sommets d'un acteur subissent la
  meme rotation (autour de l'axe vertical passant par son centre initial) puis la meme translation.
  On n'utilise pas SceneObject.orientation, qui ferait tourner chaque piece autour de son propre
  centre et disloquerait la voiture.
- Coordonnees : celles de new.xml, identiques a celles de Blender.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import drjit as dr
import mitsuba as mi

try:
    mi.set_variant("cuda_ad_mono_polarized")
except Exception:                                   # pragma: no cover
    mi.set_variant("llvm_ad_mono_polarized")

from sionna.rt import load_scene, RadioMaterial    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PFA = ROOT.parent
PHASE3_DIR = PFA / "phase3"
SCENE_DIR = ROOT / "scene_reelle"

C0 = 299_792_458.0
N_FRAMES, FRAME_S = 100, 0.1

# ---- BS : les 4 noeuds en toiture des phases 3 a 5 + TX5 (ajoute le 2026-11-10, toit a 24,42 m) ----
BS_POS = {"TX1": np.array([-119.55, 54.933, 20.79 + 1.5]),
          "TX2": np.array([47.9, 110.0, 18.0 + 1.5]),
          "TX3": np.array([43.7, 33.0, 21.8 + 1.5]),
          "TX4": np.array([107.0, -28.6, 19.7 + 1.5]),
          "TX5": np.array([81.951, -140.59, 24.42 + 1.5])}
BS_NAMES = list(BS_POS)
RX_OFFSET = np.array([0.3, 0.3, -0.2])           # reseau de reception radar co-localise

# ---- UE ----------------------------------------------------------------------------------------
UE_PED_HEIGHT = 1.3        # telephone tenu en main
UE_PED_FORWARD = 0.35      # devant le corps : le corps du porteur peut bloquer les trajets arriere
UE_CAR_ABOVE_ROOF = 0.05   # antenne de toit
NO_UE_PREFIX = ("bus",)    # le bus est un obstacle, pas un utilisateur

# ---- Monde reel : perturbation des materiaux statiques (ecart jumeau / realite) ----------------
PERTURB_MATERIALS = ("itu_brick", "itu_ceiling_board", "itu_concrete", "itu_very_dry_ground")
PERTURB_REL = 0.20          # facteur uniforme dans [1 - 0,2 ; 1 + 0,2] sur eps_r et sigma
SCATTERING = {"actor": 0.4, "structure": 0.05, "clutter": 0.2}

# ---- Materiau "peau" des pietons (Gabriel 1996, peau seche), comme en phase 5 -------------------
SKIN = {3.5e9: (37.0, 2.0), 28e9: (16.6, 25.8)}


def actor_class(name: str) -> str:
    if name.startswith("ped"):
        return "pedestrian"
    if name.startswith("bus"):
        return "bus"
    return "car"


def rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Trajectory:
    """Parcours a vitesse constante d'une polyligne XY ; arret au bout."""

    def __init__(self, xy: np.ndarray, speed: float):
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
        self.xy = xy[keep]
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]
        self.length = float(self.s[-1])
        self.speed = float(speed)

    def at_s(self, s):
        s = np.clip(s, 0.0, self.length)
        return np.c_[np.interp(s, self.s, self.xy[:, 0]), np.interp(s, self.s, self.xy[:, 1])]

    def state(self, t: int):
        s = self.speed * t * FRAME_S
        p = self.at_s(s)[0]
        a, b = self.at_s(np.array([s - 1.0, s + 1.0]))       # tangente lissee sur +-1 m
        if np.linalg.norm(b - a) < 1e-6:
            a, b = self.at_s(np.array([self.length - 1.0, self.length]))
        h = (b - a) / np.linalg.norm(b - a)
        moving = s < self.length
        return p, h, (self.speed * h if moving else np.zeros(2))


class RealWorld:
    """Scene "reelle" animee. Une seule instance par processus (elle possede la scene Sionna)."""

    def __init__(self, scene_dir: Path = SCENE_DIR, perturb_seed: int | None = 2026):
        self.scene_dir = Path(scene_dir)
        self.manifest = json.load(open(self.scene_dir / "trajectoires_blender.json", encoding="utf-8"))
        self.scene = load_scene(str(self.scene_dir / "real.xml"), merge_shapes=False,
                                remove_duplicate_vertices=False)
        self.params = self.scene.mi_scene_params

        # -- regroupement des objets par acteur / mobilier ---------------------------------------
        self.actor_objs: dict[str, list] = {}
        self.clutter_objs, self.static_objs = [], []
        for name, obj in self.scene.objects.items():
            if name.startswith("act__"):
                self.actor_objs.setdefault(name.split("__")[1], []).append(obj)
            elif name.startswith("clutter__"):
                self.clutter_objs.append(obj)
            else:
                self.static_objs.append(obj)
        self.actors = sorted(self.actor_objs, key=lambda a: (actor_class(a), a))
        self.ue_actors = [a for a in self.actors if not a.startswith(NO_UE_PREFIX)]
        self.objid_actor = {int(o.object_id): a for a, objs in self.actor_objs.items() for o in objs}
        self.clutter_ids = {int(o.object_id) for o in self.clutter_objs}

        # -- geometrie initiale de chaque acteur ---------------------------------------------------
        self._v0, self.c0, self.top_z = {}, {}, {}
        for a, objs in self.actor_objs.items():
            for o in objs:
                self._v0[o.mi_mesh.id()] = np.array(self.params[o.mi_mesh.id() + ".vertex_positions"]).reshape(-1, 3)
            allv = np.concatenate([self._v0[o.mi_mesh.id()] for o in objs])
            self.c0[a] = (allv.min(0) + allv.max(0)) / 2
            self.top_z[a] = float(allv[:, 2].max())

        # -- trajectoires : la courbe est recalee pour partir exactement du centre de l'acteur ----
        self.traj: dict[str, Trajectory] = {}
        for a, tr in self.manifest["trajectoires"].items():
            if a not in self.actor_objs:
                raise KeyError(f"trajectoire '{a}' sans acteur dans real.xml (acteurs : {self.actors})")
            xy = np.array(tr["points"])[:, :2]
            xy = xy - xy[0] + self.c0[a][:2]
            self.traj[a] = Trajectory(xy, tr["vitesse_m_s"])
        self.yaw0 = {a: float(np.arctan2(*self.traj[a].state(0)[1][::-1])) for a in self.traj}

        # -- materiaux -----------------------------------------------------------------------------
        self._setup_materials(perturb_seed)
        self.frame = None
        self.apply_frame(0)

    # ----------------------------------------------------------------------------------------------
    def _setup_materials(self, seed):
        rm = self.scene.radio_materials
        self.perturbation = {}
        if seed is not None:
            gen = np.random.default_rng(seed)
            for m in PERTURB_MATERIALS:
                if m not in rm:
                    continue
                f_eps, f_sig = 1 + PERTURB_REL * gen.uniform(-1, 1, 2)
                base_cb = rm[m].frequency_update_callback

                def cb(f, base_cb=base_cb, f_eps=f_eps, f_sig=f_sig):
                    eps, sig = base_cb(f)
                    return dr_max1(eps * f_eps), sig * f_sig
                rm[m].frequency_update_callback = cb
                self.perturbation[m] = {"facteur_eps_r": float(f_eps), "facteur_sigma": float(f_sig)}
        for name, mat in rm.items():
            if name.startswith("itu_clutter"):
                mat.scattering_coefficient = SCATTERING["clutter"]
            elif name in ("itu_brick", "itu_ceiling_board"):
                mat.scattering_coefficient = SCATTERING["structure"]
            elif name in ("itu_metal", "itu_glass", "itu_car_wood", "itu_ped_a_wood", "itu_ped_B_wood") \
                    or name.startswith("itu_bus"):
                mat.scattering_coefficient = SCATTERING["actor"]
        # Couleurs de rendu des pietons : celles des pietons A et B de la phase 3 (YOLO y est habitue)
        col = {"a": tuple(float(c) for c in rm["itu_ped_a_wood"].color),
               "b": tuple(float(c) for c in rm["itu_ped_B_wood"].color)}
        peds = [a for a in self.actors if actor_class(a) == "pedestrian"]
        self.ped_color = {p: col["b"] if (p == "ped_b" or (p not in ("ped_a",) and i % 2 == 1)) else col["a"]
                          for i, p in enumerate(peds)}
        self._skin = {}

    def set_band(self, freq_hz: float, bw_hz: float):
        """Frequence porteuse + materiau peau des pietons pour cette bande."""
        self.scene.frequency, self.scene.bandwidth = freq_hz, bw_hz
        eps, sig = SKIN[min(SKIN, key=lambda f: abs(f - freq_hz))]
        for p, color in self.ped_color.items():
            key = (p, freq_hz)
            if key not in self._skin:
                self._skin[key] = RadioMaterial(name=f"skin_{p}_{freq_hz / 1e9:g}GHz", relative_permittivity=eps,
                                                conductivity=sig, scattering_coefficient=SCATTERING["actor"],
                                                color=color)
            for o in self.actor_objs[p]:
                o.radio_material = self._skin[key]

    # ----------------------------------------------------------------------------------------------
    def actor_state(self, t: int) -> dict:
        """Etat vrai des acteurs a la frame t : position (centre de boite), cap, vitesse."""
        out = {}
        for a in self.actors:
            if a in self.traj:
                p, h, v = self.traj[a].state(t)
                yaw = float(np.arctan2(h[1], h[0]))
                out[a] = {"pos": np.r_[p, self.c0[a][2]], "heading": h, "vel": np.r_[v, 0.0],
                          "dyaw": yaw - self.yaw0[a], "moving": bool(np.any(v))}
            else:
                out[a] = {"pos": self.c0[a].copy(), "heading": np.array([1.0, 0.0]), "vel": np.zeros(3),
                          "dyaw": 0.0, "moving": False}
        return out

    def apply_frame(self, t: int) -> dict:
        state = self.actor_state(t)
        for a in self.traj:
            R = rotz(state[a]["dyaw"])
            shift = state[a]["pos"]
            for o in self.actor_objs[a]:
                key = o.mi_mesh.id()
                v = (self._v0[key] - self.c0[a]) @ R.T + shift
                self.params[key + ".vertex_positions"] = mi.Float(v.astype(np.float32).ravel())
        self.params.update()
        self.scene.scene_geometry_updated()
        for a in self.actors:
            vel = mi.Vector3f(*state[a]["vel"].tolist())
            for o in self.actor_objs[a]:
                o.velocity = vel
        self.frame, self.state = t, state
        return state

    def ue_positions(self, t: int | None = None) -> dict:
        """Position de l'antenne de chaque UE (pieton : en main devant lui ; voiture : sur le toit)."""
        st = self.state if t is None or t == self.frame else self.actor_state(t)
        out = {}
        for a in self.ue_actors:
            p = st[a]["pos"]
            if actor_class(a) == "pedestrian":
                xy = p[:2] + UE_PED_FORWARD * st[a]["heading"]
                out[a] = np.r_[xy, UE_PED_HEIGHT]
            else:
                out[a] = np.r_[p[:2], self.top_z[a] + UE_CAR_ABOVE_ROOF]
        return out

    # ----------------------------------------------------------------------------------------------
    def first_hit(self, origins: np.ndarray, targets: np.ndarray):
        """Lancer de rayons origine -> cible. Retourne (bloque, distance du premier obstacle, id objet)."""
        d = targets - origins
        dist = np.linalg.norm(d, axis=1)
        d = d / dist[:, None]
        o = origins + 0.02 * d
        ray = mi.Ray3f(o=mi.Point3f(*[mi.Float(o[:, i].astype(np.float32)) for i in range(3)]),
                       d=mi.Vector3f(*[mi.Float(d[:, i].astype(np.float32)) for i in range(3)]))
        si = self.scene.mi_scene.ray_intersect(ray)
        valid = np.array(si.is_valid()).astype(bool)
        t_hit = np.array(si.t)
        blocked = valid & (t_hit < dist - 0.04)
        hit_ptr = np.array(dr.reinterpret_array(mi.UInt, si.shape))
        if not hasattr(self, "_ptr_to_obj"):
            self._ptr_to_obj = {int(np.array(dr.reinterpret_array(mi.UInt, mi.ShapePtr(ob.mi_mesh)))[0]):
                                int(ob.object_id) for ob in self.scene.objects.values()}
        obj_ids = np.array([self._ptr_to_obj.get(int(p), -2) for p in hit_ptr])
        obj_ids[~blocked] = -1
        return blocked, t_hit, obj_ids

    def blocker_class(self, obj_id: int) -> str:
        if obj_id < 0:
            return "aucun" if obj_id == -1 else "inconnu"
        if obj_id in self.objid_actor:
            return f"acteur:{self.objid_actor[obj_id]}"
        if obj_id in self.clutter_ids:
            return "mobilier"
        return "batiment/sol"

    def summary(self) -> dict:
        return {"acteurs": {a: {"classe": actor_class(a), "centre_initial": self.c0[a].round(2).tolist(),
                                "trajectoire_m": round(self.traj[a].length, 1) if a in self.traj else 0.0,
                                "vitesse_m_s": self.traj[a].speed if a in self.traj else 0.0,
                                "ue": a in self.ue_actors}
                            for a in self.actors},
                "mobilier": [o.name for o in self.clutter_objs],
                "perturbation_materiaux": self.perturbation}


def dedupe_paths(paths) -> dict:
    """Supprime les trajets speculaires / diffractes comptes plusieurs fois par Sionna (en place).

    Sionna RT 1.2.2 reconnait une reflexion deja trouvee par un hachage du plan reflechissant, dont
    l'offset est quantifie a 1 mm. A ~100 m de la source, l'erreur d'intersection en simple precision
    est du meme ordre : des rayons frappant le meme triangle tombent de part et d'autre d'un pas de
    quantification et la meme reflexion est stockee plusieurs fois (jusqu'a 8 copies observees sur le
    bus, +9 a +15 dB parasites). Une chaine sans interaction diffuse est entierement determinee par
    la suite (type, objet, primitive) de ses interactions : on n'en garde qu'une. Les trajets diffus
    sont des echantillons distincts par construction et ne sont pas touches.
    Retourne le nombre de trajets supprimes et la part de puissance qu'ils representaient."""
    inter = np.array(paths.interactions)
    if inter.size == 0:
        return {"supprimes": 0, "part_puissance": 0.0}
    obj = np.array(paths.objects)
    prim = np.array(paths.primitives)
    a_re, a_im = np.array(paths.a[0]), np.array(paths.a[1])
    sa = paths.synthetic_array
    # geometrie : [depth, rx, tx, paths] (synthetique) ou [depth, rx, rx_ant, tx, tx_ant, paths]
    D = inter.shape[0]
    inter2, obj2, prim2 = (x.reshape(D, -1, x.shape[-1]) for x in (inter, obj, prim))
    keep = np.ones(inter2.shape[1:], bool)                    # [links, paths]
    for l in range(inter2.shape[1]):
        seen = set()
        for k in range(inter2.shape[2]):
            it = inter2[:, l, k]
            if not it.any() or (it & 2).any():               # LOS, trajet vide, ou diffus : on garde
                continue
            key = (tuple(it), tuple(obj2[:, l, k]), tuple(prim2[:, l, k]))
            if key in seen:
                keep[l, k] = False
            else:
                seen.add(key)
    keep = keep.reshape(inter.shape[1:])
    if sa:                                                    # a : [rx, rx_ant, tx, tx_ant, paths]
        m = keep[:, None, :, None, :]
    else:
        m = keep
    p_all = float(np.sum(a_re ** 2 + a_im ** 2))
    p_rm = float(np.sum((a_re ** 2 + a_im ** 2) * (~m)))
    paths._a_real = mi.TensorXf(np.ascontiguousarray(a_re * m, dtype=np.float32))
    paths._a_imag = mi.TensorXf(np.ascontiguousarray(a_im * m, dtype=np.float32))
    return {"supprimes": int((~keep).sum()), "part_puissance": p_rm / p_all if p_all > 0 else 0.0}


def dr_max1(x):
    """eps_r >= 1 (contrainte de Sionna), compatible flottant et mi.Float."""
    try:
        return max(float(x), 1.0)
    except TypeError:
        return dr.maximum(x, 1.0)
