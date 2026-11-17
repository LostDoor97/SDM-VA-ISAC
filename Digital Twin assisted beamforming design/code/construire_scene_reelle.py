"""Construit la scene "reelle" du projet DT beamforming.

real.xml = phase3/new.xml (inchange : batiments, sol, routes, voiture, pietons A et B)
         + objets exportes de Blender par blender_export_dt.py (nouveaux acteurs, mobilier urbain).

Chaque forme recoit un identifiant explicite, repris par Sionna comme nom d'objet :
    act__<acteur>__<n>      partie d'un acteur mobile (car, ped_a, ped_b, car2, bus, ped_c, ...)
    clutter__<objet>__<n>   mobilier present dans la realite seulement (jamais dans le jumeau)
    static__<n>             scene statique de new.xml (batiments, sol, routes)

Usage : .venv\\Scripts\\python.exe "Digital Twin assisted beamforming design\\code\\construire_scene_reelle.py" [dossier]
        (dossier par defaut : scene_reelle/, qui doit contenir trajectoires_blender.json et meshes_dt/)
"""
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_XML = ROOT.parent / "phase3" / "new.xml"

# Acteurs de new.xml, reconnus par leur materiau (comme en phase 5)
EXISTING_ACTOR_OF_MAT = {"mat-itu_metal": "car", "mat-itu_glass": "car", "mat-itu_car_wood": "car",
                         "mat-itu_ped_a_wood": "ped_a", "mat-itu_ped_B_wood": "ped_b"}
# Couleur de rendu des nouveaux materiaux (gris pour le mobilier, blanc casse pour le bus)
COLORS = {"clutter": "0.55 0.55 0.58", "bus": "0.85 0.85 0.80", "default": "0.5 0.5 0.5"}


def bsdf(mat_id, itu_type, rgb):
    el = ET.Element("bsdf", {"type": "diffuse", "id": mat_id, "name": mat_id})
    ET.SubElement(el, "string", {"name": "type", "value": itu_type})
    ET.SubElement(el, "rgb", {"value": rgb, "name": "reflectance"})
    return el


def material_for(obj_name, role, itu_type, root, known):
    """Materiau d'un nouvel objet. Les acteurs reprennent les materiaux de new.xml (meme rendu, donc
    meme apparence pour YOLO) ; le bus et le mobilier ont leur propre materiau."""
    if role == "decor":
        mat_id, rgb = f"mat-itu_clutter_{itu_type}", COLORS["clutter"]
    elif obj_name.startswith("bus"):
        mat_id, rgb = f"mat-itu_bus_{itu_type}", COLORS["bus"]
    elif obj_name.startswith("car") and itu_type in ("metal", "glass"):
        return f"mat-itu_{itu_type}"
    elif obj_name.startswith("car") and itu_type == "wood":
        return "mat-itu_car_wood"
    elif obj_name.startswith("ped") and itu_type == "wood":
        return "mat-itu_ped_a_wood"          # remplace par un materiau "peau" dans le code
    else:
        mat_id, rgb = f"mat-itu_dt_{itu_type}", COLORS["default"]
    if mat_id not in known:
        root.insert(len(known) + 1, bsdf(mat_id, itu_type, rgb))
        known.add(mat_id)
    return mat_id


def main(scene_dir):
    scene_dir = Path(scene_dir).resolve()
    man = json.load(open(scene_dir / "trajectoires_blender.json", encoding="utf-8"))
    tree = ET.parse(NEW_XML)
    root = tree.getroot()
    known = {b.get("id") for b in root.findall("bsdf")}
    rel_meshes = Path(os.path.relpath(NEW_XML.parent, scene_dir))

    counts = {}
    for sh in root.findall("shape"):
        fn = sh.find("string[@name='filename']")
        fn.set("value", (rel_meshes / fn.get("value")).as_posix())
        mat = sh.find("ref").get("id")
        actor = EXISTING_ACTOR_OF_MAT.get(mat)
        prefix = f"act__{actor}" if actor else "static"
        counts[prefix] = counts.get(prefix, 0) + 1
        sid = f"{prefix}__{counts[prefix]}"
        sh.set("id", sid)
        sh.set("name", sid)

    added = []
    for o in man["objets"]:
        name = o["objet"].replace(".", "_")
        prefix = f"clutter__{name.removeprefix('clutter_')}" if o["role"] == "decor" else f"act__{name}"
        for k, f in enumerate(o["fichiers"]):
            mat_id = material_for(o["objet"], o["role"], f["materiau"], root, known)
            sid = f"{prefix}__{k + 1}"
            sh = ET.SubElement(root, "shape", {"type": "ply", "id": sid, "name": sid})
            ET.SubElement(sh, "string", {"name": "filename", "value": f["ply"]})
            ET.SubElement(sh, "boolean", {"name": "face_normals", "value": "true"})
            ET.SubElement(sh, "ref", {"id": mat_id, "name": "bsdf"})
            added.append((sid, f["ply"], mat_id))

    ET.indent(tree, "\t")
    out = scene_dir / "real.xml"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print(f"{out} : {sum(counts.values())} formes de new.xml + {len(added)} formes ajoutees")
    for k, v in sorted(counts.items()):
        print(f"  {k:14s} {v} formes")
    for sid, ply, mat in added:
        print(f"  + {sid:32s} {ply:45s} {mat}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "scene_reelle")
