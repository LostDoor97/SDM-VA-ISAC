# DT assisted beamforming: code for steps 1 and 2

Run everything from `C:\Users\AdminFix\Desktop\PFA` with `.venv\Scripts\python.exe`.

| Order | Script | Role | Time (RTX 4070) |
|---|---|---|---|
| 0 | `carte_visibilite.py` | Map of BS line of sight at 1.3 m, used to draw the trajectories | ~10 s |
| 1 | `blender_export_dt.py` | **Run inside Blender 3.6.** Exports the new objects (PLY) and the `traj_*` curves to `scene_reelle/` | a few s |
| 2 | `construire_scene_reelle.py` | `scene_reelle/real.xml` = `phase3/new.xml` + exported objects, with explicit ids (`act__*`, `clutter__*`, `static__*`) | ~1 s |
| 3 | `etape2_canal_reel.py` | Step 2 / M6: true FR2 channel, 256-beam DFT codebook, best beam and BS, LOS and blockers → `etape2/` | ~5 min |
| 4 | `etape1_jumeau_sequence.py` | Step 1 / M1: radar FR1+FR2, renders and YOLO, phase 5 twin C1–C4 at every frame (causal) → `etape1/` | ~1–1.5 h (cached, can be resumed) |

| — | `verifier_evenements.py` | Quick scenario check (no channel): actor gaps/collisions, LOS changes and blockers per UE → `etape1/verification_scenario.json` | ~1 min |
| 4b | `yolo_v3_donnees.py` | YOLO v3 training set: 1,300 renders of the real scene with random actor placements (never the test sequence), automatic labels, 3 classes | ~40 min |
| 4c | `yolo_v3_entrainement.py [--resume] [--val]` | Fine-tunes YOLO v3 (from v2, 960 px, 60 epochs) on GPU → `yolo_v3/train/weights/best.pt`. Needs the CUDA torch installed in `.venv` (torch 2.13.0+cu130) | ~85 min |
| 4d | `etape1_jumeau_sequence.py --yolo yolo_v3/train/weights/best.pt --imgsz 960 --tag _yolo_v3` | Re-runs step 1 with YOLO v3 (cached renders and radar) → `etape1/*_yolo_v3.*` | ~18 min |
| 5 | `construire_notebook.py --execute` | Builds and runs `DT_beamforming_etapes_1_2.ipynb` from the outputs of steps 1 and 2 (read only) | ~30 s |

`dt_common.py` holds the "real" world: animated scene, UEs, 5 BSs (TX5 added), perturbed materials,
and `dedupe_paths()`, which removes specular/diffracted paths that Sionna RT 1.2.2 stores several times
(plane-hash rounding at 1 mm vs float32 intersection error ~100 m from the source; up to +16 dB on ~1 %
of links). `etape2_v0_avec_doublons/` keeps the first step 2 run without this fix, for comparison.

Steps 3 and 4 are independent of each other. Step 2 does not need perception.
