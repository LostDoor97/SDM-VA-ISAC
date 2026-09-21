# Récapitulatif — SDM-VA-ISAC (Semantic Digital Mapping via Vision-Assisted ISAC)

*Mis à jour pour la réunion du 08/09/2026 — couvre Phases 2, 3 et 4.*

## 1. Contexte du projet (résumé initial)

PFE construit sur deux travaux existants — le *Telecom Brain* AGI-natif (Saad et al.) et
SA-VA-ISAC — en proposant le pont manquant entre les signaux physiques (radar ISAC + vision) et
un World Model causal hyper-dimensionnel. Le pipeline cible comporte 6 modules :

1. **1a/1b — Sensing** : ISAC (Sionna RT → nuage de points radar) + Vision (caméra RGB → YOLO)
2. **2 — Fusion multimodale** : association radar↔vision, objets `O_t = {(r_i, c_i, α_i, b_i)}`
3. **3 — Semantic Digital Map Core** : grille d'occupation bayésienne + encodage Hyper-Dimensionnel (HD)
4. **4 — Risque sémantique + JSCC** : carte de risque, compression pour le backhaul
5/6. **Interface AGI / World Model** : foncteur SemMap → WorldModel_HD, simulation contrefactuelle

**Roadmap en 6 phases** : (1) revue de littérature, (2) scène Sionna RT calibrée, (3) pipeline de
perception multimodale (YOLOv8/SegFormer + fusion KDTree + sensing Sionna), (4) grille bayésienne
3D + encodage HD + visualisation Open3D interactive, (5) risk heatmap + JSCC + beamforming informé,
(6) métriques quantitatives vs baselines.

**État d'avancement (résumé pour la réunion)** : Phases 2, 3 et 4 terminées et validées. Phase 5
pas encore commencée (prochaine étape, cf. §7).

---

## 2. Phase 2 — Environnement de simulation (terminée)

- Scène `scene/scene.xml`, carte OSM du quartier gouvernemental de Rabat, matériaux ITU
  (concrete/marble/metal), bbox X:[-805,904] Y:[-839,761] Z:[0,74m].
- Stack : Sionna 2.0.1 (dépôt local + submodule `sionna-rt`), Mitsuba 3.8.0, DrJIT 1.3.1, PyTorch
  CUDA 12.4 (RTX 4070), venv Python 3.11.9 avec `include-system-site-packages=true`.
- Notebook `sionna_rt_osm_pipeline.ipynb`, kernel Jupyter dédié **`pfa-sionna`** (pas le `python3`
  par défaut, qui pointe vers le Python global).
- API Sionna 2.x notées : `mi.set_variant("cuda_ad_mono_polarized")` (pas `cuda_ad_rgb`),
  `importlib.metadata.version("sionna")` (pas `sionna.__version__`), `PlanarArray(pattern="tr38901")`,
  `scene.add(tx)` (pas setattr), CIR via `paths.cir(...)`.

---

## 3. Phase 3 — Pipeline de perception multimodale

### 3.1 Itération 1 — scène `scene2.xml` (archivée, historique complet conservé)

Scène dynamique (voiture + 2 piétons + lampadaire), pipeline Module 1a→1b→2→3 + extension
dynamique multi-frame. **Tout le diagnostic et les correctifs de cette itération restent valables
comme méthodologie**, même si la scène elle-même a depuis été reconstruite :

- **Écart de domaine sim-vers-réel** : YOLOv8 (poids COCO stock) ne détectait rien sur les rendus
  Mitsuba — cause racine identifiée dans le code source de Sionna
  (`visual_scene_from_wireless_scene` remplace **tout** BSDF par un diffus plat de la couleur du
  matériau radio — les textures ne sont jamais visibles, quel que soit ce qui est peint dans
  Blender). **Solution retenue** : fine-tuner YOLOv8 directement sur des rendus auto-annotés générés
  par ce même pipeline (labels calculés par projection pinhole de la bbox 3D connue de chaque acteur
  — aucune annotation manuelle) plutôt que de chercher le photoréalisme.
- **Bug caméra figée** : lors de l'extension dynamique, la caméra restait sur la pose de la frame 0
  → la voiture sortait du cadre après ~6 frames. Corrigé en recalculant position ET visée à chaque
  frame à partir de la position courante des acteurs.
- **Radar aveugle aux acteurs dynamiques** : `scattering_coefficient` par défaut = 0 pour tous les
  matériaux ITU → la réflexion spéculaire seule ne peut pas "voir" de petites cibles irrégulières.
  Corrigé en fixant un coefficient de diffusion non nul (0.3–0.4) sur les matériaux des acteurs.
- **Fuite de vérité terrain côté ISAC** : le fallback matériau mappait directement voiture/piéton
  sur leurs classes fines — irréaliste (un radar réel ne connaît que sa réponse RF) et ça aurait
  faussé $\Delta_{VA}$ (gain vision-assist, Module 6). Corrigé avec une classe générique `target`
  pour le fallback ISAC, seule la vision produisant les étiquettes fines.

### 3.2 Itération 2 — nouvelle scène `phase3/new.xml`, architecture dual-nœud

Scène entièrement reconstruite (import BlOSM + véhicule/piétons), avec les exigences Sionna prises
en compte dès la conception. Bugs de construction rencontrés et corrigés en cours de route :
matériaux non conformes à la convention `itu_*` (`load_scene()` échouait), fusion involontaire de
2 piétons + voiture partageant un matériau Blender (soudés en un seul maillage de 46m par
l'exporteur — corrigé en donnant un matériau Blender dédié à chaque acteur, reconnecté aux
propriétés EM ITU réelles via `fix_new_export.py`), routes exportées en instances Mitsuba non
supportées par Sionna (corrigé via `Object → Convert → Mesh`), TX positionnés exactement à la
surface du toit (auto-intersection quasi-totale — corrigé avec +1.5m de clearance de mât), rendu
caméra entièrement noir quand la caméra est co-localisée avec sa TX (`show_devices=False` requis).

**Architecture initiale** : 2 nœuds de sensing larges (`bs0`/TX1, `bs1`/TX2, monostatiques, caméra
co-localisée cadrée sur le centroïde des 3 acteurs) + 2 nœuds capteurs légers dédiés piétons
(récepteur + caméra seulement, pas de BS complète — résout à la fois la densité de points radar et
la résolution visuelle qu'aucun FOV ne peut corriger seul à ~85-105m de distance).

**Détecteur vision** : YOLOv8n fine-tuné (v2, `gen_yolo_dataset_v2.py` + `finetune_yolo_v2.py`) —
mAP50=0.965, mAP50-95=0.887.

**Fusion (Module 2)** : implémentée avec **association par KDTree** (`scipy.spatial.cKDTree` sur
les centroïdes 2D des détections YOLO par nœud) — satisfait littéralement la méthode nommée dans le
plan initial (une première version utilisait un test point-dans-bbox, fonctionnellement équivalent
mais pas la technique nommée). Garde-fou anti-confusion de profondeur : la vision ne raffine que
les points déjà classés `target` par l'ISAC.

**Module 3** : grille bayésienne 2D top-down (résolution 10m) + encodage HD simplifié (D=2000,
vecteurs aléatoires bipolaires), mise à jour bayésienne + facteur d'oubli γ=0.98.

**Extension dynamique** : 6 frames (voiture + 2 piétons en mouvement), pose de **chaque** nœud
recalculée à chaque frame dès la conception (évite le bug de la première itération).

### 3.3 Extension à 4 nœuds BS — TX3/TX4 (`phase3_new_scene_presentation_v3.ipynb`)

Les 4 positions candidates identifiées dès la revue de scène initiale n'avaient été déployées qu'à
moitié (TX1/TX2) ; TX3 et TX4 ont ensuite été ajoutés comme nœuds BS complets (mêmes clearance de
mât +1.5m, même caméra co-localisée cadrée sur le centroïde des acteurs). Architecture finale :
**4 nœuds BS + 2 capteurs piétons dédiés = 6 nœuds de sensing.**

- Nuage de points radar : 8 038 → **20 899 points** (richesse multistatique bien plus grande).
- Objets fusionnés : voiture 165→**315**, piéton 16→**37**.
- **TX3** fonctionne comme TX1/TX2 (détection vision confirmée, ~85-120m des acteurs).
- **TX4** (~180-210m, le plus éloigné) : **0 détection vision** — la caméra de ce nœud est
  occultée par un bâtiment au premier plan (confirmé visuellement sur le rendu
  `module1b_camera_tx4.png`), pas seulement une question de distance/résolution comme envisagé au
  départ. TX4 reste utile pour la diversité radar multistatique mais n'apporte pas de raffinement
  vision — documenté honnêtement comme limitation réaliste de déploiement (un site candidat n'a pas
  toujours une ligne de vue dégagée sur toutes les cibles), pas masqué ni corrigé artificiellement.
- 100% des points `target` (352/352) restent correctement raffinés en `car`/`pedestrian`, zéro
  contamination des points statiques — la garde-fou anti-confusion de profondeur tient toujours à
  4 nœuds.

**Modules 4/5** : squelettes d'interface définis (JSCC, foncteur AGI), non implémentés — hors
périmètre de cette phase, dépendent d'un modèle entraîné / d'une interface externe.

---

## 4. Fichiers livrables — Phase 3 (dans `phase3/`)

| Fichier | Rôle |
|---|---|
| `new.xml` + `meshes/` | Scène Mitsuba (4 nœuds BS + 2 capteurs piétons) |
| `fix_new_export.py` | Patch post-export Blender (matériaux radio personnalisés → EM ITU) |
| `gen_yolo_dataset_v2.py`, `finetune_yolo_v2.py` | Génération dataset auto-annoté + fine-tuning YOLO |
| `yolo_finetune_v2/weights/best.pt` | Poids YOLOv8 fine-tunés (mAP50=0.965) |
| `phase3_new_scene.ipynb` | Pipeline core (Modules 1a→2→3), dual-nœud |
| `phase3_new_scene_report.ipynb` | + Modules 4/5 squelettes + extension dynamique, narratif technique complet |
| `phase3_new_scene_presentation_v2.ipynb` | Version recadrée (choix de conception, pas bugs), dual-nœud, fusion KDTree |
| `phase3_new_scene_presentation_v2.pdf` (+ `.tex`) | Export PDF (XeLaTeX) de la v2 |
| `phase3_new_scene_presentation_v3.ipynb` | **Version de référence actuelle** — 4 nœuds BS (TX1-TX4) + 2 capteurs piétons |
| `phase3_new_scene_kdtree.ipynb` | Copie technique de référence (fusion KDTree isolée, dual-nœud) |
| `module2_fused_objects_v2.json` | Objets fusionnés 4 nœuds (20 896 points) — réutilisé tel quel en Phase 4 |
| `archive_old_scene/` | Itération 1 complète (scène, notebook, dataset, modèle, résultats) — conservée intacte |

---

## 5. Phase 4 — Grille sémantique 3D, encodage HD complet, visualisation interactive

Construit sur la sortie validée de la Phase 3 (4 nœuds BS + 2 capteurs piétons, fusion KDTree) —
reprend les Modules 1a/1b/2 tels quels, livre les 3 cibles visées par le plan pour cette phase.

### 5.1 Module 3 — grille bayésienne 3D

Extension directe du mécanisme 2D top-down de la Phase 3 : indexation (Z,Y,X,C), résolution
10m XY / 3m Z (grille 9×169×179, 1 449 cellules observées sur 272 259).

### 5.2 Encodage Hyper-Dimensionnel complet

`torchhd` non installable proprement (nom PyPI ambigu) → implémenté en NumPy pur : hypervecteurs
de niveau (level hypervectors) bind/bundle/permute pour une position continue.

- **Piège #1 (corrigé)** : la plage des niveaux calée sur la bbox complète de la scène (~1800m)
  écrasait le signal utile (les données réelles tiennent dans ~250m) → aliasing de position.
  Corrigé en calant la plage sur l'étendue **réellement observée**.
- **Piège #2 (corrigé)** : un bundle plat (position, classe) par cellule laisse une classe
  nombreuse (façade, ~800 cellules) noyer une classe rare (voiture, 1-3 cellules) sous le seuillage
  en signe — limitation de capacité HDC réelle, pas un bug. Corrigé en bundlant les positions
  **par classe d'abord**, puis en lieant chaque bundle à son hypervecteur de classe (une seule
  contribution par classe à la carte, quel que soit le nombre de cellules brutes).
- **Résultat validé** : requête position→classe par similarité cosinus retrouve correctement `car`
  (0.379, devant ground 0.282, pedestrian 0.332, facade 0.196).

### 5.3 Fusion multi-BS par addition de vecteurs HD

Démontre le mécanisme visé par le plan : chaque nœud BS encode sa propre carte en un vecteur
compact (bipolaire), la fusion = simple **addition** des vecteurs (pas de retransmission de grille
brute — c'est tout l'intérêt de l'encodage HD pour le backhaul).

- Avec 2 nœuds (bs0+bs1), la fusion battait chaque nœud seul (validé).
- **En passant à 4 nœuds**, ce n'est plus vrai : bs2 (TX3) seul = 0.752, la fusion des 4 = 0.617
  (même après une fusion pondérée par la confiance de la caméra propre de chaque nœud, qui
  descend TX4 à un poids de 0.30 — la fusion pondérée descend même légèrement à 0.597). **Ce n'est
  pas un bug** : chaque vecteur par nœud est déjà seuillé en bipolaire avant sommation (contrainte
  de compression backhaul), et avec seulement 3-4 vecteurs à sommer, les égalités de vote
  majoritaire sont fréquentes — limitation de capacité du bundling HDC bipolaire au-delà de 2-3
  nœuds, documentée explicitement dans le notebook et le JSON de résumé plutôt que masquée.
- **Important** : cette limitation concerne **uniquement cette démo de compression HD**, pas le
  jumeau numérique lui-même — la grille bayésienne 3D (§5.1) intègre déjà correctement les 6 nœuds
  via mise à jour bayésienne directe, sans passer par ce seuillage bipolaire.

### 5.4 Visualisation

- **Open3D** (nommé explicitement dans le plan initial) : rendu hors-écran (`Visualizer(visible=False)`
  + `capture_screen_image`) pour l'exécution automatisée du notebook — nuage de points coloré par
  classe + 6 sphères pour les nœuds. Snippet `draw_geometries(...)` fourni en commentaire pour une
  exploration interactive manuelle (fenêtre bloquante, à lancer à la main).
- **Plotly, notebook séparé** (`phase4_interactive_plotly.ipynb`) : vue interactive **intégrée
  directement dans la sortie de cellule** — rotation/zoom/pan à la souris, légende cliquable par
  classe et par type de nœud, sans fenêtre externe. Un simple *Run All* suffit (contrairement au
  rendu Open3D, statique par conception, ou à `draw_geometries()`, qui bloquerait l'exécution
  automatisée). Recharge directement `module2_fused_objects_v2.json`, pas besoin de relancer
  Sionna RT / YOLO.
- Une version web autonome (Three.js, consultable dans un navigateur sans Jupyter) a aussi été
  produite à titre de démonstration partageable, hors dépôt du projet.

---

## 6. Fichiers livrables — Phase 4 (dans `phase4/`)

| Fichier | Rôle |
|---|---|
| `phase4_semantic_map.ipynb` | Pipeline complet : grille 3D, encodage HD, fusion multi-BS pondérée, Open3D |
| `phase4_interactive_plotly.ipynb` | Visualisation 3D interactive alternative (Plotly), intégrée à la sortie de cellule |
| `phase4_summary.json` | Résumé + validation (4 checks Module 3/HD, dont note explicite sur la limite de fusion multi-BS) |
| `module3_semantic_grid3d_beta.npy` | Grille bayésienne 3D sauvegardée |
| `module3_hd_map_vector3d.npy` | Vecteur HD de la carte complète |
| `module3_grid3d_topdown_v2.png`, `module3_open3d_pointcloud.png` | Figures (projection top-down, rendu Open3D) |

---

## 7. Limitations connues / écarts avec le plan initial

- **Fréquence** : le cahier des charges vise 28GHz (mmWave) ; le pipeline tourne à 3.5GHz car
  `itu_very_dry_ground` (matériau du sol) n'est défini par l'ITU-R P.2040 que pour 1–10GHz. Passer
  à 28GHz demande d'abord de changer le matériau du sol pour `itu_concrete` (valide 1–100GHz).
- **Hauteur des BS** : le cahier des charges vise 10m ; les 4 nœuds BS sont montés sur toiture
  (~18-23m), choisis pour leur visibilité directe sur les acteurs. Déploiement rooftop réaliste,
  à documenter/justifier dans le rapport plutôt qu'à corriger.
- **TX4 sans confirmation vision** : occulté par un bâtiment au premier plan (cf. §3.3) — limitation
  de déploiement réaliste, pas une erreur de conception.
- **Fusion multi-BS par addition de vecteurs HD** : ne bat plus le meilleur nœud seul au-delà de
  2-3 nœuds (cf. §5.3) — limitation de capacité du bundling bipolaire HDC, ne concerne que cette
  démonstration de compression, pas le jumeau numérique lui-même.
- **SegFormer** non utilisé (YOLOv8 seul, le plan proposait l'un ou l'autre).
- **Modules 4/5 (Phase 5)** non implémentés — squelettes d'interface seulement, comme prévu par le
  plan pour cette étape du projet.

---

## 8. Prochaine étape

**Phase 5** — carte de risque sémantique (gaussienne autour des cibles, pondérée par
classe/vitesse), JSCC (compression des deltas de carte pour le backhaul/federated learning),
complétion par diffusion des zones occultées, beamforming DRL informé par le risque sémantique.
