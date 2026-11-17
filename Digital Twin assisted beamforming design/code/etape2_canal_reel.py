"""Etape 2 -- M6 : canal reel FR2 et codebook DFT (verite terrain du beamforming).

Pour chaque frame de la sequence (scene "reelle" : mobilier, materiaux perturbes, diffusion diffuse)
et chaque couple BS--UE :
  - canal h_t en 28 GHz sur 16 sous-porteuses reparties sur 400 MHz, reseau UPA 16x16 tr38901 a la
    BS (un panneau par BS, oriente vers la zone d'interet), antenne isotrope a l'UE ;
  - gain des N = 256 faisceaux d'un codebook DFT 2D, moyenne sur la bande :
        G_b = mean_f |w_b^T h_t(f)|^2 ;
  - faisceau optimal (balayage exhaustif), SNR correspondant, BS optimale par UE ;
  - ligne de vue geometrique et nature de l'obstacle (batiment, acteur, mobilier) ;
  - faisceau geometrique (direction BS -> UE), apercu de la reference J0.

Usage : .venv\\Scripts\\python.exe "Digital Twin assisted beamforming design\\code\\etape2_canal_reel.py"
        [--scene DOSSIER] [--out DOSSIER] [--frames N] [--seeds-check K]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dt_common as dc
import mitsuba as mi
from sionna.rt import Transmitter, Receiver, PlanarArray, PathSolver, subcarrier_frequencies
from sionna.rt.utils import rotation_matrix

FREQ, BW = 28e9, 400e6
N_ROWS = N_COLS = 16
N_SUB = 16
P_TX_DBM, NF_DB = 30.0, 7.0
NOISE_DBM = -174.0 + 10 * np.log10(BW) + NF_DB
MAX_DEPTH = 4
SNR_OUTAGE_DB = 0.0
LAM = dc.C0 / FREQ


def dft_codebook(pos_local: np.ndarray, osf: int = 1):
    """Codebook DFT 2D dans le repere du panneau (boresight +x, reseau dans le plan y-z).
    Faisceau b = (i, j) pointe vers u = k_y, v = k_z, grille uniforme sur [-1, 1]."""
    u = -1 + (2 * np.arange(N_COLS * osf) + 1) / (N_COLS * osf)
    v = -1 + (2 * np.arange(N_ROWS * osf) + 1) / (N_ROWS * osf)
    U, V = np.meshgrid(u, v, indexing="ij")
    uv = np.c_[U.ravel(), V.ravel()]
    phase = 2 * np.pi / LAM * (np.outer(uv[:, 0], pos_local[:, 1]) + np.outer(uv[:, 1], pos_local[:, 2]))
    # convention de Sionna verifiee : h_n ~ exp(+j k r_n . u_dep) -> w_n = exp(-j k r_n . u) / sqrt(N)
    W = np.exp(-1j * phase) / np.sqrt(pos_local.shape[0])
    return W.astype(np.complex64), uv


def nearest_beam(uv_grid, k_local):
    if k_local[0] <= 0:                     # derriere le panneau
        return -1
    return int(np.argmin(np.sum((uv_grid - k_local[1:]) ** 2, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(dc.SCENE_DIR))
    ap.add_argument("--out", default=str(dc.ROOT / "etape2"))
    ap.add_argument("--frames", type=int, default=dc.N_FRAMES)
    ap.add_argument("--seeds-check", type=int, default=8, help="frames recalculees avec une autre graine")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    world = dc.RealWorld(Path(args.scene))
    sc = world.scene
    world.set_band(FREQ, BW)
    sc.tx_array = PlanarArray(num_rows=N_ROWS, num_cols=N_COLS, vertical_spacing=0.5, horizontal_spacing=0.5,
                              pattern="tr38901", polarization="V")
    sc.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    pos_local = np.array(sc.tx_array.positions(LAM))
    pos_local = pos_local.T if pos_local.shape[0] == 3 else pos_local
    W, uv = dft_codebook(pos_local)
    B = W.shape[0]

    T = args.frames
    ues = world.ue_actors
    U = len(ues)
    # Zone d'interet = barycentre des positions des UE sur toute la sequence (planification du site)
    target = np.mean([p for t in range(T) for p in world.ue_positions(t).values()], axis=0)
    target[2] = 1.5
    R_bs = {}
    for n, p in dc.BS_POS.items():
        tx = Transmitter(name=n, position=p.tolist())
        sc.add(tx)
        tx.look_at(mi.Point3f(*target.tolist()))
        R_bs[n] = np.array(rotation_matrix(tx.orientation)).reshape(3, 3)
    for a in ues:
        sc.add(Receiver(name=f"ue_{a}", position=world.ue_positions(0)[a].tolist()))
    solver = PathSolver()
    freqs = subcarrier_frequencies(N_SUB, BW / N_SUB)
    NB = len(dc.BS_NAMES)

    G = np.zeros((T, NB, U, B), np.float32)
    H = np.zeros((T, NB, U, N_SUB, W.shape[1]), np.complex64)
    UE_POS = np.zeros((T, U, 3))
    ACT = {a: np.zeros((T, 6)) for a in world.actors}          # x, y, z, vx, vy, cap [rad]
    LOS = np.zeros((T, NB, U), bool)
    BLOCK = np.empty((T, NB, U), dtype=object)
    GEO_BEAM = np.full((T, NB, U), -1, int)

    dedup_log = []

    def trace(seed):
        paths = solver(sc, max_depth=MAX_DEPTH, los=True, specular_reflection=True, diffuse_reflection=True,
                       refraction=False, diffraction=True, edge_diffraction=False, synthetic_array=True,
                       seed=seed)
        dedup_log.append(dc.dedupe_paths(paths))      # trajets comptes plusieurs fois par Sionna (voir dt_common)
        h = paths.cfr(frequencies=freqs, normalize_delays=True, out_type="numpy")   # [rx,1,tx,N,1,F]
        return np.transpose(h[:, 0, :, :, 0, :], (1, 0, 3, 2))                       # [tx, rx, F, N]

    t0 = time.time()
    for t in range(T):
        st = world.apply_frame(t)
        ue_pos = world.ue_positions()
        for a in ues:
            sc.get(f"ue_{a}").position = mi.Point3f(*ue_pos[a].tolist())
        UE_POS[t] = [ue_pos[a] for a in ues]
        for a in world.actors:
            ACT[a][t] = [*st[a]["pos"], *st[a]["vel"][:2], np.arctan2(st[a]["heading"][1], st[a]["heading"][0])]
        h = trace(seed=1000 + t)
        H[t] = h
        G[t] = np.mean(np.abs(np.einsum("bn,trfn->trbf", W, h)) ** 2, axis=-1)
        for i, (n, p) in enumerate(dc.BS_POS.items()):
            blocked, _, oid = world.first_hit(np.repeat(p[None], U, 0), UE_POS[t])
            LOS[t, i] = ~blocked
            BLOCK[t, i] = [world.blocker_class(int(o)) for o in oid]
            for j in range(U):
                k = UE_POS[t, j] - p
                GEO_BEAM[t, i, j] = nearest_beam(uv, R_bs[n].T @ (k / np.linalg.norm(k)))
        if t % 10 == 0 or t == T - 1:
            el = time.time() - t0
            print(f"frame {t:3d}/{T}  {el:6.0f} s ecoulees, ~{el / (t + 1) * (T - t - 1):5.0f} s restantes", flush=True)

    # ---- grandeurs derivees ------------------------------------------------------------------
    best_beam = G.argmax(-1)                                         # [T, NB, U]
    g_best = G.max(-1)
    snr_best = P_TX_DBM + 10 * np.log10(g_best + 1e-30) - NOISE_DBM  # [T, NB, U]
    best_bs = snr_best.argmax(1)                                     # [T, U]
    snr_oracle = snr_best.max(1)                                     # [T, U]
    g_geo = np.where(GEO_BEAM >= 0, np.take_along_axis(G, np.clip(GEO_BEAM, 0, None)[..., None], -1)[..., 0], 0)
    loss_geo = 10 * np.log10(g_best / np.maximum(g_geo, 1e-30))

    # ---- verifications ------------------------------------------------------------------------
    # (1) codebook : en ligne de vue, le faisceau geometrique doit etre (presque) le meilleur
    served = np.take_along_axis(LOS, best_bs[:, None, :], 1)[:, 0]
    los_links = LOS & (GEO_BEAM >= 0)
    geo_ok = float(np.mean(loss_geo[los_links] <= 1.0)) if los_links.any() else float("nan")
    # (2) variabilite de la diffusion diffuse (GPU) : memes frames, autre graine
    seeds_check = {}
    if args.seeds_check:
        frames_chk = np.linspace(0, T - 1, min(args.seeds_check, T)).astype(int)
        same, dg = [], []
        for t in frames_chk:
            world.apply_frame(int(t))
            for j, a in enumerate(ues):
                sc.get(f"ue_{a}").position = mi.Point3f(*UE_POS[t, j].tolist())
            h2 = trace(seed=5000 + int(t))
            G2 = np.mean(np.abs(np.einsum("bn,trfn->trbf", W, h2)) ** 2, axis=-1)
            same.append((G2.argmax(-1) == best_beam[t]).ravel())
            dg.append(np.abs(10 * np.log10(G2.max(-1) / g_best[t])).ravel())
        same, dg = np.concatenate(same), np.concatenate(dg)
        seeds_check = {"frames": frames_chk.tolist(), "meme_faisceau_optimal": float(same.mean()),
                       "ecart_gain_optimal_median_db": float(np.median(dg)),
                       "ecart_gain_optimal_p90_db": float(np.percentile(dg, 90))}

    # ---- evenements par UE --------------------------------------------------------------------
    events = {}
    for j, a in enumerate(ues):
        bb = best_bs[:, j]
        los_serv = served[:, j]
        cause = [BLOCK[t, best_bs[t, j], j] for t in range(T)]
        trans = [{"frame": int(t), "bs": dc.BS_NAMES[best_bs[t, j]], "obstacle": cause[t]}
                 for t in range(1, T) if los_serv[t - 1] and not los_serv[t]]
        events[a] = {
            "changements_bs_optimale": int(np.sum(bb[1:] != bb[:-1])),
            "changements_faisceau_optimal": int(np.sum((best_beam[1:, :, j] != best_beam[:-1, :, j])
                                                       [np.arange(T - 1), bb[1:]])),
            "part_frames_bs_optimale_en_LOS": float(los_serv.mean()),
            "pertes_LOS_bs_servante": trans,
            "snr_oracle_median_db": float(np.median(snr_oracle[:, j])),
            "snr_oracle_min_db": float(snr_oracle[:, j].min()),
            "part_coupure": float(np.mean(snr_oracle[:, j] < SNR_OUTAGE_DB)),
            "bs_optimales": {n: int(np.sum(bb == i)) for i, n in enumerate(dc.BS_NAMES)},
        }
    obstacles = {}
    for c in BLOCK[~LOS]:
        k = c.split(":")[0]
        obstacles[k] = obstacles.get(k, 0) + 1

    # ---- sauvegarde ---------------------------------------------------------------------------
    np.save(out / "H_fr2.npy", H)
    np.savez_compressed(out / "canal_reel_fr2.npz", G=G, best_beam=best_beam, snr_best_db=snr_best,
                        best_bs=best_bs, los=LOS, blocker=BLOCK.astype(str), geo_beam=GEO_BEAM,
                        ue_pos=UE_POS, ue_names=np.array(ues), bs_names=np.array(dc.BS_NAMES),
                        bs_pos=np.array([dc.BS_POS[n] for n in dc.BS_NAMES]),
                        bs_rot=np.array([R_bs[n] for n in dc.BS_NAMES]), codebook=W, codebook_uv=uv,
                        actor_names=np.array(world.actors), actor_state=np.array([ACT[a] for a in world.actors]),
                        subcarriers_hz=np.array(freqs), sector_target=target)
    summary = {
        "etape": "2 - canal reel FR2 et codebook (M6)",
        "scene": str(Path(args.scene) / "real.xml"), "n_frames": T, "frame_s": dc.FRAME_S,
        "bande": {"frequence_ghz": FREQ / 1e9, "largeur_mhz": BW / 1e6, "sous_porteuses": N_SUB},
        "bs": {"reseau": f"UPA {N_ROWS}x{N_COLS} tr38901, pol. V", "p_tx_dbm": P_TX_DBM,
               "bruit_dbm": round(NOISE_DBM, 1), "cible_secteur": target.round(1).tolist()},
        "codebook": {"type": "DFT 2D", "N": B},
        "trace": {"max_depth": MAX_DEPTH, "speculaire": True, "diffuse": True, "diffraction": True,
                  "trajets_dupliques_supprimes": {
                      "total": int(sum(d["supprimes"] for d in dedup_log[:T])),
                      "frames_concernees": int(sum(d["supprimes"] > 0 for d in dedup_log[:T])),
                      "part_puissance_max_frame": float(max(d["part_puissance"] for d in dedup_log[:T]))}},
        "monde_reel": world.summary(),
        "ue": ues,
        "part_liens_LOS": {n: {a: float(LOS[:, i, j].mean()) for j, a in enumerate(ues)}
                           for i, n in enumerate(dc.BS_NAMES)},
        "obstacles_liens_NLOS": obstacles,
        "evenements_par_ue": events,
        "faisceau_geometrique_J0_apercu": {
            "perte_mediane_db_liens_LOS": float(np.median(loss_geo[los_links])) if los_links.any() else None,
            "perte_mediane_db_liens_NLOS": float(np.median(loss_geo[~LOS & (GEO_BEAM >= 0)]))
            if (~LOS & (GEO_BEAM >= 0)).any() else None},
        "validation": {
            "codebook_LOS_faisceau_geometrique_a_1dB_du_meilleur": geo_ok,
            "variabilite_diffuse": seeds_check,
        },
        "temps_calcul_s": round(time.time() - t0, 1),
    }
    json.dump(summary, open(out / "etape2_summary.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    figures(out, T, ues, snr_best, snr_oracle, best_bs, best_beam, LOS, loss_geo, served)
    print(json.dumps({k: summary[k] for k in ("obstacles_liens_NLOS", "validation")}, indent=1, ensure_ascii=False))
    for a, e in events.items():
        print(f"  {a:6s} BS opt. changee {e['changements_bs_optimale']:2d}x, faisceau {e['changements_faisceau_optimal']:3d}x, "
              f"LOS {e['part_frames_bs_optimale_en_LOS']:.0%}, SNR med {e['snr_oracle_median_db']:5.1f} dB, "
              f"min {e['snr_oracle_min_db']:5.1f} dB, pertes LOS {len(e['pertes_LOS_bs_servante'])}")


def figures(out, T, ues, snr_best, snr_oracle, best_bs, best_beam, LOS, loss_geo, served):
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#1f4e79", "#c55a11", "#548235", "#7030a0", "#b8860b", "#2f4f4f"]
    t_s = np.arange(T) * dc.FRAME_S
    U = len(ues)
    fig, axes = plt.subplots(U, 1, figsize=(13, 2.3 * U), sharex=True, squeeze=False)
    for j, a in enumerate(ues):
        ax = axes[j, 0]
        for i, n in enumerate(dc.BS_NAMES):
            ax.plot(t_s, snr_best[:, i, j], color=colors[i], lw=1.2, alpha=0.8, label=n)
        ax.scatter(t_s, snr_oracle[:, j], c=[colors[b] for b in best_bs[:, j]], s=10, zorder=3)
        nlos = ~served[:, j]
        ax.fill_between(t_s, -40, 80, where=nlos, color="0.85", step="mid", zorder=0)
        ax.set_ylim(-40, 80)
        ax.set_ylabel(f"{a}\nSNR [dB]")
        ax.axhline(SNR_OUTAGE_DB, color="crimson", lw=0.8, ls="--")
    axes[0, 0].legend(ncol=4, frameon=False, loc="upper right")
    axes[-1, 0].set_xlabel("temps [s]  (gris : BS optimale hors ligne de vue)")
    fig.suptitle("SNR du meilleur faisceau par BS (balayage exhaustif, 256 faisceaux) ; points : BS optimale")
    fig.tight_layout()
    fig.savefig(out / "snr_par_ue_et_bs.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for i, n in enumerate(dc.BS_NAMES):
        l = loss_geo[:, i, :]
        for m, ls, lab in ((LOS[:, i, :], "-", "LOS"), (~LOS[:, i, :], "--", "NLOS")):
            x = np.sort(l[m & np.isfinite(l)])
            if len(x):
                axes[0].plot(x, np.arange(1, len(x) + 1) / len(x), ls, color=colors[i], label=f"{n} {lab}")
    axes[0].set(xlim=(0, 30), xlabel="Perte du faisceau geometrique vs optimal [dB]", ylabel="CDF",
                title="Apercu J0 : pointer vers la position de l'UE")
    axes[0].legend(frameon=False, fontsize=9, ncol=2)
    im = axes[1].imshow(LOS.reshape(T, -1).T, aspect="auto", cmap="Greens", interpolation="nearest",
                        extent=[0, T * dc.FRAME_S, LOS.shape[1] * U, 0])
    axes[1].set_yticks(np.arange(LOS.shape[1] * U) + 0.5)
    axes[1].set_yticklabels([f"{n}-{a}" for n in dc.BS_NAMES for a in ues], fontsize=7)
    axes[1].set(xlabel="temps [s]", title="Ligne de vue geometrique BS-UE (vert)")
    fig.tight_layout()
    fig.savefig(out / "los_et_faisceau_geometrique.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
