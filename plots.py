"""Draw the report figures from the cached arrays in results/figdata.npz.

Research-grade styling: Okabe-Ito colour-blind-safe palette, serif text with
Computer-Modern mathtext, de-spined axes, light grid, (a)/(b) panel labels, and
vector PDF output (PNG alongside for quick view). Run ``gen_data.py`` first.

The figures are composed, not scattered:
  fig1_anatomy.pdf  -- (a) 4-method bars, linear vs tabular; (b) the mechanism:
                       polluted-bit weight + success vs step, offline vs online.
  fig2_coverage.pdf -- offline/online success & gap band vs off-support ratio.
  fig3_bound.pdf    -- measured gradient gap vs the ~10x-vacuous Thm 3.5 bound,
                       with chi2 on a twin axis; success-is-flat annotation.
  fig4_patch.pdf    -- patch Pareto: success vs env-access refreshes (log x),
                       chi2-triggered vs random placement, offline/online refs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe

OUT = os.path.join(os.path.dirname(__file__), "results")

# Okabe-Ito colour-blind-safe palette.
OI = {
    "black":  "#000000",
    "orange": "#E69F00",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "verm":   "#D55E00",
    "purple": "#CC79A7",
    "grey":   "#999999",
}
C_OFFLINE = OI["verm"]    # offline OPD -- the method that fails
C_ONLINE = OI["green"]    # online OPD -- the upper bound
C_RL = OI["blue"]         # online RL
C_SFT = OI["grey"]        # SFT floor
C_TEACH = OI["black"]     # teacher ceiling


def set_style():
    rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def panel_label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left")


def load():
    d = np.load(os.path.join(OUT, "figdata.npz"))
    return {k: d[k] for k in d.files}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name + ".pdf/.png")


def fig1_anatomy(D):
    """(a) 4-method bars linear vs tabular; (b) the mechanism weight trajectory."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # -- (a) baseline bars --
    methods = [str(m) for m in D["baseline__methods"]]
    x = np.arange(len(methods)); w = 0.38
    lin_m, lin_s = D["baseline__linear_mean"], D["baseline__linear_std"]
    tab_m, tab_s = D["baseline__tabular_mean"], D["baseline__tabular_std"]
    teach = float(D["baseline__teacher"])
    axa.bar(x - w / 2, lin_m, w, yerr=lin_s, capsize=2.5, color=C_OFFLINE,
            label="linear (generalising)", error_kw=dict(lw=0.8))
    axa.bar(x + w / 2, tab_m, w, yerr=tab_s, capsize=2.5, color=C_ONLINE,
            label="tabular (no sharing)", error_kw=dict(lw=0.8))
    axa.axhline(teach, ls="--", lw=1, c=C_TEACH)
    axa.set_xticks(x); axa.set_xticklabels(methods)
    axa.set_ylabel("deployment success"); axa.set_ylim(0, 1.08)
    axa.set_title("Offline OPD barely beats SFT;\nlinear collapses harder than tabular")
    # teacher label sits just ABOVE its dashed line (was colliding with the line)
    axa.text(0.02, teach + 0.015, f"teacher ({teach:.2f})", color=C_TEACH,
             fontsize=8, va="bottom", ha="left", transform=axa.get_yaxis_transform())
    # 2-entry legend, dropped just below the teacher line so it never crosses it
    axa.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    panel_label(axa, "(a)")
    # annotate the linear<tabular offline inversion
    axa.annotate("", xy=(1 - w / 2, lin_m[1] + 0.04), xytext=(1 + w / 2, tab_m[1] + 0.04),
                 arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.8))
    axa.text(1, max(lin_m[1], tab_m[1]) + 0.10, "linear\nlower", ha="center",
             fontsize=7.5, color="0.3")

    # -- (b) mechanism: weight + success, offline vs online --
    steps = D["mechanism__steps"]
    axb.plot(steps, D["mechanism__w_off"], "-", c=C_OFFLINE, lw=1.8,
             label="offline: polluted-bit weight")
    axb.plot(steps, D["mechanism__w_on"], "-", c=C_ONLINE, lw=1.8,
             label="online: polluted-bit weight")
    axb.set_xlabel("OPD training step")
    axb.set_ylabel("polluted-bit reconcile weight")
    axb.set_title("The mechanism: offline cannot train\nthe recovery feature (zero gradient)")
    axb.axhline(0, ls=":", c="0.6", lw=0.8)
    # legend to the mid-right empty band (was at center-left, over the green curve)
    axb.legend(loc="center right", bbox_to_anchor=(0.98, 0.42))
    panel_label(axb, "(b)")
    # twin axis: success
    axc = axb.twinx()
    axc.spines["top"].set_visible(False)
    axc.plot(steps, D["mechanism__succ_off"], "--", c=C_OFFLINE, lw=1.1, alpha=0.7)
    axc.plot(steps, D["mechanism__succ_on"], "--", c=C_ONLINE, lw=1.1, alpha=0.7)
    axc.set_ylabel("success (dashed)", color="0.4")
    axc.set_ylim(0, 1.05)
    axc.tick_params(axis="y", labelcolor="0.4")
    axc.grid(False)
    fig.tight_layout()
    save(fig, "fig1_anatomy")


def fig2_coverage(D):
    osr = D["coverage__off_support"]; off = D["coverage__offline"]
    on = D["coverage__online"]; teach = D["coverage__teacher"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(osr, teach, ":", c=C_TEACH, lw=1, label="teacher ceiling")
    ax.plot(osr, on, "o-", c=C_ONLINE, lw=1.8, ms=5, label="online OPD")
    ax.plot(osr, off, "s-", c=C_OFFLINE, lw=1.8, ms=5, label="offline OPD")
    ax.fill_between(osr, off, on, color=C_OFFLINE, alpha=0.10)
    # annotate the widening gap
    i = len(osr) - 1
    ax.annotate(f"gap {on[i]-off[i]:.2f}", xy=(osr[i], (on[i] + off[i]) / 2),
                xytext=(osr[i] - 0.13, (on[i] + off[i]) / 2),
                fontsize=8, color=C_OFFLINE,
                arrowprops=dict(arrowstyle="-", color=C_OFFLINE, lw=0.7))
    ax.set_xlabel("off-support ratio  (deployment mass on uncovered states)")
    ax.set_ylabel("deployment success"); ax.set_ylim(0, 1.05)
    ax.set_title("When does offline OPD fail?  A smooth spectrum")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig2_coverage")


def fig3_bound(D):
    steps = D["bound__steps"]; gap = D["bound__gap"]
    bound = D["bound__bound"]; chi2 = D["bound__chi2"]; succ = D["bound__success"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(steps, bound, "s-", c=OI["purple"], lw=1.6, ms=4,
            label=r"Thm 3.5 bound  $G\,\sigma_A\sqrt{\chi^2}$")
    ax.plot(steps, gap, "o-", c=C_OFFLINE, lw=1.8, ms=4,
            label=r"measured $\|\nabla J_{\mathrm{on}}-\nabla J_{\mathrm{off}}\|$")
    ax.set_xlabel("offline OPD step"); ax.set_ylabel("gradient-gap magnitude")
    ax.set_title("The Theorem 3.5 bound is ~10$\\times$ vacuous")
    # vacuity annotation
    j = len(steps) - 1
    ax.annotate(f"{bound[j]/max(gap[j],1e-9):.0f}$\\times$ gap",
                xy=(steps[j], bound[j]), xytext=(steps[j] * 0.62, bound[j] * 0.95),
                fontsize=8.5, color=OI["purple"])
    ax.legend(loc="upper left")
    # twin axis: chi2 and a note that success is flat
    ax2 = ax.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(steps, chi2, "^--", c=C_ONLINE, lw=1.1, alpha=0.8, ms=3,
             label=r"$\chi^2(\pi_\theta\|\pi_{\mathrm{ref}})$")
    ax2.set_ylabel(r"$\chi^2$ divergence", color=C_ONLINE)
    ax2.tick_params(axis="y", labelcolor=C_ONLINE); ax2.grid(False)
    ax2.legend(loc="lower right")
    ax.text(0.62, 0.20, f"success flat at {succ[-1]:.2f} throughout\n(gap grows, success doesn't)",
            transform=ax.transAxes, fontsize=8, color="0.35", ha="center")
    fig.tight_layout()
    save(fig, "fig3_bound")


def fig4_patch(D):
    off = float(D["patch__offline"]); on = float(D["patch__online"])
    onstep = int(D["patch__online_steps"])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axhline(on, ls="--", c=C_ONLINE, lw=1.2, alpha=0.8,
               label=f"online OPD (~{onstep} refresh, {on:.2f})")
    ax.axhline(off, ls=":", c=C_OFFLINE, lw=1.4,
               label=f"offline OPD (0 refresh, {off:.2f})")
    ax.scatter(np.maximum(D["patch__rand_r"], 0.5), D["patch__rand_s"],
               marker="x", s=42, c=OI["orange"], label="random placement", zorder=3)
    ax.scatter(np.maximum(D["patch__chi2_r"], 0.5), D["patch__chi2_s"],
               marker="o", s=70, c=C_ONLINE, edgecolor="k", linewidth=0.6,
               label=r"$\chi^2$-triggered (ours)", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("environment-access refreshes  (log scale)")
    ax.set_ylabel("deployment success"); ax.set_ylim(0, 1.05)
    ax.set_title(r"$\chi^2$-triggered refresh: online performance at ~3% env cost")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_patch")


def fig5_anatomy(D):
    """The patch closing the loop with Fig 1b: refreshes fire when drift crosses
    the threshold, each unlocking the previously-frozen polluted-bit weight."""
    steps = D["anatomy__steps"]; drift = D["anatomy__drift"]
    w = D["anatomy__weight"]; succ = D["anatomy__success"]
    thresh = float(D["anatomy__thresh"]); fire_steps = D["anatomy__fire_steps"]
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 1.15]))
    # -- (a) drift vs threshold, with fire markers --
    axa.plot(steps, drift, "-", c=OI["blue"], lw=1.6,
             label=r"$\chi^2$ drift vs. live dataset")
    axa.axhline(thresh, ls="--", c="0.5", lw=1, label=fr"threshold $={thresh}$")
    for i, fs in enumerate(fire_steps):
        axa.axvline(fs, c=OI["orange"], lw=0.8, alpha=0.55,
                    label="refresh fires" if i == 0 else None)
    axa.set_ylabel(r"$\chi^2(\pi_\theta\|\pi_{\mathrm{data}})$")
    axa.set_title("The patch in action: drift triggers a refresh, "
                  "which unlocks the frozen weight")
    axa.legend(loc="upper right", ncol=1)
    panel_label(axa, "(a)")
    # -- (b) polluted-bit weight + success, with same fire lines --
    for fs in fire_steps:
        axb.axvline(fs, c=OI["orange"], lw=0.8, alpha=0.55)
    axb.plot(steps, w, "-", c=C_OFFLINE, lw=1.9, label="polluted-bit weight")
    axb.axhline(0, ls=":", c="0.6", lw=0.8)
    axb.set_ylabel("polluted-bit reconcile weight", color=C_OFFLINE)
    axb.tick_params(axis="y", labelcolor=C_OFFLINE)
    axb.set_xlabel("OPD training step")
    panel_label(axb, "(b)")
    axc = axb.twinx(); axc.spines["top"].set_visible(False)
    axc.plot(steps, succ, "--", c=C_ONLINE, lw=1.4, label="deployment success")
    axc.set_ylabel("success", color=C_ONLINE)
    axc.tick_params(axis="y", labelcolor=C_ONLINE); axc.set_ylim(0, 1.05)
    axc.grid(False)
    # annotate the jump
    axb.annotate("weight unlocks\n$-0.01\\to+5.8$, success $0.42\\to0.97$",
                 xy=(140, 5.8), xytext=(165, 3.0), fontsize=8, color="0.25",
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    fig.tight_layout()
    save(fig, "fig5_anatomy")


def fig6_phase(D):
    """Reward robustness phase diagram over (reconcile_cost, wrong_penalty).

    rec% recovery is the heatmap with per-cell teacher reconcile-mass underneath.
    Three regimes are tagged with small corner letters whose meaning lives in the
    LaTeX caption (keeping the plot uncluttered):
      (A) reconcile too cheap -> teacher always reconciles, no trap to fix;
      (B) trap exists but the fixed chi2 threshold never fired (transfer miss);
      (C) reconcile too costly -> teacher abandons it, recovery is partial.
    The healthy interior band (0.4<=c_rec<=1.2) recovers 80-99%."""
    rc = D["phase__rcosts"]; wp = D["phase__wpens"]
    rec = D["phase__recpct"]; trec = D["phase__t_rec"]
    nr, nc = rec.shape
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    im = ax.imshow(rec, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(nc)); ax.set_xticklabels([f"{v:g}" for v in rc])
    ax.set_yticks(range(nr)); ax.set_yticklabels([f"{v:g}" for v in wp])
    ax.set_xlabel(r"reconcile cost  $c_{\mathrm{rec}}$  (cheap $\rightarrow$ costly)")
    ax.set_ylabel(r"wrong-answer penalty  $\lambda_{\mathrm{wrong}}$")
    ax.set_title(r"Recovery is a band, not a point: $\chi^2$-refresh reward"
                 "\nrecovery across the cost structure (collect $\\approx$ 0, linear)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("reward recovered\n"
                   r"$(\mathrm{patch}-\mathrm{floor})/(\mathrm{ceil}-\mathrm{floor})$")
    cbar.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # per-cell: recovery % (bold) with teacher reconcile-mass below it.
    for i in range(nr):
        for j in range(nc):
            light = rec[i, j] < 0.6
            ax.text(j, i + 0.10, f"{rec[i, j]:.0%}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if light else "black")
            ax.text(j, i - 0.24, f"$t_{{\\mathrm{{rec}}}}\\!=\\!{trec[i, j]:.2f}$",
                    ha="center", va="center", fontsize=6.5,
                    color="0.88" if light else "0.30")

    def corner_tag(j, i, s):
        ax.text(j - 0.42, i + 0.30, s, ha="left", va="top", fontsize=9,
                fontweight="bold", color="#FFFFFF",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="#D55E00")])

    # (A) the whole rc=0.2 column: reconcile so cheap the teacher always uses it,
    #     so there is no trap and nothing to recover (floor ~ ceil).
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, nr, fill=False,
                               edgecolor="#D55E00", lw=1.4, ls="--"))
    for i in range(nr):
        corner_tag(0, i, "A")
    # (C) reconcile too costly + cheap failure: teacher abandons reconcile.
    cj = list(rc).index(1.6); ci = list(wp).index(1.5)
    ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1, fill=False,
                               edgecolor="#D55E00", lw=1.6, ls=":"))
    corner_tag(cj, ci, "C")
    # (B) trap exists but the fixed threshold never fired (transfer miss).
    bj = list(rc).index(0.4); bi = list(wp).index(3.0)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="#D55E00", lw=1.6, ls=":"))
    corner_tag(bj, bi, "B")
    ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(-0.5, nr - 0.5)
    fig.tight_layout()
    save(fig, "fig6_phase")


def _shorten(label):
    return (str(label).replace(" (ours)", "").replace(" (Rang)", "")
            .replace("offline OPD", "offline").replace("online OPD", "online")
            .replace("support-aware", "support-\naware")
            .replace("branch-replay", "branch-\nreplay")
            .replace("uncertainty-query", "uncert.-\nquery")
            .replace("chi2-refresh", r"$\chi^2$-refresh"))


# Env-access class -> colour: 0 pure-offline, 1 our collect-side refresh, 2 deploy-side.
ENVC = {0: OI["grey"], 1: OI["verm"], 2: OI["blue"]}
ENVC_NAME = {0: "pure offline (0 env)", 1: r"$\chi^2$-refresh (collect-side)",
             2: "online / query (deploy-side)"}


def fig7_crosspatch(D):
    """All five hint patches + full-dist + offline/online on the mechanism axis.

    (a) the polluted-bit reconcile weight -- the structurally-starved feature --
    bar per method, coloured by env-access class; pure-offline patches stay pinned
    at the SFT init, only env-touching methods lift it. (b) success and reward per
    method, with the teacher reward ceiling drawn in."""
    lab = [_shorten(x) for x in D["crosspatch__labels"]]
    polw = D["crosspatch__pol_w"]; succ = D["crosspatch__succ"]
    rew = D["crosspatch__reward"]; ec = D["crosspatch__env_class"].astype(int)
    t_s = float(D["crosspatch__teacher_succ"]); t_r = float(D["crosspatch__teacher_reward"])
    x = np.arange(len(lab)); cols = [ENVC[e] for e in ec]

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 1.05]))
    # -- (a) polluted-bit weight --
    axa.bar(x, polw, color=cols, edgecolor="k", linewidth=0.4)
    axa.axhline(polw[0], ls=":", c="0.5", lw=1)
    axa.text(len(lab) - 0.4, polw[0], " SFT init", va="center", fontsize=7.5, c="0.4")
    axa.set_ylabel("polluted-bit\nreconcile weight")
    axa.set_title("Only coverage-buying methods can lift the structurally-"
                  "starved feature\n(pure-offline patches stay pinned at init)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ENVC[k]) for k in (0, 1, 2)]
    axa.legend(handles, [ENVC_NAME[k] for k in (0, 1, 2)], loc="upper left",
               fontsize=8)
    panel_label(axa, "(a)")
    # -- (b) success + reward --
    w = 0.4
    axb.bar(x - w / 2, succ, w, color=OI["sky"], edgecolor="k", linewidth=0.4,
            label="success")
    axb.axhline(t_s, ls="--", lw=1, c=C_TEACH)
    axb.text(0.02, t_s + 0.01, f"teacher success {t_s:.2f}", fontsize=7.5,
             c=C_TEACH, transform=axb.get_yaxis_transform())
    axb.set_ylabel("deployment success", color=OI["sky"])
    axb.set_ylim(0, 1.08); axb.tick_params(axis="y", labelcolor=OI["sky"])
    panel_label(axb, "(b)")
    axr = axb.twinx(); axr.spines["top"].set_visible(False)
    axr.bar(x + w / 2, rew, w, color=OI["orange"], edgecolor="k", linewidth=0.4,
            label="reward")
    axr.axhline(t_r, ls="--", lw=1, c=OI["orange"])
    axr.axhline(0, ls="-", lw=0.6, c="0.7")
    axr.text(0.98, t_r, f"teacher reward {t_r:+.2f}", fontsize=7.5, ha="right",
             c=OI["orange"], transform=axr.get_yaxis_transform())
    axr.set_ylabel("deployment reward", color=OI["orange"])
    axr.tick_params(axis="y", labelcolor=OI["orange"]); axr.grid(False)
    axb.set_xticks(x); axb.set_xticklabels(lab, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig7_crosspatch")


def fig8_metric(D):
    """Reward-vs-success scatter: why success is the gameable proxy. The teacher
    is the reward-optimal point; methods northwest of it (high success, lower
    reward) have bought accuracy by over-reconciling at a cost the teacher
    refuses. The six pure-offline methods all collapse to one point (the mechanism
    claim), so we draw them as a single labelled cluster rather than six
    unreadable overlapping dots."""
    lab = [str(x) for x in D["crosspatch__labels"]]
    succ = D["crosspatch__succ"]; rew = D["crosspatch__reward"]
    ec = D["crosspatch__env_class"].astype(int)
    t_s = float(D["crosspatch__teacher_succ"]); t_r = float(D["crosspatch__teacher_reward"])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    # gaming region: success above teacher.
    ax.axhspan(t_s, 1.07, color=OI["grey"], alpha=0.07)
    ax.axvline(t_r, ls=":", c="0.6", lw=1); ax.axhline(t_s, ls=":", c="0.6", lw=1)

    # The top-right is crowded (4 near-identical-success points); fan their
    # labels out into the empty left region, each with a thin leader, instead of
    # stacking text on the markers. Pure-offline methods collapse to one point.
    PLACE = {  # label -> (text_x, text_y) in data coords, right-aligned, leader to point
        "chi2-refresh (ours)": (-0.55, 1.00),
        "online OPD":          (-0.55, 0.92),
        "uncertainty-query":   (-0.55, 0.84),
    }
    pts = list(zip(rew, succ, ec, lab))
    used = [False] * len(pts)
    for i, (r, s, e, name) in enumerate(pts):
        if used[i]:
            continue
        group = [j for j in range(len(pts))
                 if not used[j] and abs(pts[j][0] - r) < 0.05 and abs(pts[j][1] - s) < 0.02]
        for j in group:
            used[j] = True
        if len(group) == 1:
            ax.scatter(r, s, s=80, c=ENVC[e], edgecolor="k", linewidth=0.5, zorder=3)
            short = name.replace(" (ours)", "").replace(" (Rang)", "")
            tx, ty = PLACE.get(name, (r + 0.05, s))
            ax.annotate(short, xy=(r, s), xytext=(tx, ty), fontsize=8,
                        va="center", ha="right", color="0.15",
                        arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6))
        else:
            ax.scatter(r, s, s=150, c=ENVC[e], edgecolor="k", linewidth=0.9,
                       marker="o", zorder=3)
            members = [pts[j][3].replace(" (Rang)", "") for j in group]
            txt = (f"{len(group)} pure-offline methods\ncollapse to one point:\n"
                   + "\n".join("· " + m for m in members))
            ax.annotate(txt, xy=(r, s), xytext=(r + 0.12, s + 0.30), fontsize=7.2,
                        va="top", ha="left", color="0.25",
                        arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6))
    # teacher star, label placed just below it (clear region, outside the band).
    ax.scatter(t_r, t_s, marker="*", s=260, c=OI["yellow"], edgecolor="k",
               linewidth=0.7, zorder=4)
    ax.annotate("teacher\n(reward-optimal)", xy=(t_r, t_s), xytext=(t_r, t_s - 0.13),
                fontsize=8, ha="center", va="top", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6))
    ax.set_xlabel("deployment reward  (the honest objective)")
    ax.set_ylabel("deployment success  (the gameable proxy)")
    ax.set_title("Reward vs. success: the teacher is the reward ceiling;\n"
                 "high-success outliers are over-reconcilers")
    handles = [plt.Line2D([], [], marker="o", ls="", mec="k", mfc=ENVC[k], ms=8)
               for k in (0, 1, 2)]
    ax.legend(handles, [ENVC_NAME[k] for k in (0, 1, 2)], loc="lower right",
              fontsize=8)
    ax.set_ylim(0.33, 1.08); ax.set_xlim(-1.9, 0.55)
    fig.tight_layout()
    save(fig, "fig8_metric")


METHOD_STYLE = {
    "SFT":           (C_SFT,    "o", "-"),
    "offline":       (C_OFFLINE, "s", "-"),
    "full-dist":     (OI["purple"], "D", "--"),
    "branch-replay": (OI["orange"], "^", "--"),
    "chi2-refresh":  (C_ONLINE, "o", "-"),
    "online":        (C_RL,     "v", ":"),
}
METHOD_LABEL = {"SFT": "SFT", "offline": "offline OPD", "full-dist": "full-dist (Rang)",
                "branch-replay": "branch-replay", "chi2-refresh": r"$\chi^2$-refresh (ours)",
                "online": "online OPD"}


def _sweep_panel(ax, x, mat, methods, teach, ylabel, xlabel, logx=False):
    for i, m in enumerate(methods):
        c, mk, ls = METHOD_STYLE[m]
        lw = 2.2 if m == "chi2-refresh" else 1.4
        ax.plot(x, mat[i], ls=ls, marker=mk, color=c, lw=lw, ms=5,
                label=METHOD_LABEL[m], zorder=4 if m == "chi2-refresh" else 3)
    if np.ndim(teach) == 0:
        ax.axhline(float(teach), ls=":", c=C_TEACH, lw=1.2, label="teacher")
    else:
        ax.plot(x, teach, ls=":", c=C_TEACH, lw=1.2, marker="*", ms=7, label="teacher")
    if logx:
        ax.set_xscale("symlog", linthresh=0.04)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)


def fig9_coverage_sweep(D):
    """Ablation A: success and reward vs collection coverage (deploy 0.45)."""
    m = [str(s) for s in D["sweeps__methods"]]
    x = D["sweeps__A_x"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    _sweep_panel(axa, x, D["sweeps__A_succ"], m, float(D["sweeps__A_teach_s"]),
                 "deployment success", r"collection noise (coverage) $\rightarrow$")
    axa.set_ylim(0.30, 1.05); axa.set_title("Success vs collection coverage")
    panel_label(axa, "(a)")
    _sweep_panel(axb, x, D["sweeps__A_rew"], m, float(D["sweeps__A_teach_r"]),
                 "deployment reward", r"collection noise (coverage) $\rightarrow$")
    axb.axhline(0, ls="-", lw=0.6, c="0.7"); axb.set_title("Reward vs collection coverage")
    panel_label(axb, "(b)")
    axb.legend(loc="lower right", fontsize=7.5, ncol=1)
    axa.annotate("collect $\\approx$ 0:\nonly coverage-buyers escape", xy=(0, 0.42),
                 xytext=(0.10, 0.345), fontsize=7.5, color="0.35", ha="center",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))
    fig.tight_layout(); save(fig, "fig9_coverage_sweep")


def fig10_deploy_sweep(D):
    """Ablation B: success and reward vs deployment shift (collect ~0). The
    headline appendix figure: under growing shift with clean collection, every
    zero-env method collapses together; only chi2-refresh and online survive."""
    m = [str(s) for s in D["sweeps__methods"]]
    x = D["sweeps__B_x"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    _sweep_panel(axa, x, D["sweeps__B_succ"], m, D["sweeps__B_teach_s"],
                 "deployment success", r"deployment shift (noise) $\rightarrow$")
    axa.set_ylim(0.30, 1.05); axa.set_title("Success vs deployment shift")
    panel_label(axa, "(a)")
    _sweep_panel(axb, x, D["sweeps__B_rew"], m, D["sweeps__B_teach_r"],
                 "deployment reward", r"deployment shift (noise) $\rightarrow$")
    axb.axhline(0, ls="-", lw=0.6, c="0.7"); axb.set_title("Reward vs deployment shift")
    panel_label(axb, "(b)")
    axb.legend(loc="lower left", fontsize=7.5, ncol=1)
    # highlight the collapse band: offline/full-dist/branch-replay converge.
    axa.annotate("all zero-env methods\ncollapse together", xy=(0.45, 0.42),
                 xytext=(0.30, 0.345), fontsize=7.5, color="0.35", ha="center",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))
    fig.tight_layout(); save(fig, "fig10_deploy_sweep")


def main():
    set_style()
    D = load()
    fig1_anatomy(D)
    fig2_coverage(D)
    fig3_bound(D)
    fig4_patch(D)
    fig5_anatomy(D)
    if "phase__recpct" in D:
        fig6_phase(D)
    if "crosspatch__succ" in D:
        fig7_crosspatch(D)
        fig8_metric(D)
    if "sweeps__A_succ" in D:
        fig9_coverage_sweep(D)
        fig10_deploy_sweep(D)
    print("all figures written to", OUT)


if __name__ == "__main__":
    main()
