"""Appendix B: detailed controlled-variable sweeps, on the honest metric.

This mirrors the two-axis ablation a sceptic expects (collection coverage and
deployment shift) but carries two things a Monte-Carlo study cannot:
  * every cell reports BOTH success (the gameable proxy) and reward (the honest,
    teacher-optimal objective);
  * every number is exact (sampling-free), so there are no confidence intervals
    to report -- the curves are facts, not estimates.

Honest caveat we state in the report: being exact/deterministic, we do not have
the "variance is the tell" finding a seeded MC study has; we trade that for
exactness and the gradient/chi^2 layer available at every cell.

Ablation A: success/reward vs collection coverage, deploy fixed at 0.45.
Ablation B: success/reward vs deployment shift, collect fixed at ~0 (our worst
            case: clean collection = the literal definition of distribution
            shift, where only coverage-buying methods survive).
Linear student throughout (the generalising, ~real-LLM case).
"""

import numpy as np

from opd_toy import (RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, build_features, train_sft, train_offline_opd,
    train_online_opd, train_refresh)
from opd_toy import exact
from patch_ablation_all import base_cfg, patch_fulldist, patch_branch_replay

DEPLOY_DEFAULT = 0.45


def _methods(env, teacher, rp, feats, cfg):
    """Return {name: (success, reward)} for every method at this operating point."""
    def fr():
        s = LinearSoftmaxStudent(env, feats, seed=0); s.set_params(rp); return s

    def sr(pol):
        d = exact.occupancy(env, pol, cfg.deploy_noise)
        return d.success, d.avg_reward

    out = {}
    out["SFT"] = sr(fr())
    out["offline"] = sr(train_offline_opd(env, teacher, fr(), fr(), cfg).student)
    out["full-dist"] = sr(patch_fulldist(env, teacher, fr(), fr(), cfg))
    out["branch-replay"] = sr(patch_branch_replay(env, teacher, fr(), fr(), cfg))
    out["chi2-refresh"] = sr(train_refresh(env, teacher, fr(), fr(), cfg,
                             trigger="chi2", chi2_thresh=0.2, budget=50,
                             check_every=10).student)
    out["online"] = sr(train_online_opd(env, teacher, fr(), fr(), cfg).student)
    return out


ORDER = ["SFT", "offline", "full-dist", "branch-replay", "chi2-refresh", "online"]


def _print_block(title, axis_name, axis_vals, rows, teach):
    print(f"\n### {title}")
    head = f"{axis_name:>10}" + "".join(f"{m:>16}" for m in ORDER) + f"{'teacher':>10}"
    print(head); print("-" * len(head))
    for v, r in zip(axis_vals, rows):
        cells = "".join(f"  {r[m][0]:.3f}/{r[m][1]:+.2f}" for m in ORDER)
        print(f"{v:>10.3g}{cells}{teach[v][0]:>6.3f}/{teach[v][1]:+.2f}")


def ablation_A():
    """Collection coverage sweep, deploy fixed at 0.45 (linear)."""
    env = RetrievalQAEnv(base_cfg()); feats = build_features(env)
    teacher = TeacherPolicy(env, noise=DEPLOY_DEFAULT, temperature=0.03)
    td = exact.occupancy(env, teacher, DEPLOY_DEFAULT)
    collects = (1e-9, 0.05, 0.15, 0.30, 0.45)
    rows, teach = [], {}
    for c in collects:
        cfg = TrainConfig(steps=300, lr=0.5, collect_noise=c, deploy_noise=DEPLOY_DEFAULT,
                          dataset_size=2000, record_every=300)
        sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=0), cfg)
        rows.append(_methods(env, teacher, sft.student.get_params(), feats, cfg))
        teach[c] = (td.success, td.avg_reward)
    _print_block("Ablation A: collection coverage (deploy=0.45)", "collect", collects, rows, teach)
    return collects, rows


def ablation_B():
    """Deployment shift sweep, collect fixed at ~0 (linear)."""
    env = RetrievalQAEnv(base_cfg()); feats = build_features(env)
    deploys = (0.10, 0.20, 0.30, 0.45)
    rows, teach = [], {}
    for dp in deploys:
        teacher = TeacherPolicy(env, noise=dp, temperature=0.03)
        td = exact.occupancy(env, teacher, dp)
        cfg = TrainConfig(steps=300, lr=0.5, collect_noise=1e-9, deploy_noise=dp,
                          dataset_size=2000, record_every=300)
        sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=0), cfg)
        rows.append(_methods(env, teacher, sft.student.get_params(), feats, cfg))
        teach[dp] = (td.success, td.avg_reward)
    _print_block("Ablation B: deployment shift (collect~0)", "deploy", deploys, rows, teach)
    return deploys, rows


def main():
    print("Detailed sweeps -- each cell is success/reward, exact (no CIs). Linear student.")
    ablation_A()
    ablation_B()


if __name__ == "__main__":
    main()
