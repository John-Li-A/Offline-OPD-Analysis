"""Robustness phase diagram: is the offline-collapse + patch-recovery story an
artifact of one hand-picked cost point, or does it hold across a band?

The reward's *ordinal* structure step_cost < reconcile_cost < wrong_penalty is
forced by the semantics (a cheap step, an expensive-but-reliable recovery, a
costly failure). The *cardinal* values are a modelling choice. This sweep shows
the conclusion is a property of the band, not the point: we vary the two cost
magnitudes a reviewer would challenge -- reconcile_cost and wrong_penalty -- and
report, per cell:

    t_rec   teacher reconcile-mass at deploy
            -> proves we sit in the SELECTIVE-USE band (not 0 = never, not 1 =
               always); the degenerate limits are the named boundaries.
    floor   offline-OPD deployment reward (the collapse)
    ceil    teacher deployment reward (reward-optimal, the honest ceiling)
    patch   chi2-refresh deployment reward
    rec%    (patch - floor)/(ceil - floor)  -- the scale-FREE recovery fraction,
            invariant to any affine rescaling a*R+b of the reward.

No conclusion hard-coded; whatever prints is what we report. collect~0 (the
worst case = the literal definition of distribution shift), linear student.
"""

import numpy as np

from opd_toy import (RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, build_features, train_sft, train_offline_opd, train_refresh)
from opd_toy import exact
from patch_ablation_all import base_cfg

DEPLOY = 0.45


def teacher_recmass(env, teacher):
    dep = exact.occupancy(env, teacher, DEPLOY)
    m = 0.0
    for s in range(env.num_states):
        if dep.visit[s] > 0 and env.legal_actions(s)[env.reconcile_action]:
            m += dep.visit[s] * teacher.probs(s)[env.reconcile_action]
    return float(m)


def cell(rcost, wpen, steps=250):
    env = RetrievalQAEnv(base_cfg(reconcile_cost=rcost, wrong_penalty=wpen))
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=DEPLOY, temperature=0.03)
    ceil = exact.occupancy(env, teacher, DEPLOY).avg_reward
    t_rec = teacher_recmass(env, teacher)
    cfg = TrainConfig(steps=steps, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                      dataset_size=2000, record_every=steps)
    sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=0), cfg)
    rp = sft.student.get_params()

    def fr():
        s = LinearSoftmaxStudent(env, feats, seed=0); s.set_params(rp); return s

    floor = train_offline_opd(env, teacher, fr(), fr(), cfg).student
    floor_r = exact.occupancy(env, floor, DEPLOY).avg_reward
    pat = train_refresh(env, teacher, fr(), fr(), cfg, trigger="chi2",
                        chi2_thresh=0.2, budget=50, check_every=10)
    patch_r = exact.occupancy(env, pat.student, DEPLOY).avg_reward
    denom = ceil - floor_r
    recpct = (patch_r - floor_r) / denom if abs(denom) > 1e-9 else float("nan")
    return t_rec, floor_r, ceil, patch_r, recpct, pat.final["refreshes"]


def main():
    rcosts = (0.2, 0.4, 0.8, 1.2, 1.6)
    wpens = (1.5, 3.0, 5.0)
    print("collect~0, linear.  t_rec=teacher reconcile-mass (band check); "
          "rec%=(patch-floor)/(ceil-floor)\n")
    for wpen in wpens:
        print(f"### wrong_penalty = {wpen}")
        print(f"{'rcost':>7}{'t_rec':>8}{'floor':>9}{'ceil':>9}{'patch':>9}{'rec%':>8}{'env':>6}")
        print("-" * 56)
        for rc in rcosts:
            t_rec, fl, ce, pa, rp, nr = cell(rc, wpen)
            print(f"{rc:>7.1f}{t_rec:>8.3f}{fl:>+9.3f}{ce:>+9.3f}{pa:>+9.3f}{rp:>7.0%}{nr:>6}")
        print()


if __name__ == "__main__":
    main()
