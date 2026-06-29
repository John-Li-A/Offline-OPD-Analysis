"""Can the chi2-refresh patch recover reward (not just success) without paying
more environment access?

Background: at collect~0 the patch hits success 1.000 but reward only +0.022,
below online OPD's +0.144 and the teacher's +0.193. The suspicion is
over-reconciliation: the patch refreshes hard, drives the polluted-bit weight
past online's, and the linear student spills extra reconciles onto clean states
(success up, reward down). This sweep tests whether a milder refresh schedule
(higher chi2 threshold -> fewer refreshes, or fewer steps -> less overshoot)
recovers reward while keeping env cost ~3%.

No conclusion hard-coded; whatever prints is what we report. Reward is the
honest objective (teacher = reward-optimal ceiling); success is the gameable
proxy.
"""

import numpy as np

from opd_toy import (RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, build_features, train_sft, train_offline_opd,
    train_online_opd, train_refresh)
from opd_toy import exact
from patch_ablation_all import base_cfg, diagnostics

DEPLOY = 0.45


def main():
    env = RetrievalQAEnv(base_cfg())
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=DEPLOY, temperature=0.03)
    td = exact.occupancy(env, teacher, DEPLOY)

    def fr(rp):
        s = LinearSoftmaxStudent(env, feats, seed=0)
        s.set_params(rp)
        return s

    def line(name, pol, cost):
        d = exact.occupancy(env, pol, DEPLOY)
        _, w, r = diagnostics(env, pol)
        print(f"{name:<26}{d.success:>7.3f}{d.avg_reward:>+9.4f}{r:>9.3f}{w:>9.2f}{cost:>9}")

    print(f"teacher: succ={td.success:.3f} reward={td.avg_reward:+.4f}  (reward is the ceiling)\n")
    print(f"{'method':<26}{'succ':>7}{'reward':>9}{'rec-m':>9}{'pol-w':>9}{'env':>9}")
    print("-" * 69)

    # references
    cfg0 = TrainConfig(steps=300, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                       dataset_size=2000, record_every=300)
    sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=0), cfg0)
    rp = sft.student.get_params()
    line("SFT", fr(rp), 0)
    line("offline OPD", train_offline_opd(env, teacher, fr(rp), fr(rp), cfg0).student, 0)
    on = train_online_opd(env, teacher, fr(rp), fr(rp), cfg0).student
    line("online OPD (target)", on, 300)
    print("-" * 69)

    # sweep chi2 threshold (higher -> fewer refreshes -> milder)
    for th in (0.2, 0.5, 1.0, 2.0, 5.0):
        res = train_refresh(env, teacher, fr(rp), fr(rp), cfg0, trigger="chi2",
                            chi2_thresh=th, budget=50, check_every=10)
        line(f"chi2 thresh={th}", res.student, res.final["refreshes"])
    print("-" * 69)

    # sweep steps at the default threshold (early stop -> less overshoot)
    for steps in (100, 150, 200):
        cfg = TrainConfig(steps=steps, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                          dataset_size=2000, record_every=steps)
        res = train_refresh(env, teacher, fr(rp), fr(rp), cfg, trigger="chi2",
                            chi2_thresh=0.2, budget=50, check_every=10)
        line(f"chi2 thresh=0.2 steps={steps}", res.student, res.final["refreshes"])


if __name__ == "__main__":
    main()
