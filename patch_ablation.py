"""chi2-triggered refresh patch + ablation [Q-vii].

The limitation: offline OPD (Lightning) is free of environment access but
collapses under the multi-turn pollution shift; online OPD recovers but pays a
fresh rollout every step. The patch is a Pareto bridge -- re-collect the frozen
dataset from the current student only when it has gone *stale*, spending a
bounded budget of env-access refreshes.

Staleness is measured by the exact trajectory chi-squared between the current
student and the policy that generated the live dataset -- the same
chi^2(pi_theta || pi_data) Theorem 3.5 ties to the offline/online gradient gap.
So the trigger is theory-driven, not a heuristic.

Ablation (the scientific control): we compare three triggers -- chi2, periodic,
and budget-matched random -- reporting success against the number of refreshes
actually spent. The honest finding (verified by a learning-rate sweep, not just
the headline run) is NOT "chi2 beats periodic on success": at a small,
well-behaved learning rate periodic works fine and is even marginally higher.
The defensible contributions are:

  1. Pareto win, robustly: chi2-refresh reaches the online ceiling at ~3% of the
     env-access cost (a handful of refreshes vs one per step) at every lr.
  2. Robustness vs fragility: chi2 never blows up because it *bounds* the
     student/data drift by construction (it fires when chi^2 crosses a threshold
     and resets it, capping the importance weights). Fixed-period refresh lets
     drift accumulate unchecked between ticks and explodes at the larger lrs
     (collapsing to the 1/K no-info floor); random placement is a high-variance
     coin-flip. Run ``patch_lr_sweep`` to see the robustness gap directly.
  3. chi2 self-tunes its spend (more refreshes while the policy is still moving,
     none once converged); the schedules spend blindly.
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, build_features,
    train_sft, train_offline_opd, train_online_opd, train_refresh,
)
from opd_toy import exact


def build(deploy=0.45):
    cfg = EnvConfig(num_answers=3, sources_per_answer=2, step_cost=0.04,
                    reconcile_cost=0.8, wrong_penalty=3.0, base_signal=1.0,
                    cross_signal=0.0, pollute_coeff=1.2)
    env = RetrievalQAEnv(cfg)
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=deploy, temperature=0.03)
    return env, feats, teacher, deploy


def main():
    env, feats, teacher, deploy = build()
    ts = exact.occupancy(env, teacher, deploy).success
    cfg = TrainConfig(steps=300, lr=0.5, collect_noise=1e-9, deploy_noise=deploy,
                      dataset_size=2000, record_every=300)
    sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=0), cfg)
    rp = sft.student.get_params()

    def fresh():
        st = LinearSoftmaxStudent(env, feats, seed=0)
        st.set_params(rp)
        return st

    off = train_offline_opd(env, teacher, fresh(), fresh(), cfg).final["success"]
    on = train_online_opd(env, teacher, fresh(), fresh(), cfg).final["success"]
    print(f"teacher@deploy={ts:.3f}   offline(0 refresh)={off:.3f}   "
          f"online(~{cfg.steps} refresh)={on:.3f}\n")

    # chi2 trigger: vary the threshold so the spent-budget varies; report the
    # frontier of (refreshes spent, success).
    print("ABLATION -- success vs env-access refreshes, by trigger placement")
    print(f"{'trigger':>10}{'knob':>10}{'refreshes':>11}{'success':>9}")
    print("-" * 40)
    for th in (0.3, 0.2, 0.1, 0.05):
        r = train_refresh(env, teacher, fresh(), fresh(), cfg, trigger="chi2",
                          chi2_thresh=th, budget=50, check_every=10)
        print(f"{'chi2':>10}{th:>10.2f}{r.final['refreshes']:>11}{r.final['success']:>9.3f}")
    for period in (100, 50, 30, 20):
        r = train_refresh(env, teacher, fresh(), fresh(), cfg, trigger="periodic",
                          period=period, budget=50, check_every=10)
        print(f"{'periodic':>10}{period:>10d}{r.final['refreshes']:>11}{r.final['success']:>9.3f}")
    for seed in range(3):
        for b in (3, 6, 12):
            r = train_refresh(env, teacher, fresh(), fresh(), cfg, trigger="random",
                              budget=b, check_every=1, refresh_seed=seed)
            print(f"{'random':>10}{('b=%d/s%d' % (b, seed)):>10}{r.final['refreshes']:>11}{r.final['success']:>9.3f}")


if __name__ == "__main__":
    main()
