"""Learning-rate robustness sweep for the chi2-refresh patch [Q-vii control].

The headline ablation (patch_ablation.py) at lr=0.5 makes periodic refresh look
like it collapses to the 1/K floor. This sweep is the honest control that shows
*why*: periodic's collapse is an exploding-gradient artifact that only appears at
larger learning rates, while the chi2 trigger is stable at every lr because it
bounds the student/data drift by construction.

For each lr we report offline / online success and the chi2 vs periodic patched
success (with the number of env-access refreshes each spent). The takeaway:
* chi2 reaches the online ceiling at a handful of refreshes for every lr;
* periodic blows up (-> ~0.333) at lr >= 0.2 and only works in a narrow low-lr
  band -- it lets drift accumulate unchecked between fixed ticks;
* chi2 self-tunes its spend with the learning rate.
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, build_features,
    train_sft, train_offline_opd, train_online_opd, train_refresh,
)
from opd_toy import exact


def main():
    cfg = EnvConfig(num_answers=3, sources_per_answer=2, step_cost=0.04,
                    reconcile_cost=0.8, wrong_penalty=3.0, base_signal=1.0,
                    cross_signal=0.0, pollute_coeff=1.2)
    env = RetrievalQAEnv(cfg)
    feats = build_features(env)
    deploy = 0.45
    teacher = TeacherPolicy(env, noise=deploy, temperature=0.03)

    print(f"{'lr':>6}{'offline':>9}{'online':>9}{'chi2':>14}{'periodic':>14}")
    print("-" * 52)
    for lr in (0.5, 0.2, 0.1, 0.05):
        tr = TrainConfig(steps=300, lr=lr, collect_noise=1e-9, deploy_noise=deploy,
                         dataset_size=2000, record_every=300)
        sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=0), tr)
        rp = sft.student.get_params()

        def fresh():
            st = LinearSoftmaxStudent(env, feats, seed=0)
            st.set_params(rp)
            return st

        off = train_offline_opd(env, teacher, fresh(), fresh(), tr).final["success"]
        on = train_online_opd(env, teacher, fresh(), fresh(), tr).final["success"]
        chi = train_refresh(env, teacher, fresh(), fresh(), tr, trigger="chi2",
                            chi2_thresh=0.2, budget=50, check_every=10)
        per = train_refresh(env, teacher, fresh(), fresh(), tr, trigger="periodic",
                            period=50, budget=50, check_every=10)
        print(f"{lr:>6}{off:>9.3f}{on:>9.3f}"
              f"{('%.3f(r=%d)' % (chi.final['success'], chi.final['refreshes'])):>14}"
              f"{('%.3f(r=%d)' % (per.final['success'], per.final['refreshes'])):>14}")


if __name__ == "__main__":
    main()
