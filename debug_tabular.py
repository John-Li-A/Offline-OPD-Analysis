"""Decisive test: linear (generalising) vs tabular (no generalisation) student.

Hypothesis after the smoke runs: in a small smooth MDP there is no *hard* coverage
hole, so the linear student's generalisation fills the weakly-covered deep states
and offline OPD succeeds. Strip generalisation (tabular: gradient is exactly zero
where pi_ref never visits) and offline OPD should collapse toward the SFT floor on
the deep deploy states, while online OPD -- which visits them -- still recovers.

If this holds, the linear-vs-tabular contrast IS the phenomenon: offline OPD's
success is a generalisation effect, not a property of the distilled signal.
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, TabularStudent, build_features,
    train_sft, train_offline_opd, train_online_opd,
)
from opd_toy import exact


def make_student(kind, env, feats, seed):
    if kind == "linear":
        return LinearSoftmaxStudent(env, feats, seed=seed)
    return TabularStudent(env, seed=seed)


def main():
    env_cfg = EnvConfig(num_answers=3, sources_per_answer=2, horizon=6,
                        step_cost=0.04, wrong_penalty=3.0,
                        base_signal=0.97, cross_signal=0.05)
    env = RetrievalQAEnv(env_cfg)
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=0.55, temperature=0.03)
    teacher_succ = exact.occupancy(env, teacher, 0.55).success
    print(f"teacher@deploy success={teacher_succ:.3f}\n")
    print(f"{'student':>8}{'method':>9}{'steps':>7}{'success':>10}{'off_supp':>10}")
    print("-" * 46)
    for kind in ("linear", "tabular"):
        steps = 200 if kind == "linear" else 600
        lr = 0.5 if kind == "linear" else 1.0
        cfg = TrainConfig(steps=steps, lr=lr, collect_noise=0.02, deploy_noise=0.55,
                          dataset_size=2000, record_every=steps)
        sft = train_sft(env, teacher, make_student(kind, env, feats, 0), cfg)
        ref_params = sft.student.get_params()

        def fresh():
            st = make_student(kind, env, feats, 0)
            st.set_params(ref_params)
            return st

        reference = fresh()
        off = train_offline_opd(env, teacher, reference, fresh(), cfg)
        on = train_online_opd(env, teacher, reference, fresh(), cfg)
        print(f"{kind:>8}{'sft':>9}{steps:>7}{sft.final['success']:>10.3f}{'-':>10}")
        print(f"{kind:>8}{'offline':>9}{steps:>7}{off.final['success']:>10.3f}"
              f"{off.final.get('off_support_ratio', 0):>10.3f}")
        print(f"{kind:>8}{'online':>9}{steps:>7}{on.final['success']:>10.3f}"
              f"{on.final.get('off_support_ratio', 0):>10.3f}")
        print()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
