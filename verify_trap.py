"""Crux verification of the multi-turn trap: does offline OPD collapse for a
reason bound to multi-turn dynamics rather than to "tabular can't generalise"?

The decisive prediction (the thing the competitor's report cannot show): even
the LINEAR (generalising) student collapses offline, because generalisation
extrapolates the on-path "never reconcile" behaviour into the deployment
conflict states -- turning generalisation from rescuer into accomplice. Online
OPD, which resamples into conflict states and sees the teacher reconcile,
recovers for BOTH students.

Clean collection (collect_noise~0, base_signal=1, cross_signal=0) produces no
conflicted states, so pi_ref never reconciles. Deployment noise makes conflicted
states common; reconcile is the only reliable recovery there. The coverage hole
emerges from the noise gap and the multi-turn evidence dynamics, not from any
hand-wired action probability.
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, TabularStudent, build_features,
    train_sft, train_offline_opd, train_online_opd,
)
from opd_toy import exact


def make_student(kind, env, feats):
    return LinearSoftmaxStudent(env, feats, seed=0) if kind == "linear" else TabularStudent(env, seed=0)


def reconcile_mass(env, student, noise):
    """Deployment probability mass placed on the reconcile action (a diagnostic
    of whether the policy actually recovers via reconcile or not)."""
    stats = exact.occupancy(env, student, noise)
    m = 0.0
    for s in range(env.num_states):
        if stats.visit[s] <= 0:
            continue
        m += stats.visit[s] * student.probs(s)[env.reconcile_action]
    return m


def main():
    env_cfg = EnvConfig(num_answers=3, sources_per_answer=2, horizon=8,
                        step_cost=0.04, wrong_penalty=3.0,
                        base_signal=1.0, cross_signal=0.0)
    env = RetrievalQAEnv(env_cfg)
    feats = build_features(env)
    deploy = 0.45
    teacher = TeacherPolicy(env, noise=deploy, temperature=0.03)
    tstats = exact.occupancy(env, teacher, deploy)
    print(f"states={env.num_states}  teacher@deploy({deploy}) success={tstats.success:.3f}  "
          f"teacher reconcile-mass={reconcile_mass(env, teacher, deploy):.3f}\n")

    for kind in ("linear", "tabular"):
        steps = 500 if kind == "linear" else 1200
        lr = 0.5 if kind == "linear" else 1.0
        cfg = TrainConfig(steps=steps, lr=lr, collect_noise=1e-9, deploy_noise=deploy,
                          dataset_size=2000, record_every=steps // 4)
        sft = train_sft(env, teacher, make_student(kind, env, feats), cfg)
        ref_params = sft.student.get_params()

        def fresh():
            st = make_student(kind, env, feats)
            st.set_params(ref_params)
            return st

        reference = fresh()
        off = train_offline_opd(env, teacher, reference, fresh(), cfg)
        on = train_online_opd(env, teacher, reference, fresh(), cfg)
        print(f"--- {kind} student (steps={steps}) ---")
        print(f"  SFT floor: {sft.final['success']:.3f}   off_support@SFT-ref: "
              f"{off.history['off_support_ratio'][0]:.3f}")
        print(f"  {'step':>6}{'offline':>10}{'online':>10}")
        for i in range(len(off.history['step'])):
            print(f"  {off.history['step'][i]:>6}{off.history['success'][i]:>10.3f}"
                  f"{on.history['success'][i]:>10.3f}")
        rec_off = reconcile_mass(env, off.student, deploy)
        rec_on = reconcile_mass(env, on.student, deploy)
        print(f"  final: offline={off.final['success']:.3f}  online={on.final['success']:.3f}  "
              f"teacher={tstats.success:.3f}  (online-offline={on.final['success']-off.final['success']:.3f})")
        print(f"  reconcile-mass: offline={rec_off:.3f}  online={rec_on:.3f}\n")


if __name__ == "__main__":
    main()
