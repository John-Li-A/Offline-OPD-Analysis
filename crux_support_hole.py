"""Decisive test of the principled emergent support hole.

Make collection (near-)deterministic: collect_noise = 0, base_signal = 1.0,
cross_signal = 0.0. Then per answer exactly one evidence pattern is reachable
during collection, so an entire REGION of contradictory-evidence info-states has
literally zero probability under pi_ref. Deployment noise makes that region
common. This violates support coverage (Asm 3.2) by construction of the noise
gap -- not by a hand-wired action -- and the "recovery" uses ordinary actions in
states pi_ref never saw.

Predictions:
* tabular offline: zero gradient on the off-support region forever -> stuck at
  SFT there -> fails, and does NOT close with more steps (unlike the covered
  case where offline crept up to online).
* tabular online: visits the region -> recovers toward teacher.
* linear offline: generalisation may rescue it (the hidden-variable finding).
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


def main():
    # Deterministic collection: clean signatures, exact support hole at deploy.
    env_cfg = EnvConfig(num_answers=3, sources_per_answer=2, horizon=6,
                        step_cost=0.04, wrong_penalty=3.0,
                        base_signal=1.0, cross_signal=0.0)
    env = RetrievalQAEnv(env_cfg)
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=0.45, temperature=0.03)
    tsucc = exact.occupancy(env, teacher, 0.45).success
    print(f"teacher@deploy(0.45) success={tsucc:.3f}  (deterministic collection)\n")

    for kind in ("tabular", "linear"):
        steps = 1200 if kind == "tabular" else 400
        lr = 1.0 if kind == "tabular" else 0.5
        cfg = TrainConfig(steps=steps, lr=lr, collect_noise=1e-9, deploy_noise=0.45,
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
        print(f"  final: offline={off.final['success']:.3f}  online={on.final['success']:.3f}  "
              f"teacher={tsucc:.3f}  (online-offline={on.final['success']-off.final['success']:.3f})\n")


if __name__ == "__main__":
    main()
