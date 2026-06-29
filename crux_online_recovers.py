"""Crux test: does tabular ONLINE recover toward teacher with enough steps,
while offline plateaus below? This decides whether the clean
"offline fails / online recovers" story holds, or whether both are stuck
(making "generalisation is the hidden variable" + vacuous-bound the real thesis).

We train tabular offline OPD, online OPD, and online RL for many steps at the
highest-ceiling operating point, logging the success trajectory so we can see
whether online is still climbing or has plateaued.
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    TabularStudent, build_features,
    train_sft, train_offline_opd, train_online_opd, train_online_rl,
)
from opd_toy import exact


def main():
    env_cfg = EnvConfig(num_answers=3, sources_per_answer=2, horizon=6,
                        step_cost=0.04, wrong_penalty=3.0,
                        base_signal=0.97, cross_signal=0.05)
    env = RetrievalQAEnv(env_cfg)
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=0.45, temperature=0.03)
    tsucc = exact.occupancy(env, teacher, 0.45).success
    print(f"teacher@deploy(0.45) success={tsucc:.3f}\n")

    cfg = TrainConfig(steps=1500, lr=1.0, collect_noise=0.02, deploy_noise=0.45,
                      dataset_size=2000, record_every=250)
    sft = train_sft(env, teacher, TabularStudent(env, seed=0), cfg)
    ref_params = sft.student.get_params()
    print(f"tabular SFT floor: {sft.final['success']:.3f}\n")

    def fresh():
        st = TabularStudent(env, seed=0)
        st.set_params(ref_params)
        return st

    reference = fresh()
    print("Training tabular offline / online_opd / online_rl for 1500 steps...\n")
    off = train_offline_opd(env, teacher, reference, fresh(), cfg)
    on = train_online_opd(env, teacher, reference, fresh(), cfg)
    rl = train_online_rl(env, teacher, reference, fresh(), cfg)

    print(f"{'step':>6}{'offline':>10}{'online_opd':>12}{'online_rl':>12}")
    print("-" * 40)
    for i in range(len(off.history["step"])):
        print(f"{off.history['step'][i]:>6}{off.history['success'][i]:>10.3f}"
              f"{on.history['success'][i]:>12.3f}{rl.history['success'][i]:>12.3f}")
    print(f"\nteacher={tsucc:.3f}  "
          f"final: offline={off.final['success']:.3f} "
          f"online_opd={on.final['success']:.3f} online_rl={rl.final['success']:.3f}")


if __name__ == "__main__":
    main()
