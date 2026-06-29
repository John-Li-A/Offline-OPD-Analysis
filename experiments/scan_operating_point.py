"""Operating-point scan: find the cleanest tabular offline failure.

The failure lives in the tabular student (no generalisation to rescue offline
OPD on off-support deep states). We scan deploy_noise x wrong_penalty and report,
for the tabular student, the SFT floor / offline / online triple plus the teacher
ceiling. We want: a healthy ceiling (room to fail), online clearly recovering
toward the teacher, and offline stuck well below online -- the largest clean
(online - offline) gap.
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    TabularStudent, build_features,
    train_sft, train_offline_opd, train_online_opd,
)
from opd_toy import exact


def trial(deploy_noise, wpen, cost, steps=400, lr=1.0):
    env_cfg = EnvConfig(num_answers=3, sources_per_answer=2, horizon=6,
                        step_cost=cost, wrong_penalty=wpen,
                        base_signal=0.97, cross_signal=0.05)
    env = RetrievalQAEnv(env_cfg)
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=deploy_noise, temperature=0.03)
    tsucc = exact.occupancy(env, teacher, deploy_noise).success
    cfg = TrainConfig(steps=steps, lr=lr, collect_noise=0.02, deploy_noise=deploy_noise,
                      dataset_size=2000, record_every=steps)

    sft = train_sft(env, teacher, TabularStudent(env, seed=0), cfg)
    ref_params = sft.student.get_params()

    def fresh():
        st = TabularStudent(env, seed=0)
        st.set_params(ref_params)
        return st

    reference = fresh()
    off = train_offline_opd(env, teacher, reference, fresh(), cfg)
    on = train_online_opd(env, teacher, reference, fresh(), cfg)
    return tsucc, sft.final["success"], off.final["success"], on.final["success"], \
        off.final.get("off_support_ratio", 0.0)


def main():
    print(f"{'dep':>5}{'wpen':>6}{'cost':>6}{'teach':>8}{'sft':>8}{'off':>8}{'on':>8}"
          f"{'on-off':>8}{'teach-off':>10}{'offsup':>8}")
    print("-" * 76)
    best = None
    for dep in (0.45, 0.55, 0.65):
        for wpen in (3.0, 5.0):
            tsucc, sft, off, on, osr = trial(dep, wpen, 0.04)
            gap = on - off
            flag = ""
            if tsucc > 0.5 and gap > 0.05:
                flag = "  <=="
                if best is None or gap > best[0]:
                    best = (gap, dep, wpen)
            print(f"{dep:>5}{wpen:>6}{0.04:>6}{tsucc:>8.3f}{sft:>8.3f}{off:>8.3f}"
                  f"{on:>8.3f}{gap:>8.3f}{tsucc-off:>10.3f}{osr:>8.3f}{flag}")
    if best:
        print(f"\nbest clean gap: dep={best[1]} wpen={best[2]} (online-offline={best[0]:.3f})")


if __name__ == "__main__":
    main()
