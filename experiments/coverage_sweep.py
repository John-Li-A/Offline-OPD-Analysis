"""Coverage sweep: when does offline OPD work, and when does it fail? [Q-v]

The pollution rate ``rho(noise) = pollute_coeff * noise`` is the single knob that
controls how much deployment mass lands in the off-support trap region. Sweeping
``pollute_coeff`` from 0 upward traces a continuous spectrum:

* pollute_coeff = 0: no pollution, deployment == collection support. Offline OPD
  should match online (Theorem 3.6 shared fixed point) -- offline does NOT fail.
* growing pollute_coeff: more deployment mass falls on polluted states the clean
  reference never covers, the off-support ratio rises, and offline OPD's success
  peels away from online's while online keeps recovering.

So "offline OPD fails under distribution shift" is not a binary caricature: it is
a smooth function of how far deployment drifts off the collection support, with
``off_support_ratio`` as the controlled independent variable. We report the
linear student (the headline: even a generaliser collapses) and use few steps +
endpoint-only diagnostics to stay fast.
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, build_features,
    train_sft, train_offline_opd, train_online_opd,
)
from opd_toy import exact


def trial(pollute_coeff, deploy=0.45, steps=300, lr=0.5, seed=0):
    cfg = EnvConfig(num_answers=3, sources_per_answer=2, step_cost=0.04,
                    reconcile_cost=0.8, wrong_penalty=3.0, base_signal=1.0,
                    cross_signal=0.0, pollute_coeff=pollute_coeff)
    env = RetrievalQAEnv(cfg)
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=deploy, temperature=0.03)
    tsucc = exact.occupancy(env, teacher, deploy).success
    cfg_tr = TrainConfig(steps=steps, lr=lr, collect_noise=1e-9, deploy_noise=deploy,
                         dataset_size=2000, record_every=steps)  # endpoint-only diag

    sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=seed), cfg_tr)
    ref_params = sft.student.get_params()

    def fresh():
        st = LinearSoftmaxStudent(env, feats, seed=seed)
        st.set_params(ref_params)
        return st

    off = train_offline_opd(env, teacher, fresh(), fresh(), cfg_tr)
    on = train_online_opd(env, teacher, fresh(), fresh(), cfg_tr)
    return {
        "rho_deploy": env.rho(deploy),
        "teacher": tsucc,
        "sft": sft.final["success"],
        "offline": off.final["success"],
        "online": on.final["success"],
        "off_support": off.final["off_support_ratio"],
    }


def main():
    print(f"{'p_coef':>7}{'rho_dep':>8}{'offsup':>8}{'teacher':>8}{'sft':>7}"
          f"{'offline':>8}{'online':>8}{'on-off':>8}")
    print("-" * 64)
    for pc in (0.0, 0.2, 0.4, 0.8, 1.2, 1.6):
        r = trial(pc)
        print(f"{pc:>7.1f}{r['rho_deploy']:>8.3f}{r['off_support']:>8.3f}"
              f"{r['teacher']:>8.3f}{r['sft']:>7.3f}{r['offline']:>8.3f}"
              f"{r['online']:>8.3f}{r['online']-r['offline']:>8.3f}")


if __name__ == "__main__":
    main()
