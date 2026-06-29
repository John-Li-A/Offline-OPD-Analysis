"""Control: does the *free* full-distribution advantage patch (Rang-style,
Lightning OPD patch 2) rescue our task --- or does the feature collision defeat
it? [supports Q-vii / Q-vi discussion]

Lightning offline OPD weights the advantage by the behaviour policy at the
sampled action; the full-distribution fix instead weights by the *teacher*
distribution, with the same frozen reference visitation and zero env access:

    g_offline   = sum_s d_ref(s) sum_a pi_ref(a|s) A(s,a) grad log pi(a|s)
    g_fulldist  = sum_s d_ref(s) sum_a pi_T (a|s) A(s,a) grad log pi(a|s)   <-- patch

That is exactly opd_gradient(., ref_stats, action_policy=teacher, adv).

Prediction (because in our task generalisation is the *accomplice*, not the
rescuer): the free patch cannot recover here.
  * At collect_noise ~ 0 the polluted states have ~0 mass in the frozen dataset,
    so there is nothing to distil there -> stuck near the SFT/offline floor.
  * Raising collect_noise surfaces polluted states, but the linear student then
    generalises "commit the peak" from the feature-identical clean states into
    the polluted ones, fighting the teacher's reconcile signal -> still poor.
If so, the conclusion is sharp: in a feature-collision multi-turn trap, NO
zero-env-access patch suffices; one must buy back coverage (our chi2-refresh).
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, build_features, train_sft, train_offline_opd, train_online_opd,
)
from opd_toy import exact

DEPLOY = 0.45


def base_cfg(**kw):
    d = dict(num_answers=3, sources_per_answer=2, step_cost=0.04, reconcile_cost=0.8,
             wrong_penalty=3.0, base_signal=1.0, cross_signal=0.0, pollute_coeff=1.2)
    d.update(kw)
    return EnvConfig(**d)


def train_fulldist(env, teacher, reference, student, cfg):
    """Offline OPD but with the teacher distribution as the action weighting
    (full-distribution advantage). Frozen reference visitation; zero env access."""
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    for _ in range(cfg.steps):
        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        grad = exact.opd_gradient(env, student, ref_stats, teacher, adv)  # teacher-weighted
        student.set_params(student.get_params() + cfg.lr * grad)
    return exact.occupancy(env, student, DEPLOY).success


def main():
    print(f"{'collect':>8}{'offsup@ref':>12}{'offline':>9}{'fulldist':>10}{'online':>9}{'teacher':>9}")
    print("-" * 57)
    for collect in (1e-9, 0.05, 0.15, 0.30, 0.45):
        env = RetrievalQAEnv(base_cfg())
        feats = build_features(env)
        teacher = TeacherPolicy(env, noise=DEPLOY, temperature=0.03)
        tsucc = exact.occupancy(env, teacher, DEPLOY).success
        cfg = TrainConfig(steps=400, lr=0.5, collect_noise=collect, deploy_noise=DEPLOY,
                          dataset_size=2000, record_every=400)
        sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=0), cfg)
        rp = sft.student.get_params()

        def fresh():
            st = LinearSoftmaxStudent(env, feats, seed=0); st.set_params(rp); return st

        off = train_offline_opd(env, teacher, fresh(), fresh(), cfg)
        fd = train_fulldist(env, teacher, fresh(), fresh(), cfg)
        on = train_online_opd(env, teacher, fresh(), fresh(), cfg)
        osr = off.final["off_support_ratio"]
        print(f"{collect:>8.3g}{osr:>12.3f}{off.final['success']:>9.3f}"
              f"{fd:>10.3f}{on.final['success']:>9.3f}{tsucc:>9.3f}")


if __name__ == "__main__":
    main()
