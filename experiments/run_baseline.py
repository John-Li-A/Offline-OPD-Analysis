"""Baseline comparison (Question iv): SFT / online RL / offline OPD / online OPD.

Reports the four required metrics -- success rate, average reward, trajectory
length, off-support state ratio -- as mean +- std over seeds, at a fixed
(collect_noise, deploy_noise) operating point. Online OPD and online RL are the
upper-bound references; SFT is the floor; offline OPD (Lightning OPD) is the
method under test.

The operating point is chosen to expose a *regional* coverage hole (not a wired
secret action): collection is near-noiseless so the reference policy commits
after 1-2 retrievals and concentrates on shallow states, while deployment is
noisy enough that avoiding the wrong-answer penalty requires polling many
sources -- pushing the deployed student into deep states the frozen dataset
effectively never covered. Offline OPD has no gradient there; online methods,
which resample at deployment, do.
"""

from __future__ import annotations

import numpy as np

from opd_toy import (
    EnvConfig,
    LinearSoftmaxStudent,
    RetrievalQAEnv,
    TeacherPolicy,
    TrainConfig,
    build_features,
    train_offline_opd,
    train_online_opd,
    train_online_rl,
    train_sft,
)
from opd_toy import exact


def run_seed(seed: int, cfg: TrainConfig, env_cfg: EnvConfig) -> dict:
    env = RetrievalQAEnv(env_cfg)
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=cfg.deploy_noise, temperature=0.03)

    sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=seed), cfg)
    ref_params = sft.student.get_params()

    def fresh():
        st = LinearSoftmaxStudent(env, feats, seed=seed)
        st.set_params(ref_params)
        return st

    reference = fresh()
    off = train_offline_opd(env, teacher, reference, fresh(), cfg)
    on = train_online_opd(env, teacher, reference, fresh(), cfg)
    rl = train_online_rl(env, teacher, reference, fresh(), cfg)

    # SFT off-support needs ref_stats too; recompute against the same reference.
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    sft_eval = _full_eval(env, teacher, sft.student, ref_stats, cfg)
    return {
        "teacher": exact.occupancy(env, teacher, cfg.deploy_noise).success,
        "sft": sft_eval,
        "offline": off.final,
        "online_opd": on.final,
        "online_rl": rl.final,
        "teacher_collect_len": exact.occupancy(env, teacher, cfg.collect_noise).avg_length,
        "teacher_deploy_len": exact.occupancy(env, teacher, cfg.deploy_noise).avg_length,
    }


def _full_eval(env, teacher, student, ref_stats, cfg):
    from opd_toy.methods import evaluate
    return evaluate(env, teacher, student, ref_stats, cfg)


def main():
    env_cfg = EnvConfig(num_answers=3, sources_per_answer=2, horizon=6,
                        step_cost=0.04, wrong_penalty=3.0,
                        base_signal=0.97, cross_signal=0.05)
    cfg = TrainConfig(steps=200, lr=0.5, collect_noise=0.02, deploy_noise=0.55,
                      dataset_size=2000, record_every=200)
    seeds = list(range(5))

    rows = {m: [] for m in ("sft", "offline", "online_opd", "online_rl")}
    tlist, tcl, tdl = [], [], []
    for sd in seeds:
        out = run_seed(sd, cfg, env_cfg)
        tlist.append(out["teacher"])
        tcl.append(out["teacher_collect_len"])
        tdl.append(out["teacher_deploy_len"])
        for m in rows:
            rows[m].append(out[m])

    print(f"\n=== Baseline (iv): K={env_cfg.num_answers} r={env_cfg.sources_per_answer} "
          f"M={env_cfg.num_answers*env_cfg.sources_per_answer} H={env_cfg.horizon} | "
          f"collect={cfg.collect_noise} deploy={cfg.deploy_noise} "
          f"wpen={env_cfg.wrong_penalty} cost={env_cfg.step_cost} | {len(seeds)} seeds ===")
    print(f"teacher: success={np.mean(tlist):.3f}  collect_len={np.mean(tcl):.2f}  "
          f"deploy_len={np.mean(tdl):.2f}  (shallow-collect/deep-deploy gap = "
          f"{np.mean(tdl)-np.mean(tcl):.2f})\n")
    print(f"{'method':<14}{'success':>16}{'avg_reward':>16}{'traj_len':>14}{'off_support':>14}")
    print("-" * 74)
    for m in ("online_rl", "online_opd", "offline", "sft"):
        succ = np.array([r["success"] for r in rows[m]])
        rew = np.array([r["reward"] for r in rows[m]])
        ln = np.array([r["length"] for r in rows[m]])
        osr = np.array([r.get("off_support_ratio", 0.0) for r in rows[m]])
        print(f"{m:<14}{succ.mean():>8.3f}+-{succ.std():<6.3f}"
              f"{rew.mean():>10.3f}{ln.mean():>14.2f}{osr.mean():>14.3f}")


if __name__ == "__main__":
    main()
