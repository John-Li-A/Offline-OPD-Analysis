"""Stage-1 smoke test: does the phenomenon exist before we invest in the report?

Runs SFT, offline OPD, online OPD, and online RL across a handful of seeds at a
fixed (collect_noise, deploy_noise) operating point and prints the four required
metrics plus the gradient-level diagnostics. We are checking three predictions:

1. offline OPD lands at or below the SFT floor (it fails), with high variance;
2. online OPD and online RL recover (near the teacher);
3. during offline training chi^2 grows and the measured gradient gap stays under
   -- but trends toward -- the Theorem 3.5 bound.

If these hold, the design is sound and we proceed to the full study. If not, we
tune the noise gap / overlap / step cost here, cheaply, before writing anything.
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


def run_seed(seed: int, cfg: TrainConfig, env_cfg: EnvConfig) -> dict:
    env = RetrievalQAEnv(env_cfg)
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=cfg.deploy_noise, temperature=0.03)

    # SFT -> pi_ref
    sft_student = LinearSoftmaxStudent(env, feats, seed=seed)
    sft = train_sft(env, teacher, sft_student, cfg)
    ref_params = sft.student.get_params()

    def fresh():
        st = LinearSoftmaxStudent(env, feats, seed=seed)
        st.set_params(ref_params)
        return st

    reference = fresh()  # frozen pi_ref for OPD

    off = train_offline_opd(env, teacher, reference, fresh(), cfg)
    on = train_online_opd(env, teacher, reference, fresh(), cfg)
    rl = train_online_rl(env, teacher, reference, fresh(), cfg)

    teacher_succ = _teacher_success(env, teacher, cfg.deploy_noise)
    return {
        "teacher": teacher_succ,
        "sft": sft.final,
        "offline": off.final,
        "online_opd": on.final,
        "online_rl": rl.final,
        "offline_hist": off.history,
    }


def _teacher_success(env, teacher, noise):
    from opd_toy import exact

    return exact.occupancy(env, teacher, noise).success


def main():
    env_cfg = EnvConfig(num_answers=3, sources_per_answer=2, horizon=6,
                        step_cost=0.1, wrong_penalty=2.0,
                        base_signal=0.95, cross_signal=0.08)
    cfg = TrainConfig(steps=150, lr=0.5, collect_noise=0.02, deploy_noise=0.6, record_every=50)
    seeds = list(range(2))

    # Mechanism check: same teacher is shallow at collect, deep at deploy.
    from opd_toy import exact
    _env = RetrievalQAEnv(env_cfg)
    _t = TeacherPolicy(_env, noise=cfg.deploy_noise, temperature=0.03)
    lc = exact.occupancy(_env, _t, cfg.collect_noise)
    ld = exact.occupancy(_env, _t, cfg.deploy_noise)
    print(f"teacher depth: collect_len={lc.avg_length:.2f} (succ {lc.success:.3f}) | "
          f"deploy_len={ld.avg_length:.2f} (succ {ld.success:.3f})")

    rows = {m: [] for m in ("teacher", "sft", "offline", "online_opd", "online_rl")}
    last_hist = None
    for sd in seeds:
        out = run_seed(sd, cfg, env_cfg)
        rows["teacher"].append(out["teacher"])
        for m in ("sft", "offline", "online_opd", "online_rl"):
            rows[m].append(out[m])
        last_hist = out["offline_hist"]

    print(f"\n=== Smoke test: K={env_cfg.num_answers}, H={env_cfg.horizon}, "
          f"collect={cfg.collect_noise}, deploy={cfg.deploy_noise}, {len(seeds)} seeds ===\n")
    print(f"{'method':<14}{'success':>18}{'reward':>16}{'length':>14}{'off_supp':>12}")
    print("-" * 74)
    print(f"{'teacher':<14}{np.mean(rows['teacher']):>18.3f}{'':>16}{'':>14}{'':>12}")
    for m in ("online_rl", "online_opd", "offline", "sft"):
        succ = np.array([r["success"] for r in rows[m]])
        rew = np.array([r["reward"] for r in rows[m]])
        ln = np.array([r["length"] for r in rows[m]])
        osr = np.array([r.get("off_support_ratio", 0.0) for r in rows[m]])
        print(f"{m:<14}{succ.mean():>10.3f}+-{succ.std():<5.3f}"
              f"{rew.mean():>10.3f}{ln.mean():>14.2f}{osr.mean():>12.3f}")

    print("\n--- offline OPD training dynamics (last seed) ---")
    print(f"{'step':>6}{'success':>10}{'off_supp':>10}{'chi2':>12}{'grad_gap':>12}{'bound':>12}")
    h = last_hist
    for i in range(len(h["step"])):
        print(f"{h['step'][i]:>6}{h['success'][i]:>10.3f}{h['off_support_ratio'][i]:>10.3f}"
              f"{h['chi2'][i]:>12.4f}{h['grad_gap'][i]:>12.5f}{h['bound'][i]:>12.5f}")


if __name__ == "__main__":
    main()
