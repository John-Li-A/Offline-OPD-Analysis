"""Four-method baseline table [Q-iv]: SFT / offline OPD / online OPD / online RL.

Compares the four recipes on the four metrics the assignment asks for -- success
rate, average reward, trajectory length, and off-support ratio -- for both the
linear (generalising) and tabular (no-sharing) students, averaged over seeds
with standard deviations.

* SFT: behaviour-clone the teacher on clean collection. The reference floor.
* offline OPD (Lightning): frozen clean reference visitation + action weights.
* online OPD: resample from the student at deployment noise (the upper bound).
* online RL: REINFORCE on terminal reward at deployment (non-distillation ref).

The headline this table backs up: offline OPD barely moves off the SFT floor and
sits far below the online methods, and the linear student -- which generalises --
collapses at least as hard as the tabular one. Online OPD and online RL both
recover toward the teacher ceiling because they resample into the polluted trap
states the clean reference never covers.
"""

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, TabularStudent, build_features,
    train_sft, train_offline_opd, train_online_opd, train_online_rl,
)
from opd_toy import exact

METHODS = ("sft", "offline", "online_opd", "online_rl")
METRICS = ("success", "reward", "length", "off_support_ratio")


def make_student(kind, env, feats, seed):
    if kind == "linear":
        return LinearSoftmaxStudent(env, feats, seed=seed)
    return TabularStudent(env, seed=seed)


def run_one(kind, env, feats, teacher, cfg, seed):
    """Train all four methods from a shared SFT reference; return final metrics."""
    sft = train_sft(env, teacher, make_student(kind, env, feats, seed), cfg)
    ref_params = sft.student.get_params()

    def fresh():
        st = make_student(kind, env, feats, seed)
        st.set_params(ref_params)
        return st

    reference = fresh()
    results = {"sft": sft.final}
    results["offline"] = train_offline_opd(env, teacher, fresh(), fresh(), cfg).final
    results["online_opd"] = train_online_opd(env, teacher, fresh(), fresh(), cfg).final
    results["online_rl"] = train_online_rl(env, teacher, fresh(), fresh(), cfg).final
    # SFT has no off_support recorded (no reference yet); fill from a ref eval.
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    results["sft"] = {**results["sft"],
                      "off_support_ratio": exact_off_support(env, sft.student, ref_stats, cfg)}
    return results


def exact_off_support(env, student, ref_stats, cfg):
    dep = exact.occupancy(env, student, cfg.deploy_noise)
    expected = ref_stats.visit * cfg.dataset_size
    covered = expected >= 1.0
    off = float(dep.visit[~covered].sum())
    tot = float(dep.visit.sum())
    return off / tot if tot > 0 else 0.0


def main(seeds=(0,)):
    """Single seed by default: the gradients are exact (sampling-free), so runs
    from a fixed init are deterministic and seed variance is ~0. Pass more seeds
    only to vary the student initialisation."""
    base = EnvConfig(num_answers=3, sources_per_answer=2, step_cost=0.04,
                     reconcile_cost=0.8, wrong_penalty=3.0, base_signal=1.0,
                     cross_signal=0.0, pollute_coeff=1.2)
    env = RetrievalQAEnv(base)
    feats = build_features(env)
    deploy = 0.45
    teacher = TeacherPolicy(env, noise=deploy, temperature=0.03)
    tstats = exact.occupancy(env, teacher, deploy)
    print(f"teacher@deploy: success={tstats.success:.3f} reward={tstats.avg_reward:.3f} "
          f"length={tstats.avg_length:.2f}\n")

    for kind in ("linear", "tabular"):
        steps = 400 if kind == "linear" else 800
        lr = 0.5 if kind == "linear" else 1.0
        cfg = TrainConfig(steps=steps, lr=lr, collect_noise=1e-9, deploy_noise=deploy,
                          dataset_size=2000, record_every=steps)
        # acc[method][metric] = list over seeds
        acc = {m: {k: [] for k in METRICS} for m in METHODS}
        for seed in seeds:
            res = run_one(kind, env, feats, teacher, cfg, seed)
            for m in METHODS:
                for k in METRICS:
                    acc[m][k].append(res[m][k])
        print(f"=== {kind} student (steps={steps}, seeds={list(seeds)}) ===")
        print(f"{'method':>11}{'success':>16}{'reward':>16}{'length':>14}{'off_supp':>14}")
        for m in METHODS:
            cells = []
            for k in METRICS:
                arr = np.array(acc[m][k])
                cells.append(f"{arr.mean():.3f}+-{arr.std():.3f}")
            print(f"{m:>11}{cells[0]:>16}{cells[1]:>16}{cells[2]:>14}{cells[3]:>14}")
        print()


if __name__ == "__main__":
    main()
