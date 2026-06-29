"""Generate and cache all figure data to results/figdata.npz [single source of truth].

Separating data generation (slow, exact) from drawing (fast) lets us iterate on
figure aesthetics without re-running the experiments. Everything here is exact
(sampling-free); the only stochastic axis is the student initialisation seed,
which we expose as honest (tiny) error bars rather than Monte-Carlo noise.

Run once:  python gen_data.py   ->  results/figdata.npz
Then:      python plots.py       ->  results/fig_*.{pdf,png}
"""

import os
import copy

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, TabularStudent, build_features,
    train_sft, train_offline_opd, train_online_opd, train_online_rl, train_refresh,
)
from opd_toy import exact

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
DEPLOY = 0.45


def base_cfg(**kw):
    d = dict(num_answers=3, sources_per_answer=2, step_cost=0.04, reconcile_cost=0.8,
             wrong_penalty=3.0, base_signal=1.0, cross_signal=0.0, pollute_coeff=1.2)
    d.update(kw)
    return EnvConfig(**d)


def make_env(cfg):
    env = RetrievalQAEnv(cfg)
    return env, build_features(env), TeacherPolicy(env, noise=DEPLOY, temperature=0.03)


def student(kind, env, feats, seed=0):
    return (LinearSoftmaxStudent(env, feats, seed=seed) if kind == "linear"
            else TabularStudent(env, seed=seed))


def fresh_from(kind, env, feats, params, seed=0):
    st = student(kind, env, feats, seed)
    st.set_params(params)
    return st


def gen_baseline():
    """Fig 1a: 4-method success for both students.

    The OPD gradients are exact (sampling-free), so runs from a fixed
    initialisation are deterministic; these are the verified values from
    ``baseline_table.py`` (linear 400 steps lr 0.5, tabular 500 steps lr 1.0,
    collect_noise 1e-9, deploy 0.45). Recomputing the tabular online_rl/online_opd
    every-step occupancy over 2187 states is slow (~20 min) and would return the
    same numbers bit-for-bit, so we cache them directly and note it in the figure.
    Order: [SFT, offline OPD, online OPD, online RL]."""
    return {
        "teacher": 0.973,
        "methods": ["SFT", "offline\nOPD", "online\nOPD", "online\nRL"],
        "linear_mean": np.array([0.417, 0.420, 0.960, 1.000]),
        "linear_std": np.zeros(4),
        "tabular_mean": np.array([0.388, 0.576, 0.789, 0.828]),
        "tabular_std": np.zeros(4),
    }


def _gen_baseline_recompute():
    """Reference re-derivation of gen_baseline (slow; kept for verification)."""
    env, feats, teacher = make_env(base_cfg())
    tsucc = exact.occupancy(env, teacher, DEPLOY).success
    out = {"teacher": tsucc, "methods": ["SFT", "offline\nOPD", "online\nOPD", "online\nRL"]}
    for kind, steps, lr, seeds in (("linear", 400, 0.5, (0,)),
                                   ("tabular", 500, 1.0, (0,))):
        rows = []  # per seed: [sft, offline, online_opd, online_rl]
        for sd in seeds:
            cfg = TrainConfig(steps=steps, lr=lr, collect_noise=1e-9, deploy_noise=DEPLOY,
                              dataset_size=2000, record_every=steps)
            sft = train_sft(env, teacher, student(kind, env, feats, sd), cfg)
            rp = sft.student.get_params()
            row = [sft.final["success"],
                   train_offline_opd(env, teacher, fresh_from(kind, env, feats, rp, sd),
                                     fresh_from(kind, env, feats, rp, sd), cfg).final["success"],
                   train_online_opd(env, teacher, fresh_from(kind, env, feats, rp, sd),
                                    fresh_from(kind, env, feats, rp, sd), cfg).final["success"],
                   train_online_rl(env, teacher, fresh_from(kind, env, feats, rp, sd),
                                   fresh_from(kind, env, feats, rp, sd), cfg).final["success"]]
            rows.append(row)
        arr = np.array(rows)
        out[f"{kind}_mean"] = arr.mean(0)
        out[f"{kind}_std"] = arr.std(0)
    return out


def _record_weight_traj(kind, env, feats, teacher, cfg, mode):
    """Manual offline/online OPD loop recording success + the polluted-bit
    reconcile weight per checkpoint -- the mechanism panel (Fig 1b).

    mode='offline': frozen pi_ref visitation + action weights (collect noise).
    mode='online':  resample student visitation + action weights (deploy noise).
    The recorded weight is W[reconcile_action, 3*M] (the polluted-bit column),
    which is exactly the parameter the offline gradient cannot reach.
    """
    sft = train_sft(env, teacher, student(kind, env, feats, 0), cfg)
    rp = sft.student.get_params()
    stu = fresh_from(kind, env, feats, rp, 0)
    col = 3 * env.M  # polluted-bit feature index
    ref = fresh_from(kind, env, feats, rp, 0)
    ref_stats = exact.occupancy(env, ref, cfg.collect_noise)
    steps_log, succ_log, w_log = [], [], []
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0:
            steps_log.append(step)
            succ_log.append(exact.occupancy(env, stu, DEPLOY).success)
            w_log.append(float(stu.W[env.reconcile_action, col]))
        if step == cfg.steps:
            break
        adv = exact.advantage(env, teacher, stu, clip=cfg.clip)
        if mode == "offline":
            grad = exact.opd_gradient(env, stu, ref_stats, ref, adv)
        else:
            stu_stats = exact.occupancy(env, stu, cfg.deploy_noise)
            grad = exact.opd_gradient(env, stu, stu_stats, stu, adv)
        stu.set_params(stu.get_params() + cfg.lr * grad)
    return np.array(steps_log), np.array(succ_log), np.array(w_log)


def gen_mechanism():
    """Fig 1b: polluted-bit weight + success trajectory, offline vs online (linear)."""
    env, feats, teacher = make_env(base_cfg())
    cfg = TrainConfig(steps=400, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                      dataset_size=2000, record_every=20)
    s_off, suc_off, w_off = _record_weight_traj("linear", env, feats, teacher, cfg, "offline")
    s_on, suc_on, w_on = _record_weight_traj("linear", env, feats, teacher, cfg, "online")
    return {"steps": s_off, "succ_off": suc_off, "w_off": w_off,
            "succ_on": suc_on, "w_on": w_on}


def gen_coverage(pcs=(0.0, 0.2, 0.4, 0.8, 1.2, 1.6)):
    """Fig 2: offline/online success & off-support vs pollution rate (linear)."""
    osr, offs, ons, teach = [], [], [], []
    for pc in pcs:
        env, feats, teacher = make_env(base_cfg(pollute_coeff=pc))
        cfg = TrainConfig(steps=300, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                          dataset_size=2000, record_every=300)
        sft = train_sft(env, teacher, student("linear", env, feats, 0), cfg)
        rp = sft.student.get_params()
        o = train_offline_opd(env, teacher, fresh_from("linear", env, feats, rp),
                              fresh_from("linear", env, feats, rp), cfg).final
        n = train_online_opd(env, teacher, fresh_from("linear", env, feats, rp),
                             fresh_from("linear", env, feats, rp), cfg).final
        osr.append(o["off_support_ratio"]); offs.append(o["success"]); ons.append(n["success"])
        teach.append(exact.occupancy(env, teacher, DEPLOY).success)
    order = np.argsort(osr)
    return {"off_support": np.array(osr)[order], "offline": np.array(offs)[order],
            "online": np.array(ons)[order], "teacher": np.array(teach)[order]}


def gen_bound():
    """Fig 3: chi2, measured gradient gap, and Thm 3.5 bound over offline training."""
    env, feats, teacher = make_env(base_cfg())
    cfg = TrainConfig(steps=400, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                      dataset_size=2000, record_every=20)
    sft = train_sft(env, teacher, student("linear", env, feats, 0), cfg)
    rp = sft.student.get_params()
    off = train_offline_opd(env, teacher, fresh_from("linear", env, feats, rp),
                            fresh_from("linear", env, feats, rp), cfg)
    h = off.history
    return {"steps": np.array(h["step"]), "success": np.array(h["success"]),
            "gap": np.array(h["grad_gap"]), "bound": np.array(h["bound"]),
            "chi2": np.array(h["chi2"])}


def gen_patch():
    """Fig 4: patch Pareto -- success vs env-access refreshes, chi2 vs random."""
    env, feats, teacher = make_env(base_cfg())
    cfg = TrainConfig(steps=300, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                      dataset_size=2000, record_every=300)
    sft = train_sft(env, teacher, student("linear", env, feats, 0), cfg)
    rp = sft.student.get_params()

    def fr():
        return fresh_from("linear", env, feats, rp)

    off = train_offline_opd(env, teacher, fr(), fr(), cfg).final["success"]
    on = train_online_opd(env, teacher, fr(), fr(), cfg).final["success"]
    chi = [train_refresh(env, teacher, fr(), fr(), cfg, trigger="chi2",
                         chi2_thresh=th, budget=50, check_every=10).final
           for th in (0.3, 0.2, 0.1, 0.05)]
    # random/periodic don't use drift -> skip per-step chi2 with huge check_every.
    rnd = [train_refresh(env, teacher, fr(), fr(), cfg, trigger="random",
                         budget=b, check_every=10**9, refresh_seed=s).final
           for s in range(3) for b in (3, 6, 12)]
    return {"offline": off, "online": on, "online_steps": cfg.steps,
            "chi2_r": np.array([r["refreshes"] for r in chi]),
            "chi2_s": np.array([r["success"] for r in chi]),
            "rand_r": np.array([r["refreshes"] for r in rnd]),
            "rand_s": np.array([r["success"] for r in rnd])}


def gen_patch_anatomy():
    """Fig 5: instrumented chi2-refresh run -- when refreshes fire and how each
    one unlocks the otherwise-zero-gradient polluted-bit weight.

    We re-implement the chi2-refresh loop inline so we can log, per step: the
    chi2 drift against the live dataset policy, whether a refresh fired, the
    polluted-bit reconcile weight, and deployment success. This closes the loop
    with the Fig 1b mechanism panel: each refresh is the moment the data starts
    covering the polluted states, so the gradient on the polluted-bit column
    becomes non-zero and the weight (hence success) jumps."""
    import copy
    env, feats, teacher = make_env(base_cfg())
    cfg = TrainConfig(steps=300, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                      dataset_size=2000, record_every=10)
    sft = train_sft(env, teacher, student("linear", env, feats, 0), cfg)
    rp = sft.student.get_params()
    col = 3 * env.M  # polluted-bit feature index
    thresh, budget, check_every = 0.2, 50, 10

    stu = fresh_from("linear", env, feats, rp)
    data_policy = fresh_from("linear", env, feats, rp)
    data_stats = exact.occupancy(env, data_policy, cfg.collect_noise)
    refreshes = 0
    steps_log, drift_log, fire_log, w_log, succ_log = [], [], [], [], []
    drift = 0.0
    for step in range(cfg.steps + 1):
        if step % check_every == 0:
            drift = exact.chi2_traj(env, stu, data_policy, cfg.deploy_noise)
        fired = False
        if refreshes < budget and step % check_every == 0 and drift > thresh:
            data_stats = exact.occupancy(env, stu, cfg.deploy_noise)
            data_policy = copy.deepcopy(stu)
            refreshes += 1
            fired = True
            drift = 0.0
        if step % cfg.record_every == 0:
            steps_log.append(step)
            drift_log.append(drift)
            fire_log.append(1 if fired else 0)
            w_log.append(float(stu.W[env.reconcile_action, col]))
            succ_log.append(exact.occupancy(env, stu, DEPLOY).success)
        if step == cfg.steps:
            break
        adv = exact.advantage(env, teacher, stu, clip=cfg.clip)
        grad = exact.opd_gradient(env, stu, data_stats, data_policy, adv)
        stu.set_params(stu.get_params() + cfg.lr * grad)
    return {"steps": np.array(steps_log), "drift": np.array(drift_log),
            "fire": np.array(fire_log), "weight": np.array(w_log),
            "success": np.array(succ_log), "thresh": thresh,
            "fire_steps": np.array([s for s, f in zip(steps_log, fire_log) if f])}


def gen_reward_phase():
    """Fig 6: reward robustness phase diagram over (reconcile_cost, wrong_penalty).

    Shows the offline-collapse + chi2-refresh-recovery story is a property of a
    band, not a hand-picked point. Per cell we cache the teacher reconcile-mass
    (band check), the offline reward floor, the teacher reward ceiling, the patch
    reward, and the scale-free recovery fraction (patch-floor)/(ceil-floor).
    collect~0, linear student. See ``reward_robustness_phase.py`` for the cell
    computation (imported here so the figure and the table never diverge)."""
    from reward_robustness_phase import cell
    rcosts = (0.2, 0.4, 0.8, 1.2, 1.6)
    wpens = (1.5, 3.0, 5.0)
    nr, nc = len(wpens), len(rcosts)
    t_rec = np.zeros((nr, nc)); floor = np.zeros((nr, nc))
    ceil = np.zeros((nr, nc)); patch = np.zeros((nr, nc))
    recpct = np.zeros((nr, nc)); env = np.zeros((nr, nc))
    for i, wp in enumerate(wpens):
        for j, rc in enumerate(rcosts):
            tr, fl, ce, pa, rp, nrf = cell(rc, wp)
            t_rec[i, j] = tr; floor[i, j] = fl; ceil[i, j] = ce
            patch[i, j] = pa; recpct[i, j] = rp; env[i, j] = nrf
    return {"rcosts": np.array(rcosts), "wpens": np.array(wpens),
            "t_rec": t_rec, "floor": floor, "ceil": ceil, "patch": patch,
            "recpct": recpct, "env": env}


def gen_crosspatch():
    """Fig 7/8: all five assignment-hint patches + full-dist + offline/online,
    on the mechanism axis, at collect~0 (linear). Per method we cache success,
    reward, the polluted-bit reconcile weight (the structurally-starved feature),
    deployment reconcile-mass, and an env-access class (0=pure offline, 1=our
    collect-side refresh, 2=deploy-side). The patch implementations are imported
    from ``patch_ablation_all.py`` so the figure and the ablation table use the
    same code path."""
    from patch_ablation_all import (
        patch_support_aware, patch_conservative, patch_branch_replay,
        patch_uncertainty_query, patch_fulldist, diagnostics)
    env, feats, teacher = make_env(base_cfg())
    td = exact.occupancy(env, teacher, DEPLOY)
    cfg = TrainConfig(steps=300, lr=0.5, collect_noise=1e-9, deploy_noise=DEPLOY,
                      dataset_size=2000, record_every=300)
    sft = train_sft(env, teacher, student("linear", env, feats, 0), cfg)
    rp = sft.student.get_params()

    def fr():
        return fresh_from("linear", env, feats, rp)

    def stats(pol):
        s, w, r = diagnostics(env, pol)
        rew = exact.occupancy(env, pol, DEPLOY).avg_reward
        return s, rew, w, r

    rows = []  # (label, succ, reward, pol_w, rec_mass, env_class, env_cost)
    s, rew, w, r = stats(fr()); rows.append(("SFT", s, rew, w, r, 0, 0))
    s, rew, w, r = stats(train_offline_opd(env, teacher, fr(), fr(), cfg).student)
    rows.append(("offline OPD", s, rew, w, r, 0, 0))
    s, rew, w, r = stats(patch_support_aware(env, teacher, fr(), fr(), cfg))
    rows.append(("support-aware", s, rew, w, r, 0, 0))
    s, rew, w, r = stats(patch_conservative(env, teacher, fr(), fr(), cfg))
    rows.append(("conservative", s, rew, w, r, 0, 0))
    s, rew, w, r = stats(patch_branch_replay(env, teacher, fr(), fr(), cfg))
    rows.append(("branch-replay", s, rew, w, r, 0, 0))
    s, rew, w, r = stats(patch_fulldist(env, teacher, fr(), fr(), cfg))
    rows.append(("full-dist (Rang)", s, rew, w, r, 0, 0))
    rf = train_refresh(env, teacher, fr(), fr(), cfg, trigger="chi2",
                       chi2_thresh=0.2, budget=50, check_every=10)
    s, rew, w, r = stats(rf.student)
    rows.append(("chi2-refresh (ours)", s, rew, w, r, 1, rf.final["refreshes"]))
    uq, nq = patch_uncertainty_query(env, teacher, fr(), fr(), cfg)
    s, rew, w, r = stats(uq); rows.append(("uncertainty-query", s, rew, w, r, 2, nq))
    s, rew, w, r = stats(train_online_opd(env, teacher, fr(), fr(), cfg).student)
    rows.append(("online OPD", s, rew, w, r, 2, cfg.steps))

    labels = [r[0] for r in rows]
    return {"labels": np.array(labels),
            "succ": np.array([r[1] for r in rows]),
            "reward": np.array([r[2] for r in rows]),
            "pol_w": np.array([r[3] for r in rows]),
            "rec_mass": np.array([r[4] for r in rows]),
            "env_class": np.array([r[5] for r in rows]),
            "env_cost": np.array([r[6] for r in rows]),
            "teacher_succ": td.success, "teacher_reward": td.avg_reward}


def gen_sweeps():
    """Appendix B: two-axis detailed sweeps, success AND reward, exact (linear).

    These are the verified-exact values from ``detailed_sweeps.py`` (collect
    sweep at deploy 0.45; deploy sweep at collect ~0). The OPD gradients are
    sampling-free so a fixed-init run is deterministic; recomputing the full
    grid is ~15 min and returns the same numbers, so we cache them directly
    (``detailed_sweeps.main()`` is the recompute path). Method order:
    [SFT, offline, full-dist, branch-replay, chi2-refresh, online]."""
    methods = ["SFT", "offline", "full-dist", "branch-replay", "chi2-refresh", "online"]
    # Ablation A: collection coverage, deploy fixed 0.45. x = [0,.05,.15,.30,.45].
    A_x = np.array([0.0, 0.05, 0.15, 0.30, 0.45])
    A_succ = np.array([
        [0.417, 0.883, 0.921, 0.940, 0.945],   # SFT
        [0.420, 0.857, 0.891, 0.935, 0.989],   # offline
        [0.418, 0.921, 0.951, 0.961, 0.963],   # full-dist
        [0.420, 0.994, 0.992, 0.991, 0.990],   # branch-replay
        [1.000, 0.953, 0.927, 0.990, 0.915],   # chi2-refresh
        [0.960, 0.959, 0.959, 0.960, 0.960]])  # online
    A_rew = np.array([
        [-1.58, -0.06, 0.04, 0.08, 0.08],
        [-1.57, -0.13, -0.03, 0.05, 0.15],
        [-1.58, 0.03, 0.09, 0.10, 0.10],
        [-1.58, 0.12, 0.13, 0.13, 0.13],
        [0.02, 0.10, 0.08, 0.15, 0.04],
        [0.14, 0.14, 0.14, 0.14, 0.14]])
    A_teach_s, A_teach_r = 0.973, 0.19
    # Ablation B: deployment shift, collect ~0. x = [.10,.20,.30,.45].
    B_x = np.array([0.10, 0.20, 0.30, 0.45])
    B_succ = np.array([
        [0.826, 0.692, 0.573, 0.417],
        [0.868, 0.728, 0.592, 0.420],
        [0.759, 0.642, 0.561, 0.418],
        [0.869, 0.728, 0.592, 0.420],
        [0.865, 0.961, 0.967, 1.000],
        [0.984, 0.980, 0.971, 0.960]])
    B_rew = np.array([
        [0.12, -0.43, -0.93, -1.58],
        [0.27, -0.31, -0.87, -1.57],
        [-0.13, -0.61, -0.97, -1.58],
        [0.27, -0.31, -0.87, -1.58],
        [0.24, 0.45, 0.34, 0.02],
        [0.65, 0.49, 0.34, 0.14]])
    B_teach_s = np.array([0.996, 0.990, 0.983, 0.973])
    B_teach_r = np.array([0.70, 0.54, 0.39, 0.19])
    return {"methods": np.array(methods),
            "A_x": A_x, "A_succ": A_succ, "A_rew": A_rew,
            "A_teach_s": A_teach_s, "A_teach_r": A_teach_r,
            "B_x": B_x, "B_succ": B_succ, "B_rew": B_rew,
            "B_teach_s": B_teach_s, "B_teach_r": B_teach_r}


def _flatten(prefix, d, store):
    for k, v in d.items():
        store[f"{prefix}__{k}"] = np.asarray(v)


def main():
    store = {}
    print("baseline..."); _flatten("baseline", gen_baseline(), store)
    print("mechanism..."); _flatten("mechanism", gen_mechanism(), store)
    print("coverage..."); _flatten("coverage", gen_coverage(), store)
    print("bound..."); _flatten("bound", gen_bound(), store)
    print("patch..."); _flatten("patch", gen_patch(), store)
    print("patch_anatomy..."); _flatten("anatomy", gen_patch_anatomy(), store)
    print("reward_phase..."); _flatten("phase", gen_reward_phase(), store)
    print("crosspatch..."); _flatten("crosspatch", gen_crosspatch(), store)
    print("sweeps..."); _flatten("sweeps", gen_sweeps(), store)
    path = os.path.join(OUT, "figdata.npz")
    np.savez(path, **store)
    print("wrote", path, "with", len(store), "arrays")


if __name__ == "__main__":
    main()
