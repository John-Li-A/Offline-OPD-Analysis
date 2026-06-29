"""Training methods, each driven entirely by the exact machinery in ``exact.py``.

All four recipes optimise a student policy by plain gradient ascent on an exact
(sampling-free) gradient. They differ only in which distribution the gradient is
taken under -- the single axis the whole study turns on.

* ``train_sft`` -- behaviour cloning. Maximise teacher-action log-likelihood
  under the teacher's own visitation at the *collection* noise level. The result
  is the reference policy ``pi_ref`` that seeds every OPD run.
* ``train_offline_opd`` -- Lightning OPD. Freeze the rollout visitation and the
  action weighting at ``pi_ref`` (collection noise), recompute the advantage
  ``log pi_T - log pi_theta`` each step, and ascend. No environment, no live
  teacher beyond the one-time precompute.
* ``train_online_opd`` -- standard OPD upper bound. Re-derive the rollout
  visitation and action weighting from the *current* student at deployment noise
  each step. Same advantage, on-policy state coverage.
* ``train_online_rl`` -- REINFORCE on the environment's terminal reward at
  deployment noise, as a non-distillation reference point.

Metrics are evaluated at deployment noise so that distribution shift between
collection and deployment is visible. ``off_support_ratio`` is the student's
deployment visitation mass that lands on states the frozen reference dataset
effectively never covered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import exact
from .env import RetrievalQAEnv
from .policies import TeacherPolicy


@dataclass
class TrainConfig:
    """Hyperparameters shared across the training recipes.

    Attributes:
        steps: number of gradient steps.
        lr: learning rate for vanilla gradient ascent.
        clip: advantage clip range for OPD.
        collect_noise: distractor rate at data collection (low -> clean).
        deploy_noise: distractor rate at deployment/eval (high -> conflicts).
        dataset_size: number of rollouts in the frozen offline dataset. A state
            is "off-support" when the deployed student visits it but its expected
            count in N reference rollouts is below one -- the honest finite-data
            notion of "the precomputed dataset effectively never covered it".
        record_every: cadence for logging training-dynamics metrics.
    """

    steps: int = 300
    lr: float = 0.5
    clip: float = 10.0
    collect_noise: float = 0.05
    deploy_noise: float = 0.35
    dataset_size: int = 2000
    record_every: int = 10


@dataclass
class TrainResult:
    """Outputs of a training run.

    Attributes:
        student: the trained policy object.
        history: per-checkpoint dict of arrays (step, success, reward, length,
            off_support_ratio, chi2, grad_gap, bound).
        final: final-step scalar metrics.
    """

    student: object
    history: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)


def evaluate(env: RetrievalQAEnv, teacher, student, ref_stats, cfg: TrainConfig) -> dict:
    """Exact deployment metrics plus the Theorem 3.5 bound terms.

    ``ref_stats`` is the frozen reference visitation (collection noise); passing
    ``None`` skips the off-support computation (used before pi_ref exists).
    """
    dep = exact.occupancy(env, student, cfg.deploy_noise)
    metrics = {
        "success": dep.success,
        "reward": dep.avg_reward,
        "length": dep.avg_length,
    }
    if ref_stats is not None:
        # Finite-dataset off-support: a state the deployed student visits whose
        # expected count in N reference rollouts is below one is effectively
        # uncovered by the precomputed dataset.
        expected_count = ref_stats.visit * cfg.dataset_size
        covered = expected_count >= 1.0
        off_mass = float(dep.visit[~covered].sum())
        total = float(dep.visit.sum())
        metrics["off_support_ratio"] = off_mass / total if total > 0 else 0.0
    return metrics


def _ascend(student, grad: np.ndarray, lr: float) -> None:
    student.set_params(student.get_params() + lr * grad)


def _student_qv(env: RetrievalQAEnv, student, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact Q^pi and V^pi under the student policy at a noise level.

    Mirrors the teacher's backward induction over potential but takes the policy
    *expectation* instead of the max, and books the same rewards (per-action step
    cost; an answer pays the belief mass on its candidate; reconcile averages its
    successor value under the belief). Returns ``(Q, V)`` so an exact REINFORCE
    gradient can use the value as its baseline.
    """
    A = env.num_actions
    Q = np.zeros((env.num_states, A))
    V = np.zeros(env.num_states)
    order = np.argsort(-env.potential)  # highest potential first
    for s in order:
        b_eff = env.belief_eff(s, noise)
        b_true = env.belief_true(s, noise)
        mask = env.legal_actions(s)
        for a in range(A):
            if not mask[a]:
                continue
            if env.is_answer(a):
                bj = b_true[env.answer_of(a)]
                Q[s, a] = (bj * env.cfg.correct_reward
                           - (1.0 - bj) * env.cfg.wrong_penalty
                           - env.cfg.step_cost)
            elif env.is_reconcile(a):
                v_next = sum(b_true[j] * V[env.reconcile_state(j)] for j in range(env.K))
                Q[s, a] = -env.cfg.reconcile_cost + v_next
            else:
                out = env.retrieve_outcome_dist(s, a, noise)
                s_sup = env.next_state(s, a, 1)
                s_ref = env.next_state(s, a, 2)
                Q[s, a] = -env.cfg.step_cost + out[0] * V[s_sup] + out[1] * V[s_ref]
        p = student.probs(s)
        legalQ = np.where(mask, Q[s], 0.0)
        V[s] = float((p * legalQ).sum())
    return Q, V


def train_sft(env: RetrievalQAEnv, teacher, student, cfg: TrainConfig) -> TrainResult:
    """Behaviour-clone the teacher under its own visitation at collection noise.

    Gradient: ``sum_s d^teacher(s) sum_a pi_T(a|s) grad log pi_theta(a|s)`` -- the
    advantage-free score function, i.e. weighted maximum likelihood.
    """
    teacher_stats = exact.occupancy(env, teacher, cfg.collect_noise)
    hist = {k: [] for k in ("step", "success", "reward", "length")}
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0:
            m = evaluate(env, teacher, student, None, cfg)
            hist["step"].append(step)
            for k in ("success", "reward", "length"):
                hist[k].append(m[k])
        if step == cfg.steps:
            break
        grad = np.zeros(student.num_params)
        for s in range(env.num_states):
            w = teacher_stats.visit[s]
            if w <= 0.0:
                continue
            coeffs = w * teacher.probs(s)
            grad += student.grad_logpi_weighted(s, coeffs)
        _ascend(student, grad, cfg.lr)
    res = TrainResult(student=student, history=hist)
    res.final = evaluate(env, teacher, student, None, cfg)
    return res


def theorem_diagnostics(env: RetrievalQAEnv, teacher, student, reference, cfg: TrainConfig) -> dict:
    """Exact Theorem 3.5 check: gap ``||grad J_on - grad J_off||`` vs the bound.

    Both gradients are taken under the *same* deployment dynamics so the only
    difference is the rollout/action measure (student vs frozen reference) -- the
    setting the theorem actually covers. We return the measured gap, the exact
    chi-squared divergence, the bound constants, and the bound value, so the
    report can plot whether ``gap <= G * sigma_A * sqrt(chi2)`` stays informative
    as the student drifts.
    """
    noise = cfg.deploy_noise
    adv = exact.advantage(env, teacher, student, clip=cfg.clip)
    stu_stats = exact.occupancy(env, student, noise)
    ref_stats = exact.occupancy(env, reference, noise)
    g_on = exact.opd_gradient(env, student, stu_stats, student, adv)
    g_off = exact.opd_gradient(env, student, ref_stats, reference, adv)
    gap = float(np.linalg.norm(g_on - g_off))
    chi2 = exact.chi2_traj(env, student, reference, noise)
    G, sigma_A = exact.bound_terms(env, student, reference, adv, noise)
    bound = G * sigma_A * np.sqrt(max(chi2, 0.0))
    return {"grad_gap": gap, "chi2": chi2, "G": G, "sigma_A": sigma_A, "bound": bound}


def train_offline_opd(env: RetrievalQAEnv, teacher, reference, student, cfg: TrainConfig) -> TrainResult:
    """Lightning OPD: ascend the offline gradient with frozen pi_ref rollouts.

    The rollout visitation and the action weighting are both fixed at the
    reference (collection noise) and never refreshed. Only the advantage
    ``log pi_T - log pi_theta`` is recomputed as the student moves.
    """
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    keys = ("step", "success", "reward", "length", "off_support_ratio", "grad_gap", "chi2", "bound")
    hist = {k: [] for k in keys}
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0:
            m = evaluate(env, teacher, student, ref_stats, cfg)
            diag = theorem_diagnostics(env, teacher, student, reference, cfg)
            hist["step"].append(step)
            for k in ("success", "reward", "length", "off_support_ratio"):
                hist[k].append(m[k])
            for k in ("grad_gap", "chi2", "bound"):
                hist[k].append(diag[k])
        if step == cfg.steps:
            break
        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        grad = exact.opd_gradient(env, student, ref_stats, reference, adv)
        _ascend(student, grad, cfg.lr)
    res = TrainResult(student=student, history=hist)
    res.final = evaluate(env, teacher, student, ref_stats, cfg)
    res.final.update(theorem_diagnostics(env, teacher, student, reference, cfg))
    return res


def train_online_opd(env: RetrievalQAEnv, teacher, reference, student, cfg: TrainConfig) -> TrainResult:
    """Standard OPD upper bound: refresh rollouts from the student each step.

    Same advantage as offline OPD, but the visitation and action weighting come
    from the current student at deployment noise, so on-policy state coverage is
    restored. ``reference`` is kept only for the off-support metric.
    """
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    keys = ("step", "success", "reward", "length", "off_support_ratio")
    hist = {k: [] for k in keys}
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0:
            m = evaluate(env, teacher, student, ref_stats, cfg)
            hist["step"].append(step)
            for k in ("success", "reward", "length", "off_support_ratio"):
                hist[k].append(m[k])
        if step == cfg.steps:
            break
        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        stu_stats = exact.occupancy(env, student, cfg.deploy_noise)
        grad = exact.opd_gradient(env, student, stu_stats, student, adv)
        _ascend(student, grad, cfg.lr)
    res = TrainResult(student=student, history=hist)
    res.final = evaluate(env, teacher, student, ref_stats, cfg)
    return res


def train_refresh(env: RetrievalQAEnv, teacher, reference, student, cfg: TrainConfig,
                  trigger: str = "chi2", chi2_thresh: float = 1.0, period: int = 50,
                  budget: int = 5, check_every: int = 10, refresh_seed: int = 0) -> TrainResult:
    """chi2-triggered dataset refresh -- the Pareto bridge between offline and
    online OPD [Q-vii patch].

    Offline OPD is free (zero env access) but collapses; online OPD recovers but
    pays a fresh rollout every step. This patch starts from the frozen reference
    dataset and re-collects it from the *current* student only when the data has
    gone stale, spending a bounded ``budget`` of env-access refreshes.

    Staleness is measured by the exact trajectory chi-squared between the current
    student and the snapshot policy that generated the live dataset -- the same
    chi^2(pi_theta || pi_data) that Theorem 3.5 says controls the offline/online
    gradient gap. When it exceeds ``chi2_thresh`` (and budget remains) we refresh:
    re-derive the visitation and action weighting from the current student at
    deployment noise, and reset the drift snapshot. Between refreshes the gradient
    is taken offline against the frozen snapshot, exactly like Lightning OPD.

    Ablation levers:
        trigger: "chi2" (theory-driven), "periodic" (every ``period`` steps), or
            "random" (Bernoulli matched to the chi2 trigger's spend rate).
        chi2_thresh: drift threshold for the chi2 trigger.
        period: refresh cadence for the periodic trigger.
        budget: max number of refreshes (the env-access cost axis).
    """
    import copy

    rng = np.random.default_rng(refresh_seed)
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    # Live dataset + the policy snapshot that generated it (starts at reference).
    data_stats = ref_stats
    data_policy = copy.deepcopy(reference)
    refreshes_used = 0
    # Match a random trigger's spend rate to roughly budget/steps.
    rand_rate = budget / max(cfg.steps, 1)

    keys = ("step", "success", "reward", "length", "off_support_ratio", "chi2", "refreshes")
    hist = {k: [] for k in keys}
    drift = 0.0  # chi2 drift is only re-measured every ``check_every`` steps
    for step in range(cfg.steps + 1):
        if step % check_every == 0:
            drift = exact.chi2_traj(env, student, data_policy, cfg.deploy_noise)
        if step % cfg.record_every == 0:
            m = evaluate(env, teacher, student, ref_stats, cfg)
            hist["step"].append(step)
            for k in ("success", "reward", "length", "off_support_ratio"):
                hist[k].append(m[k])
            hist["chi2"].append(drift)
            hist["refreshes"].append(refreshes_used)
        if step == cfg.steps:
            break

        # Decide whether to spend an env-access refresh this step. The chi2
        # trigger only acts on a freshly-measured drift (the check-step), so it
        # cannot fire on a stale value between checks.
        fresh_drift = (step % check_every == 0)
        if refreshes_used < budget:
            if trigger == "chi2":
                fire = fresh_drift and drift > chi2_thresh
            elif trigger == "periodic":
                fire = (step > 0) and (step % period == 0)
            else:  # random, budget-matched
                fire = rng.random() < rand_rate
        else:
            fire = False
        if fire:
            data_stats = exact.occupancy(env, student, cfg.deploy_noise)
            data_policy = copy.deepcopy(student)
            refreshes_used += 1
            drift = 0.0  # data is fresh again; reset until the next check

        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        grad = exact.opd_gradient(env, student, data_stats, data_policy, adv)
        _ascend(student, grad, cfg.lr)

    res = TrainResult(student=student, history=hist)
    res.final = evaluate(env, teacher, student, ref_stats, cfg)
    res.final["refreshes"] = refreshes_used
    return res


def train_online_rl(env: RetrievalQAEnv, teacher, reference, student, cfg: TrainConfig) -> TrainResult:
    """REINFORCE on terminal reward at deployment noise (non-distillation ref).

    Exact policy gradient ``sum_s d^pi(s) sum_a pi(a|s) A^pi(s,a) grad log pi``
    with ``A^pi = Q^pi - V^pi`` from ``_student_qv`` -- the advantage baseline
    makes it a clean, low-variance exact REINFORCE.
    """
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    keys = ("step", "success", "reward", "length", "off_support_ratio")
    hist = {k: [] for k in keys}
    for step in range(cfg.steps + 1):
        if step % cfg.record_every == 0:
            m = evaluate(env, teacher, student, ref_stats, cfg)
            hist["step"].append(step)
            for k in ("success", "reward", "length", "off_support_ratio"):
                hist[k].append(m[k])
        if step == cfg.steps:
            break
        Q, V = _student_qv(env, student, cfg.deploy_noise)
        stu_stats = exact.occupancy(env, student, cfg.deploy_noise)
        grad = np.zeros(student.num_params)
        for s in range(env.num_states):
            w = stu_stats.visit[s]
            if w <= 0.0:
                continue
            mask = env.legal_actions(s)
            advs = np.where(mask & np.isfinite(Q[s]), Q[s] - V[s], 0.0)
            coeffs = w * student.probs(s) * advs
            grad += student.grad_logpi_weighted(s, coeffs)
        _ascend(student, grad, cfg.lr)
    res = TrainResult(student=student, history=hist)
    res.final = evaluate(env, teacher, student, ref_stats, cfg)
    return res
