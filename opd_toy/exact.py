"""Exact (sampling-free) analysis machinery -- the intellectual core.

Because the information-state DAG is small and the dynamics are known, every
quantity the OPD theory talks about can be computed in closed form instead of
estimated by Monte Carlo:

* ``occupancy`` -- exact state-visitation distribution d^pi(s) and exact episode
  metrics (success, reward, length), by forward-propagating the joint
  distribution over (true answer, information state).
* ``opd_gradient`` -- the exact OPD surrogate gradient
  ``sum_s d^rollout(s) sum_a pi_action(a|s) A(s,a) grad log pi_theta(a|s)``.
  Online OPD instantiates this with the student's own visitation and action
  weights; offline OPD instantiates it with the frozen reference's visitation
  and action weights. The action weight is the lever: offline weights actions by
  pi_ref(a|s), which puts near-zero mass on the corroborate-more action at
  conflicting states, so that gradient term is not high-variance -- it is
  exactly, provably, near-zero. We can read the coefficient off directly.
* ``chi2_traj`` -- the exact trajectory-level chi-squared divergence
  chi^2(pi_theta || pi_ref) that appears in Theorem 3.5.
* ``bound_terms`` -- the exact constants G and sigma_A in the Theorem 3.5 bound
  ``||grad J_on - grad J_off|| <= G * sigma_A * sqrt(chi^2)``.

Nothing here samples. That is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import POLLUTED, REFUTES, SUPPORTS, VERIFIED, RetrievalQAEnv


@dataclass
class RolloutStats:
    """Exact episode statistics under a fixed policy and noise level.

    Attributes:
        visit: ``visit[s]`` = probability the episode visits observable state
            ``s`` (also the expected number of visits, since every action raises
            the potential so each state is entered at most once).
        joint_visit: ``joint_visit[a_tilde, s]`` = P(effective answer is a_tilde
            AND visit s). Indexed by the *effective* answer (which governs the
            reads), not the true answer; the true-answer split is reconstructed
            from the phase at commit time.
        success: probability the committed answer is the true answer a*.
        avg_reward: expected terminal reward (correctness minus step costs).
        avg_length: expected number of actions per episode.
    """

    visit: np.ndarray
    joint_visit: np.ndarray
    success: float
    avg_reward: float
    avg_length: float


def _init_joint(env: RetrievalQAEnv, noise: float) -> np.ndarray:
    """Start-of-episode joint ``[a_tilde, state]`` with the pollution prior.

    With probability ``1 - rho`` the episode is clean (a_tilde = a*, uniform) and
    starts at the clean empty state; with probability ``rho`` it is polluted
    (a_tilde uniform over wrong answers, marginally uniform) and starts at the
    polluted empty state.
    """
    K = env.K
    rho = env.rho(noise)
    j = np.zeros((K, env.num_states))
    clean_empty = 0
    polluted_empty = POLLUTED * env.n_evidence
    j[:, clean_empty] = (1.0 - rho) / K
    j[:, polluted_empty] = rho / K
    return j


def _commit_correct(env: RetrievalQAEnv, phase: int, flow: np.ndarray, ans: int) -> float:
    """Correct-answer mass when committing ``ans`` from ``flow`` over a_tilde.

    Clean/verified: a* = a_tilde so the correct mass is ``flow[ans]``. Polluted:
    a* is uniform over the wrong answers ``{!= a_tilde}``, so committing ``ans``
    is correct with prob ``[a_tilde != ans]/(K-1)`` -- summing the off-ans mass
    divided by ``K-1``.
    """
    if phase == POLLUTED:
        return float((flow.sum() - flow[ans]) / (env.K - 1))
    return float(flow[ans])


def occupancy(env: RetrievalQAEnv, policy, noise: float, horizon: int | None = None) -> RolloutStats:
    """Forward-propagate the exact (effective answer, observable state) joint.

    The joint is indexed by the effective answer ``a_tilde`` (which governs the
    noisy reads). At a commit the true-answer split is reconstructed from the
    phase (``_commit_correct``). Reconcile reveals a* and lands on the verified
    clean signature: from a clean state a_tilde is unchanged; from a polluted
    state the mass splits uniformly over the true answers ``a* != a_tilde``, and
    the new effective answer equals a* (verified phase has a* = a_tilde).

    States are processed in increasing potential ``phi = depth + (M+1)*verified``;
    every action raises ``phi``, so incoming mass is complete before a state is
    expanded and the episode is structurally bounded. A per-node turn-moment
    ``tmoment`` carries the mass-weighted action count for exact length/cost.
    """
    K = env.K
    M = env.M
    psup = env.support_prob(noise)  # [source, a_tilde]
    joint = _init_joint(env, noise)
    tmoment = np.zeros((K, env.num_states))
    visit = np.zeros(env.num_states)
    success = 0.0
    reward = 0.0
    length = 0.0

    order = np.argsort(env.potential)  # ascending potential
    for s in order:
        mass_s = joint[:, s]  # [a_tilde]
        total_s = mass_s.sum()
        if total_s <= 0.0:
            continue
        visit[s] += total_s
        tm_s = tmoment[:, s]
        phase = int(env.phase_of[s])
        p = policy.probs(s)
        mask = env.legal_actions(s)

        # Answer branches: terminate, book metrics.
        for a in range(M, M + K):
            pa = p[a]
            if pa <= 0.0:
                continue
            ans = env.answer_of(a)
            flow = mass_s * pa  # [a_tilde]
            total = flow.sum()
            correct = _commit_correct(env, phase, flow, ans)
            wrong = total - correct
            success += correct
            steps_mass = float(pa * (tm_s + mass_s).sum())  # action count incl. answer
            length += steps_mass
            reward += (correct * env.cfg.correct_reward
                       - wrong * env.cfg.wrong_penalty
                       - steps_mass * env.cfg.step_cost)

        # Retrieve branches: split on the a_tilde-governed outcome, flow forward.
        for i in range(M):
            pi = p[i]
            if pi <= 0.0 or not mask[i]:
                continue
            flow = mass_s * pi  # [a_tilde]
            tflow = pi * (tm_s + mass_s)
            p_sup = psup[i]  # [a_tilde]
            s_sup = env.next_state(s, i, SUPPORTS)
            s_ref = env.next_state(s, i, REFUTES)
            joint[:, s_sup] += flow * p_sup
            joint[:, s_ref] += flow * (1.0 - p_sup)
            tmoment[:, s_sup] += tflow * p_sup
            tmoment[:, s_ref] += tflow * (1.0 - p_sup)

        # Reconcile branch: reveal a* and jump to the verified clean signature.
        if mask[env.reconcile_action]:
            pr = p[env.reconcile_action]
            if pr > 0.0:
                flow = mass_s * pr  # [a_tilde]
                tflow = pr * (tm_s + mass_s)
                # The turn-moment already charges step_cost per action; reconcile
                # costs more, so book the premium (reconcile_cost - step_cost) on
                # the reconcile flow.
                reward -= (env.cfg.reconcile_cost - env.cfg.step_cost) * float(flow.sum())
                if phase == POLLUTED:
                    # a* uniform over {!= a_tilde}; new effective answer = a*.
                    for at in range(K):
                        if flow[at] <= 0.0 and tflow[at] <= 0.0:
                            continue
                        for astar in range(K):
                            if astar == at:
                                continue
                            tgt = env.reconcile_state(astar)
                            joint[astar, tgt] += flow[at] / (K - 1)
                            tmoment[astar, tgt] += tflow[at] / (K - 1)
                else:  # CLEAN: a* = a_tilde, unchanged.
                    for at in range(K):
                        tgt = env.reconcile_state(at)
                        joint[at, tgt] += flow[at]
                        tmoment[at, tgt] += tflow[at]

    return RolloutStats(
        visit=visit,
        joint_visit=joint,
        success=float(success),
        avg_reward=float(reward),
        avg_length=float(length),
    )


def advantage(env: RetrievalQAEnv, teacher, student, clip: float = 10.0) -> np.ndarray:
    """Per-(state, action) OPD advantage ``A(s,a) = log pi_T - log pi_theta``.

    Computed densely on every legal (state, action) pair and clipped to
    ``[-clip, clip]`` exactly as the slime implementation does. Illegal actions
    are left at 0 (they carry no visitation mass anyway).
    """
    A = np.zeros((env.num_states, env.num_actions))
    for s in range(env.num_states):
        mask = env.legal_actions(s)
        lt = teacher.log_probs(s)
        ls = student.log_probs(s)
        a = lt - ls
        a = np.clip(a, -clip, clip)
        A[s] = np.where(mask, a, 0.0)
    return A


def opd_gradient(env: RetrievalQAEnv, student, stats: RolloutStats, action_policy, adv: np.ndarray) -> np.ndarray:
    """Exact OPD surrogate gradient under a given rollout/action distribution.

    ``sum_s visit(s) * sum_a action_policy(a|s) * A(s,a) * grad log pi_theta(a|s)``.

    * Online OPD: ``stats`` from the student's own visitation and
      ``action_policy = student`` (resamples from the current policy).
    * Offline OPD: ``stats`` from the frozen reference visitation and
      ``action_policy = reference`` (the behaviour policy that wrote the data).

    The advantage ``adv`` is treated as a fixed scalar (stop-gradient), matching
    standard OPD practice.
    """
    grad = np.zeros(student.num_params)
    for s in range(env.num_states):
        w = stats.visit[s]
        if w <= 0.0:
            continue
        pa = action_policy.probs(s)
        coeffs = w * pa * adv[s]  # [num_actions]
        grad += student.grad_logpi_weighted(s, coeffs)
    return grad


def chi2_traj(env: RetrievalQAEnv, student, reference, noise: float, horizon: int | None = None) -> float:
    """Exact trajectory-level chi-squared divergence chi^2(pi_theta || pi_ref).

    ``chi^2 = E_{x ~ pi_ref}[ w(x)^2 ] - 1`` with the trajectory importance ratio
    ``w(x) = prod_t pi_theta(a_t|s_t) / pi_ref(a_t|s_t)``. Transition probs cancel
    inside the ratio, so we propagate ``E_pi_ref[w^2 up to s]`` forward over the
    potential DAG, indexed by the effective answer. The episode-start pollution
    split is part of the environment, not the policy, so it seeds both measures
    identically (it carries no ratio). Reconcile redistributes to the verified
    signature exactly as in ``occupancy``.

    This is the quantity inside the square root of the Theorem 3.5 bound.
    """
    K = env.K
    M = env.M
    psup = env.support_prob(noise)
    # m[a_tilde, s] = sum over partial paths to s of P_ref(path|a_tilde) * w(path)^2
    m = _init_joint(env, noise)
    e_w2 = 0.0
    order = np.argsort(env.potential)
    for s in order:
        mass = m[:, s]
        if mass.sum() <= 0.0:
            continue
        phase = int(env.phase_of[s])
        mask = env.legal_actions(s)
        pth = student.probs(s)
        prf = reference.probs(s)

        # Answer branches close the trajectory: fold in (pi_theta/pi_ref)^2.
        for a in range(M, M + K):
            if prf[a] <= 0.0:
                continue
            ratio2 = (pth[a] / prf[a]) ** 2
            e_w2 += mass.sum() * prf[a] * ratio2

        # Retrieve branches advance and split on the noisy outcome.
        for i in range(M):
            if not mask[i] or prf[i] <= 0.0:
                continue
            ratio2 = (pth[i] / prf[i]) ** 2
            step = mass * prf[i] * ratio2  # [a_tilde]
            p_sup = psup[i]
            s_sup = env.next_state(s, i, SUPPORTS)
            s_ref = env.next_state(s, i, REFUTES)
            m[:, s_sup] += step * p_sup
            m[:, s_ref] += step * (1.0 - p_sup)

        # Reconcile branch advances to the verified signature.
        if mask[env.reconcile_action] and prf[env.reconcile_action] > 0.0:
            a = env.reconcile_action
            ratio2 = (pth[a] / prf[a]) ** 2
            step = mass * prf[a] * ratio2  # [a_tilde]
            if phase == POLLUTED:
                for at in range(K):
                    if step[at] <= 0.0:
                        continue
                    for astar in range(K):
                        if astar == at:
                            continue
                        m[astar, env.reconcile_state(astar)] += step[at] / (K - 1)
            else:
                for at in range(K):
                    m[at, env.reconcile_state(at)] += step[at]
    return float(e_w2 - 1.0)


def bound_terms(env: RetrievalQAEnv, student, reference, adv: np.ndarray, noise: float) -> tuple[float, float]:
    """Exact constants ``G`` and ``sigma_A`` in the Theorem 3.5 bound.

    ``G = max_{s,a} ||grad log pi_theta(a|s)||`` (bounded score function).
    ``sigma_A = sqrt(E_{x ~ pi_ref}[ (sum_t |A_t|)^2 ])`` (bounded absolute
    advantage), computed exactly over reference trajectories at ``noise``.
    """
    G = student.max_score_norm()
    sigma_A = _sigma_A(env, reference, adv, noise)
    return G, sigma_A


def _sigma_A(env: RetrievalQAEnv, reference, adv: np.ndarray, noise: float) -> float:
    """sqrt(E_{x ~ pi_ref}[(sum_t |A_t|)^2]) by exact forward moment recursion.

    For every (effective answer, state) node we carry three path-weighted sums
    over reference trajectories reaching it: ``base`` (probability mass), ``s1``
    (mass times running sum of |A|), and ``s2`` (mass times running sum squared).
    Taking action ``a`` with |A| = ``add`` updates a child via the algebra of
    ``(R + add)`` and ``(R + add)^2``; an answer closes the trajectory and books
    ``E[(sum|A|)^2]``. Retrieve and reconcile advance the recursion; reconcile
    redistributes to the verified signature as in ``occupancy``.
    """
    K = env.K
    M = env.M
    psup = env.support_prob(noise)
    base = _init_joint(env, noise)
    s1 = np.zeros((K, env.num_states))
    s2 = np.zeros((K, env.num_states))
    total = 0.0
    order = np.argsort(env.potential)
    for s in order:
        mass = base[:, s]
        if mass.sum() <= 0.0:
            continue
        phase = int(env.phase_of[s])
        mask = env.legal_actions(s)
        prf = reference.probs(s)
        absA = np.abs(adv[s])
        cur1 = s1[:, s]
        cur2 = s2[:, s]
        for a in range(env.num_actions):
            if not mask[a] or prf[a] <= 0.0:
                continue
            add = absA[a]
            m1 = cur1 + add * mass
            m2 = cur2 + 2 * add * cur1 + add * add * mass
            if env.is_answer(a):
                total += prf[a] * float(m2.sum())
            elif env.is_reconcile(a):
                if phase == POLLUTED:
                    for at in range(K):
                        if mass[at] <= 0.0:
                            continue
                        for astar in range(K):
                            if astar == at:
                                continue
                            tgt = env.reconcile_state(astar)
                            base[astar, tgt] += prf[a] * mass[at] / (K - 1)
                            s1[astar, tgt] += prf[a] * m1[at] / (K - 1)
                            s2[astar, tgt] += prf[a] * m2[at] / (K - 1)
                else:
                    for at in range(K):
                        tgt = env.reconcile_state(at)
                        base[at, tgt] += prf[a] * mass[at]
                        s1[at, tgt] += prf[a] * m1[at]
                        s2[at, tgt] += prf[a] * m2[at]
            else:
                p_sup = psup[a]
                s_sup = env.next_state(s, a, SUPPORTS)
                s_ref = env.next_state(s, a, REFUTES)
                base[:, s_sup] += prf[a] * p_sup * mass
                base[:, s_ref] += prf[a] * (1.0 - p_sup) * mass
                s1[:, s_sup] += prf[a] * p_sup * m1
                s1[:, s_ref] += prf[a] * (1.0 - p_sup) * m1
                s2[:, s_sup] += prf[a] * p_sup * m2
                s2[:, s_ref] += prf[a] * (1.0 - p_sup) * m2
    return float(np.sqrt(max(total, 0.0)))
