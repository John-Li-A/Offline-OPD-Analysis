"""Policies for the retrieval QA POMDP.

Three policies live here:

* ``TeacherPolicy`` -- the exact oracle. Because retrievals strictly grow the
  information state (each source is queried at most once), the decision process
  is a DAG of bounded depth, so the optimal action-value function is solved in a
  single exact backward pass over depth -- no iteration, no sampling. The teacher
  is a Boltzmann policy over those Q-values, defined on *every* information
  state, including the conflicting-evidence states an improving student drifts
  into. One teacher object supplies both the SFT demonstrations and the OPD
  targets, so teacher consistency holds by construction.

* ``LinearSoftmaxStudent`` -- the trainable policy. A linear-softmax over
  hand-built features of the information state. Shared weights mean on-path
  gradients generalise into unseen states, the way a real model does, so an
  early error compounds deterministically rather than being escaped by luck.

* ``TabularStudent`` -- a control with one independent logit row per state. Its
  gradient at a state touches only that state's logits, so off-support states
  get zero gradient. Comparing the two isolates parameter generalisation from
  literal state coverage.

Every policy exposes the same primitive the exact-gradient code needs:
``grad_logpi_weighted(state, coeffs)`` returns ``sum_a coeffs[a] * grad_theta
log pi(a | state)`` as a flat parameter-gradient vector. Both the offline and
online OPD gradients are assembled from this primitive.
"""

from __future__ import annotations

import numpy as np

from .env import REFUTES, SUPPORTS, UNQUERIED, RetrievalQAEnv

NEG_INF = -1e30


class TeacherPolicy:
    """Exact Boltzmann-optimal oracle over information states.

    Args:
        env: the retrieval QA environment.
        noise: distractor rate the teacher assumes when forming beliefs and
            predicting retrieval outcomes (the canonical task condition).
        temperature: Boltzmann temperature; small -> near-greedy, large ->
            softer. The teacher stays fixed throughout the study.
        discount: discount applied to future value in the backward pass.
    """

    def __init__(self, env: RetrievalQAEnv, noise: float, temperature: float = 0.25, discount: float = 0.99):
        self.env = env
        self.noise = noise
        self.temperature = temperature
        self.discount = discount
        self._solve()

    def _solve(self) -> None:
        """Exact backward induction over potential to get Q(s, a) and the policy.

        States are processed in order of decreasing potential ``phi = depth +
        (M+1)*[verified]``. Every non-terminal action strictly increases ``phi``
        (retrieve grows depth; reconcile jumps to a verified depth-M signature),
        so each state's successors are solved when it is reached -- one exact
        pass. Read predictions use the effective-answer belief; answer rewards
        use the true-answer belief; reconcile reveals a* (drawn from the
        true-answer belief) and lands on the verified clean signature.
        """
        env = self.env
        A = env.num_actions
        Q = np.full((env.num_states, A), NEG_INF)
        V = np.zeros(env.num_states)
        order = np.argsort(-env.potential)  # highest potential first
        for s in order:
            b_eff = env.belief_eff(s, self.noise)
            b_true = env.belief_true(s, self.noise)
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
                    # Authoritative re-read reveals a* ~ belief_true and lands on
                    # the verified clean signature e*(a*). Its own (higher) cost
                    # keeps it off the clean path yet worth it when polluted.
                    v_next = sum(b_true[j] * V[env.reconcile_state(j)] for j in range(env.K))
                    Q[s, a] = -env.cfg.reconcile_cost + self.discount * v_next
                else:
                    out = env.retrieve_outcome_dist(s, a, self.noise)
                    s_sup = env.next_state(s, a, SUPPORTS)
                    s_ref = env.next_state(s, a, REFUTES)
                    Q[s, a] = -env.cfg.step_cost + self.discount * (out[0] * V[s_sup] + out[1] * V[s_ref])
            V[s] = Q[s, mask].max()
        self.Q = Q
        self.V = V
        self._log_probs = self._boltzmann_log_probs(Q)

    def _boltzmann_log_probs(self, Q: np.ndarray) -> np.ndarray:
        """Row-wise log-softmax of Q / temperature over legal actions only."""
        logits = Q / self.temperature
        log_probs = np.full_like(logits, NEG_INF)
        for s in range(self.env.num_states):
            mask = self.env.legal_actions(s)
            z = logits[s, mask]
            z = z - z.max()
            lp = z - np.log(np.exp(z).sum())
            log_probs[s, mask] = lp
        return log_probs

    def log_probs(self, state_idx: int) -> np.ndarray:
        return self._log_probs[state_idx]

    def probs(self, state_idx: int) -> np.ndarray:
        return np.exp(self._log_probs[state_idx])


def build_features(env: RetrievalQAEnv) -> np.ndarray:
    """One-hot encode every observable state into a feature matrix.

    Each source contributes a 3-way one-hot (unqueried / supports / refutes);
    then a ``polluted`` phase bit, a ``verified`` phase bit, and a constant bias
    term are appended. Returns ``[num_states, 3*M + 3]``.

    The ``polluted`` column is the crux of the linear-student collapse: clean
    collection is essentially never polluted, so this column receives no gradient
    under offline OPD and stays at its initial weight. A polluted state then has
    the same trained logits as the clean state with identical evidence -- whose
    optimal action is ``commit the evidence peak`` -- so the student commits the
    (wrong) peak instead of reconciling. Online OPD visits polluted states, the
    teacher advantage drives this column up, and the student learns to reconcile
    when polluted while keeping the clean states correct.
    """
    from .env import POLLUTED, VERIFIED
    M = env.M
    n = env.num_states
    d = 3 * M + 3
    feats = np.zeros((n, d), dtype=np.float64)
    for s in range(n):
        row = env.states[s]
        for m in range(M):
            feats[s, 3 * m + int(row[m])] = 1.0
        feats[s, 3 * M] = 1.0 if env.phase_of[s] == POLLUTED else 0.0
        feats[s, 3 * M + 1] = 1.0 if env.phase_of[s] == VERIFIED else 0.0
        feats[s, -1] = 1.0
    return feats


class LinearSoftmaxStudent:
    """Linear-softmax policy: logits = W @ phi(s), masked to legal actions.

    The parameter vector is ``W`` of shape ``[num_actions, feature_dim]``,
    flattened. ``grad_logpi_weighted`` returns the advantage-weighted score
    function used by both OPD gradients.
    """

    def __init__(self, env: RetrievalQAEnv, features: np.ndarray, seed: int = 0):
        self.env = env
        self.features = features
        self.A = env.num_actions
        self.d = features.shape[1]
        rng = np.random.default_rng(seed)
        self.W = rng.normal(scale=0.01, size=(self.A, self.d))

    @property
    def num_params(self) -> int:
        return self.W.size

    def get_params(self) -> np.ndarray:
        return self.W.ravel().copy()

    def set_params(self, theta: np.ndarray) -> None:
        self.W = theta.reshape(self.A, self.d).copy()

    def logits(self, state_idx: int) -> np.ndarray:
        z = self.W @ self.features[state_idx]
        mask = self.env.legal_actions(state_idx)
        z = np.where(mask, z, NEG_INF)
        return z

    def log_probs(self, state_idx: int) -> np.ndarray:
        z = self.logits(state_idx)
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    def probs(self, state_idx: int) -> np.ndarray:
        return np.exp(self.log_probs(state_idx))

    def grad_logpi_weighted(self, state_idx: int, coeffs: np.ndarray) -> np.ndarray:
        """Return ``sum_a coeffs[a] * grad_theta log pi(a | s)`` flattened.

        For a softmax, ``grad_theta log pi(a|s) = (e_a - pi(.|s)) outer phi(s)``.
        Summing over actions with weights ``coeffs`` gives
        ``(coeffs - (sum_a coeffs) * pi) outer phi``.
        """
        p = self.probs(state_idx)
        phi = self.features[state_idx]
        g_actions = coeffs - coeffs.sum() * p  # [A]
        return np.outer(g_actions, phi).ravel()

    def max_score_norm(self) -> float:
        """Exact ``max_{s,a} ||grad log pi(a|s)||_F`` without params-sized work.

        The score is the rank-1 outer product ``(e_a - pi) outer phi(s)``, whose
        Frobenius norm factorises as ``||e_a - pi|| * ||phi(s)||``.
        """
        G = 0.0
        for s in range(self.env.num_states):
            mask = self.env.legal_actions(s)
            phi_norm = float(np.linalg.norm(self.features[s]))
            p = self.probs(s)
            for a in range(self.A):
                if not mask[a]:
                    continue
                e = np.zeros(self.A)
                e[a] = 1.0
                G = max(G, float(np.linalg.norm(e - p)) * phi_norm)
        return G


class TabularStudent:
    """One independent logit row per information state.

    Highest-capacity student in the study (zero approximation error, teacher
    exactly representable), but with no parameter sharing: the gradient at state
    ``s`` touches only row ``s``. Off-support states therefore receive zero
    gradient and stay frozen at their initial value -- the control that isolates
    generalisation from literal coverage.
    """

    def __init__(self, env: RetrievalQAEnv, seed: int = 0):
        self.env = env
        self.A = env.num_actions
        self.n = env.num_states
        rng = np.random.default_rng(seed)
        self.logits_table = rng.normal(scale=0.01, size=(self.n, self.A))

    @property
    def num_params(self) -> int:
        return self.logits_table.size

    def get_params(self) -> np.ndarray:
        return self.logits_table.ravel().copy()

    def set_params(self, theta: np.ndarray) -> None:
        self.logits_table = theta.reshape(self.n, self.A).copy()

    def logits(self, state_idx: int) -> np.ndarray:
        mask = self.env.legal_actions(state_idx)
        return np.where(mask, self.logits_table[state_idx], NEG_INF)

    def log_probs(self, state_idx: int) -> np.ndarray:
        z = self.logits(state_idx)
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    def probs(self, state_idx: int) -> np.ndarray:
        return np.exp(self.log_probs(state_idx))

    def grad_logpi_weighted(self, state_idx: int, coeffs: np.ndarray) -> np.ndarray:
        """Sparse gradient: only state ``state_idx``'s row is non-zero."""
        p = self.probs(state_idx)
        g = np.zeros_like(self.logits_table)
        g[state_idx] = coeffs - coeffs.sum() * p
        return g.ravel()

    def max_score_norm(self) -> float:
        """Exact ``max_{s,a} ||grad log pi(a|s)||``; for tabular ``phi`` is e_s, norm 1."""
        G = 0.0
        for s in range(self.env.num_states):
            mask = self.env.legal_actions(s)
            p = self.probs(s)
            for a in range(self.A):
                if not mask[a]:
                    continue
                e = np.zeros(self.A)
                e[a] = 1.0
                G = max(G, float(np.linalg.norm(e - p)))
        return G
