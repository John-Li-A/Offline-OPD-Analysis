"""Multi-hop retrieval QA POMDP with an undirected-spoof pollution trap.

A hidden answer ``a*`` is drawn uniformly from K candidates. The agent issues
*retrieve* actions against M information sources (M > K, several independent
sources per answer) and each returns a noisy binary signal -- does this source
support its answer-group. After gathering evidence the agent commits one
*answer* and the episode ends. A source cannot be re-queried, so an early read
is locked in and biases every later decision.

The trap: undirected start-latent spoof
---------------------------------------
At episode start the episode is *polluted* with probability ``rho(noise)``. When
polluted, an *effective answer* ``a_tilde`` is drawn uniformly from the wrong
answers ``{!= a*}`` and **all reads in the episode are governed by a_tilde
instead of a***. So a polluted episode accumulates a clean, confident evidence
signature for the WRONG answer -- an early derailment that compounds over every
turn. The agent observes only a *polluted* phase bit, never which wrong answer
``a_tilde`` it was spoofed toward.

Because ``rho(noise) = pollute_coeff * noise``, clean collection (noise ~ 0) is
essentially never polluted, so offline data never exercises the polluted bit;
deployment noise makes pollution common. This is the distribution shift, and it
is emergent from the noise level, not a hand-wired action probability.

The feature collision (why even a generalising student collapses)
-----------------------------------------------------------------
A polluted state with evidence ``e`` (a confident signature of some wrong answer
x) is *feature-identical* to a clean state with the same ``e`` -- they differ
only in the polluted bit. The clean state's optimal action is ``commit x``; the
polluted state's truth is ``!= x`` so its optimal action is ``reconcile``. A
student trained only on clean data learns ``commit the evidence peak`` and never
trains the polluted-bit weight, so at deployment it commits ``x`` in the
polluted state too (wrong) -- generalisation becomes the accomplice. Online OPD,
which resamples into polluted states and sees the teacher reconcile, trains the
polluted-bit weight and recovers, *while keeping the clean states correct*.

The closed forms that keep everything exact (no Monte Carlo)
------------------------------------------------------------
* ``belief_eff(a_tilde | e) ~ prod L(e | a_tilde)`` (uniform prior) drives read
  dynamics -- identical to a plain retrieval belief.
* ``belief_true(a* | e, polluted) = (1 - belief_eff(a*)) / (K - 1)`` -- derived,
  because a* is conditionally independent of the reads given a_tilde. In clean
  or verified phases ``a* = a_tilde`` so ``belief_true = belief_eff``.
* ``reconcile`` is an authoritative re-read that reveals a* and jumps to a
  *verified* clean signature ``e*(a*)``; from there the peak is the truth.

State: ``(e, phase)`` with phase in {CLEAN, POLLUTED, VERIFIED}; index
``s = e_idx + phase * 3^M``; potential ``phi = depth(e) + (M+1)*[verified]``
makes the reachable graph a DAG, so the teacher is one exact backward pass and
every occupancy / chi-squared / sigma quantity is one exact forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Retrieval outcome symbols stored in the information state, one slot per source.
UNQUERIED = 0
SUPPORTS = 1  # the retrieved passage supports this source's answer group
REFUTES = 2  # the retrieved passage argues against it

# Episode phases (the high bits of the state index).
CLEAN = 0
POLLUTED = 1
VERIFIED = 2


@dataclass(frozen=True)
class EnvConfig:
    """Static description of a multi-hop retrieval QA task with a spoof trap.

    Attributes:
        num_answers: number of candidate answers K. Needs K>=3 for genuine
            polluted-state ambiguity (a* is uniform over the K-1 wrong answers).
        sources_per_answer: independent sources per answer; ``M = K *
            sources_per_answer`` total sources.
        horizon: generous cap; episode length is structurally bounded by M+2.
        step_cost: per-action cost; makes reconcile genuinely costly on the clean
            path (so the teacher never reconciles there) yet worth it when
            polluted.
        reconcile_cost: cost of the reconcile action specifically. Set above the
            evidence-gathering it would replace so the teacher never reconciles
            on the clean path (keeping it off the reference demonstrations), yet
            below the wrong-answer regret so a polluted state still makes it
            worth it. This single knob is what keeps the polluted-bit feature
            untrained offline.
        correct_reward: reward for committing the true answer a*.
        wrong_penalty: penalty for committing a wrong answer; makes a confident
            wrong commit worse than recovering, so reconcile is value-optimal in
            polluted states.
        base_signal / cross_signal: P(source SUPPORTS | own / other answer) at
            zero noise.
        pollute_coeff: ``rho(noise) = clip(pollute_coeff * noise, 0, 1)`` is the
            probability an episode starts polluted. Linear in noise so clean
            collection (noise ~ 0) is ~never polluted but deployment is.
        enable_reconcile: if False the reconcile action is illegal and the
            polluted/verified machinery is inert (plain retrieval task).
        seed: reserved.
    """

    num_answers: int = 3
    sources_per_answer: int = 2
    horizon: int = 8
    step_cost: float = 0.04
    reconcile_cost: float = 0.8
    correct_reward: float = 1.0
    wrong_penalty: float = 3.0
    base_signal: float = 1.0
    cross_signal: float = 0.0
    pollute_coeff: float = 1.2
    enable_reconcile: bool = True
    seed: int = 0


class RetrievalQAEnv:
    """Enumerable multi-hop retrieval QA POMDP with an undirected-spoof trap.

    Action layout: ``0..M-1`` retrievals (source m in group ``m //
    sources_per_answer``), ``M..M+K-1`` answers, ``M+K`` reconcile. Observable
    state is ``(e, phase)``; the latent effective answer ``a_tilde`` and true
    answer ``a*`` are never in the state (a fair POMDP oracle), but the exact
    occupancy machinery carries ``a_tilde`` so the noisy dynamics stay closed
    form.
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.K = cfg.num_answers
        self.M = cfg.num_answers * cfg.sources_per_answer
        self.reconcile_action = self.M + self.K
        self.num_actions = self.M + self.K + 1
        self.prior = np.full(self.K, 1.0 / self.K)
        self.source_group = np.array([m // cfg.sources_per_answer for m in range(self.M)])
        self._build_evidence_matrix()
        self._enumerate_states()

    # -- construction -----------------------------------------------------
    def _build_evidence_matrix(self) -> None:
        """E[m, a] = P(source_m SUPPORTS | effective answer a) at zero noise."""
        E = np.full((self.M, self.K), self.cfg.cross_signal)
        for m in range(self.M):
            E[m, self.source_group[m]] = self.cfg.base_signal
        self.E = E

    def support_prob(self, noise: float) -> np.ndarray:
        """P(source_m SUPPORTS | effective answer a) at a given distractor rate."""
        return (1.0 - noise) * self.E + noise * 0.5

    def rho(self, noise: float) -> float:
        """Probability an episode starts polluted at this noise level."""
        if not self.cfg.enable_reconcile:
            return 0.0
        return float(np.clip(self.cfg.pollute_coeff * noise, 0.0, 1.0))

    def _enumerate_states(self) -> None:
        """Build the ``3 * 3^M`` states ``(e, phase)`` and the potential order."""
        M, K = self.M, self.K
        base = 3**M
        self.n_evidence = base
        self.num_states = 3 * base
        e_states = np.empty((base, M), dtype=np.int8)
        for idx in range(base):
            rem = idx
            for m in range(M):
                e_states[idx, m] = rem % 3
                rem //= 3
        self.states = np.vstack([e_states, e_states, e_states])  # CLEAN, POLLUTED, VERIFIED
        self.phase_of = np.concatenate([
            np.full(base, CLEAN, dtype=np.int8),
            np.full(base, POLLUTED, dtype=np.int8),
            np.full(base, VERIFIED, dtype=np.int8)])
        depth_e = (e_states != UNQUERIED).sum(axis=1).astype(np.int64)
        self.depth = np.concatenate([depth_e, depth_e, depth_e])
        verified = (self.phase_of == VERIFIED).astype(np.int64)
        self.potential = self.depth + (M + 1) * verified
        self.e_index = {tuple(int(x) for x in e_states[idx]): idx for idx in range(base)}
        # e*(a): clean fully-queried signature of answer a.
        self.clean_target = np.empty(K, dtype=np.int64)
        for a in range(K):
            e = np.array([SUPPORTS if self.source_group[m] == a else REFUTES
                          for m in range(M)], dtype=np.int8)
            self.clean_target[a] = self.e_index[tuple(int(x) for x in e)]

    # -- beliefs ----------------------------------------------------------
    def belief_eff(self, state_idx: int, noise: float) -> np.ndarray:
        """Posterior over the effective answer a_tilde given evidence (uniform
        prior). Drives read dynamics; identical to a plain retrieval belief."""
        s = self.states[state_idx]
        psup = self.support_prob(noise)  # [M, K]
        logp = np.log(self.prior)
        for m in range(self.M):
            if s[m] == SUPPORTS:
                logp = logp + np.log(psup[m])
            elif s[m] == REFUTES:
                logp = logp + np.log(1.0 - psup[m])
        logp -= logp.max()
        p = np.exp(logp)
        return p / p.sum()

    def belief_true(self, state_idx: int, noise: float) -> np.ndarray:
        """Posterior over the TRUE answer a* given the observable state.

        Clean/verified: a* == a_tilde, so it is the effective belief. Polluted:
        a* is uniform over the wrong answers given a_tilde, marginalising to
        ``(1 - belief_eff(a*)) / (K - 1)``.
        """
        be = self.belief_eff(state_idx, noise)
        if self.phase_of[state_idx] == POLLUTED:
            return (1.0 - be) / (self.K - 1)
        return be

    # -- action typing ----------------------------------------------------
    def is_retrieve(self, action: int) -> bool:
        return action < self.M

    def is_answer(self, action: int) -> bool:
        return self.M <= action < self.M + self.K

    def is_reconcile(self, action: int) -> bool:
        return action == self.reconcile_action

    def answer_of(self, action: int) -> int:
        return action - self.M

    def legal_actions(self, state_idx: int) -> np.ndarray:
        """Answers always legal; a source legal only if unqueried; reconcile
        legal only in CLEAN/POLLUTED phases when enabled."""
        s = self.states[state_idx]
        mask = np.ones(self.num_actions, dtype=bool)
        for m in range(self.M):
            if s[m] != UNQUERIED:
                mask[m] = False
        phase = self.phase_of[state_idx]
        if not (self.cfg.enable_reconcile and phase in (CLEAN, POLLUTED)):
            mask[self.reconcile_action] = False
        return mask

    # -- dynamics ---------------------------------------------------------
    def next_state(self, state_idx: int, source: int, outcome: int) -> int:
        """State after a retrieval returns ``outcome`` -- phase is preserved."""
        phase = int(self.phase_of[state_idx])
        e = self.states[state_idx].copy()
        e[source] = outcome
        return self.e_index[tuple(int(x) for x in e)] + phase * self.n_evidence

    def reconcile_state(self, true_answer: int) -> int:
        """State after reconcile reveals ``true_answer``: verified clean signature."""
        return int(self.clean_target[true_answer]) + VERIFIED * self.n_evidence

    def retrieve_outcome_dist(self, state_idx: int, source: int, noise: float) -> np.ndarray:
        """Predictive ``[p_supports, p_refutes]`` under the effective belief."""
        b = self.belief_eff(state_idx, noise)
        psup = self.support_prob(noise)[source]
        p_sup = float(b @ psup)
        return np.array([p_sup, 1.0 - p_sup])
