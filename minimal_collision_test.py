"""Minimal feasibility test of the feature-collision trap mechanism.

Before rewriting the whole env, verify in isolation the single claim the full
design rests on: a LINEAR (generalising) student collapses under offline OPD,
while online OPD recovers -- driven by a feature collision, not by capacity.

Setup (the smallest thing that captures the mechanism):
  Actions: 0=commit_X, 1=commit_Y, 2=reconcile.
  States (as student feature rows [is_sigX, is_sigY, b_bit, bias]):
    C_X = [1,0,0,1]  true answer X  -> optimal commit_X
    C_Y = [0,1,0,1]  true answer Y  -> optimal commit_Y
    P   = [1,0,1,1]  true answer Y  -> optimal reconcile
         (P is identical to C_X except the pollution bit b=1)

  Clean collection visits {C_X, C_Y} only (b is never set), so the b-column
  receives no offline gradient and stays at init. At deployment the student also
  reaches P. Because P's features equal C_X's except the untrained b-column, the
  offline-trained student treats P like C_X and commits X (wrong). Online OPD
  visits P, the teacher's reconcile advantage trains the b-column, and the
  student learns to separate P from C_X and reconcile.

This is the principled analogue of "an early error poisons later states": the
polluted state looks exactly like a confident clean state for the WRONG answer,
so generalisation -- which rescued the student elsewhere -- becomes the
accomplice that drives it into the trap.
"""

import numpy as np

NEG_INF = -1e30
A = 3  # commit_X, commit_Y, reconcile
COMMIT_X, COMMIT_Y, RECONCILE = 0, 1, 2

# Feature rows.
phi = {
    "C_X": np.array([1.0, 0.0, 0.0, 1.0]),
    "C_Y": np.array([0.0, 1.0, 0.0, 1.0]),
    "P":   np.array([1.0, 0.0, 1.0, 1.0]),  # == C_X except b-bit
}
optimal = {"C_X": COMMIT_X, "C_Y": COMMIT_Y, "P": RECONCILE}
D = 4


def near_greedy_logprobs(opt_action, temp=0.1):
    """Teacher: near-greedy on the optimal action over the 3 actions."""
    z = np.full(A, 0.0)
    z[opt_action] = 1.0
    z = z / temp
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


teacher_lp = {s: near_greedy_logprobs(optimal[s]) for s in phi}


class Lin:
    def __init__(self, seed=0):
        self.W = np.random.default_rng(seed).normal(scale=0.01, size=(A, D))

    def logp(self, s):
        z = self.W @ phi[s]
        z = z - z.max()
        return z - np.log(np.exp(z).sum())

    def p(self, s):
        return np.exp(self.logp(s))

    def grad_weighted(self, s, coeffs):
        p = self.p(s)
        return np.outer(coeffs - coeffs.sum() * p, phi[s])


def opd_grad(student, visit, action_weight):
    """sum_s visit(s) sum_a aw(a|s) * A(s,a) * grad log pi(a|s), A=logT-logS."""
    g = np.zeros((A, D))
    for s, w in visit.items():
        if w <= 0:
            continue
        adv = np.clip(teacher_lp[s] - student.logp(s), -10, 10)
        aw = action_weight(s)
        g += student.grad_weighted(s, w * aw * adv)
    return g


def train(student, visit, action_weight, steps=400, lr=0.5):
    for _ in range(steps):
        student.W += lr * opd_grad(student, visit, action_weight)
    return student


def report(tag, student):
    pP = student.p("P")
    pCX = student.p("C_X")
    print(f"  {tag:8s} | P: commitX={pP[COMMIT_X]:.3f} reconcile={pP[RECONCILE]:.3f}"
          f"  | C_X: commitX={pCX[COMMIT_X]:.3f}"
          f"  | b-col(reconcile)={student.W[RECONCILE,2]:+.3f}")


def main():
    # Reference/teacher visitation at clean collection: only b=0 states.
    clean_visit = {"C_X": 0.5, "C_Y": 0.5}
    # Deployment visitation: the pollution dynamic routes some wrong-answer mass
    # into the polluted state P (b=1) that clean collection never produced.
    deploy_visit = {"C_X": 0.4, "C_Y": 0.3, "P": 0.3}

    print("teacher (oracle): P->reconcile, C_X->commitX, C_Y->commitY\n")

    # --- Offline OPD: frozen clean visitation + reference action weights. ---
    off = Lin(seed=1)
    # seed the student near the reference by a quick SFT-like warmup on clean.
    train(off, clean_visit, lambda s: np.exp(teacher_lp[s]), steps=400, lr=0.5)
    print("OFFLINE OPD (visits only b=0 clean states):")
    report("offline", off)

    # --- Online OPD: deploy visitation incl. P + student action weights. ---
    on = Lin(seed=1)
    train(on, clean_visit, lambda s: np.exp(teacher_lp[s]), steps=400, lr=0.5)  # same warmup
    train(on, deploy_visit, lambda s: on.p(s), steps=400, lr=0.5)
    print("ONLINE OPD (resamples into polluted state P):")
    report("online", on)

    print("\nVerdict:")
    print(f"  offline reconcile@P = {off.p('P')[RECONCILE]:.3f}  (low => collapsed into the trap)")
    print(f"  online  reconcile@P = {on.p('P')[RECONCILE]:.3f}  (high => recovered)")


if __name__ == "__main__":
    main()
