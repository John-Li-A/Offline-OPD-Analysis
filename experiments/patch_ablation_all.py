"""Unified ablation of all five assignment-hint patches for Q-vii.

The assignment lists five candidate fixes for offline OPD's multi-turn collapse:

    1. support-aware loss          (reshape/zero the advantage off-support)
    2. DAgger-style refresh        (= our chi2-triggered refresh)
    3. conservative regularization (pull pi_theta toward pi_ref)
    4. branch-aware replay         (upweight teacher/ref-disagreement states)
    5. uncertainty-triggered query (query the teacher where the student is unsure)

We evaluate all five (plus the offline floor, the online ceiling, and the
free full-distribution control) on the *same* feature-collision trap, at two
collection conditions, with the linear (generalising, ~real-LLM) student.

The organising question is mechanistic, not just "what is the success number":
**can the patch put a non-zero gradient on the polluted-bit column** -- the
single feature the offline gradient cannot reach (clean collection never sets
it, so its gradient is structurally zero). A patch can only lift success if it
either (a) buys coverage of the polluted states (refresh / query), or (b) finds
some other route to that column. So for each patch we report, exactly:

    success            -- deployment success (sampling-free)
    pol-bit w          -- W[reconcile, polluted-bit]; SFT init is ~ -0.013
    reconcile mass     -- deployment visitation-weighted reconcile probability
    env @ deploy?      -- does the patch touch the environment after collection?
    env cost           -- rollout refreshes / teacher queries spent (0 = pure offline)

No conclusion is hard-coded here. Whatever the loop prints is what we report.
"""

import copy

import numpy as np

from opd_toy import (
    EnvConfig, RetrievalQAEnv, TeacherPolicy, TrainConfig,
    LinearSoftmaxStudent, build_features,
    train_sft, train_offline_opd, train_online_opd, train_refresh,
)
from opd_toy import exact

DEPLOY = 0.45


def base_cfg(**kw):
    d = dict(num_answers=3, sources_per_answer=2, step_cost=0.04, reconcile_cost=0.8,
             wrong_penalty=3.0, base_signal=1.0, cross_signal=0.0, pollute_coeff=1.2)
    d.update(kw)
    return EnvConfig(**d)


def diagnostics(env, stu):
    """Crux diagnostics: polluted-bit reconcile weight + deployment reconcile mass."""
    col = 3 * env.M
    w_pol = float(stu.W[env.reconcile_action, col])
    dep = exact.occupancy(env, stu, DEPLOY)
    rmass = 0.0
    for s in range(env.num_states):
        if dep.visit[s] <= 0.0 or not env.legal_actions(s)[env.reconcile_action]:
            continue
        rmass += dep.visit[s] * stu.probs(s)[env.reconcile_action]
    return dep.success, w_pol, float(rmass)


# ---------------------------------------------------------------------------
# Patch implementations. Each takes (env, teacher, reference, student, cfg) and
# returns the trained student. All gradients are exact (sampling-free).
# ---------------------------------------------------------------------------

def patch_support_aware(env, teacher, reference, student, cfg):
    """Support-aware loss: zero the OPD advantage on states the frozen dataset
    effectively never covers (expected count < 1 in N reference rollouts).

    The textbook conservative move -- do not trust advantages extrapolated onto
    unsupported states. Note it can only *remove* gradient signal, never create
    it: off-support states already carry ~zero reference visitation, so the
    masked term was already ~zero. Pure offline; zero env access."""
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    covered = (ref_stats.visit * cfg.dataset_size) >= 1.0  # [num_states]
    for _ in range(cfg.steps):
        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        adv = np.where(covered[:, None], adv, 0.0)
        grad = exact.opd_gradient(env, student, ref_stats, reference, adv)
        student.set_params(student.get_params() + cfg.lr * grad)
    return student


def patch_conservative(env, teacher, reference, student, cfg, lam=0.3):
    """Conservative regularization: offline OPD gradient minus lambda * grad of
    KL(pi_theta || pi_ref), under the frozen reference visitation.

    grad_theta KL(pi_theta||pi_ref) at s has coeffs pi_theta(a)*(log pi_theta -
    log pi_ref)(a) (the +1 term cancels). This pulls the student toward the
    behaviour policy. Because pi_ref itself extrapolates 'commit the peak' into
    the polluted states (same feature collision), pulling toward it is expected
    to entrench the failure, not fix it. Pure offline; zero env access."""
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    for _ in range(cfg.steps):
        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        grad = exact.opd_gradient(env, student, ref_stats, reference, adv)
        kl_grad = np.zeros(student.num_params)
        for s in range(env.num_states):
            w = ref_stats.visit[s]
            if w <= 0.0:
                continue
            lp_t = student.log_probs(s)
            lp_r = reference.log_probs(s)
            mask = env.legal_actions(s)
            pt = student.probs(s)
            coeffs = np.where(mask, pt * (lp_t - lp_r), 0.0)
            kl_grad += w * student.grad_logpi_weighted(s, coeffs)
        student.set_params(student.get_params() + cfg.lr * (grad - lam * kl_grad))
    return student


def patch_branch_replay(env, teacher, reference, student, cfg, gamma=4.0):
    """Branch-aware replay: reweight the frozen reference visitation by
    alpha(s) = 1 + gamma * KL(pi_T(.|s) || pi_ref(.|s)), upweighting states where
    the teacher and the behaviour policy disagree most (the 'interesting'
    branches). This is the competitor's replay-reweighting patch.

    Crucially it only redistributes mass that the dataset already has: a polluted
    state with zero reference visitation stays at zero however large its
    disagreement, so at clean collection there is nothing to upweight. Pure
    offline; zero env access."""
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    alpha = np.ones(env.num_states)
    for s in range(env.num_states):
        mask = env.legal_actions(s)
        if not mask.any():
            continue
        lt = teacher.log_probs(s)
        lr = reference.log_probs(s)
        pt = teacher.probs(s)
        kl = float(np.where(mask, pt * (lt - lr), 0.0).sum())
        alpha[s] = 1.0 + gamma * max(kl, 0.0)
    reweighted = copy.copy(ref_stats)
    reweighted.visit = ref_stats.visit * alpha
    for _ in range(cfg.steps):
        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        grad = exact.opd_gradient(env, student, reweighted, reference, adv)
        student.set_params(student.get_params() + cfg.lr * grad)
    return student


def patch_uncertainty_query(env, teacher, reference, student, cfg, ent_frac=0.5, beta=1.0):
    """Uncertainty-triggered teacher query (SafeDAgger-style): each step, roll out
    the current student at deployment noise, flag the states it visits where its
    normalised entropy exceeds ``ent_frac`` of the max, and add a teacher-cloning
    gradient there (a teacher query at those states).

    This DOES touch the environment at deployment (student rollouts) and queries
    the live teacher -- it breaks the pure-offline contract. We include it to
    test the gating idea on its own terms. We also report what fraction of
    visited deployment mass the gate actually fires on, because in a
    feature-collision trap the student may be *confidently wrong* (low entropy)
    at exactly the polluted states, in which case the gate never fires there."""
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    queries = 0
    for _ in range(cfg.steps):
        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        grad = exact.opd_gradient(env, student, ref_stats, reference, adv)
        dep = exact.occupancy(env, student, cfg.deploy_noise)
        for s in range(env.num_states):
            w = dep.visit[s]
            if w <= 0.0:
                continue
            mask = env.legal_actions(s)
            nlegal = int(mask.sum())
            if nlegal <= 1:
                continue
            p = student.probs(s)
            ent = -float(np.sum(p * np.log(p, where=p > 0, out=np.zeros_like(p))))
            if ent < ent_frac * np.log(nlegal):
                continue
            coeffs = w * teacher.probs(s)
            grad += beta * student.grad_logpi_weighted(s, coeffs)
            queries += 1
        student.set_params(student.get_params() + cfg.lr * grad)
    return student, queries


def patch_fulldist(env, teacher, reference, student, cfg):
    """Full-distribution advantage (Rang-style): offline gradient weighted by the
    *teacher* action distribution instead of the behaviour policy, frozen
    reference visitation, zero env access. Included as the 'free' control."""
    ref_stats = exact.occupancy(env, reference, cfg.collect_noise)
    for _ in range(cfg.steps):
        adv = exact.advantage(env, teacher, student, clip=cfg.clip)
        grad = exact.opd_gradient(env, student, ref_stats, teacher, adv)
        student.set_params(student.get_params() + cfg.lr * grad)
    return student


def run_point(collect, steps=300, lr=0.5):
    env = RetrievalQAEnv(base_cfg())
    feats = build_features(env)
    teacher = TeacherPolicy(env, noise=DEPLOY, temperature=0.03)
    tsucc = exact.occupancy(env, teacher, DEPLOY).success
    cfg = TrainConfig(steps=steps, lr=lr, collect_noise=collect, deploy_noise=DEPLOY,
                      dataset_size=2000, record_every=steps)
    sft = train_sft(env, teacher, LinearSoftmaxStudent(env, feats, seed=0), cfg)
    rp = sft.student.get_params()

    def fresh():
        st = LinearSoftmaxStudent(env, feats, seed=0)
        st.set_params(rp)
        return st

    rows = []  # (label, success, w_pol, rmass, env_at_deploy, env_cost)

    s, w, r = diagnostics(env, fresh())
    rows.append(("SFT (floor)", s, w, r, "no", "0"))

    off = train_offline_opd(env, teacher, fresh(), fresh(), cfg).student
    s, w, r = diagnostics(env, off)
    rows.append(("offline OPD", s, w, r, "no", "0"))

    sa = patch_support_aware(env, teacher, fresh(), fresh(), cfg)
    s, w, r = diagnostics(env, sa)
    rows.append(("1 support-aware", s, w, r, "no", "0"))

    cons = patch_conservative(env, teacher, fresh(), fresh(), cfg)
    s, w, r = diagnostics(env, cons)
    rows.append(("3 conservative", s, w, r, "no", "0"))

    br = patch_branch_replay(env, teacher, fresh(), fresh(), cfg)
    s, w, r = diagnostics(env, br)
    rows.append(("4 branch-replay", s, w, r, "no", "0"))

    fd = patch_fulldist(env, teacher, fresh(), fresh(), cfg)
    s, w, r = diagnostics(env, fd)
    rows.append(("full-dist (Rang)", s, w, r, "no", "0"))

    rf = train_refresh(env, teacher, fresh(), fresh(), cfg, trigger="chi2",
                       chi2_thresh=0.2, budget=50, check_every=10)
    s, w, r = diagnostics(env, rf.student)
    rows.append(("2 chi2-refresh (ours)", s, w, r, "collect", f"{rf.final['refreshes']} refresh"))

    uq, nq = patch_uncertainty_query(env, teacher, fresh(), fresh(), cfg)
    s, w, r = diagnostics(env, uq)
    rows.append(("5 uncertainty-query", s, w, r, "deploy", f"{nq} queries"))

    on = train_online_opd(env, teacher, fresh(), fresh(), cfg).student
    s, w, r = diagnostics(env, on)
    rows.append(("online OPD (ceiling)", s, w, r, "deploy", f"{steps} rollouts"))

    return tsucc, rows


def main():
    for collect in (1e-9, 0.05):
        tsucc, rows = run_point(collect)
        print(f"\n=== collect_noise = {collect:g}   (deploy = {DEPLOY}, "
              f"teacher@deploy = {tsucc:.3f}) ===")
        print(f"{'patch':<24}{'succ':>7}{'pol-bit w':>11}{'rec-mass':>10}"
              f"{'env@dep':>9}{'env cost':>14}")
        print("-" * 75)
        for label, s, w, r, env_at, cost in rows:
            print(f"{label:<24}{s:>7.3f}{w:>11.3f}{r:>10.3f}{env_at:>9}{cost:>14}")


if __name__ == "__main__":
    main()
