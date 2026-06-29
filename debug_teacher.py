"""Verify the multi-source design produces a *recoverable* shift.

The single number that decides whether the study is viable:
    collect-policy @ deploy   vs   deploy-policy @ deploy
If the deploy-optimal policy clearly beats the collect-optimal one *at deploy*,
then "retrieve more" is a learnable recovery and online methods have somewhere
to climb to. (Under the old blur-noise design these were equal -> no story.)
"""

import numpy as np

from opd_toy import EnvConfig, RetrievalQAEnv, TeacherPolicy
from opd_toy import exact


def bayes_ceiling(env, noise):
    """Probe-everything MAP accuracy: an upper bound on achievable success."""
    psup = env.support_prob(noise)  # [M, K]
    M, K = env.M, env.K
    acc = 0.0
    for pat in range(2 ** M):
        bits = [(pat >> m) & 1 for m in range(M)]
        joint = np.array(env.prior)
        for m in range(M):
            joint = joint * (psup[m] if bits[m] else (1 - psup[m]))
        acc += joint.max()
    return acc


def probe_table(env, noise, temp=0.03):
    t = TeacherPolicy(env, noise=noise, temperature=temp)
    st = exact.occupancy(env, t, noise)
    return t, st


def main():
    collect, deploy = 0.05, 0.40
    K, r = 3, 3
    M = K * r
    print(f"collect={collect} deploy={deploy}  K={K} r={r} M={M}"
          f"  -- sweeping (wrong_penalty, step_cost)\n")
    print(f"{'wpen':>6}{'cost':>6}{'ceil_d':>8}"
          f"{'cc':>7}{'cd':>7}{'dd':>7}{'gap':>7}{'len_c':>7}{'len_d':>7}")
    print("-" * 70)
    for wpen in (1.0, 2.0, 4.0, 8.0):
        for cost in (0.02, 0.05, 0.1):
            cfg = EnvConfig(num_answers=K, sources_per_answer=r, horizon=M,
                            step_cost=cost, wrong_penalty=wpen,
                            base_signal=0.95, cross_signal=0.08)
            env = RetrievalQAEnv(cfg)
            ceil_d = bayes_ceiling(env, deploy)
            t_c, st_c = probe_table(env, collect)
            t_d, st_d = probe_table(env, deploy)
            cd = exact.occupancy(env, t_c, deploy)
            gap = st_d.success - cd.success
            flag = "  <==" if gap > 0.08 and ceil_d > 0.7 else ""
            print(f"{wpen:>6}{cost:>6}{ceil_d:>8.3f}"
                  f"{st_c.success:>7.3f}{cd.success:>7.3f}{st_d.success:>7.3f}"
                  f"{gap:>7.3f}{st_c.avg_length:>7.2f}{st_d.avg_length:>7.2f}{flag}")
    print("\ncc=collect-pol@collect  cd=collect-pol@deploy(pi_ref-like)  dd=deploy-pol@deploy(target)")
    print("Want gap=dd-cd large AND ceil_d high. Higher wrong_penalty makes early commits costly,")
    print("so deploy-optimal retrieves more (len_d up) while collect-optimal stays overconfident.")


if __name__ == "__main__":
    main()
