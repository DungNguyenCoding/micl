import math

import torch

from bayesfl.posterior.scale_mixture import normal_log_prob, scale_mixture_log_prob


def test_scale_mixture_log_prob_matches_direct_density():
    x = torch.tensor([0.0, 0.5], dtype=torch.float64)
    pi = 0.5
    sigma1 = 1.0
    sigma2 = math.exp(-6.0)
    got = scale_mixture_log_prob(x, pi=pi, sigma1=sigma1, sigma2=sigma2)
    direct = torch.log(
        pi * torch.exp(normal_log_prob(x, 0.0, sigma1))
        + (1 - pi) * torch.exp(normal_log_prob(x, 0.0, sigma2))
    )
    assert torch.allclose(got, direct, atol=1e-10, rtol=1e-10)
