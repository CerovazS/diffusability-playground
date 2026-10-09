import math
import unittest

import numpy as np
import torch

from diffusability.posterior import covariance_profiles, exact_velocity, generator, heun, log_density, make_centers, stream_seed


class PosteriorMathTests(unittest.TestCase):
    def test_constraints_and_paper_anisotropy(self):
        profiles = covariance_profiles(16, 8, 16*math.log(.5)-4, [2, 4, 8])
        mu = make_centers(16, 8, 3, 42)
        rates, scales, anisotropies = [], [], []
        for s in profiles.values():
            self.assertTrue((s>0).all())
            self.assertAlmostEqual(s.sum(), 8, places=10)
            self.assertAlmostEqual(np.log(s).sum(), 16*math.log(.5)-4, places=9)
            rates.append(.5*(float(mu.square().sum(1).mean())+s.sum()-16-np.log(s).sum()))
            scales.append(float(mu.square().sum(1).mean())+s.sum())
            anisotropies.append(np.var(np.log(s)))
        self.assertLess(np.ptp(rates), 1e-9)
        self.assertLess(np.ptp(scales), 1e-10)
        self.assertTrue(anisotropies[0]<anisotropies[1]<anisotropies[2])

    def test_velocity_matches_autodifferentiated_mixture_score(self):
        mu = make_centers(4, 3, 2, 42)
        s = torch.tensor([.1, .3, .7, 1.2], dtype=torch.float64)
        x = torch.randn(64, 4, dtype=torch.float64, generator=generator(0, "check")).requires_grad_()
        t = torch.linspace(.01, .99, len(x), dtype=torch.float64)[:, None]
        score = torch.autograd.grad(log_density(x, t, mu, s).sum(), x)[0]
        torch.testing.assert_close(t*exact_velocity(x, t, mu, s), x+(1-t)*score, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(exact_velocity(x, torch.zeros_like(t), mu, s), mu.mean(0)-x)
        torch.testing.assert_close(exact_velocity(x, torch.ones_like(t), mu, s), x)

    def test_single_gaussian_flow_converges_to_exact_transport(self):
        mu = torch.tensor([[.4, -.8]], dtype=torch.float64)
        s = torch.tensor([.15, 1.7], dtype=torch.float64)
        noise = torch.randn(32, 2, dtype=torch.float64, generator=generator(1, "check"))
        exact = mu+noise*s.sqrt()
        velocity = lambda x,t:exact_velocity(x,t,mu,s)
        coarse = (heun(velocity, noise, 32)-exact).square().mean()
        fine = (heun(velocity, noise, 128)-exact).square().mean()
        self.assertLess(float(fine), float(coarse)/50)
        self.assertLess(float(fine), 1e-7)

    def test_separate_reproducible_streams(self):
        namespaces = ["train", "validation:velocity", "test:velocity", "test:reference", "test:reference2", "test:noise", "test:projections"]
        seeds = [stream_seed(seed, n) for seed in range(3) for n in namespaces]
        self.assertEqual(len(seeds), len(set(seeds)))
        a = torch.randn(20, generator=generator(0,"train"))
        b = torch.randn(20, generator=generator(0,"train"))
        c = torch.randn(20, generator=generator(0,"test:velocity"))
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(torch.equal(a,c))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
