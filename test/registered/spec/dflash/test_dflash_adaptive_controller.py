"""CPU unit tests for the DFlash adaptive shared-budget controller.

These exercise the pure-Python controller + cost model (no GPU / no server):

* `select_shared_budget` finds the true argmax of the batch speedup surrogate,
* the selected budget is invariant to any global latency scale,
* an adaptively-built, truncated best-first tree is structurally valid,
* the analytical roofline cost model is convex in N and amortizes weights, and
* the BASTION-style startup fit (`fit_from_samples`) recovers an affine roofline
  per batch-size bucket, round-trips through JSON, and yields a deterministic N*.

Run: ``python -m pytest test/registered/spec/dflash/test_dflash_adaptive_controller.py``
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from sglang.srt.speculative.dflash_cost_model import (
    DFlashAdaptiveCostModel,
    ModelDims,
    fit_roofline_affine,
    resolve_gpu_roofline,
)
from sglang.srt.speculative.dflash_tree_builder import (
    beam_tree_size,
    build_adaptive_best_first_trees_batched,
    build_beam_search_tree,
    build_beam_search_trees_batched,
    build_best_first_tree,
    select_shared_budget,
    topk_logprobs_per_depth,
)


class _FakeCost:
    """Convex verify V(N)=a*N^2+b*N, constant overhead. `scale` multiplies all."""

    def __init__(self, a, b, overhead, scale=1.0, ready=True):
        self.a, self.b, self.overhead, self.scale, self.ready = a, b, overhead, scale, ready

    def fixed_overhead_s(self, B):
        return self.overhead * self.scale if self.ready else None

    def estimate_verify(self, B, N, ctx):
        return (self.a * N * N + self.b * N) * self.scale

    def ar_latency_s(self, B, ctx):
        return self.estimate_verify(B, 1, ctx)

    def is_ready(self, B):
        return self.ready


def _brute_force_budget(marginals, cost, B, ctx, lo, hi):
    """argmax_N S_B(N) = mean_b A_b(N) * L_AR / (overhead + V(N))."""
    best_n, best_s = lo, -1.0
    for N in range(lo, hi + 1):
        acc = sum(sum(pp[:N]) for pp in marginals) / len(marginals)
        cyc = cost.fixed_overhead_s(B) + cost.estimate_verify(B, N, ctx)
        s = acc * cost.ar_latency_s(B, ctx) / cyc
        if s > best_s + 1e-12:
            best_s, best_n = s, N
    return best_n


def _make_marginals(B, L, seed=0):
    """Per-request non-increasing path-prob sequences (root=1.0)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(B):
        edges = (torch.rand(L - 1, generator=g) * 0.6 + 0.2).tolist()
        pp, cur = [1.0], 1.0
        for e in sorted(edges, reverse=True):
            cur *= e
            pp.append(cur)
        out.append(pp)
    return out


def _dims():
    # Qwen3-8B-ish.
    return ModelDims(num_layers=36, hidden_size=4096, num_attention_heads=32,
                     num_kv_heads=8, head_dim=128, intermediate_size=12288,
                     vocab_size=151936)


def _affine_samples(cm, buckets, affine, *, N_grid=(1, 2, 4, 8, 16, 32, 48, 64),
                    ctx_per=2048, draft_s=0.0036, overhead_s=0.0044):
    """Synthesize sweep samples whose verify_s is a known affine roofline."""
    ac, bc, am, bm = affine
    samples = []
    for B in buckets:
        for N in N_grid:
            ctx = B * ctx_per
            cs, ms = cm._analytical_branches(B, N, ctx)
            y = max(ac * cs + bc, am * ms + bm)
            samples.append(dict(batch=B, tree_size=N, context_sum=ctx,
                                verify_s=y, draft_s=draft_s, overhead_s=overhead_s))
    return samples


class TestSelectSharedBudget(unittest.TestCase):
    def test_matches_brute_force_argmax(self):
        ctx, B, L = 1000, 4, 40
        marg = _make_marginals(B, L)
        for a, b, ov in [(2e-4, 1e-3, 0.05), (5e-5, 2e-3, 0.2), (1e-3, 5e-4, 0.01)]:
            cost = _FakeCost(a, b, ov)
            got = select_shared_budget(
                accept_marginals=marg, batch_size=B, context_sum=ctx,
                cost_model=cost, min_budget=1, max_budget=L,
            )
            want = _brute_force_budget(marg, cost, B, ctx, 1, L)
            self.assertEqual(got, want, f"a={a} b={b} ov={ov}")

    def test_scale_invariance(self):
        ctx, B, L = 800, 3, 32
        marg = _make_marginals(B, L, seed=7)
        base = _FakeCost(3e-4, 1e-3, 0.1)
        n0 = select_shared_budget(
            accept_marginals=marg, batch_size=B, context_sum=ctx,
            cost_model=base, min_budget=1, max_budget=L,
        )
        for k in (0.01, 5.0, 137.0):
            nk = select_shared_budget(
                accept_marginals=marg, batch_size=B, context_sum=ctx,
                cost_model=_FakeCost(3e-4, 1e-3, 0.1, scale=k), min_budget=1, max_budget=L,
            )
            self.assertEqual(nk, n0, f"scale {k} changed N*")

    def test_batch_average_is_concave(self):
        B, L = 5, 30
        marg = _make_marginals(B, L, seed=3)
        avg = [sum(pp[n] if n < len(pp) else 0.0 for pp in marg) / B for n in range(L)]
        for i in range(L - 1):
            self.assertGreaterEqual(avg[i] + 1e-9, avg[i + 1])

    def test_bounds_respected(self):
        marg = _make_marginals(3, 30, seed=1)
        cost = _FakeCost(1e-4, 1e-3, 0.1)
        n = select_shared_budget(
            accept_marginals=marg, batch_size=3, context_sum=500,
            cost_model=cost, min_budget=8, max_budget=20,
        )
        self.assertTrue(8 <= n <= 20)

    def test_uncalibrated_falls_back_to_max(self):
        marg = _make_marginals(3, 30, seed=1)
        for cm in (None, _FakeCost(1e-4, 1e-3, 0.1, ready=False)):
            n = select_shared_budget(
                accept_marginals=marg, batch_size=3, context_sum=500,
                cost_model=cm, min_budget=1, max_budget=20,
            )
            self.assertEqual(n, 20)


class TestAdaptiveBuild(unittest.TestCase):
    def _draft_topk(self, bs, depth, vocab, K, seed=1):
        torch.manual_seed(seed)
        lp = torch.log_softmax(torch.randn(bs, depth, vocab), dim=-1)
        vals, ids = torch.topk(lp, k=K, dim=-1)
        return vals, ids

    def test_truncated_tree_is_valid(self):
        bs, depth = 3, 15
        vals, ids = self._draft_topk(bs, depth, 200, 32)
        trees, budget, stats = build_adaptive_best_first_trees_batched(
            bonus_tokens=torch.arange(bs), sorted_logprobs=vals, sorted_token_ids=ids,
            max_budget=64, min_budget=1, cost_model=_FakeCost(2e-4, 1e-3, 0.05),
            context_sum=900,
        )
        self.assertEqual(len(trees), bs)
        self.assertEqual(stats["budget"], budget)
        for t in trees:
            self.assertEqual(t.tree_size, budget)
            parents = t.parent_indices.tolist()
            self.assertEqual(parents[0], -1)
            for i, p in enumerate(parents):
                self.assertTrue(-1 <= p < i, f"parent {p} not before child {i}")
            fc = t.first_child.tolist()
            ns = t.next_sibling.tolist()
            for i, p in enumerate(parents):
                if p >= 0:
                    c, seen = fc[p], set()
                    while c != -1 and c not in seen:
                        if c == i:
                            break
                        seen.add(c)
                        c = ns[c]
                    self.assertEqual(c, i, f"child {i} unreachable from parent {p}")

    def test_uncalibrated_build_uses_max_budget(self):
        vals, ids = self._draft_topk(2, 15, 200, 32, seed=5)
        trees, budget, stats = build_adaptive_best_first_trees_batched(
            bonus_tokens=torch.arange(2), sorted_logprobs=vals, sorted_token_ids=ids,
            max_budget=48, min_budget=1, cost_model=None, context_sum=500,
        )
        self.assertEqual(budget, 48)
        self.assertFalse(stats["calibrated"])
        for t in trees:
            self.assertEqual(t.tree_size, 48)

    def test_budget_never_exceeds_reachable(self):
        vals, ids = self._draft_topk(2, 2, 50, 2, seed=9)
        trees, budget, _ = build_adaptive_best_first_trees_batched(
            bonus_tokens=torch.arange(2), sorted_logprobs=vals, sorted_token_ids=ids,
            max_budget=1000, min_budget=1, cost_model=None, context_sum=100,
        )
        for t in trees:
            self.assertEqual(t.tree_size, budget)
            self.assertLessEqual(budget, t.tokens.numel())


class TestCostModel(unittest.TestCase):
    def test_verify_convex_in_n(self):
        cm = DFlashAdaptiveCostModel(_dims(), "NVIDIA H100 80GB HBM3")
        vs = [cm.estimate_verify(8, N, 8000) for N in range(1, 48)]
        deltas = [vs[i + 1] - vs[i] for i in range(len(vs) - 1)]
        for i in range(len(deltas) - 1):
            self.assertLessEqual(deltas[i], deltas[i + 1] + 1e-15)

    def test_weight_amortization_across_batch(self):
        cm = DFlashAdaptiveCostModel(_dims(), "a100")

        def per_token(B):
            return cm.estimate_verify(B, 8, 1000 * B) / (B * 8)

        self.assertLess(per_token(16), per_token(1))
        self.assertLess(per_token(64), per_token(16))

    def test_bucketing_and_unknown_gpu(self):
        self.assertEqual(DFlashAdaptiveCostModel.bucket_of(3), 3)
        self.assertEqual(DFlashAdaptiveCostModel.bucket_of(30), 16)
        self.assertEqual(DFlashAdaptiveCostModel.bucket_of(200), 128)
        peak, bw = resolve_gpu_roofline("Some Mystery Accelerator 9000")
        self.assertGreater(peak, 0)
        self.assertGreater(bw, 0)


class TestRooflineAffineFit(unittest.TestCase):
    def test_fit_recovers_affine_when_both_regimes_present(self):
        # Construct a grid whose analytical branches cross over, so all four
        # parameters are identifiable, then check the fit reproduces the target.
        comp = np.linspace(0.001, 0.02, 16)
        mem = np.linspace(0.012, 0.013, 16)  # dominates at small idx, loses at large
        affine = (1.5, 0.002, 2.0, 0.001)
        ac, bc, am, bm = affine
        y = np.maximum(ac * comp + bc, am * mem + bm)
        got = fit_roofline_affine(comp, mem, y)
        pred = np.maximum(got[0] * comp + got[1], got[2] * mem + got[3])
        # The max() kink at the crossover keeps the least-squares fit from being
        # exact; a small relative residual is expected and fine for N* selection.
        self.assertLess(float(np.max(np.abs(pred - y))), 1e-3 * float(np.max(y)))

    def test_fit_from_samples_sets_affine_and_ready(self):
        cm = DFlashAdaptiveCostModel(_dims(), "a6000")
        affine = (1.7, 0.0008, 2.3, 0.0011)
        summary = cm.fit_from_samples(_affine_samples(cm, (1, 4, 8, 16), affine))
        for B in (1, 4, 8, 16):
            self.assertEqual(summary[cm.bucket_of(B)], "affine")
            self.assertTrue(cm.is_ready(B))
            st = cm._buckets[cm.bucket_of(B)]
            self.assertIsNotNone(st.roofline_affine)
            self.assertAlmostEqual(st.draft_s, 0.0036, places=9)
            self.assertAlmostEqual(st.overhead_s, 0.0044, places=9)
        # A bucket where the memory branch dominates the whole sweep should still
        # predict the (dominant-branch) truth within the swept range.
        cs, ms = cm._analytical_branches(1, 32, 2048)
        expect = max(affine[0] * cs + affine[1], affine[2] * ms + affine[3])
        self.assertAlmostEqual(cm.estimate_verify(1, 32, 2048), expect, delta=expect * 1e-3)

    def test_scalar_fallback_with_too_few_points(self):
        cm = DFlashAdaptiveCostModel(_dims(), "a100")
        # Only three points for the bucket -> below the affine threshold.
        affine = (1.4, 0.0, 2.0, 0.0)
        samples = _affine_samples(cm, (2,), affine, N_grid=(1, 8, 64))
        summary = cm.fit_from_samples(samples)
        self.assertEqual(summary[2], "scalar")
        st = cm._buckets[2]
        self.assertIsNone(st.roofline_affine)
        self.assertIsNotNone(st.verify_scale)
        self.assertTrue(cm.is_ready(2))


class TestCalibrationSerialize(unittest.TestCase):
    def _fitted_model(self, gpu="a100"):
        cm = DFlashAdaptiveCostModel(_dims(), gpu)
        cm.fit_from_samples(_affine_samples(cm, (1, 8), (1.6, 0.001, 2.2, 0.002)))
        return cm

    def test_dump_load_round_trip(self):
        cm = self._fitted_model()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "calib.json")  # nested dir must be created
            cm.dump_calibration(path)
            self.assertTrue(os.path.isfile(path))
            with open(path) as f:
                blob = json.load(f)
            self.assertIn("buckets", blob)
            self.assertIn("1", blob["buckets"])
            self.assertIn("roofline_affine", blob["buckets"]["1"])

            cm2 = DFlashAdaptiveCostModel(_dims(), "a100")
            self.assertFalse(cm2.is_ready(1))
            cm2.load_calibration(path)
            self.assertTrue(cm2.is_ready(1))
            self.assertTrue(cm2.is_ready(8))
            self.assertAlmostEqual(cm2.estimate_verify(1, 12, 500),
                                   cm.estimate_verify(1, 12, 500), places=12)
            self.assertAlmostEqual(cm2.fixed_overhead_s(8),
                                   cm.fixed_overhead_s(8), places=12)

    def test_fitted_budget_survives_round_trip(self):
        cm = self._fitted_model()
        marg = _make_marginals(8, 40, seed=11)
        n_before = select_shared_budget(
            accept_marginals=marg, batch_size=8, context_sum=4000,
            cost_model=cm, min_budget=1, max_budget=40,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calib.json")
            cm.dump_calibration(path)
            cm2 = DFlashAdaptiveCostModel(_dims(), "a100")
            cm2.load_calibration(path)
        n_after = select_shared_budget(
            accept_marginals=marg, batch_size=8, context_sum=4000,
            cost_model=cm2, min_budget=1, max_budget=40,
        )
        self.assertEqual(n_before, n_after)

    def test_hand_written_affine_json_estimate_path(self):
        cm = DFlashAdaptiveCostModel(_dims(), "a100")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "affine.json")
            with open(path, "w") as f:
                json.dump({
                    "schema_version": 2,
                    "gpu_name": "a100",
                    "dims": None,
                    "buckets": {
                        "4": {"draft_s": 0.002, "overhead_s": 0.001,
                              "roofline_affine": [2.0, 0.01, 3.0, 0.02], "count": 5},
                    },
                }, f)
            cm.load_calibration(path)
        self.assertTrue(cm.is_ready(4))
        cs, ms = cm._analytical_branches(4, 10, 3000)
        expect = max(2.0 * cs + 0.01, 3.0 * ms + 0.02)
        self.assertAlmostEqual(cm.estimate_verify(4, 10, 3000), expect, places=12)


class TestRawModeAblation(unittest.TestCase):
    """Uncalibrated 'raw roofline' mode used by the calibration ablation."""

    def test_raw_mode_ready_zero_overhead_raw_verify(self):
        cm = DFlashAdaptiveCostModel(_dims(), "a6000")
        cm.raw_mode = True
        self.assertTrue(cm.is_ready(1))
        self.assertTrue(cm.is_ready(32))
        self.assertEqual(cm.fixed_overhead_s(8), 0.0)
        # estimate_verify is exactly the raw analytical roofline (no scale/affine).
        self.assertAlmostEqual(cm.estimate_verify(8, 16, 8000),
                               cm._analytical_verify_s(8, 16, 8000), places=12)

    def test_raw_mode_is_adaptive_not_nmax_fallback(self):
        marg = _make_marginals(8, 64, seed=2)
        raw = DFlashAdaptiveCostModel(_dims(), "a6000")
        raw.raw_mode = True
        n_raw = select_shared_budget(accept_marginals=marg, batch_size=8,
                                     context_sum=8 * 2048, cost_model=raw,
                                     min_budget=1, max_budget=64)
        self.assertTrue(1 <= n_raw <= 64)
        # The uncalibrated NON-raw model is not ready and must fall back to N_max,
        # which is the degenerate case raw mode is designed to avoid.
        empty = DFlashAdaptiveCostModel(_dims(), "a6000")
        n_fallback = select_shared_budget(accept_marginals=marg, batch_size=8,
                                          context_sum=8 * 2048, cost_model=empty,
                                          min_budget=1, max_budget=64)
        self.assertEqual(n_fallback, 64)


class TestBeamSearchTree(unittest.TestCase):
    """Width-pruned (beam) builder: full-depth, width nodes per level.

    Matches spec-dllm build_width_pruned_tree_from_draft_logits.
    """

    def _per_depth_topk(self, depth=15, vocab=500, k=64, seed=0):
        torch.manual_seed(seed)
        lp, ids = topk_logprobs_per_depth(torch.randn(depth, vocab), k)
        return lp.tolist(), ids.tolist()

    def _assert_valid_tree(self, tree):
        n = tree.tree_size
        par = tree.parent_indices.tolist()
        dep = tree.depths.tolist()
        self.assertEqual(par[0], -1)
        self.assertEqual(dep[0], 0)
        for i in range(1, n):
            self.assertTrue(0 <= par[i] < i, f"parent {par[i]} not before child {i}")
            self.assertEqual(dep[i], dep[par[i]] + 1, "depth must be parent depth + 1")
            self.assertLessEqual(dep[i - 1], dep[i], "admission must be level order")

    def test_full_depth_and_size(self):
        # THE bug fix: beam reaches the full depth, width nodes per level, so the
        # tree size is 1 + width*depth and paths can be up to `depth` long.
        lp, ids = self._per_depth_topk(depth=15)
        for w in (1, 2, 4, 8):
            t = build_beam_search_tree(bonus_token_id=7, sorted_logprobs=lp,
                                       sorted_token_ids=ids, width=w)
            self.assertEqual(t.tree_size, beam_tree_size(w, 15))
            self.assertEqual(max(t.depths.tolist()), 15)  # NOT capped at width
            self._assert_valid_tree(t)

    def test_width_per_level(self):
        lp, ids = self._per_depth_topk(depth=15)
        t = build_beam_search_tree(bonus_token_id=1, sorted_logprobs=lp,
                                   sorted_token_ids=ids, width=4)
        from collections import Counter
        per_level = Counter(t.depths.tolist())
        self.assertEqual(per_level[0], 1)
        for d in range(1, 16):
            self.assertEqual(per_level[d], 4)  # exactly width at every depth

    def test_width_one_is_a_chain(self):
        lp, ids = self._per_depth_topk(depth=15)
        t = build_beam_search_tree(bonus_token_id=1, sorted_logprobs=lp,
                                   sorted_token_ids=ids, width=1)
        self.assertEqual(t.depths.tolist(), list(range(16)))
        self.assertEqual(t.parent_indices.tolist(), [-1] + list(range(15)))

    def test_max_depth_override(self):
        lp, ids = self._per_depth_topk(depth=15)
        t = build_beam_search_tree(bonus_token_id=1, sorted_logprobs=lp,
                                   sorted_token_ids=ids, width=3, max_depth=5)
        self.assertEqual(t.tree_size, beam_tree_size(3, 5))
        self.assertEqual(max(t.depths.tolist()), 5)

    def test_differs_from_best_first(self):
        lp, ids = self._per_depth_topk(depth=15, seed=3)
        beam = build_beam_search_tree(bonus_token_id=9, sorted_logprobs=lp,
                                      sorted_token_ids=ids, width=4)  # 61 nodes
        bf = build_best_first_tree(bonus_token_id=9, sorted_logprobs=lp,
                                   sorted_token_ids=ids, verify_length=beam.tree_size)
        self.assertEqual(beam.tree_size, bf.tree_size)  # matched budget
        self.assertNotEqual(beam.tokens.tolist(), bf.tokens.tolist())

    def test_batched_rectangular(self):
        torch.manual_seed(4)
        bs, depth, vocab, k = 3, 15, 400, 64
        lp, ids = topk_logprobs_per_depth(torch.randn(bs, depth, vocab), k)
        trees = build_beam_search_trees_batched(
            bonus_tokens=torch.arange(bs), sorted_logprobs=lp, sorted_token_ids=ids,
            width=2,
        )
        self.assertEqual(len(trees), bs)
        sizes = {t.tree_size for t in trees}
        self.assertEqual(sizes, {beam_tree_size(2, 15)})  # all identical -> rectangular
        for t in trees:
            self._assert_valid_tree(t)


if __name__ == "__main__":
    unittest.main()
