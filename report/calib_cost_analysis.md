# Rebuttal to W2: calibration cost and recalibration conditions

**Reviewer.** *"...the default latency estimator still relies on offline calibration
for each target-model/GPU pair. Further details on the calibration cost and the
conditions under which recalibration are required."*

We concede "no per-setting tuning" is too strong and will state **"no per-task or
per-concurrency tuning"**: the estimator is calibrated once per (model, GPU), then
reused across every task and concurrency level we evaluate.

**Calibration cost analysis.** The calibration is a one-time startup step, not a
per-request one. It measured 70 s end to end (Qwen3-4B, one A6000), alongside the
weight loading and graph capture the server already performs at boot. The model is
frozen after the fit, so it adds no per-token overhead during serving. The procedure
is automatic: it requires no labeled data, no search, and no manual configuration.

| Calibration (Qwen3-4B, one A6000) | Cost |
|---|---|
| Sweep | 70.27 s (42 grid points, 126 cycles, ~558 ms/cycle) |
| Curve fit | 0.02 s (18.7 ms) |
| **Total added to startup** | **70.29 s** (one-time) |

**Conditions under which recalibration is required.** Only when a static property of
the deployment changes; otherwise it is reused unchanged.

| Recalibrate (deployment constants) | Reuse as-is |
|---|---|
| GPU, target model, draft model / block size, attention backend, precision | task, dataset, prompt distribution, context length, concurrency / batch size, budget cap N_max |

<!-- **What it buys.** It removes a harder tuning problem: a fixed tree budget must be
re-chosen per load, since none is competitive across concurrency.

| Math500, Qwen3-4B | conc. 1 (tok/s / τ) | conc. 64 (tok/s / τ) |
|---|---|---|
| Fixed small tree (16) | 376 / 8.61 | 2328 / 8.81 |
| Fixed large tree (64) | 361 / 9.87 | 1135 / 9.91 |
| **BASTION (adaptive)** | 360 / **9.41** | **2374** / 7.46 |

From one calibration, BASTION tracks the best fixed budget at each end: τ≈9.4 at low
load (**+0.8** over the throughput-tuned small tree) and 2374 tok/s at high load
(**2.1×** the large tree, which collapses under concurrency; HumanEval identical, 2188
vs 1036). Matching this with a fixed budget needs per-concurrency retuning, exactly
what the calibration avoids.

**Limitation.** This offline step is a genuine per-(model, GPU) dependency; without
it, budget selection falls back to a coarser analytical roofline. We will add this
analysis to the appendix. -->
