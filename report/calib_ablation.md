# Calibration ablation: how to run it

**Goal.** Isolate what the startup calibration buys by running the *same* adaptive
tree controller with and without it. Only the cost model changes.

## Conditions

| Condition | How | Cost model used for N* |
|---|---|---|
| **Calibrated** (full method) | `CALIB_MODE=fit` | per-batch affine roofline, fit on the real kernels at startup |
| **Uncalibrated (raw roofline)** | `CALIB_MODE=raw` | raw analytical roofline, zero fixed overhead (no sweep) |
| Fixed budget N (reference) | `run_dflash_best_first_tree.sh`, `MAX_TOKENS=N` | non-adaptive, constant tree |
| Chain (reference) | `run_dflash.sh` | no tree |

The first two are the ablation; the last two are optional context for the same plot.

## Run (one GPU + port per condition, then benchmark both identically)

```bash
cd /home/namcao/sglang
# Calibrated
CUDA_VISIBLE_DEVICES=6 CALIB_MODE=fit PORT=30006 ./run_dflash_calib_ablation.sh &
# Uncalibrated (raw roofline)
CUDA_VISIBLE_DEVICES=7 CALIB_MODE=raw PORT=30007 ./run_dflash_calib_ablation.sh &

# Bench each server with the identical greedy client, e.g. (in ~/dflash):
#   ./scripts/run_bastion.sh 1 4 8 16 32 64      # point it at each PORT in turn
```

Confirm the active condition in each server log:
- fit: `calibration=FIT ... calibration added <ms>ms ... buckets={1:'affine',...}`
- raw: `calibration=RAW (uncalibrated). Skipping the startup sweep ...`

## Controls

- **Hold fixed:** model, draft model, GPU, attention backends, `N_max` (`MAX_TOKENS`),
  `min-tokens`, sampling (greedy, temperature 0), client, and dataset.
- **Vary:** only `CALIB_MODE`. Sweep concurrency (1, 4, 8, 16, 32, 64) as the x-axis.

Because everything but the cost model is identical, any throughput / acceptance-length
gap between fit and raw is attributable to the calibration alone.

## What to measure

For each condition and concurrency, report:
1. **Throughput (tok/s)** and **acceptance length τ** (the main plot).
2. **Chosen budget N\*** vs concurrency, from the per-cycle telemetry line
   `DFLASH adaptive cycle #k: N*=.../64 ...`. This shows *where* the two cost models
   place the budget, which is the mechanism behind any throughput/τ difference.

## Expected result and how it supports W2

- **Both run and stay adaptive.** Raw mode selects a real N\* that shrinks with batch
  (it is not the degenerate `N* = N_max` fallback that an entirely absent cost model
  produces). This alone shows the controller does not *require* calibration.
- **Calibrated should give the better throughput/τ trade-off**, because the fit
  corrects the roofline magnitude and per-branch shape, so N\* is placed more
  accurately; raw should still land well above a fixed large tree.
- **Interpretation:** calibration is a second-order refinement, not a prerequisite.
  This is the concrete evidence behind the W2 claim that the method also runs from the
  analytical roofline alone, and it quantifies how much the offline step actually buys.

## Note on scope (optional stricter variant)

`raw` drops *both* the verify-magnitude fit and the measured draft/overhead (it uses
zero overhead), so it is the clean "no calibration at all" condition. If you want to
isolate *only* the verify-magnitude fit while holding overhead fixed, that needs a
third mode (fit the sweep's draft/overhead but keep the raw verify roofline); ask and
it is a small addition.
