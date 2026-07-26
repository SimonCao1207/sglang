"""Batched latency cost model for DFlash adaptive tree drafting.

The adaptive best-first controller (`dflash_tree_builder.select_shared_budget`)
picks one shared tree budget ``N`` for a whole speculative batch by maximizing

    S_B(N) = (mean_b A_b(N)) * L_AR(B) / C_B(N)

where ``C_B(N) = D_B + O_B + V_B(N)`` is the per-cycle latency: draft forward
``D_B``, non-verify overhead ``O_B`` (tree build + KV realignment), and the
batched target-verify forward ``V_B(N)``. This module supplies ``V_B``,
``D_B``, ``O_B`` and an ``L_AR(B)`` proxy.

Design (ported/generalized from BASTION `bastion/cost_model.py`, which only
models batch size 1):

* ``V_B(N)`` is an analytical **roofline** — ``max(compute_bound,
  memory_bound)`` — generalized to a batch of ``B`` trees of shared size ``N``
  over per-request contexts. Model-weight loads are counted **once** (amortized
  across the batch), which is exactly why ``C_B(N) != B * C_1(N)`` and why the
  optimal budget shrinks as concurrency grows.
* Magnitude is calibrated by a **BASTION-style least-squares fit** of an affine
  roofline ``max(alpha_c * compute + beta_c, alpha_m * memory + beta_m)`` to
  real verify latencies measured on the actual serving kernels during a
  one-shot startup sweep (`DFlashWorker._run_startup_calibration_sweep`). The
  fit is done **per batch-size bucket**, so the calibration captures how
  batching amortizes weight loads. ``D_B`` / ``O_B`` come from the same sweep.
  After the sweep the model is fixed, so ``N*`` is a deterministic function of
  ``(batch, context, N_max)`` with zero per-step measurement overhead.

All latencies are in seconds. FLOP / byte formulas follow the dense-transformer
accounting in BASTION's cost model (GQA attention + gated FFN + LM head).
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bump when the on-disk calibration schema changes incompatibly.
_CALIBRATION_SCHEMA_VERSION = 2

# A bucket needs at least this many distinct sweep points to fit the four affine
# parameters; below it we fall back to a single scalar magnitude.
_MIN_POINTS_FOR_AFFINE = 5


# ---------------------------------------------------------------------------
# GPU roofline specs (peak bf16 FLOP/s, HBM bandwidth in bytes/s).
# Ported from bastion/cost_model.py; aliases are matched case/punct-insensitively
# against ``torch.cuda.get_device_name``.
# ---------------------------------------------------------------------------
_GPU_SPECS = {
    "a5000": (111e12, 768e9),
    "a6000": (155e12, 768e9),
    "a100": (312e12, 1935e9),
    "h100": (989e12, 3350e9),
    "h200": (989e12, 4800e9),
    "b200": (2250e12, 8000e9),
    "b6000": (504e12, 1792e9),
}

_GPU_ALIASES = {
    "rtxa5000": "a5000",
    "nvidiartxa5000": "a5000",
    "rtxa6000": "a6000",
    "nvidiartxa6000": "a6000",
    "nvidiaa100": "a100",
    "a10080gb": "a100",
    "nvidiaa10080gb": "a100",
    "a10080gbpcie": "a100",
    "nvidiaa10080gbpcie": "a100",
    "h10080gb": "h100",
    "h10080gbsxm": "h100",
    "nvidiah100": "h100",
    "nvidiah10080gb": "h100",
    "nvidiah10080gbsxm": "h100",
    "nvidiah10080gbhbm3": "h100",
    "h200141gb": "h200",
    "nvidiah200": "h200",
    "nvidiah200141gb": "h200",
    "nvidiah200141gbsxm": "h200",
    "nvidiah200141gbhbm3e": "h200",
    "nvidiab200": "b200",
    "b200sxm": "b200",
    "rtxpro6000": "b6000",
    "nvidiartxpro6000": "b6000",
}

# Fallback roofline when the device is unrecognized. Never crash the server for
# a missing spec — the startup fit corrects the magnitude against real kernels
# regardless of which roofline shape we start from.
_DEFAULT_GPU = "a100"

_BYTES_PER_PARAM = 2  # bf16


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_gpu_roofline(gpu_name: str) -> tuple[float, float]:
    """Return ``(peak_flops_per_s, memory_bandwidth_bytes_per_s)`` for a GPU.

    Falls back to a default spec (with a one-time warning) for unknown devices.
    """
    norm = _normalize(gpu_name)
    key = norm if norm in _GPU_SPECS else _GPU_ALIASES.get(norm)
    if key is None:
        # Substring match handles names like "NVIDIA H100 80GB HBM3 MIG ...".
        for cand in _GPU_SPECS:
            if cand in norm:
                key = cand
                break
    if key is None:
        logger.warning(
            "DFLASH adaptive cost model: unknown GPU '%s'; falling back to '%s' "
            "roofline. The startup calibration fit will correct the magnitude.",
            gpu_name,
            _DEFAULT_GPU,
        )
        key = _DEFAULT_GPU
    return _GPU_SPECS[key]


@dataclass(frozen=True)
class ModelDims:
    """Dense-transformer dimensions needed for the roofline."""

    num_layers: int          # L
    hidden_size: int         # h
    num_attention_heads: int  # n_q
    num_kv_heads: int        # n_kv
    head_dim: int            # d
    intermediate_size: int   # h_ffn (gated FFN inner width)
    vocab_size: int          # V

    @classmethod
    def from_model_config(cls, model_config) -> "ModelDims":
        hf = model_config.hf_text_config
        intermediate_size = int(
            getattr(hf, "intermediate_size", None)
            or getattr(hf, "ffn_hidden_size", None)
            or 4 * int(model_config.hidden_size)
        )
        return cls(
            num_layers=int(model_config.num_hidden_layers),
            hidden_size=int(model_config.hidden_size),
            num_attention_heads=int(model_config.num_attention_heads),
            num_kv_heads=int(model_config.num_key_value_heads),
            head_dim=int(model_config.head_dim),
            intermediate_size=intermediate_size,
            vocab_size=int(model_config.vocab_size),
        )


def _batched_verify_flops(dims: ModelDims, batch: int, tree_size: int, context_sum: int) -> float:
    """Total FLOPs of one batched verify forward.

    ``batch`` trees each of ``tree_size`` nodes (query tokens), request ``b``
    attending to its own ``c_b`` context tokens; ``context_sum = sum_b c_b``.
    Non-attention terms scale with the total query tokens ``S = batch*tree_size``;
    attention QK/SV scale with ``sum_b tree_size*(c_b + tree_size)``.
    """
    h = dims.hidden_size
    h_q = dims.num_attention_heads * dims.head_dim
    h_kv = dims.num_kv_heads * dims.head_dim
    h_ffn = dims.intermediate_size
    L = dims.num_layers
    V = dims.vocab_size

    S = batch * tree_size
    # attn_span = sum_b N*(c_b + N) = N*context_sum + batch*N^2
    attn_span = tree_size * context_sum + batch * tree_size * tree_size

    q_proj = 2 * S * h * h_q
    kv_proj = 4 * S * h * h_kv
    qk_matmul = 2 * attn_span * h_q
    sv_matmul = 2 * attn_span * h_q
    o_proj = 2 * S * h_q * h
    attn = q_proj + kv_proj + qk_matmul + sv_matmul + o_proj

    gate_up = 4 * S * h * h_ffn
    down = 2 * S * h_ffn * h
    ffn = gate_up + down

    head = 2 * S * h * V
    return float(L * (attn + ffn) + head)


def _batched_verify_bytes(dims: ModelDims, batch: int, tree_size: int, context_sum: int) -> float:
    """Total HBM traffic (bytes) of one batched verify forward.

    Model weights are read **once** regardless of batch/tree size — the
    amortization that makes batched verify cheaper per token.
    """
    h = dims.hidden_size
    h_q = dims.num_attention_heads * dims.head_dim
    h_kv = dims.num_kv_heads * dims.head_dim
    h_ffn = dims.intermediate_size
    n_q = dims.num_attention_heads
    L = dims.num_layers
    V = dims.vocab_size
    bp = _BYTES_PER_PARAM

    S = batch * tree_size
    attn_span = tree_size * context_sum + batch * tree_size * tree_size

    embed_param = V * h
    attn_param = 2 * h * h_q + 2 * h * h_kv
    ffn_param = 3 * h * h_ffn
    output_param = h * V
    footprint_param = bp * (embed_param + L * (attn_param + ffn_param) + output_param)

    kv_read = 2 * context_sum * h_kv
    kv_write = 2 * S * h_kv
    footprint_kv = bp * L * (kv_read + kv_write)

    qkv_proj_io = S * h + (S * h_q + 2 * S * h_kv)
    attn_score_io = S * h_q + n_q * attn_span
    attn_value_io = n_q * attn_span + S * h_q
    o_proj_io = S * h_q + S * h
    attn_act = qkv_proj_io + attn_score_io + attn_value_io + o_proj_io

    gate_up_io = S * h + 2 * S * h_ffn
    down_io = 2 * S * h_ffn + S * h
    ffn_act = gate_up_io + down_io

    head_io = S * h + S * V
    footprint_act = bp * (L * (attn_act + ffn_act) + head_io)

    return float(footprint_param + footprint_kv + footprint_act)


def fit_roofline_affine(
    compute_s,
    memory_s,
    measured_s,
) -> Tuple[float, float, float, float]:
    """Least-squares fit of the calibrated roofline to measured verify latencies.

    Fits ``measured ~= max(alpha_c * compute_s + beta_c, alpha_m * memory_s +
    beta_m)`` with all four parameters constrained non-negative, mirroring
    BASTION's ``fit_roofline_calibration`` (``bastion/cost_model.py``) but meant
    to be called **per batch-size bucket**. Returns the tuple in the order
    :attr:`_BucketState.roofline_affine` expects: ``(alpha_c, beta_c, alpha_m,
    beta_m)``.

    Requires numpy + scipy, imported lazily so the serving path (which only
    evaluates an already-fit model) does not depend on scipy.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    compute_s = np.asarray(compute_s, dtype=np.float64)
    memory_s = np.asarray(memory_s, dtype=np.float64)
    measured_s = np.asarray(measured_s, dtype=np.float64)

    # curve_fit passes an index vector so the model can gather the paired
    # analytical branches: the two regressors are not one shared x-axis.
    def _model(indices, alpha_c, beta_c, alpha_m, beta_m):
        idx = np.round(indices).astype(int)
        return np.maximum(
            alpha_c * compute_s[idx] + beta_c,
            alpha_m * memory_s[idx] + beta_m,
        )

    indices = np.arange(measured_s.shape[0], dtype=np.float64)
    popt, _ = curve_fit(
        _model,
        indices,
        measured_s,
        p0=[1.0, 0.0, 1.0, 0.0],
        bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]),
        maxfev=5000,
    )
    alpha_c, beta_c, alpha_m, beta_m = (float(x) for x in popt)
    return alpha_c, beta_c, alpha_m, beta_m


@dataclass
class _BucketState:
    """Calibration for one batch-size bucket. ``None`` fields until fitted.

    Magnitude of ``V_B(N)`` is corrected by an affine roofline
    ``roofline_affine = (alpha_c, beta_c, alpha_m, beta_m)`` applied per branch
    (BASTION-style). A scalar ``verify_scale`` is a fallback used only when a
    bucket has too few sweep points to fit the four affine parameters; when both
    are present the affine form wins.
    """

    roofline_affine: Optional[Tuple[float, float, float, float]] = None
    verify_scale: Optional[float] = None   # fallback: median measured / analytical
    draft_s: Optional[float] = None        # D_B
    overhead_s: Optional[float] = None     # O_B (build + realign + select)
    count: int = 0


class DFlashAdaptiveCostModel:
    """Batched cost model for the shared-budget controller.

    The analytical roofline provides the (convex) shape of ``V_B(N)``; a
    one-shot startup sweep supplies the magnitude via :meth:`fit_from_samples`
    (per batch-size affine fit) and the non-verify constants ``D_B`` / ``O_B``.
    Once fit the model is fixed — evaluation is pure and deterministic.
    """

    def __init__(self, dims: ModelDims, gpu_name: str) -> None:
        self.dims = dims
        self.gpu_name = gpu_name
        self.peak_flops, self.mem_bandwidth = resolve_gpu_roofline(gpu_name)
        self._buckets: Dict[int, _BucketState] = {}

    @classmethod
    def from_model_config(cls, model_config, gpu_name: str) -> "DFlashAdaptiveCostModel":
        return cls(ModelDims.from_model_config(model_config), gpu_name)

    # -- roofline ----------------------------------------------------------
    def _analytical_branches(
        self, batch: int, tree_size: int, context_sum: int
    ) -> Tuple[float, float]:
        """Uncalibrated (compute_bound_s, memory_bound_s) roofline branches."""
        flops = _batched_verify_flops(self.dims, batch, tree_size, context_sum)
        nbytes = _batched_verify_bytes(self.dims, batch, tree_size, context_sum)
        return flops / self.peak_flops, nbytes / self.mem_bandwidth

    def _analytical_verify_s(self, batch: int, tree_size: int, context_sum: int) -> float:
        compute_s, memory_s = self._analytical_branches(batch, tree_size, context_sum)
        return max(compute_s, memory_s)

    @staticmethod
    def bucket_of(batch: int) -> int:
        """Group batch sizes so calibration generalizes across nearby sizes."""
        if batch <= 8:
            return int(batch)
        return 1 << (int(batch).bit_length() - 1)  # largest power of two <= batch

    # -- estimates used by the controller ----------------------------------
    def estimate_verify(self, batch: int, tree_size: int, context_sum: int) -> float:
        """Calibrated ``V_B(N)`` in seconds.

        Applies, in priority order: the affine roofline fit if present, else the
        scalar ``verify_scale`` fallback, else the raw analytical roofline.
        """
        st = self._buckets.get(self.bucket_of(batch))
        if st is not None and st.roofline_affine is not None:
            compute_s, memory_s = self._analytical_branches(batch, tree_size, context_sum)
            alpha_c, beta_c, alpha_m, beta_m = st.roofline_affine
            return max(alpha_c * compute_s + beta_c, alpha_m * memory_s + beta_m)
        raw = self._analytical_verify_s(batch, tree_size, context_sum)
        scale = st.verify_scale if (st and st.verify_scale is not None) else 1.0
        return raw * scale

    def verify_next_delta(self, batch: int, tree_size: int, context_sum: int) -> float:
        """Marginal verify cost of growing the shared budget by one node."""
        return self.estimate_verify(batch, tree_size + 1, context_sum) - self.estimate_verify(
            batch, tree_size, context_sum
        )

    def fixed_overhead_s(self, batch: int) -> Optional[float]:
        """``D_B + O_B`` from the fitted sweep, or ``None`` if not yet fitted."""
        st = self._buckets.get(self.bucket_of(batch))
        if st is None or st.draft_s is None or st.overhead_s is None:
            return None
        return st.draft_s + st.overhead_s

    def ar_latency_s(self, batch: int, context_sum: int) -> float:
        """Proxy for one autoregressive decode step ``L_AR(B)``.

        A verify forward at ``N = 1`` processes exactly one query token per
        request — i.e. an ordinary decode step — so its calibrated cost is a
        faithful AR-latency proxy for the ``S_B`` speedup readout.
        """
        return self.estimate_verify(batch, 1, context_sum)

    def is_ready(self, batch: int) -> bool:
        """True once the bucket has the constants the controller needs."""
        return self.fixed_overhead_s(batch) is not None

    # -- startup calibration fit -------------------------------------------
    def fit_from_samples(
        self,
        samples: Iterable[dict],
        *,
        min_points_for_affine: int = _MIN_POINTS_FOR_AFFINE,
    ) -> Dict[int, str]:
        """Fit per-batch-size calibration from measured startup-sweep samples.

        ``samples`` is an iterable of dicts with keys ``batch``, ``tree_size``,
        ``context_sum``, ``verify_s`` and (optionally) ``draft_s`` /
        ``overhead_s``. For each batch-size bucket this fits a BASTION-style
        affine roofline (:func:`fit_roofline_affine`) when the bucket has enough
        distinct points, else falls back to a scalar ``verify_scale`` (median
        measured / analytical). ``draft_s`` / ``overhead_s`` are aggregated by
        median so the controller has the fixed constants it needs. Returns a
        per-bucket summary ``{bucket: "affine"|"scalar"}`` for logging.
        """
        import numpy as np

        grouped: Dict[int, List[dict]] = defaultdict(list)
        for s in samples:
            grouped[self.bucket_of(int(s["batch"]))].append(s)

        summary: Dict[int, str] = {}
        for bucket, pts in sorted(grouped.items()):
            compute_s: List[float] = []
            memory_s: List[float] = []
            measured: List[float] = []
            drafts: List[float] = []
            overs: List[float] = []
            for p in pts:
                verify_s = float(p["verify_s"])
                if verify_s <= 0.0:
                    continue
                c_s, m_s = self._analytical_branches(
                    int(p["batch"]), int(p["tree_size"]), int(p["context_sum"])
                )
                compute_s.append(c_s)
                memory_s.append(m_s)
                measured.append(verify_s)
                if p.get("draft_s") is not None:
                    drafts.append(float(p["draft_s"]))
                if p.get("overhead_s") is not None:
                    overs.append(float(p["overhead_s"]))
            if not measured:
                continue

            st = self._buckets.get(bucket) or _BucketState()
            n_distinct = len(
                {(round(c, 15), round(m, 15)) for c, m in zip(compute_s, memory_s)}
            )
            affine = None
            if n_distinct >= min_points_for_affine:
                try:
                    affine = fit_roofline_affine(
                        np.asarray(compute_s),
                        np.asarray(memory_s),
                        np.asarray(measured),
                    )
                except Exception as e:  # pragma: no cover - scipy numerics
                    logger.warning(
                        "DFLASH affine fit failed for bucket %s (%d pts): %s; "
                        "falling back to scalar verify_scale.",
                        bucket,
                        len(measured),
                        e,
                    )
            if affine is not None:
                st.roofline_affine = affine
                st.verify_scale = None
                summary[bucket] = "affine"
            else:
                raws = [max(c, m) for c, m in zip(compute_s, memory_s)]
                scales = [v / r for v, r in zip(measured, raws) if r > 0.0]
                st.roofline_affine = None
                st.verify_scale = float(np.median(scales)) if scales else 1.0
                summary[bucket] = "scalar"
            if drafts:
                st.draft_s = float(np.median(drafts))
            if overs:
                st.overhead_s = float(np.median(overs))
            st.count = len(pts)
            self._buckets[bucket] = st
        return summary

    # -- serialization (optional dump for inspection / reuse) --------------
    def dump_calibration(self, path: str) -> None:
        """Write the current per-bucket calibration to `path` (atomic).

        Only buckets with the constants the controller needs are written. The
        GPU roofline and model dims are embedded so a later load can warn on a
        mismatch. Written to a temp file and renamed so a hard kill mid-write
        cannot corrupt the file.
        """
        buckets = {
            str(b): {
                "verify_scale": st.verify_scale,
                "draft_s": st.draft_s,
                "overhead_s": st.overhead_s,
                "count": st.count,
                **(
                    {"roofline_affine": list(st.roofline_affine)}
                    if st.roofline_affine is not None
                    else {}
                ),
            }
            for b, st in sorted(self._buckets.items())
            if st.count > 0 and st.draft_s is not None and st.overhead_s is not None
        }
        if not buckets:
            return
        data = {
            "schema_version": _CALIBRATION_SCHEMA_VERSION,
            "gpu_name": self.gpu_name,
            "gpu_peak_flops": self.peak_flops,
            "gpu_mem_bandwidth": self.mem_bandwidth,
            "dims": asdict(self.dims),
            "buckets": buckets,
        }
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def load_calibration(self, path: str) -> None:
        """Load per-bucket calibration from `path` into this model.

        Warns — but does not fail — if the calibration's model dims or GPU
        differ from this run's. Used by tests to round-trip a fit; the serving
        path always fits fresh at startup.
        """
        with open(os.fspath(path)) as f:
            data = json.load(f)

        ver = int(data.get("schema_version", 0))
        if ver != _CALIBRATION_SCHEMA_VERSION:
            logger.warning(
                "DFLASH calibration %s has schema_version=%s (expected %s); "
                "attempting to load anyway.",
                path,
                ver,
                _CALIBRATION_SCHEMA_VERSION,
            )
        loaded_dims = data.get("dims")
        if loaded_dims is not None and loaded_dims != asdict(self.dims):
            logger.warning(
                "DFLASH calibration %s was fit for different model dims (%s) than "
                "this run (%s). Verify-cost estimates may be off.",
                path,
                loaded_dims,
                asdict(self.dims),
            )
        if data.get("gpu_name") not in (None, self.gpu_name):
            logger.warning(
                "DFLASH calibration %s was recorded on GPU '%s' but this run is "
                "'%s'. Loading anyway.",
                path,
                data.get("gpu_name"),
                self.gpu_name,
            )

        for b_str, rec in data.get("buckets", {}).items():
            affine = rec.get("roofline_affine")
            self._buckets[int(b_str)] = _BucketState(
                roofline_affine=(
                    tuple(float(x) for x in affine) if affine is not None else None
                ),
                verify_scale=rec.get("verify_scale"),
                draft_s=rec.get("draft_s"),
                overhead_s=rec.get("overhead_s"),
                count=max(1, int(rec.get("count", 1))),
            )
        logger.info(
            "DFLASH loaded calibration from %s (%d buckets).",
            path,
            len(data.get("buckets", {})),
        )


__all__ = [
    "ModelDims",
    "DFlashAdaptiveCostModel",
    "fit_roofline_affine",
    "resolve_gpu_roofline",
]
