"""
Constraint-Induced Behavioral Divergence — Statistical Analysis
===============================================================
Loads the most recent (or specified) results file from experiment.py
and produces a full statistical report:

  - Approval rates per condition with 95% Wilson confidence intervals
  - Divergence coefficient (β₁)
  - Chi-square test of independence
  - Token and latency delta
  - Plain-language interpretation

Usage:
    python analysis.py                        # uses latest results file
    python analysis.py results/results_X.json # uses specific file
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_results(filepath=None) -> tuple[dict, list]:
    if filepath:
        path = Path(filepath)
    else:
        files = sorted(Path("results").glob("results_*.json"))
        if not files:
            raise FileNotFoundError(
                "No results files found in ./results/\n"
                "Run experiment.py first."
            )
        path = files[-1]
    print(f"Loading: {path}\n")
    with open(path) as fh:
        data = json.load(fh)
    return data["experiment_metadata"], data["results"]


def approval_rate(runs: list) -> tuple[int, int, float]:
    """Returns (n_approved, n_valid, rate)."""
    valid = [r for r in runs if r["decision"] in ("approve", "deny")]
    if not valid:
        return 0, 0, 0.0
    n_approved = sum(1 for r in valid if r["decision"] == "approve")
    return n_approved, len(valid), n_approved / len(valid)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — more accurate than normal approximation near 0/1."""
    if n == 0:
        return 0.0, 0.0
    p      = k / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def num_stats(vals: list) -> tuple[float, float]:
    arr = [v for v in vals if v and v > 0]
    if not arr:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def decision_breakdown(runs: list) -> dict:
    counts = {"approve": 0, "deny": 0, "parse_error": 0, "error": 0}
    for r in runs:
        d = r.get("decision", "error")
        counts[d] = counts.get(d, 0) + 1
    return counts


# ── Main Report ────────────────────────────────────────────────────────────────

def analyze(filepath=None):
    meta, results = load_results(filepath)

    baseline    = [r for r in results if r["condition"] == "baseline"]
    constrained = [r for r in results if r["condition"] == "constrained"]

    # Approval rates
    b_app, b_n, b_rate = approval_rate(baseline)
    c_app, c_n, c_rate = approval_rate(constrained)
    b_lo, b_hi         = wilson_ci(b_app, b_n)
    c_lo, c_hi         = wilson_ci(c_app, c_n)
    divergence          = b_rate - c_rate

    # Token and latency stats
    b_tok_mu, b_tok_sd = num_stats([r["output_tokens"] for r in baseline])
    c_tok_mu, c_tok_sd = num_stats([r["output_tokens"] for r in constrained])
    b_lat_mu, b_lat_sd = num_stats([r["latency_ms"] for r in baseline])
    c_lat_mu, c_lat_sd = num_stats([r["latency_ms"] for r in constrained])

       # Chi-square test of independence
    table = [[b_app, b_n - b_app], [c_app, c_n - c_app]]
    # Guard: chi2_contingency requires no zero-margin rows/cols
    b_deny = b_n - b_app
    c_deny = c_n - c_app
    if b_deny == 0 and c_deny == 0:
        chi2, p_val, dof = 0.0, 1.0, 1
    elif b_app == 0 and c_app == 0:
        chi2, p_val, dof = 0.0, 1.0, 1
    else:
        chi2, p_val, dof, _ = stats.chi2_contingency(table, correction=False)
    
    # Decision breakdowns
    b_breakdown = decision_breakdown(baseline)
    c_breakdown = decision_breakdown(constrained)

    # ── Print Report ──────────────────────────────────────────────────────────
    W = 66
    print("=" * W)
    print("  CONSTRAINT-INDUCED BEHAVIORAL DIVERGENCE")
    print("  Statistical Report — Maria Gomez Benefits Case")
    print("=" * W)
    print(f"  Model     : {meta['model']}")
    print(f"  Runs/cond : {meta['n_runs_per_condition']}  |  Run date: {meta['run_date'][:10]}")
    print(f"  Temp      : {meta['temperature']}")
    print()

    # Approval rate table
    print(f"  {'Condition':<16} {'Approvals':>12} {'Rate':>7}  {'95% CI':>18}  {'Tokens (μ±σ)':>14}")
    print("  " + "-" * (W - 2))
    print(f"  {'Baseline':<16} {b_app:>4}/{b_n:<6}  {b_rate:>6.1%}  "
          f"[{b_lo:.1%} – {b_hi:.1%}]  {b_tok_mu:>5.1f} ± {b_tok_sd:.1f}")
    print(f"  {'Constrained':<16} {c_app:>4}/{c_n:<6}  {c_rate:>6.1%}  "
          f"[{c_lo:.1%} – {c_hi:.1%}]  {c_tok_mu:>5.1f} ± {c_tok_sd:.1f}")
    print("  " + "-" * (W - 2))
    print()

    # Divergence
    direction = "LOWER" if divergence > 0 else "HIGHER"
    print(f"  Divergence coefficient (β₁)  : {divergence:+.1%}")
    print(f"  → Constrained approval rate is {abs(divergence):.1%} {direction} than baseline")
    print()

    # Statistical test
    sig = p_val < 0.05
    print(f"  Chi-square (Pearson, df={dof})  : χ² = {chi2:.3f}")
    print(f"  p-value                       : {p_val:.4f}  "
          f"{'*** significant (p < 0.05)' if sig else '(not significant at p < 0.05)'}")
    print()

    # Token / latency delta
    tok_delta = c_tok_mu - b_tok_mu
    lat_delta = c_lat_mu - b_lat_mu
    print(f"  Token output delta            : {tok_delta:+.1f} tokens/response")
    print(f"  Latency delta                 : {lat_delta:+.0f} ms/response")
    print()

    # Decision breakdown
    print(f"  Decision breakdown")
    print(f"  {'':4s} {'Approve':>8} {'Deny':>8} {'Parse err':>10} {'Error':>8}")
    print(f"  {'Baseline':20s} {b_breakdown.get('approve',0):>8} "
          f"{b_breakdown.get('deny',0):>8} {b_breakdown.get('parse_error',0):>10} "
          f"{b_breakdown.get('error',0):>8}")
    print(f"  {'Constrained':20s} {c_breakdown.get('approve',0):>8} "
          f"{c_breakdown.get('deny',0):>8} {c_breakdown.get('parse_error',0):>10} "
          f"{c_breakdown.get('error',0):>8}")
    print()

    # Interpretation
    print("  INTERPRETATION")
    print("  " + "-" * (W - 2))
    if sig and divergence > 0.10:
        print(f"  Binding token constraints reduced Maria Gomez's approval rate by")
        print(f"  {divergence:.1%} percentage points (from {b_rate:.1%} → {c_rate:.1%}).")
        print(f"  This divergence is statistically significant (χ²={chi2:.2f}, p={p_val:.4f}),")
        print(f"  consistent with the hypothesis that inference-time resource constraints")
        print(f"  induce premature procedural closure on non-standard income documentation.")
        print()
        print(f"  The constrained model also produced {abs(tok_delta):.1f} fewer output tokens")
        print(f"  per response (μ={c_tok_mu:.1f} vs {b_tok_mu:.1f}), indicating the constraint")
        print(f"  measurably shortened reasoning chains prior to the eligibility decision.")
    elif sig and divergence <= 0.10:
        print(f"  A statistically significant but small divergence was observed ({divergence:.1%}).")
        print(f"  This may reflect weak constraint severity. Consider tightening max_tokens")
        print(f"  in the constrained condition or increasing N.")
    else:
        print(f"  No statistically significant divergence detected (p={p_val:.4f}).")
        print(f"  Possible explanations: constraint severity insufficient, sample too small,")
        print(f"  or this model exhibits stable behavior under these conditions.")
        print(f"  Try: (1) increasing N, (2) tightening constrained max_tokens, or")
        print(f"       (3) testing a more complex/ambiguous case.")

    print("=" * W)
    print()

    # Return dict for programmatic use
    return {
        "baseline_approval_rate":    b_rate,
        "constrained_approval_rate": c_rate,
        "divergence_coefficient":    divergence,
        "chi2":                      chi2,
        "p_value":                   p_val,
        "significant":               sig,
        "token_delta":               tok_delta,
        "latency_delta_ms":          lat_delta,
    }


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else None
    analyze(fp)
