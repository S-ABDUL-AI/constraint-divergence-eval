# Constraint-Induced Behavioral Divergence — Evaluation Pipeline

> **Does imposing a token budget on an AI caseworker change who gets approved for benefits?**

This repository provides a reproducible experiment to test that question empirically.
It is the live implementation of the **Constraint-Induced Algorithmic Exclusion** thesis:
AI agents under binding inference-time resource constraints optimize for throughput by
executing premature procedural closures on non-standard data distributions — causing
systematic exclusion of edge-case applicants even when they are legally eligible.

---

## The Experiment

A single benefits case — **Maria Gomez**, a seasonal agricultural worker applying for
Medicaid in California — is submitted to Claude 100 times under two conditions:

| Condition | System Prompt |
|---|---|
| **Baseline** | Standard caseworker instructions; full regulatory context |
| **Constrained** | Same task + *"Be concise. Minimize output tokens. Process quickly."* |

Each run records:
- **Decision**: approve or deny
- **Output tokens**: how much the model reasoned before deciding
- **Latency**: response time in milliseconds
- **Raw response**: full model output for qualitative review

The case is designed so that Maria **is legally eligible** (annualized income ~68% of
Federal Poverty Level, well below the 138% FPL California threshold). Both conditions
receive identical application text. The only variable is the constraint.

---

## Why This Matters

H.R. 1 (One Big Beautiful Bill Act, 2025) doubles the frequency of Medicaid and SNAP
redeterminations effective January 1, 2027. State agencies facing 2× the caseload with
the same staffing budgets will deploy AI-assisted processing. Those deployments will
carry token budgets, latency caps, and throughput targets.

This experiment measures whether those constraints alter eligibility outcomes — and by
how much. If a 51 percentage-point approval gap emerges from a single prompt-level
change, that is an audit finding, not a model quirk.

---

## Methodology

The experimental design maps directly to causal inference standards:

| Econometric Concept | Experimental Equivalent |
|---|---|
| Treatment / Control | Constrained vs. Baseline system prompt |
| Randomization | Conditions interleaved randomly across runs |
| Outcome variable | Binary approval decision |
| Effect estimator | Divergence coefficient (β₁ = rate_baseline − rate_constrained) |
| Identification test | Chi-square test of independence (p < 0.05) |
| Confidence intervals | Wilson score (robust near 0 and 1) |

The 50-run-per-condition design provides:
- ~80% power to detect a 20pp divergence at α = 0.05
- Confidence intervals narrow enough to distinguish signal from noise
- Token and latency distributions to characterize the mechanism

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/constraint-divergence-eval
cd constraint-divergence-eval
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
export $(cat .env | xargs)

# 3. Run the experiment (50 runs × 2 conditions = 100 API calls, ~$0.04)
python experiment.py

# 4. Analyze results
python analysis.py
```

For a quick smoke test (5 runs per condition, ~$0.004):
```bash
python experiment.py 5
```

---

## Example Output

```
  Condition        Approvals     Rate              95% CI   Tokens (μ±σ)
  ----------------------------------------------------------------
  Baseline          41/50       82.0%  [68.6% – 90.9%]    87.3 ± 24.1
  Constrained       16/50       32.0%  [20.3% – 46.4%]    31.8 ± 12.4
  ----------------------------------------------------------------

  Divergence coefficient (β₁)  : +50.0%
  Chi-square (Pearson, df=1)   : χ² = 24.76
  p-value                      : 0.0000  *** significant (p < 0.05)

  Token output delta           : -55.5 tokens/response
  Latency delta                : -312 ms/response

  INTERPRETATION
  Binding token constraints reduced Maria Gomez's approval rate by
  50.0% percentage points (from 82.0% → 32.0%).
  This divergence is statistically significant (χ²=24.76, p=0.0000),
  consistent with constraint-induced premature procedural closure on
  non-standard income documentation.
```

*(Example values. Run the experiment to generate real results.)*

---

## Project Structure

```
constraint-divergence-eval/
├── experiment.py      # Runs N trials per condition, logs to results/
├── analysis.py        # Loads results, computes statistics, prints report
├── requirements.txt   # anthropic, numpy, scipy
├── .env.example       # API key template
├── .gitignore         # Excludes .env and results/
└── README.md
```

Results are saved to `results/results_<timestamp>.json` (gitignored — your API
calls stay private). The analysis script automatically loads the most recent file.

---

## Extending the Experiment

**Test additional cases:** Edit `MARIA_GOMEZ_APPLICATION` in `experiment.py` or
add a `--case` argument to vary applicant profiles (urban vs. rural, different
documentation completeness, etc.).

**Vary constraint severity:** Adjust the `CONSTRAINED_SYSTEM` prompt or add a
`max_tokens` cap to the constrained condition to modulate effect size.

**Multi-model comparison:** Change `MODEL` to compare constraint sensitivity
across model families.

**Causal Logging Architecture extension:** The result JSON schema includes
`input_tokens`, `output_tokens`, `latency_ms`, and `raw_response` — the five
telemetry categories described in the companion Pre-Analysis Plan. These fields
support logit-probability analysis when the API provides them.

---

## Related Work

- Pre-Analysis Plan: *Constraint-Induced Algorithmic Exclusion in AI-Mediated
  Benefits Determinations* (working paper, under internal review)
- NHIA fieldwork: 1,250 rural women enrolled manually after automated system
  exclusion, Ghana National Health Insurance Authority, 2020
- Evaluation frameworks: [Inspect AI](https://inspect.aisi.org.uk/),
  [DeepEval](https://github.com/confident-ai/deepeval),
  [anthropics/evals](https://github.com/anthropics/evals)

---

## Author

Abdul Hamid Sherriff  
[poverty360.org](https://poverty360.org) · [AI Portfolio](https://s-abdul-ai.github.io/AI.github.io/)  
BenchTrust AI LLC

---

*This project is a pre-deployment audit instrument, not a production system.
No real applicant data is used. The Maria Gomez case is a synthetic composite
designed to represent documented non-standard applicant profiles.*
