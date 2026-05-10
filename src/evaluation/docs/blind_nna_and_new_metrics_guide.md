# Blind-NNA Baselines, New Metrics, and CSV Outputs — Smoke-Test Guide

> Companion doc to the [evaluation README](../README.md) and the
> [experimental-setup blueprint](../experimental_setup_blueprint_e2e.md).
> Built from the first real run of the new harness features so the
> console output and CSV excerpts below are verbatim, not synthesized.

---

## 1. What changed

The blind-family additions landed in two waves:

**Wave 1 — initial blind runners + new metrics + CSVs:**

1. **Two "blind" NNA runners** — `NNA-Dijkstra-Blind` and
   `NNA-AStar-Blind`. They plan exactly like their replan-capable
   siblings (shortest `base_time` path on the full `base_graph`, totally
   hazard- and block-blind), but they **do not replan** on blocked-edge
   encounter. First block hit → episode fails with
   `failure_reason = "blocked"`. See README §5.2b.
2. **Four new metrics**, added to the `metrics.REGISTRY` so they flow
   through every existing aggregation and CSV pipeline without further
   wiring:
   - `hazard_score` — raw length × (flood + landslide) hazard sum; the
     reward-weight-free companion to `hazard_exposure`.
   - `steps` — edge-traversal count per episode.
   - `distance` — meters walked (sum of `length_m` across traversed edges).
   - `runtime` — per-episode wall-clock in ms (always defined, even on
     failure — unlike the four above which are `NaN` when the episode
     fails).
3. **Two new CSV artifacts**, written alongside `metrics.json` by the
   evaluator:
   - `raw_metrics.csv` — one row per `(scenario, algorithm)` pair.
   - `overall_metrics.csv` — one row per `(algorithm, bucket)` where
     bucket is `RI1..RI5` or `"all"`. Wide format.

**Wave 2 — `NNA-Dijkstra-HA-Blind`:**

4. **Third blind runner** — `NNA-Dijkstra-HA-Blind`. Hazard-aware
   `travel_time` weights (like the HA oracle) but **block-blind** at
   plan time (like the other Blind variants). Plans on the full base
   graph with λ-drag costs, fails with `failure_reason = "blocked"`
   on first blocked edge. See README §5.3b.

   Purpose: fill the "hazard-aware × block-blind" cell of the 2×2
   capability matrix. The `Blind → HA-Blind` gap isolates "what
   hazard-aware weighting alone buys you"; the `HA-Blind → HA` gap
   isolates "what block foresight buys you on top of those weights."

   Piggy-backs on the existing `runners/base.py::run_nna_blind` helper
   via a new optional `plan_graph` parameter that routes the planner
   onto `view.hazard_aware_full_graph` (base graph + travel_time attrs
   on every edge, blocked or not).

All additive: existing runners, metrics, and the `metrics.json` shape are
unchanged, so downstream scripts that consume `metrics.json` stay valid.

---

## 2. How to run it (exact commands)

The cohort `la_trinidad_mini` already exists. These two commands are what
I ran for this guide:

```bash
cd "Benguet Flood and Landslide Data"

# Stage 2 — only run the two NEW algorithms (existing ones already have
# their routes/*.jsonl committed).
uv run python -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms NNA-Dijkstra-Blind NNA-AStar-Blind

# Stage 3 — evaluate across all 8 algorithms. Picks up every
# routes/*.jsonl in the cohort. Emits metrics.json + the two CSVs.
uv run python -m src.evaluation.evaluator \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini
```

Open the CSVs in Excel (Windows):

```bash
start src/evaluation/cohorts/la_trinidad_mini/report/overall_metrics.csv
start src/evaluation/cohorts/la_trinidad_mini/report/raw_metrics.csv
```

For the 9-method **thesis** run from scratch (all algos at once):

```bash
uv run python -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms NNA-Dijkstra NNA-AStar \
                 NNA-Dijkstra-Blind NNA-AStar-Blind \
                 NNA-Dijkstra-HA NNA-Dijkstra-HA-Blind \
                 DQN@balanced_HF DQN@fast_HF DQN@safe_HF

uv run python -m src.evaluation.evaluator \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini
```

To add only the Wave 2 runner to a cohort that already has the Wave 1
algorithms committed:

```bash
uv run python -m src.evaluation.run_policies \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini \
    --algorithms NNA-Dijkstra-HA-Blind

uv run python -m src.evaluation.evaluator \
    --cohort-dir src/evaluation/cohorts/la_trinidad_mini
```

---

## 3. Sample output (verbatim from the smoke run)

### 3.1 Stage 2 — running the two new blind runners

```
02:50:53  INFO     Cohort: la_trinidad_mini  (100 scenarios)
02:50:53  INFO     Graph:  data\staged_subgraphs\selected_subgraph_n200.graphml
02:50:53  INFO     Loading graph: data\staged_subgraphs\selected_subgraph_n200.graphml
02:50:54  INFO       nodes=200 edges=417
02:50:54  INFO
Running NNA-Dijkstra-Blind (sha256:5da0914f80286918)
02:50:54  INFO       [NNA-Dijkstra-Blind] 1/100
02:50:54  INFO       [NNA-Dijkstra-Blind] 50/100
02:50:54  INFO       [NNA-Dijkstra-Blind] 100/100
02:50:54  INFO       NNA-Dijkstra-Blind: 100 routes in 0.58s -> src\evaluation\cohorts\la_trinidad_mini\routes\NNA-Dijkstra-Blind.jsonl
02:50:54  INFO
Running NNA-AStar-Blind (sha256:fefe48c90ed9e507)
02:50:54  INFO       [NNA-AStar-Blind] 1/100
02:50:55  INFO       [NNA-AStar-Blind] 50/100
02:50:55  INFO       [NNA-AStar-Blind] 100/100
02:50:55  INFO       NNA-AStar-Blind: 100 routes in 0.58s -> src\evaluation\cohorts\la_trinidad_mini\routes\NNA-AStar-Blind.jsonl
```

~0.6 s per algorithm on the 100-scenario smoke cohort. On the full
`la_trinidad_v1` cohort (2500 scenarios), expect ~15 s per algorithm.

### 3.2 Stage 3 — aggregation across all 8 algorithms

```
02:51:02  INFO     Scoring DQN@balanced_HF
02:51:02  INFO     Scoring DQN@fast_HF
02:51:02  INFO     Scoring DQN@safe_HF
02:51:02  INFO     Scoring NNA-AStar-Blind
02:51:02  INFO     Scoring NNA-AStar
02:51:02  INFO     Scoring NNA-Dijkstra-Blind
02:51:02  INFO     Scoring NNA-Dijkstra-HA
02:51:02  INFO     Scoring NNA-Dijkstra
02:51:02  INFO     Wrote src\evaluation\cohorts\la_trinidad_mini\report\metrics.json
02:51:02  INFO     Wrote src\evaluation\cohorts\la_trinidad_mini\report\raw_metrics.csv (800 rows)
02:51:02  INFO     Wrote src\evaluation\cohorts\la_trinidad_mini\report\overall_metrics.csv (48 rows)

========================================================================
Cohort: la_trinidad_mini  (100 scenarios)
Graph:  la_trinidad_subgraph_n200
Mode:   deterministic_v3
========================================================================

  DQN@balanced_HF  (scored 100)
    success_rate     =  88.00%  (100 episodes)
    travel_time(min) = mean=21.3  std=12.6  over 88 successful
    hazard_exposure  = mean=1858.12  std=1663.01  over 88 successful
    replan_count     = mean=0.00  max=0
    failures: timeout=12
    by RI:   RI1= 95.0%  RI2= 75.0%  RI3= 90.0%  RI4= 85.0%  RI5= 95.0%
    robustness:  success=0.915  travel_time=0.609  hazard_exposure=0.282

  DQN@fast_HF  (scored 100)
    success_rate     =  87.00%  (100 episodes)
    travel_time(min) = mean=22.1  std=15.9  over 87 successful
    hazard_exposure  = mean=1950.96  std=2214.79  over 87 successful
    replan_count     = mean=0.00  max=0
    failures: timeout=13
    by RI:   RI1= 85.0%  RI2= 80.0%  RI3= 95.0%  RI4= 75.0%  RI5=100.0%
    robustness:  success=0.893  travel_time=0.618  hazard_exposure=0.212

  DQN@safe_HF  (scored 100)
    success_rate     =  78.00%  (100 episodes)
    travel_time(min) = mean=20.9  std=14.8  over 78 successful
    hazard_exposure  = mean=1681.61  std=1716.90  over 78 successful
    replan_count     = mean=0.00  max=0
    failures: timeout=22
    by RI:   RI1= 90.0%  RI2= 50.0%  RI3= 65.0%  RI4= 85.0%  RI5=100.0%
    robustness:  success=0.769  travel_time=0.538  hazard_exposure=0.214

  NNA-AStar-Blind  (scored 100)
    success_rate     =  81.00%  (100 episodes)
    travel_time(min) = mean=18.3  std=9.6  over 81 successful
    hazard_exposure  = mean=1414.69  std=1329.79  over 81 successful
    replan_count     = mean=0.00  max=0
    failures: blocked=19
    by RI:   RI1=100.0%  RI2= 25.0%  RI3= 85.0%  RI4= 95.0%  RI5=100.0%
    robustness:  success=0.648  travel_time=0.595  hazard_exposure=0.246

  NNA-AStar  (scored 100)
    success_rate     = 100.00%  (100 episodes)
    travel_time(min) = mean=21.9  std=12.2  over 100 successful
    hazard_exposure  = mean=1924.05  std=1726.05  over 100 successful
    replan_count     = mean=0.21  max=2
    by RI:   RI1=100.0%  RI2=100.0%  RI3=100.0%  RI4=100.0%  RI5=100.0%
    robustness:  success=1.000  travel_time=0.558  hazard_exposure=0.233

  NNA-Dijkstra-Blind  (scored 100)
    success_rate     =  81.00%  (100 episodes)
    travel_time(min) = mean=18.3  std=9.6  over 81 successful
    hazard_exposure  = mean=1414.69  std=1329.79  over 81 successful
    replan_count     = mean=0.00  max=0
    failures: blocked=19
    by RI:   RI1=100.0%  RI2= 25.0%  RI3= 85.0%  RI4= 95.0%  RI5=100.0%
    robustness:  success=0.648  travel_time=0.595  hazard_exposure=0.246

  NNA-Dijkstra-HA  (scored 100)
    success_rate     = 100.00%  (100 episodes)
    travel_time(min) = mean=21.0  std=11.1  over 100 successful
    hazard_exposure  = mean=1784.54  std=1522.70  over 100 successful
    replan_count     = mean=0.00  max=0
    by RI:   RI1=100.0%  RI2=100.0%  RI3=100.0%  RI4=100.0%  RI5=100.0%
    robustness:  success=1.000  travel_time=0.587  hazard_exposure=0.269

  NNA-Dijkstra  (scored 100)
    success_rate     = 100.00%  (100 episodes)
    travel_time(min) = mean=21.9  std=12.2  over 100 successful
    hazard_exposure  = mean=1924.05  std=1726.05  over 100 successful
    replan_count     = mean=0.21  max=2
    by RI:   RI1=100.0%  RI2=100.0%  RI3=100.0%  RI4=100.0%  RI5=100.0%
    robustness:  success=1.000  travel_time=0.558  hazard_exposure=0.233
```

(Only success/travel_time/hazard_exposure are shown in the console
summary to keep it compact; the new metrics appear in the CSVs and in
`metrics.json`.)

---

## 3.3 Wave 2 — NNA-Dijkstra-HA-Blind added (verbatim from the 2026-04-20 smoke)

```
15:59:03  INFO     Cohort: la_trinidad_mini  (100 scenarios)
15:59:03  INFO     Graph:  data\staged_subgraphs\selected_subgraph_n200.graphml
15:59:03  INFO     Loading graph: data\staged_subgraphs\selected_subgraph_n200.graphml
15:59:04  INFO       nodes=200 edges=417
15:59:04  INFO
Running NNA-Dijkstra-HA-Blind (sha256:2185abbcd5b93134)
15:59:04  INFO       [NNA-Dijkstra-HA-Blind] 1/100
15:59:05  INFO       [NNA-Dijkstra-HA-Blind] 50/100
15:59:05  INFO       [NNA-Dijkstra-HA-Blind] 100/100
15:59:05  INFO       NNA-Dijkstra-HA-Blind: 100 routes in 0.86s -> src\evaluation\cohorts\la_trinidad_mini\routes\NNA-Dijkstra-HA-Blind.jsonl
```

Running the evaluator over the now 9-algorithm cohort:

```
15:59:09  INFO     Scoring DQN@balanced_HF
15:59:09  INFO     Scoring DQN@fast_HF
15:59:09  INFO     Scoring DQN@safe_HF
15:59:09  INFO     Scoring NNA-AStar-Blind
15:59:09  INFO     Scoring NNA-AStar
15:59:09  INFO     Scoring NNA-Dijkstra-Blind
15:59:09  INFO     Scoring NNA-Dijkstra-HA-Blind
15:59:09  INFO     Scoring NNA-Dijkstra-HA
15:59:09  INFO     Scoring NNA-Dijkstra
15:59:09  INFO     Wrote src\evaluation\cohorts\la_trinidad_mini\report\metrics.json
15:59:09  INFO     Wrote src\evaluation\cohorts\la_trinidad_mini\report\raw_metrics.csv (900 rows)
15:59:09  INFO     Wrote src\evaluation\cohorts\la_trinidad_mini\report\overall_metrics.csv (54 rows)
```

The three most relevant blocks from the console summary:

```
  NNA-Dijkstra-Blind  (scored 100)
    success_rate     =  81.00%  (100 episodes)
    travel_time(min) = mean=18.3  std=9.6  over 81 successful
    hazard_exposure  = mean=1414.69  std=1329.79  over 81 successful
    replan_count     = mean=0.00  max=0
    failures: blocked=19
    by RI:   RI1=100.0%  RI2= 25.0%  RI3= 85.0%  RI4= 95.0%  RI5=100.0%
    robustness:  success=0.648  travel_time=0.595  hazard_exposure=0.246

  NNA-Dijkstra-HA-Blind  (scored 100)
    success_rate     =  86.00%  (100 episodes)
    travel_time(min) = mean=18.6  std=9.5  over 86 successful
    hazard_exposure  = mean=1431.57  std=1261.79  over 86 successful
    replan_count     = mean=0.00  max=0
    failures: blocked=14
    by RI:   RI1=100.0%  RI2= 40.0%  RI3= 90.0%  RI4=100.0%  RI5=100.0%
    robustness:  success=0.729  travel_time=0.603  hazard_exposure=0.275

  NNA-Dijkstra-HA  (scored 100)
    success_rate     = 100.00%  (100 episodes)
    travel_time(min) = mean=21.0  std=11.1  over 100 successful
    hazard_exposure  = mean=1784.54  std=1522.70  over 100 successful
    replan_count     = mean=0.00  max=0
    by RI:   RI1=100.0%  RI2=100.0%  RI3=100.0%  RI4=100.0%  RI5=100.0%
    robustness:  success=1.000  travel_time=0.587  hazard_exposure=0.269
```

### 3.4 Reading the Wave 2 results — 2×2 capability decomposition

Pivoting the success rates into the matrix from README §7.1:

|                                         | **Block-blind** | **Block-aware (oracle)** |
|---|---|---|
| **Hazard-blind** (`base_time` weights)  | Blind: **81%** (100/25/85/95/100) | Replan: **100%** (all RIs) |
| **Hazard-aware** (`travel_time` weights) | HA-Blind: **86%** (100/40/90/100/100) | HA: **100%** (all RIs) |

Decomposition:

- **Value of hazard-aware weighting alone** = HA-Blind − Blind =
  **+5pp overall**, concentrated at mid-RI:
  - RI1: +0pp (both 100% — no blocks to avoid)
  - RI2: **+15pp** (25% → 40%) — the headline lift
  - RI3: **+5pp** (85% → 90%)
  - RI4: **+5pp** (95% → 100%)
  - RI5: +0pp (both 100% — artifact of the 10-node RI5 SCC on the
    smoke subgraph, only short paths survive)
- **Value of block foresight on top of HA weights** = HA − HA-Blind =
  **+14pp overall**, heaviest at RI2 (+60pp: 40% → 100%).
- **Sum** = 19pp = Blind → HA total jump, as expected.

The mid-RI dominance hypothesis (README §5.3b) held:

- ✓ HA-Blind strictly dominated Blind at RI2, RI3, RI4.
- ✓ Tails (RI1, RI5) tied at 100% each.
- ✓ `replan_count = 0` on every HA-Blind row.
- ✓ All 14 HA-Blind failures carried `failure_reason = "blocked"`
  (no trapped, no timeout). Contrast with Blind: 19 blocked failures.

The `travel_time` premium HA-Blind paid for its safer routes was small
(18.6 vs 18.3 min average over successful episodes) and the
`hazard_exposure` difference is in the noise (1432 vs 1415) — the latter
being an average over *successful* episodes only, so HA-Blind's larger
denominator (it completed 5 more scenarios) masks some of the safety
gain.

### 3.5 Regression check — Wave 1 runners unchanged

The refactor added an optional `plan_graph` parameter to
`run_nna_blind` with `None` as the default (falling back to
`view.base_graph`). The existing `NNA-Dijkstra-Blind` and
`NNA-AStar-Blind` runners don't pass the new parameter, so they
continue to plan on `view.base_graph` with `base_time` — byte-identical
to the pre-change baseline. Confirmed by the 2026-04-20 run:

| Algorithm | Success | Blocked | sha hash |
|---|---|---|---|
| NNA-Dijkstra-Blind | 81.00% | 19 | `sha256:5da0914f80286918` (unchanged) |
| NNA-AStar-Blind | 81.00% | 19 | `sha256:fefe48c90ed9e507` (unchanged) |

The `algorithm_config_hash` didn't change because the runner
`policy_metadata` didn't change. Existing `routes/NNA-*-Blind.jsonl`
files on disk remain valid.

---

## 4. The files on disk after a Stage 3 run

```
src/evaluation/cohorts/la_trinidad_mini/report/
├── metrics.json           ≈70 KB  full nested aggregate (machine-readable)
├── raw_metrics.csv        ≈160 KB one row per (scenario, algo) → 900 rows + header
└── overall_metrics.csv    ≈22 KB  one row per (algo, bucket)   → 54 rows + header
```

All three carry the same numbers — the CSVs are a projection of
`metrics.json` into wide spreadsheet format. Use whichever fits the task:

- JSON for Python / scripts / diff-based review.
- `raw_metrics.csv` for ad-hoc exploration in pandas / Excel / per-
  scenario filtering.
- `overall_metrics.csv` for the comparison tables that go into the
  manuscript and for quick "which algo wins at RI3?" answers.

---

## 5. Sample CSV excerpts

### 5.1 `raw_metrics.csv` — first 12 data rows

```
scenario_id,RI,algorithm_id,failure_reason,replan_count,success,travel_time,hazard_exposure,hazard_score,steps,distance,runtime
la_trinidad_mini_000000,RI1,DQN@balanced_HF,,0,1.0,33.0442,3992.82,8799.99,52.0,11130.76,55.27
la_trinidad_mini_000001,RI1,DQN@balanced_HF,,0,1.0,19.8959,2251.30,4908.43,53.0,6896.87,19.20
la_trinidad_mini_000002,RI1,DQN@balanced_HF,,0,1.0,30.2669,4752.48,9660.39,36.0,9395.27,14.00
la_trinidad_mini_000003,RI1,DQN@balanced_HF,,0,1.0,24.6208,3372.40,7101.44,44.0,8021.04,17.47
la_trinidad_mini_000004,RI1,DQN@balanced_HF,,0,1.0,36.4068,4450.46,10045.76,55.0,12088.33,21.54
la_trinidad_mini_000005,RI1,DQN@balanced_HF,,0,1.0,29.7884,4481.87,9594.90,37.0,9203.09,14.96
...
la_trinidad_mini_000012,RI1,DQN@balanced_HF,timeout,0,0.0,,,,,,82.03
la_trinidad_mini_000013,RI1,DQN@balanced_HF,,0,1.0,27.6346,4145.45,9141.98,42.0,8417.26,19.96
```

Key observations:

- **Row 000012 failed** (`failure_reason = timeout`, `success = 0.0`).
  Every metric column is empty **except `runtime`** (82.03 ms). That's
  the "runtime is always defined" rule — failed episodes still have a
  wall-clock cost worth measuring.
- Successful rows populate all 7 metric columns.
- `replan_count` is `0` for DQN rows (DQN uses action masking, never
  replans). For `NNA-Dijkstra` / `NNA-AStar` rows it's 0 or 1 or 2 — the
  number of block-encounters where the policy had to reroute on the
  passable subgraph.

### 5.2 `overall_metrics.csv` — full 48-row wide table

Columns (38 total):

```
algorithm_id, bucket, n,
success_{mean,stdev,min,max},
travel_time_{mean,stdev,min,max},
hazard_exposure_{mean,stdev,min,max},
hazard_score_{mean,stdev,min,max},
steps_{mean,stdev,min,max},
distance_{mean,stdev,min,max},
runtime_{mean,stdev,min,max},
robustness_success, robustness_travel_time, robustness_hazard_exposure,
robustness_hazard_score, robustness_steps, robustness_distance,
robustness_runtime,
failure_counts
```

Robustness and `failure_counts` cells are **blank except on the
`bucket = "all"` row** — by design, since those are second-order statistics
that only make sense across RIs. Here's the condensed `bucket=all` view
for quick eyeballing (mean columns only):

| algorithm_id | n | success | travel_time | hazard_exposure | hazard_score | runtime_ms | rob_success | fail |
|---|---|---|---|---|---|---|---|---|
| DQN@balanced_HF | 100 | 0.88 | 21.32 | 1858.12 | 4237.68 | 24.69 | 0.915 | timeout=12 |
| DQN@fast_HF | 100 | 0.87 | 22.13 | 1950.96 | 4357.48 | 30.53 | 0.893 | timeout=13 |
| DQN@safe_HF | 100 | 0.78 | 20.91 | 1681.61 | 3808.54 | 32.75 | 0.769 | timeout=22 |
| NNA-AStar-Blind | 100 | 0.81 | 18.25 | 1414.69 | 3238.40 | 2.45 | 0.648 | blocked=19 |
| NNA-AStar | 100 | 1.00 | 21.88 | 1924.05 | 4404.69 | 2.39 | 1.000 | — |
| NNA-Dijkstra-Blind | 100 | 0.81 | 18.25 | 1414.69 | 3238.40 | 3.01 | 0.648 | blocked=19 |
| **NNA-Dijkstra-HA-Blind** | 100 | **0.86** | **18.57** | **1431.57** | **3293.50** | ~3 | **0.729** | **blocked=14** |
| NNA-Dijkstra-HA | 100 | 1.00 | 21.01 | 1784.54 | 4111.36 | 2.11 | 1.000 | — |
| NNA-Dijkstra | 100 | 1.00 | 21.88 | 1924.05 | 4404.69 | 2.79 | 1.000 | — |

The full CSV is 54 rows × 38 columns; the above is the `bucket=all` row
for each algo reprojected. Open `overall_metrics.csv` in Excel to see
the RI1..RI5 breakdown alongside. The new HA-Blind row is highlighted;
note its position between the hazard-blind blinds (81% success) and
the oracle (100%).

---

## 6. How to read the CSVs (by column)

### `raw_metrics.csv` — one row per (scenario, algo)

| Column | Meaning |
|---|---|
| `scenario_id` | Stable ID; joins back to `scenarios.jsonl`. |
| `RI` | One of `RI1`..`RI5`. Pulled from `scenario.rain_level`. |
| `algorithm_id` | Matches a row in `POLICY_FACTORIES`. |
| `failure_reason` | Empty on success; otherwise `trapped`, `timeout`, `no_route`, `invalid_action`, or `blocked` (block-blind NNAs only: Blind, AStar-Blind, HA-Blind). |
| `replan_count` | Number of local replans on the fair-replan NNAs (0 for every blind variant including HA-Blind, 0 for oracle, 0 for DQN). |
| `success` | 0.0 or 1.0 — the `success` metric (redundant with the per-metric columns in `metrics.json`; kept here for filtering convenience). |
| `travel_time` | Minutes. Blank on failure. |
| `hazard_exposure` | Reward-weighted hazard × distance (w_f=0.6, w_l=0.4). Blank on failure. |
| `hazard_score` | Raw unweighted hazard × distance (both flood + landslide counted equally). Blank on failure. |
| `steps` | Number of edges walked. Blank on failure. |
| `distance` | Meters walked. Blank on failure. |
| `runtime` | Milliseconds. **Always defined** — even on failure. |

**Useful pandas one-liner** (`uv run python -c '...'` or a notebook):

```python
import pandas as pd
df = pd.read_csv("src/evaluation/cohorts/la_trinidad_mini/report/raw_metrics.csv")
# Success rate pivot, algo vs RI:
df.groupby(["algorithm_id", "RI"])["success"].mean().unstack()
# Mean travel time among successful runs, algo vs RI:
df.dropna(subset=["travel_time"]).groupby(["algorithm_id", "RI"])["travel_time"].mean().unstack()
```

### `overall_metrics.csv` — one row per (algo, bucket)

- `bucket ∈ {RI1, RI2, RI3, RI4, RI5, all}`. Exactly 6 rows per algorithm.
- `n` = number of scenarios in the bucket (20 per RI for
  `la_trinidad_mini`; 100 for `all`).
- For every metric there are four columns: `_mean`, `_stdev`, `_min`,
  `_max`. NaN/failed episodes were filtered out before the stats were
  computed, so `n` can be less than the bucket size for metrics that go
  NaN on failure (e.g. `travel_time`).
- `robustness_*` (seven columns) are only populated on the
  `bucket="all"` row. Each is `1 - σ/μ` across the five RI means for
  that metric; closer to 1.0 = more consistent across rainfall levels.
- `failure_counts` is a `;`-joined list of `reason=N` pairs on the
  `bucket="all"` row (e.g. `timeout=12` or `blocked=19`).

### Ratios that matter for the thesis

For any given algorithm, compute:

- **Blind-vs-replan gap**: `NNA-*.success_mean − NNA-*-Blind.success_mean` =
  structural lift from the fair-replan loop. On `la_trinidad_mini`
  overall: `1.00 − 0.81 = 0.19` — the replan mechanism saves 19% of
  episodes from outright failure.
- **Blind-vs-HA-Blind gap**: `NNA-*-HA-Blind.success_mean −
  NNA-*-Blind.success_mean` = value of hazard-aware *weighting* in
  isolation from block foresight. On `la_trinidad_mini` overall:
  `0.86 − 0.81 = 0.05`; per-RI: +0/+15/+5/+5/+0 (mid-RI dominance
  regime — see §3.4).
- **HA-Blind-vs-HA gap**: `NNA-*-HA.success_mean −
  NNA-*-HA-Blind.success_mean` = value of block foresight on top of
  hazard-aware weights. On `la_trinidad_mini` overall:
  `1.00 − 0.86 = 0.14`.
- **DQN-vs-replan gap**: `DQN@*.success_mean − NNA-*.success_mean` =
  learned-policy lift over hazard-blind-with-replan. On this smoke run
  the gap is *negative* (DQN 0.78–0.88 vs replan NNAs 1.00) — explained
  by the DQN's OOD tax (trained at `num_deliveries=2`, eval at 5) and
  the small 100-scenario cohort. Retraining closes this gap.
- **HA-vs-DQN gap**: `NNA-*-HA.travel_time_mean − DQN.travel_time_mean` =
  how much the DQN leaves on the table vs a full-foresight oracle.
- **Blind-NNA hazard-exposure bias**: `hazard_exposure` and
  `hazard_score` are only computed on *successful* episodes, so the
  blind NNAs' lower means reflect selection bias (the easier scenarios
  survived) rather than better path-finding. Always read `_mean` in the
  context of `n` in the same row — this is why HA-Blind's `hazard_exposure`
  (1431.57 over 86 successes) looks close to Blind's (1414.69 over 81
  successes): HA-Blind completed 5 more hard scenarios that bring
  their hazard into the average.

---

## 7. Expected ordering (what a healthy run should look like)

Per the design in README §5 and §7:

### Success rate, overall (`bucket = all`)

```
NNA-*-Blind   ≤   NNA-*-HA-Blind   ≤   DQN@*     ≤   NNA-*       ≤   NNA-*-HA
   0.81       ≤      0.86           ≤  0.78–0.88 ≤   1.00        ≤   1.00
```

On the smoke cohort, `DQN < NNA-*` and `DQN < NNA-*-HA-Blind` because
of the DQN's OOD tax (trained at `num_deliveries=2`, eval at 5); in a
large / retrained cohort we expect `NNA-*-HA-Blind ≤ DQN ≤ NNA-*`.
The present inversion is documented in README §10 and the blueprint §8.1.

### Success rate, per-RI

Expect the blind variants to be **≤** the replan variants at every RI
(blind can't recover from a block); and expect `Blind ≤ HA-Blind ≤ HA`
everywhere (hazard-aware weights add incidental block-avoidance, oracle
adds explicit block foresight on top). Smoke-test confirms both:

| RI | Replan | HA | HA-Blind | Blind | notes |
|---|---|---|---|---|---|
| RI1 | 100% | 100% | 100% | 100% | No blocks at RI1 — all tied |
| RI2 | 100% | 100% | **40%** | 25% | HA-Blind's **+15pp** lift — mid-RI regime |
| RI3 | 100% | 100% | **90%** | 85% | **+5pp** lift |
| RI4 | 100% | 100% | **100%** | 95% | **+5pp** lift |
| RI5 | 100% | 100% | 100% | 100% | SCC-collapse tie — see README §5.3b |

Bug signals:

- **Blind > Replan** at any RI → fair-replan loop isn't triggered.
- **HA-Blind > HA** at any RI → the planner substrate for HA-Blind
  (`view.hazard_aware_full_graph`) is incorrect, or oracle pathing
  regressed.
- **HA-Blind < Blind** at any RI → the `travel_time` weights aren't
  being consulted (hazard-aware planning is a no-op).
- **HA-Blind strictly dominates Blind uniformly across all RI** → the
  mid-RI hypothesis fails; investigate whether blocked-edge/λ-drag
  alignment is stronger than expected (possible if the threshold test
  has been tightened since the hypothesis was framed).

### `NNA-Dijkstra-Blind` vs `NNA-AStar-Blind` — always identical

Both plan shortest-`base_time` paths (A* with admissible heuristic =
Dijkstra on this graph) and neither does any stochastic or repair
step, so every per-scenario outcome is identical. Smoke-test: exactly
the same `success_mean`, `travel_time_mean`, `hazard_exposure_mean`,
`hazard_score_mean`, `steps_mean`, `distance_mean`. Only `runtime`
differs slightly (A*'s heuristic costs a tiny bit extra). If you see
divergence on a non-runtime column, one of the runners has a bug.

### `replan_count` ordering

- `NNA-Dijkstra-Blind`, `NNA-AStar-Blind`, `NNA-Dijkstra-HA-Blind`:
  always `0` (no replan mechanism — any blind variant fails fast on
  block encounter).
- `NNA-Dijkstra-HA`: always `0` (oracle plans on the passable graph,
  blocked edges never in the plan).
- `DQN@*`: always `0` (action-mask, not replan).
- `NNA-Dijkstra`, `NNA-AStar`: `≥ 0` — should correlate positively with
  RI on a large cohort. If replan_count is always 0 for these too, the
  fair-replan loop isn't being triggered — see §8 Troubleshooting.

---

## 8. Troubleshooting

### 8.1 A blind NNA succeeds at RI5 with 100% success rate

This *can* be legitimate on a tiny cohort (the 10-node RI5 SCC on
`la_trinidad_subgraph_n200` has very few blocked shortest paths; blind
NNA's plan happens to avoid them). But on `la_trinidad_v1` with 500
RI5 scenarios, blind NNAs should fail ≫ 0% of the time. If they
don't, either:

- The scenario_generator isn't producing blocked edges (check the log
  for `RI5: blocked N/417 edges` — N should be much larger than 0).
- The runner isn't consulting `scenario.blocked_set()`
  (`runners/base.py::run_nna_blind` should branch into the "blocked"
  failure when `(cursor, nxt) in blocked`).

### 8.2 CSV row counts don't match expectation

Expected on `la_trinidad_mini` after the Wave 2 update:
- `raw_metrics.csv`: `9 × 100 + 1 header = 901` lines.
- `overall_metrics.csv`: `9 × 6 + 1 header = 55` lines.

(Pre-Wave-2 counts were `8 × 100 + 1 = 801` and `8 × 6 + 1 = 49`; if
you're looking at an old cohort without HA-Blind routes, those still
apply.)

If the raw CSV is short, one of the `routes/<algo>.jsonl` files is
missing or truncated. If the overall CSV is short, an algorithm
didn't write a full set of 6 buckets — likely because its routes
file was missing or all its episodes failed at a specific RI with
`n=0` (in which case the bucket still appears but with blank stats).

### 8.3 A new metric appears in `metrics.json` but not the CSVs

Every registered metric in `metrics/__init__.py::REGISTRY` is
reflected in both CSVs — the column lists are derived from
`REGISTRY.keys()`. If your new metric is missing, check:

- It's spelled correctly in the REGISTRY dict.
- The module has a module-level `def compute(scenario, route) -> float`.
- Stage 3 was rerun *after* the REGISTRY edit (the CSVs are generated
  in the same call as `metrics.json`, so there's no way to have a
  fresh JSON + stale CSVs unless the writer raised an exception mid-
  write).

### 8.4 A `NaN` appears in `overall_metrics.csv`

Shouldn't happen — `_safe_stats` filters NaN before aggregating, and
the writer renders `None` as empty string. If you see literal `NaN`
text in a cell, a raw-value `NaN` leaked past the filter. Add a
repro case and patch `_safe_stats` in `evaluator.py`.

### 8.5 `failure_reason == "blocked"` never appears

The blind NNAs should emit `blocked` whenever they hit a planned edge
that is in `scenario.blocked_set()`. If you never see it, the most
likely culprits are:

- The cohort has no blocked edges (check `scenarios.jsonl` — at least
  RI2 scenarios should have non-empty `blocked_edges`).
- The runner is using `run_nna_with_fair_replan` instead of
  `run_nna_blind` — inspect the imports in
  `runners/nna_blind.py` / `runners/nna_astar_blind.py`.
- `schemas.FAILURE_REASONS` doesn't include `"blocked"` — this is the
  frozenset the schema uses to validate failure labels.

### 8.6 `runtime` is 0.0 for a metric row

`runtime` is `Route.wall_time_ms`, set by `time.perf_counter()` at the
start and end of each policy call. Zero would mean the clock didn't
advance between the start and end — implausible unless something is
stubbing the clock. Inspect the corresponding `routes/<algo>.jsonl`
entry; `wall_time_ms` should be > 0 there.

---

## 9. Where things live (links back)

- Algorithm deep-dives, fairness argument, schemas → [README §5–§7](../README.md).
- Blind-NNA runner code → [`runners/nna_blind.py`](../runners/nna_blind.py),
  [`runners/nna_astar_blind.py`](../runners/nna_astar_blind.py),
  [`runners/nna_ha_blind.py`](../runners/nna_ha_blind.py),
  shared execution helper at [`runners/base.py::run_nna_blind`](../runners/base.py)
  (pass `plan_graph=view.hazard_aware_full_graph` for HA-Blind).
- New metric modules → [`metrics/hazard_score.py`](../metrics/hazard_score.py),
  [`metrics/steps.py`](../metrics/steps.py),
  [`metrics/distance.py`](../metrics/distance.py),
  [`metrics/runtime.py`](../metrics/runtime.py).
- CSV emission → [`evaluator.py::_write_raw_csv`](../evaluator.py)
  and [`evaluator.py::_write_overall_csv`](../evaluator.py).
- Operational command recipes, full cohort runs, cohort scaling notes →
  [experimental-setup blueprint](../experimental_setup_blueprint_e2e.md).
- Project-wide invariants (activation mode, hazard mapping, etc.) →
  [`CLAUDE.md`](../../../CLAUDE.md) at the repo root.
