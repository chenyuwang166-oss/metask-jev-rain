# calib_v2 -- per-precision calibration blocks for metask-jev-rain (Gemma-4-12B-it, verbalized one-word read-out)

> Shipped copy of the calibration study report (2026-10-03). Terminology: in `split_231_v2.json` the 115-item **check half** is stored under the key `local_sealed`; it is a half of the **public** 231 items, not the sealed set. Where the tables below say `local_sealed`, read "check half (key `local_sealed` in split_231_v2.json)". Every number is a local estimate, not an official score.

**All numbers are 231-public-item estimates from a local re-implementation of METHOD v1.5 scoring on cached logits. Nothing here is an official score. ECE on 74 noul / 18 score items is noisy; the TVD half of choice-C rests on 7 gold_probs items (3 dev / 4 check).**

## 0. Pre-registered rule (verbatim from the script docstring)

```
calib_v2.py -- per-precision read-out calibration for the metask-jev-rain adapter (CPU only, cached logits).

PRE-REGISTERED RULE (written before any number below was computed; identical for both caches)
  Data      : the 231 public JevBench items, split_231_v2.json (dev 116 / check half 115, stored under the key local_sealed).
              Every parameter is fit on dev ONLY.  The check half is scored ONCE per cache, at the picked values.
  Scoring   : local re-implementation of the METHOD v1.5 formulas (score_v15.py / calib_gemma.py, unchanged);
              all numbers are 231-public-item ESTIMATES, not official scores.
  Objective : per type, maximise H = harmonic mean of (I_type, C_type) on dev; ties -> first grid point ascending.
  choice    : P = softmax(m / T), m_j = cached per-label log-mass; T grid 0.5 .. 6.0 step 0.25 (23 points).
  score     : P = softmax(m / T), m_k = logit of digit token k; T grid {1, 1.5, 2, 2.3, 3, 4, 5, 6, 8}.
  noul      : d = lse(m["yes"], m[" yes"]) - lse(m["no"], m[" no"]) ("lower" form set).  Check that sign(d) equals
              the sign of the full 12+12 aggregate on all 74 items; if not, fall back to the full set and say so.
              One-bin histogram calibration (Zadrozny & Elkan 2001, one bin per decision side):
              p_cal = dev precision of the rule "yes iff d > 0" (d == 0 counts as "no"; never observed);
              reported P(yes) = p_cal if d > 0 else 1 - p_cal.  No free parameter beyond p_cal.
              Requires p_cal >= 0.8 so that no item abstains under the 0.8 / 0.2 thresholds (asserted, reported).
  Alternative kept for the maintainers only (not used): noul temperature T = 0.05, P(yes) = sigmoid(d / 0.05).
  Agreement : decision agreement between the two caches at their own picks (choice argmax, noul sign, score argmax
              and rounded expected value), over all 231 items and over the check half.
  Prior look: on int8 the check half was looked at once in calib_gemma (2026-10-02) and preferred score T 4.0 over
              the dev pick 5.0; we keep the pre-registered dev pick and re-state that observation, we do not re-pick.
```

Split: `split_231_v2.json` dev 116 / check half (key `local_sealed`) 115; units = task.group; 0 groups straddle halves.

## int8: cache `logits_231_verb_gemma12b_int8.jsonl`

sha256 `77df7e47831220595ef2d4fc5731afb30c0735b9e134de96d985ff2929f0096e`; meta quant = int8, peak VRAM 12.9 GB, 78 s for 231 items; items ok 231/231.

### int8 choice: dev grid (T, I, C, H) -> pick **T = 4.5**

| T | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.25 | 2.5 | 2.75 | 3.0 | 3.25 | 3.5 | 3.75 | 4.0 | 4.25 | 4.5 | 4.75 | 5.0 | 5.25 | 5.5 | 5.75 | 6.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | **77.8** | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 |
| C | 54.0 | 53.7 | 54.5 | 57.9 | 57.1 | 61.5 | 62.8 | 67.2 | 70.0 | 72.2 | 71.0 | 78.2 | 77.3 | 79.4 | 79.9 | 82.4 | **84.0** | 80.4 | 79.2 | 76.7 | 77.3 | 79.3 | 78.7 |
| H | 63.8 | 63.6 | 64.1 | 66.4 | 65.9 | 68.7 | 69.5 | 72.1 | 73.7 | 74.9 | 74.2 | 78.0 | 77.6 | 78.6 | 78.9 | 80.0 | **80.8** | 79.1 | 78.5 | 77.3 | 77.6 | 78.6 | 78.2 |

### int8 score: dev grid (T, I, C, H) -> pick **T = 5.0**

| T | 1.0 | 1.5 | 2.0 | 2.3 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 28.4 | 31.3 | 34.5 | 36.3 | 39.8 | 43.2 | **44.9** | 45.4 | 44.8 |
| C | 69.8 | 73.4 | 76.1 | 77.3 | 79.2 | 79.9 | **78.7** | 76.2 | 69.9 |
| H | 40.4 | 43.9 | 47.5 | 49.4 | 53.0 | 56.1 | **57.2** | 56.9 | 54.6 |

### int8 noul: one-bin histogram calibration

* form-set check: sign(d_lower) == sign(d_full) on 74/74 items -> **lower set used** ({yes, " yes"} vs {no, " no"}); items with d == 0: 0; min |d| = 0.250.
* dev precision of sign rule: 33/37 = **p_cal = 0.8919** (yes-side 100.0% of 14; no-side 82.6% of 23).
* reported P(yes) = 0.8919 if d > 0 else 0.1081; p_cal >= 0.8 -> no abstention: yes.

### int8 results at the picks (check half scored once)

| subset | n | I_choice | C_choice | I_noul | C_noul | abst% | I_score | C_score | I_231 | C_axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev (fit) | 116 | 77.8 | 84.0 | 77.2 | 83.6 | 0.0 | 44.9 | 78.7 | 66.6 | 82.1 |
| check half (once) | 115 | 78.9 | 75.7 | 59.1 | 83.8 | 0.0 | 78.0 | 82.4 | 72.0 | 80.6 |
| full 231 (reference) | 231 | 78.3 | 83.7 | 68.2 | 91.9 | 0.0 | 62.4 | 86.0 | 69.6 | 87.2 |

Per-tier I (chance-corrected): check half (key local_sealed): choice [easy:100.0 standard:100.0 hard:63.0] / noul [easy:100.0 standard:83.3 hard:36.8] / score [standard:95.0 hard:69.5]; full: choice [easy:100.0 standard:100.0 hard:62.1] / noul [easy:100.0 standard:83.3 hard:52.6] / score [standard:93.5 hard:46.9]

noul alternative (maintainers only, T = 0.05): dev I 77.2 / C 78.3 / abst 0.0%; check half (key local_sealed) I 59.1 / C 62.2 / abst 0.0%; full I 68.2 / C 70.3 / abst 0.0%

## bf16: cache `logits_231_verb_gemma12b_bf16off.jsonl`

sha256 `4a1811e264dd5220119692f050e622ccc44fa2c58c59c8f0a7231d7db7948a21`; meta quant = bf16, peak VRAM 19.7 GB, 128 s for 231 items; items ok 231/231.

### bf16 choice: dev grid (T, I, C, H) -> pick **T = 4.5**

| T | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.25 | 2.5 | 2.75 | 3.0 | 3.25 | 3.5 | 3.75 | 4.0 | 4.25 | 4.5 | 4.75 | 5.0 | 5.25 | 5.5 | 5.75 | 6.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | **77.8** | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 | 77.8 |
| C | 57.4 | 59.2 | 61.0 | 62.9 | 64.9 | 67.1 | 69.6 | 70.6 | 75.0 | 74.9 | 77.1 | 81.7 | 81.6 | 82.0 | 84.9 | 81.3 | **86.7** | 85.9 | 79.6 | 79.8 | 79.7 | 78.6 | 77.6 |
| H | 66.1 | 67.2 | 68.4 | 69.6 | 70.8 | 72.1 | 73.5 | 74.0 | 76.4 | 76.3 | 77.5 | 79.7 | 79.7 | 79.9 | 81.2 | 79.5 | **82.0** | 81.7 | 78.7 | 78.8 | 78.7 | 78.2 | 77.7 |

### bf16 score: dev grid (T, I, C, H) -> pick **T = 5.0**

| T | 1.0 | 1.5 | 2.0 | 2.3 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I | 30.0 | 31.8 | 34.6 | 36.3 | 39.8 | 43.2 | **44.9** | 45.4 | 44.8 |
| C | 67.6 | 71.6 | 74.6 | 76.0 | 78.3 | 79.2 | **78.1** | 75.7 | 69.6 |
| H | 41.6 | 44.0 | 47.3 | 49.2 | 52.8 | 55.9 | **57.0** | 56.8 | 54.5 |

### bf16 noul: one-bin histogram calibration

* form-set check: sign(d_lower) == sign(d_full) on 74/74 items -> **lower set used** ({yes, " yes"} vs {no, " no"}); items with d == 0: 0; min |d| = 0.250.
* dev precision of sign rule: 32/37 = **p_cal = 0.8649** (yes-side 100.0% of 13; no-side 79.2% of 24).
* reported P(yes) = 0.8649 if d > 0 else 0.1351; p_cal >= 0.8 -> no abstention: yes.

### bf16 results at the picks (check half scored once)

| subset | n | I_choice | C_choice | I_noul | C_noul | abst% | I_score | C_score | I_231 | C_axis |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev (fit) | 116 | 77.8 | 86.7 | 72.4 | 81.0 | 0.0 | 44.9 | 78.1 | 65.1 | 81.9 |
| check half (once) | 115 | 78.9 | 77.7 | 53.1 | 83.8 | 0.0 | 81.1 | 84.5 | 71.1 | 82.0 |
| full 231 (reference) | 231 | 78.3 | 84.0 | 62.8 | 91.6 | 0.0 | 64.1 | 86.3 | 68.4 | 87.3 |

Per-tier I (chance-corrected): check half (key local_sealed): choice [easy:100.0 standard:100.0 hard:63.0] / noul [easy:100.0 standard:83.3 hard:26.3] / score [standard:95.8 hard:73.8]; full: choice [easy:100.0 standard:100.0 hard:62.1] / noul [easy:100.0 standard:75.0 hard:47.4] / score [standard:93.9 hard:49.1]

noul alternative (maintainers only, T = 0.05): dev I 72.4 / C 72.9 / abst 0.0%; check half (key local_sealed) I 53.1 / C 56.8 / abst 0.0%; full I 62.8 / C 64.9 / abst 0.0%

## Decision agreement int8 vs bf16 (each at its own picks)

| subset | choice argmax | noul decision | score argmax | score round(EV) |
|---|---:|---:|---:|---:|
| full | 135/139 | 72/74 | 18/18 | 18/18 |
| check half (key local_sealed) | 68/69 | 36/37 | 9/9 | 9/9 |

max |d_int8 - d_bf16| over 74 noul items: 9.875. Items that flip: choice ['hard-opus-b-multi_hop-04', 'hard-opus-c-temporal_numeric-06', 'hard-sol-a-multi_hop-07', 'hard-sol-b-temporal_numeric-04']; noul ['hard-sol-b-judge_hard-05', 'original-adequacy-01-0']; score argmax []; score round(EV) [].

## Disclosure: prior look at the check half (int8, score temperature)

In calib_gemma (2026-10-02, int8 cache, score grid 1..16 step 0.5) the dev pick was T_score = 5.0 and the check half was scored once at that pick; alongside, the then-current value T 4.0 was reported on the check half for reference (check-half score I/C 81.4/86.3 at T 4.0 vs 78.0/82.4 at T 5.0; dev 43.2/79.9 vs 44.9/78.7). The check half therefore preferred 4.0. We keep the pre-registered dev pick (this run re-derives it on the coarser pre-registered grid) and do not re-pick on the check half.

## calibration.json blocks

```json
{
 "int8": {
  "temperature_by_kind": {
   "choice": 4.5,
   "noul": "one_bin",
   "score": 5.0
  },
  "noul_calibration": {
   "method": "one_bin_histogram",
   "forms": {
    "yes": [
     "yes",
     " yes"
    ],
    "no": [
     "no",
     " no"
    ]
   },
   "decision": "yes iff d = logsumexp(yes forms) - logsumexp(no forms) > 0",
   "p_cal": 0.8919,
   "fitted_on": "dev half (116 items) of split_231_v2",
   "reported": "P(yes)=p_cal if d>0 else 1-p_cal",
   "reference": "Zadrozny & Elkan 2001 (histogram binning), one bin per decision side"
  },
  "noul_alternative": {
   "method": "temperature",
   "T": 0.05,
   "note": "not used by the adapter; kept for comparison"
  },
  "fitted_on_cache": "logits_231_verb_gemma12b_int8.jsonl",
  "fitted_on_cache_sha256": "77df7e47831220595ef2d4fc5731afb30c0735b9e134de96d985ff2929f0096e",
  "selection_rule": "dev-only; maximise harmonic(I_type, C_type); choice grid 0.5..6 step 0.25; score grid {1,1.5,2,2.3,3,4,5,6,8}; ties -> first ascending",
  "notes": "Fit on dev 116 of split_231_v2 only; check half (115) scored once at these values: I_231 72.0, C_axis 80.6, noul abstention 0%. Full 231 for reference: I_231 69.6, C_axis 87.2. Local reconstruction of METHOD v1.5 on the 231 public items; estimates, not official. noul form-set check: sign(lower) == sign(full 12+12) on 74/74 items."
 },
 "bf16": {
  "temperature_by_kind": {
   "choice": 4.5,
   "noul": "one_bin",
   "score": 5.0
  },
  "noul_calibration": {
   "method": "one_bin_histogram",
   "forms": {
    "yes": [
     "yes",
     " yes"
    ],
    "no": [
     "no",
     " no"
    ]
   },
   "decision": "yes iff d = logsumexp(yes forms) - logsumexp(no forms) > 0",
   "p_cal": 0.8649,
   "fitted_on": "dev half (116 items) of split_231_v2",
   "reported": "P(yes)=p_cal if d>0 else 1-p_cal",
   "reference": "Zadrozny & Elkan 2001 (histogram binning), one bin per decision side"
  },
  "noul_alternative": {
   "method": "temperature",
   "T": 0.05,
   "note": "not used by the adapter; kept for comparison"
  },
  "fitted_on_cache": "logits_231_verb_gemma12b_bf16off.jsonl",
  "fitted_on_cache_sha256": "4a1811e264dd5220119692f050e622ccc44fa2c58c59c8f0a7231d7db7948a21",
  "selection_rule": "dev-only; maximise harmonic(I_type, C_type); choice grid 0.5..6 step 0.25; score grid {1,1.5,2,2.3,3,4,5,6,8}; ties -> first ascending",
  "notes": "Fit on dev 116 of split_231_v2 only; check half (115) scored once at these values: I_231 71.1, C_axis 82.0, noul abstention 0%. Full 231 for reference: I_231 68.4, C_axis 87.3. Local reconstruction of METHOD v1.5 on the 231 public items; estimates, not official. noul form-set check: sign(lower) == sign(full 12+12) on 74/74 items."
 }
}
```

Full dev curves: `calib_v2_results.json` (available on request).
