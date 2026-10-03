#!/usr/bin/env python
"""Parity check for metask-jev-rain: does the adapter reproduce the frozen cache's decisions?

Two modes, both CPU-only (no torch needed):

  offline  (no harness run yet)      python parity_check.py --cache logits_231_verb_gemma12b_int8.jsonl \
                                        --calibration calibration.json --precision int8 --harness <upstream repo root>
      Re-applies the adapter's own read-out code (aggregate + readout_probs, imported from
      jevbench/adapters/metask_rain.py) with the calibration block of the given precision (form sets,
      temperatures, one-bin noul p_cal) to the per-form logits stored in the cache, and prints accuracy /
      decisions / abstentions per type. This is what the GPU run must reproduce.

  results  (after selftest.sh)       ... --results <out>/results.jsonl
      Compares the harness per-item records (predicted label, probs) with the offline expectation item by
      item. Tolerance-based: reports decision agreement, |delta p| quantiles, and classifies every
      disagreement as a near-tie (the cache's own margin below --tie-tol in log-mass: noul |d|, choice/score
      top-2 log-mass gap) or not. The precision is taken from the results' raw.runtime (or --precision).
      Exit code: 0 if every decision agrees or every disagreement is a near-tie, 5 otherwise.

The cache rows carry `forms` = {"<label>|<form>": logit} (single-token forms only), `labels`, `type`,
`expected`; `n_tokens` per item (prompt length) is compared with the harness usage.input_tokens.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import string
import sys
from pathlib import Path


def load_adapter_module(harness_root, adapter_file=None):
    sys.path.insert(0, str(harness_root))
    import jevbench.adapters  # noqa: F401  (package context for the relative import of .base)
    path = Path(adapter_file) if adapter_file else Path(harness_root) / "jevbench" / "adapters" / "metask_rain.py"
    spec = importlib.util.spec_from_file_location("jevbench.adapters.metask_rain", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def expected_from_cache(row, mod, block, noul_mode):
    """Adapter read-out applied to the cache's per-form logits -> (probs, decision, logmass, margin)."""
    qtype, labels = row["type"], row["labels"]
    forms = {}
    for k, v in row["forms"].items():
        label, s = k.split("|", 1)
        forms.setdefault(label, {})[s] = v
    if qtype == "noul":
        f = block["noul_calibration"]["forms"]
        groups = {"yes": list(f["yes"]), "no": list(f["no"])}
    elif qtype == "score":
        groups = {l: [s for s in mod.digit_forms(l) if s in forms.get(l, {})] for l in labels}
    else:
        codes = string.ascii_uppercase[:len(labels)]
        groups = {l: [s for s in mod.letter_forms(c) if s in forms.get(l, {})] for c, l in zip(codes, labels)}
    flat = {s: forms[l][s] for l, fs in groups.items() for s in fs}
    logmass = mod.aggregate(flat, {l: list(fs) for l, fs in groups.items()})
    probs, _ = mod.readout_probs(qtype, logmass, labels, block, noul_mode)
    if qtype == "noul":
        # harness noul decision: P(yes) >= 0.8 yes, <= 0.2 no, else abstain
        py = probs["yes"]
        decision = "yes" if py >= 0.8 else ("no" if py <= 0.2 else "abstain")
        margin = abs(mod.noul_d(logmass))
    else:
        decision = max(sorted(probs), key=lambda k: probs[k])  # harness argmax: ties -> lexicographically smallest
        top = sorted(logmass.values(), reverse=True)
        margin = top[0] - top[1] if len(top) > 1 else float("inf")
    return probs, decision, logmass, margin


def quantiles(xs, qs=(0.5, 0.9, 0.99, 1.0)):
    if not xs:
        return {}
    s = sorted(xs)
    return {f"p{int(q * 100)}": round(s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))], 6) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--harness", required=True, help="repo root containing jevbench/ (patched copy or upstream)")
    ap.add_argument("--adapter-file", default=None, help="metask_rain.py to import (default: <harness>/jevbench/adapters/metask_rain.py)")
    ap.add_argument("--precision", default=None, choices=["bf16", "int8"],
                    help="calibration block to apply (default: from the results' raw.runtime.precision, else int8)")
    ap.add_argument("--noul-mode", default="one_bin", choices=["one_bin", "temperature"])
    ap.add_argument("--results", default=None, help="harness results.jsonl from selftest.sh")
    ap.add_argument("--tie-tol", type=float, default=0.1,
                    help="a disagreement counts as a near-tie when the cache's own margin (noul |d|, choice/score top-2 log-mass gap) is below this")
    args = ap.parse_args()

    mod = load_adapter_module(args.harness, args.adapter_file)
    cal = mod.load_calibration(args.calibration)   # same validation as the adapter
    res = [json.loads(l) for l in open(args.results, encoding="utf-8") if l.strip()] if args.results else []
    precision = args.precision
    if precision is None and res:
        seen = collections.Counter(((r.get("runtime") or {}).get("precision")) for r in res if r.get("ok"))
        precision = seen.most_common(1)[0][0] if seen else None
        print(f"[parity] precision taken from results raw.runtime: {dict(seen)}")
    precision = precision or "int8"
    block = cal[precision]
    rows = [json.loads(l) for l in open(args.cache, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    exp = {}
    stats = {}
    for r in rows:
        probs, dec, lm, margin = expected_from_cache(r, mod, block, args.noul_mode)
        exp[r["id"]] = (probs, dec, r, margin)
        # cross-check the module's aggregation against the cache's own per-label log-mass
        # (choice/score: identical form sets; noul: the cache aggregated the full 12+12 sets, so only the sign of d is compared)
        if r["type"] == "noul":
            d_full = r["logits"]["yes"] - r["logits"]["no"]; d_lower = lm["yes"] - lm["no"]
            if (d_full > 0) != (d_lower > 0):
                print(f"[offline] WARNING noul sign differs between full and lower form sets: {r['id']} d_full={d_full:.3f} d_lower={d_lower:.3f}")
        else:
            delta = max(abs(lm[l] - r["logits"][l]) for l in r["labels"])
            if delta > 1e-3:
                print(f"[offline] WARNING log-mass differs from cache aggregate: {r['id']} max delta {delta:.4f}")
        s = stats.setdefault(r["type"], {"n": 0, "correct": 0, "abstain": 0})
        s["n"] += 1
        s["correct"] += dec == str(r["expected"])
        s["abstain"] += dec == "abstain"
    nc = block["noul_calibration"]
    print(f"[offline] cache {args.cache}: {len(rows)} ok rows; block {precision} (fitted on {block.get('fitted_on_cache')}); "
          f"choice T {block['temperature_by_kind']['choice']} / score T {block['temperature_by_kind']['score']} / "
          f"noul {args.noul_mode} " + (f"p_cal {nc['p_cal']}" if args.noul_mode == "one_bin" else f"T {block['noul_alternative']['T']}")
          + f"; noul forms {nc['forms']}")
    tot_n = sum(s["n"] for s in stats.values()); tot_c = sum(s["correct"] for s in stats.values())
    for t, s in sorted(stats.items()):
        print(f"[offline] {t:6s} n={s['n']:3d} correct={s['correct']:3d} acc={s['correct']/s['n']:.3f}" + (f" abstain(0.2<P(yes)<0.8)={s['abstain']}" if t == "noul" else ""))
    print(f"[offline] all    n={tot_n} correct={tot_c} acc={tot_c/tot_n:.3f}  (raw accuracy; the I numbers in calibration.json are tier-weighted and chance-corrected)")
    if precision not in str(args.cache):
        print(f"[offline] NOTE: block {precision} applied to a cache whose name does not mention {precision}; cross-precision comparison")

    if not args.results:
        return 0
    n = agree = 0; dps = []; tok_mismatch = 0; missing = 0; failed = 0; worst = []; by_type = collections.defaultdict(lambda: [0, 0])
    for rec in res:
        if not rec.get("ok"):
            failed += 1; continue
        e = exp.get(rec["task_id"])
        if e is None:
            missing += 1; continue
        probs, dec, row, margin = e
        n += 1
        pr = rec.get("probs") or {}
        if row["type"] == "noul":
            py = pr.get("yes", 0.0)
            got = "yes" if py >= 0.8 else ("no" if py <= 0.2 else "abstain")
        else:
            got = rec.get("predicted")
        dp = max(abs(pr.get(k, 0.0) - probs[k]) for k in probs)
        dps.append(dp)
        by_type[row["type"]][1] += 1
        if got == dec:
            agree += 1; by_type[row["type"]][0] += 1
        else:
            worst.append((rec["task_id"], row["type"], got, dec, round(dp, 4), round(margin, 4), margin < args.tie_tol))
        if (rec.get("usage") or {}).get("input_tokens") != row.get("n_tokens"):
            tok_mismatch += 1
    print(f"[results] {len(res)} records: ok={len(res)-failed} failed={failed} unmatched={missing}; expected from block {precision}")
    print(f"[results] decision agreement {agree}/{n} = {agree/max(n,1):.4f}; by type " +
          ", ".join(f"{t} {a}/{m}" for t, (a, m) in sorted(by_type.items())))
    print(f"[results] |delta p| quantiles {quantiles(dps)}; prompt-length (input_tokens vs cache n_tokens) mismatches {tok_mismatch}")
    hard = [w for w in worst if not w[6]]
    if worst:
        print(f"[results] disagreements (task_id, type, harness, expected, max|dp|, cache margin, near-tie<{args.tie_tol}):")
        for w in worst[:40]:
            print("   ", w)
    print(f"[results] disagreements: {len(worst)} total, {len(worst) - len(hard)} near-tie, {len(hard)} not near-tie")
    return 0 if n and not hard else 5


if __name__ == "__main__":
    raise SystemExit(main())
