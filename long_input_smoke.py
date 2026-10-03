#!/usr/bin/env python
"""LONG-INPUT SMOKE for metask-jev-rain: one forward pass each on synthetic ~16k- and ~32k-token items.

Builds each item from a public long_policy item (its policy text repeated, numbered copies, until the
rendered prompt reaches the target token count), runs MetaskRainAdapter.run() once per item in the
precision given by METASK_RAIN_DTYPE (selftest.sh sets int8), and prints one JSON line per item:
ok, status, input_tokens, peak VRAM (torch max_memory_allocated / max_memory_reserved), forward seconds.
The synthetic items have no gold; correctness is not the point, only that the forward pass fits.

Usage (inside the patched harness copy, GPU lock held by the caller):
  python long_input_smoke.py --harness <patched harness root> --endpoint <model dir> [--targets 16000,32000]
Exit code: 0 if every item ran ok (status not 422, no OOM), 6 otherwise.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--targets", default="16000,32000")
    ap.add_argument("--source-id", default=None, help="public long_policy item to repeat (default: the longest hard long_policy choice item)")
    args = ap.parse_args()
    sys.path.insert(0, args.harness)
    import torch
    from jevbench.adapters.metask_rain import MetaskRainAdapter, render, messages_for
    from jevbench.tasks import Task

    rows = [json.loads(l) for l in open(Path(args.harness) / "datasets/public/hard.jsonl", encoding="utf-8") if l.strip()]
    cands = [r for r in rows if r.get("family") == "long_policy" and r["question"]["type"] == "choice"
             and (args.source_id is None or r["id"] == args.source_id)]
    if not cands:
        print("[long-smoke] no source item found"); return 6
    src = max(cands, key=lambda r: len(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])))
    state0 = src["state"] if isinstance(src["state"], str) else json.dumps(src["state"], ensure_ascii=False)

    ad = MetaskRainAdapter(endpoint=args.endpoint, max_input_tokens=65536)
    ad.load()
    tok = ad._loaded[1]
    print(f"[long-smoke] precision={ad.precision} selection={ad.precision_selection} attn={ad.attn_implementation} source={src['id']}")
    per_copy = len(tok.encode(state0, add_special_tokens=False))
    rc = 0
    for target in [int(t) for t in args.targets.split(",") if t.strip()]:
        k = max(1, target // max(per_copy, 1))
        while True:
            state = "\n\n".join(f"[Policy copy {i + 1} of {k}]\n{state0}" for i in range(k))
            t = Task(id=f"synthetic-long-{target}", family="long_policy", state=state,
                     question=copy.deepcopy(src["question"]), labels=list(src["labels"]),
                     expected=None, split="public", group=None)
            labels, text, _ = render(t)
            prompt = tok.apply_chat_template(messages_for(text), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            n = len(tok.encode(prompt, add_special_tokens=False))
            if n >= target or k > 1000:
                break
            k += 1
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(ad.device)
        r = ad.run(t)
        out = {"target": target, "copies": k, "input_tokens": (r.usage or {}).get("input_tokens", n), "ok": r.ok,
               "status": r.status, "error": r.error,
               "peak_allocated_gib": round(torch.cuda.max_memory_allocated(ad.device) / 2**30, 2),
               "peak_reserved_gib": round(torch.cuda.max_memory_reserved(ad.device) / 2**30, 2),
               "forward_s": ((r.raw or {}).get("runtime") or {}).get("forward_s"),
               "latency_s": r.latency_s, "precision": ad.precision,
               "argmax": max(r.probs, key=r.probs.get) if r.probs else None}
        print("[long-smoke] " + json.dumps(out), flush=True)
        if not r.ok:
            rc = 6
    print(f"[long-smoke] done rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
