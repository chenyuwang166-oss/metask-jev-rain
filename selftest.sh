#!/usr/bin/env bash
# metask-jev-rain self-test: run the upstream JevBench harness CLI over the 231 public items with the
# metask_rain adapter on a Linux CUDA box, then summarize and check parity with the frozen logits cache.
#
# What it does
#   0. re-executes itself from a private copy ($OUT/selftest.sh.used), so editing this file while a run
#      is waiting for the GPU can never change what that run executes (bash reads scripts by file offset);
#   1. makes a fresh patched COPY of the upstream harness under $WORK (the reference clone is never edited):
#      rsync upstream -> $WORK, strip CRs from the two registration files, apply registration.patch,
#      drop in jevbench/adapters/metask_rain.py and the calibration file; runs the upstream unit tests on
#      a PRISTINE copy and on the patched copy and prints both result lines side by side;
#   2. waits for the GPU (no ~/jev/gpu.lock and < 2000 MiB used), takes the lock atomically, releases it on exit;
#   3. python -m jevbench.cli run  --adapter metask_rain ... (231 items, raw/results/ledger/manifest under $OUT);
#      precision is whatever the adapter selects (bf16 with >= 27 GiB free VRAM, else int8; METASK_RAIN_DTYPE forces);
#   3b. LONG-INPUT SMOKE (still under the lock): long_input_smoke.py runs one forward pass each on two synthetic
#      items of ~16k and ~32k tokens (a public long_policy text repeated) in int8 (METASK_RAIN_DTYPE=int8 for this
#      step only) and reports ok/status, input tokens and peak VRAM; SKIP_LONG=1 skips it;
#   4. python -m jevbench.cli summarize ... --public-export $OUT/summary.json  (printed);
#   5. python parity_check.py against the frozen logits cache OF THE PRECISION THAT ACTUALLY RAN (read from
#      raw.runtime.precision in results.jsonl): int8 -> logits_231_verb_gemma12b_int8.jsonl, bf16 ->
#      logits_231_verb_gemma12b_bf16off.jsonl (numerically bf16, produced by a CPU-offload run on a 24 GB card;
#      the two caches differ on 6/231 decisions, so comparing across precisions would report false disagreements).
#      First offline (expected decisions regenerated from that cache with the CURRENT calibration block and mapping:
#      temperatures + one-bin noul), then per item against the harness results: decision agreement, |dp| quantiles,
#      near-tie classification of any disagreement; exit code 0 = every decision agrees or every disagreement is a
#      near-tie, 5 = otherwise.
#   The final "done" line always prints parity_rc and long_rc; the script exits non-zero if either is.
#
# Everything is logged to ~/jev/logs/submission_<ts>.log. Nothing under ~/jev/exp, ~/jev/runs, ~/jev/models is touched;
# WORK must lie under $JEV/submission (checked before the rm -rf that refreshes the harness copies).
#
# Usage (on the GPU box):   bash ~/jev/submission/jevbench-rain/selftest.sh
# Env overrides: JEV, UPSTREAM, MODEL, WORK, OUT, CACHE_INT8, CACHE_BF16 (default: $JEV/exp/<cache>, else $PKG/caches/<cache>),
# MAX_WAIT_MIN, METASK_RAIN_DTYPE (unset = auto: bf16 if >= 27 GiB free else int8), METASK_RAIN_NOUL (one_bin default |
# temperature), LIMIT (smoke: e.g. 5), SKIP_LONG=1, LONG_TARGETS (16000,32000)
set -euo pipefail

# ---------------------------------------------------------------- 0. run from a private copy
JEV="${JEV:-$HOME/jev}"
if [ -z "${METASK_RAIN_SELFTEST_COPY:-}" ]; then
  export PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
  export OUT="${OUT:-$JEV/submission/out/$TS}"
  mkdir -p "$OUT"
  cp "${BASH_SOURCE[0]}" "$OUT/selftest.sh.used"
  METASK_RAIN_SELFTEST_COPY=1 exec bash "$OUT/selftest.sh.used" "$@"
fi
PKG="${PKG:?}"; TS="${TS:?}"; OUT="${OUT:?}"
UPSTREAM="${UPSTREAM:-$JEV/repos/jevbench}"
MODEL="${MODEL:-$JEV/models/gemma-4-12b-it}"
WORK="${WORK:-$JEV/submission/harness}"; WORK="${WORK%/}"
case "$WORK" in "$JEV/submission/"?*) ;; *) echo "[selftest] WORK=$WORK must be a directory under $JEV/submission (it is rm -rf'ed below)"; exit 2;; esac
PRISTINE="$WORK.pristine"
pick_cache() { if [ -f "$JEV/exp/$1" ]; then echo "$JEV/exp/$1"; else echo "$PKG/caches/$1"; fi; }
CACHE_INT8="${CACHE_INT8:-$(pick_cache logits_231_verb_gemma12b_int8.jsonl)}"
CACHE_BF16="${CACHE_BF16:-$(pick_cache logits_231_verb_gemma12b_bf16off.jsonl)}"
LOCK="$JEV/gpu.lock"
LOCK_NAME="metask_rain_selftest_$TS"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-180}"
LOG="$JEV/logs/submission_$TS.log"
mkdir -p "$(dirname "$LOG")" "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "[selftest] $(date -Is) pkg=$PKG upstream=$UPSTREAM model=$MODEL work=$WORK out=$OUT log=$LOG"
echo "[selftest] executing private copy $OUT/selftest.sh.used (sha256 $(sha256sum "$OUT/selftest.sh.used" | cut -c1-16)...), source $PKG/selftest.sh (sha256 $(sha256sum "$PKG/selftest.sh" | cut -c1-16)...)"

if [ -f "$JEV/.venv/bin/activate" ]; then source "$JEV/.venv/bin/activate"; fi
python - <<'PY'
import sys, torch, transformers
print(f"[selftest] python {sys.version.split()[0]} torch {torch.__version__} transformers {transformers.__version__} cuda={torch.cuda.is_available()}")
try:
    import bitsandbytes; print(f"[selftest] bitsandbytes {bitsandbytes.__version__}")
except Exception as e:
    print(f"[selftest] bitsandbytes not importable ({e}); int8 (the precision selected below 27 GiB free VRAM) will NOT load - fix the venv")
import importlib.metadata as md
print("[selftest] versions: " + " ".join(f"{p}=={md.version(p)}" for p in ("torch", "transformers", "bitsandbytes", "accelerate", "tokenizers", "safetensors", "huggingface_hub", "numpy")))
PY
echo "[selftest] precision: METASK_RAIN_DTYPE=${METASK_RAIN_DTYPE:-<unset -> auto: bf16 if >= 27 GiB free VRAM, else int8>} noul mapping: METASK_RAIN_NOUL=${METASK_RAIN_NOUL:-<unset -> one_bin>}"

# ---------------------------------------------------------------- 1. patched harness copy (+ pristine copy for the unit-test baseline)
rm -rf "$WORK" "$PRISTINE"; mkdir -p "$WORK" "$PRISTINE"
rsync -a --exclude '.git' --exclude '__pycache__' "$UPSTREAM/" "$PRISTINE/"   # keep results/: some upstream tests read it
rsync -a --exclude '.git' --exclude '__pycache__' "$UPSTREAM/" "$WORK/"
cd "$WORK"
sed -i 's/\r$//' jevbench/cli.py jevbench/adapters/__init__.py   # the patch is LF; a CRLF checkout would not match
patch -p1 --forward < "$PKG/registration.patch"
cp "$PKG/jevbench/adapters/metask_rain.py" jevbench/adapters/metask_rain.py
cp "$PKG/calibration.json" jevbench/adapters/metask_rain_calibration.json
export METASK_RAIN_CALIBRATION="$WORK/jevbench/adapters/metask_rain_calibration.json"
python -c "import jevbench.adapters.metask_rain as m; print('[selftest] adapter import ok:', m.__file__)"
# upstream unit tests: the pristine clone fails some modules at import on a venv without pytest / without the
# historical result files; the patched copy must fail EXACTLY the same ones (none of them touches this adapter).
ut_line() { ( cd "$1" && python -m unittest discover -s tests -q 2>&1 | grep -E '^(Ran |OK|FAILED)' | sed -E 's/ in [0-9.]+s//' | tr '\n' ' ' ); }   # timing stripped: compare counts only
set +e
UT_PRISTINE="$(ut_line "$PRISTINE")"; UT_PATCHED="$(ut_line "$WORK")"
set -e
echo "[selftest] upstream unittest pristine: $UT_PRISTINE"
echo "[selftest] upstream unittest patched : $UT_PATCHED"
if [ "$UT_PRISTINE" = "$UT_PATCHED" ]; then echo "[selftest] upstream unittest: patched == pristine (OK)"; else echo "[selftest] upstream unittest: patched != pristine (INSPECT)"; fi
echo "[selftest] upstream file hashes:"; sha256sum jevbench/cli.py jevbench/adapters/__init__.py jevbench/runner.py jevbench/scoring.py jevbench/adapters/metask_rain.py jevbench/adapters/metask_rain_calibration.json

TASKS="$WORK/datasets/public/easy.jsonl,$WORK/datasets/public/original.jsonl,$WORK/datasets/public/hard.jsonl"
REV="$(python -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['revision'])" "$PKG/calibration.json")"

# ---------------------------------------------------------------- 2. GPU lock discipline
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
release() { if [ -e "$LOCK" ] && [ "$(cat "$LOCK" 2>/dev/null)" = "$LOCK_NAME" ]; then rm -f "$LOCK"; echo "[selftest] gpu.lock released"; fi; }
trap release EXIT     # installed BEFORE the lock is taken: no window in which an acquired lock could be orphaned
waited=0
while :; do
  used="$(gpu_used)"
  if [ ! -e "$LOCK" ] && [ "$used" -lt 2000 ]; then
    if ( set -o noclobber; echo "$LOCK_NAME" > "$LOCK" ) 2>/dev/null; then break; fi
  fi
  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then echo "[selftest] GPU busy for $MAX_WAIT_MIN min (lock=$(cat "$LOCK" 2>/dev/null || echo none), used=${used}MiB); giving up"; exit 4; fi
  echo "[selftest] $(date +%T) waiting for GPU (lock=$(cat "$LOCK" 2>/dev/null || echo none), used=${used}MiB)"; sleep 60; waited=$((waited+1))
done
echo "[selftest] gpu.lock taken as $LOCK_NAME"

# ---------------------------------------------------------------- 3. run (231 public items)
LIMIT_ARGS=(); if [ -n "${LIMIT:-}" ]; then LIMIT_ARGS=(--limit "$LIMIT"); fi
export JEVBENCH_WARM_LOAD=1     # load the weights before the clock (the prepare() hook does the same on the current runner)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # the run must work with no network: weights come from the local $MODEL dir
RUN_CMD=(python -m jevbench.cli run
  --tasks "$TASKS" --adapter metask_rain
  --endpoint "$MODEL" --model google/gemma-4-12B-it --revision "$REV" --key-env ''
  --results "$OUT/results.jsonl" --raw-dir "$OUT/raw" --ledger "$OUT/ledger.jsonl"
  --reserve-usd 0 --cap-usd 1 --cost-basis self_hosted_gpu
  --run-label metask-jev-rain --manifest "$OUT/manifest.json" "${LIMIT_ARGS[@]}")
echo "[selftest] run command:"; printf '  %q' "${RUN_CMD[@]}"; echo
"${RUN_CMD[@]}"

# ---------------------------------------------------------------- 3b. LONG-INPUT SMOKE (int8, lock still held)
LONG_RC=skipped
if [ -z "${SKIP_LONG:-}" ]; then
  set +e
  METASK_RAIN_DTYPE=int8 python "$PKG/long_input_smoke.py" --harness "$WORK" --endpoint "$MODEL" --targets "${LONG_TARGETS:-16000,32000}" | tee "$OUT/long_input_smoke.txt"
  LONG_RC=${PIPESTATUS[0]}
  set -e
fi
echo "[selftest] long-input smoke rc=$LONG_RC (0 = both forwards ok, 6 = a failure/OOM, skipped = SKIP_LONG)"
release; trap - EXIT

# ---------------------------------------------------------------- 4. summarize
SUM_CMD=(python -m jevbench.cli summarize --tasks "$TASKS" --results "$OUT/results.jsonl" --ledger "$OUT/ledger.jsonl" --public-export "$OUT/summary.json")
echo "[selftest] summarize command:"; printf '  %q' "${SUM_CMD[@]}"; echo
"${SUM_CMD[@]}" | tee "$OUT/summary.txt"
python - "$OUT/results.jsonl" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
ok = sum(r["ok"] for r in rows); c = sum(bool(r.get("correct")) for r in rows)
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    k = (r.get("runtime") or {}).get("temperature_applied"); by[k][0] += bool(r.get("correct")); by[k][1] += 1
lat = sorted(r["latency_s"] for r in rows if r["ok"])
rt = (rows[0].get("runtime") or {}) if rows else {}
print(f"[selftest] items={len(rows)} ok={ok} correct={c} acc={c/len(rows):.3f} 422s={sum(r.get('status_code')==422 for r in rows)}")
print("[selftest] correct/n by temperature_applied (None = noul one-bin):", dict(by))
if lat: print(f"[selftest] latency_s median {lat[len(lat)//2]:.3f} p90 {lat[int(len(lat)*0.9)]:.3f} max {lat[-1]:.3f}")
print(f"[selftest] precision={rt.get('precision')} selection={rt.get('precision_selection')} calibration_block={rt.get('calibration_block')} temperatures={rt.get('temperature_by_kind')} noul_mapping={rt.get('noul_mapping')} p_cal={rt.get('noul_p_cal')} attn={rt.get('attn_implementation')}")
print(f"[selftest] revision_verified={rt.get('revision_verified')} enable_thinking_kwarg_accepted={rt.get('enable_thinking_kwarg_accepted')} transformers_tested={rt.get('transformers_tested')} calibration_sha256={rt.get('calibration_sha256')}")
nb = [r for r in rows if r.get('ok') and isinstance(r.get('probs'), dict) and set(r['probs']) == {'no', 'yes'}]
ab = sum(0.2 < r['probs']['yes'] < 0.8 for r in nb)
print(f"[selftest] noul items {len(nb)}: abstentions (0.2 < P(yes) < 0.8) {ab}; distinct P(yes) values {sorted({round(r['probs']['yes'], 4) for r in nb})[:6]}")
nf = sorted({l for r in rows for l in ((r.get('runtime') or {}).get('labels_without_single_token_form') or [])})
print(f"[selftest] labels without a single-token form (log-mass -1e9 rule): {nf or 'none'}")
PY

# ---------------------------------------------------------------- 5. parity with the frozen logits cache OF THE PRECISION THAT RAN (never aborts the summary line)
# precision = raw.runtime.precision of the first ok row of results.jsonl
PREC="$(python -c "import json,sys; rows=[json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]; ok=[r for r in rows if r.get('ok')]; print(((ok[0].get('runtime') or {}).get('precision') or '') if ok else '')" "$OUT/results.jsonl")"
case "$PREC" in
  int8) CACHE="$CACHE_INT8";;
  bf16) CACHE="$CACHE_BF16";;
  *) CACHE=""; echo "[selftest] no ok row with raw.runtime.precision in results.jsonl (got '$PREC'); parity check skipped";;
esac
PARITY_RC=none
if [ -n "$CACHE" ] && [ -f "$CACHE" ]; then
  echo "[selftest] parity: precision that ran = $PREC -> cache $CACHE (sha256 $(sha256sum "$CACHE" | cut -c1-16)...)"
  set +e
  python "$PKG/parity_check.py" --cache "$CACHE" --calibration "$PKG/calibration.json" --harness "$WORK" --precision "$PREC" | tee "$OUT/parity_offline_$PREC.txt"
  python "$PKG/parity_check.py" --results "$OUT/results.jsonl" --cache "$CACHE" --calibration "$PKG/calibration.json" --harness "$WORK" --precision "$PREC" | tee "$OUT/parity.txt"
  PARITY_RC=${PIPESTATUS[0]}
  set -e
elif [ -n "$CACHE" ]; then
  echo "[selftest] cache $CACHE not found (set CACHE_INT8 / CACHE_BF16); parity check skipped"
fi
echo "[selftest] done: parity_rc=$PARITY_RC (0 = decisions agree with the frozen cache up to near-ties, 5 = disagreement, none = skipped) long_rc=$LONG_RC results=$OUT/results.jsonl summary=$OUT/summary.json manifest=$OUT/manifest.json parity=$OUT/parity.txt long=$OUT/long_input_smoke.txt log=$LOG"
RC=0
if [ "$PARITY_RC" != none ] && [ "$PARITY_RC" != 0 ]; then RC="$PARITY_RC"; fi
if [ "$LONG_RC" != skipped ] && [ "$LONG_RC" != 0 ] && [ "$RC" = 0 ]; then RC="$LONG_RC"; fi
exit "$RC"
