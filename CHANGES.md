# CHANGES — metask-jev-rain package

## Round 1 review (2026-10-02) — findings and what was done

Severity as reported by the reviewer. "Status" is what this revision did.

| # | Severity | File | Finding (short) | Status / change |
|---|---|---|---|---|
| 1 | major | `selftest.sh` | A queued selftest executes a file that was later overwritten in place; bash reads scripts by offset, so after the GPU-wait loop it would resume at a shifted offset. Its harness copy also lacked `results/` (10 unittest errors vs 6 pristine). | **Fixed in the script**: step 0 copies itself to `$OUT/selftest.sh.used` and `exec`s that copy (guarded by `METASK_RAIN_SELFTEST_COPY`), exporting `PKG`/`TS`/`OUT` so the copy uses the package dir and the same output dir; later edits of `selftest.sh` can no longer touch a running instance. The harness copy keeps `results/` (some upstream tests read it). The self-test was relaunched and completed (see "Self-test (2026-10-03)" below). |
| 2 | major | `metask_rain.py` L205 (and duplicate finding L205) | Precision auto-selected from free VRAM (bf16 ≥ 30 GiB), but only int8 was ever run; an A100/H100 evaluator would silently get an unvalidated configuration. | **Fixed**: `_pick_precision()` returns `DEFAULT_PRECISION = "int8"` unless `METASK_RAIN_DTYPE=bf16` is set explicitly (nf4 refusal kept; no `mem_get_info` call). `raw.runtime` gains `precision_validated` (true only for int8). `calibration.json`: `precision_policy` rewritten, `precision_declared: "int8"`. README §1 (new Precision row), §5 knobs, §6 table (int8 first, bf16 "opt-in only / untested / never run"), §10 item 5. selftest prints the precision in effect. **Superseded in Round 3** (bf16 primary / int8 fallback by free VRAM, two calibration blocks). |
| 3 | minor | `README` L128 | §5 run snippet broken by an inline `# comment` without trailing `\`. | **Fixed**: comment moved above the command, line ends with ` \`. |
| 4 | minor | `metask_rain.py` L339 | `raw.runtime.enable_thinking: False` recorded unconditionally even on the TypeError fallback. | **Fixed**: `build_prompt()` sets `self._enable_thinking_kwarg_ok` True/False per branch; `raw.runtime.enable_thinking_kwarg_accepted` records it next to `enable_thinking: False`. selftest prints it. README §2 updated. |
| 5 | minor | `metask_rain.py` L221 | Local `--endpoint` dir: pinned revision neither passed nor verified, yet reported. | **Fixed**: `calibration.json` gains `config_sha256` (`478c46e8d2c5…e936bd9`, measured on the server copy), `config_bytes` 4423, `model_safetensors_bytes` 23919549408. `load()` with a local dir calls `_verify_local_revision()`: sha256 of `<dir>/config.json` vs the pin, plus single-file `model.safetensors` size when present; result recorded as `raw.runtime.revision_verified` + `revision_check` {expected/local hashes, method}. Mismatch → warning + `revision_verified: false`; `METASK_RAIN_STRICT_REVISION=1` raises. HF-id path records `revision_verified: true`, method "hf-hub revision= kwarg". README §5 and `revision_note` describe it. |
| 6 | minor | `selftest.sh` L56 | Unittest comment asserts parity with pristine without measuring; parity rc 5 aborts the script before the "done" line. | **Fixed**: the script rsyncs a second, PRISTINE copy (`$WORK.pristine`), runs `unittest discover` on both and prints the two `Ran …/FAILED (errors=N)` lines side by side with an `OK`/`INSPECT` verdict (pristine baseline measured this round on the reference clone: `Ran 35 tests … FAILED (errors=6)`). Parity step wrapped in `set +e`, rc captured from `PIPESTATUS[0]`, the final `done:` line always prints and carries `parity_rc=`; the script exits with that rc. |
| 7 | major | `metask_rain.py` L262 | A label with no single-token form raised ValueError → `ok=False` without status → counts toward the runner's 3-consecutive-error stop; the cache wrote −1e9 instead. | **Fixed (mirrors the cache)**: `_effective_groups()` keeps an empty list for such labels (raises only if *every* label is unreadable); `aggregate()` returns `NO_FORM_LOGMASS = -1e9` for an empty form list, so `tempered_softmax` gives P = 0 and the distribution stays valid. `raw.runtime.labels_without_single_token_form` lists them; selftest prints the union. README §3 paragraph + `calibration.json: readout_no_single_token_form`. `parity_check.py` needs no change (its `groups` may already be empty lists and now flow through the same code). Unit-checked locally: 11 score labels → label "10" gets P = 0, sum = 1, argmax unchanged. |
| 8 | major | `selftest.sh` L62 | Same as #1 (stale running instance; shifted offset after the wait loop). | See #1. |
| 9 | minor | `README` L126 | "transformers ≥ 4.5x" too loose: `dtype=` kwarg needs ≥ 4.56, else fp32 loads silently. | **Fixed**: README §5 states ≥ 4.56 (tested 5.17.0); `load()` enforces `MIN_TRANSFORMERS = (4, 56)` with a clear RuntimeError rather than passing both `dtype` and `torch_dtype` (transformers 5.x treats `torch_dtype` as a deprecated alias; passing both is not safe across versions). `calibration.json: min_transformers`. **Superseded in Round 3 / corrected in Round 4** (gate on model_type `gemma4_unified`, ≥ 5.10.0). |
| 10 | minor | `README` L42 | "`<bos>` is the only special token added" is inaccurate. | **Fixed**: reworded as suggested (tokenizer adds nothing; the template writes `<bos>`, `<\|turn>`/`<turn\|>`, and the empty thought channel). |
| 11 | minor | `metask_rain.py` L112 | float64 vs float32 logsumexp: no code change, document the residual. | **Fixed (docs)**: README §8 item 1 names the residual sources (float64 adapter vs float32 cache, cache's 4-decimal rounding). |

### Re-verification done this round (local, Python 3.12)

* `py_compile` of `jevbench/adapters/metask_rain.py` and `parity_check.py`: OK.
* `bash -n selftest.sh`: OK.
* `patch -p1 --dry-run < registration.patch` against LF-normalized pristine copies of upstream `cli.py` / `adapters/__init__.py` (bb05a33): both hunks apply cleanly.
* `calibration.json` parses; `load_calibration()` drift guard passes.
* Offline parity (`parity_check.py --cache <int8 cache> --calibration calibration.json --harness <upstream clone> --adapter-file …`): 197/231 (choice 120/139, noul 63/74, abstain 0, score 14/18), no log-mass warnings — unchanged by this revision.

### Self-test (2026-10-03)

* `selftest.sh` run end-to-end on one RTX 4090 (int8): 231/231 ok, 197/231 correct, 0 × 422, precision=int8,
  revision_verified=True, parity_rc=0 (decision agreement with the frozen int8 cache 231/231, max |Δp| 0.0000);
  upstream unit tests `Ran 35 tests … FAILED (errors=6)` on the pristine and on the patched copy (identical).
  p50 0.264 s / p95 0.616 s raw.

## Round 2 (2026-10-03) — documentation audit and frozen-state decisions

* `README-metask-rain.md` rewritten against the 29-finding consistency audit: Prior-work section (NInfer → Cygnet),
  precision policy bf16 primary / int8 fallback with two calibration blocks, noul one-bin histogram calibration
  disclosed plainly, score temperature back to the pre-registered dev pick 5.0 (check-half preference for 4.0
  disclosed), noul form-set check disclosed, environment pins, offline use, context limit 65536, all composite/rank
  numbers removed, run command covers all three public files.

## Round 3 (2026-10-03): per-precision calibration, one-bin noul, environment pins

Supersedes the Round 1 precision (#2) and transformers-gate (#9) decisions.

| File | Change |
|---|---|
| `calibration.json` | Restructured: two blocks `"bf16"` and `"int8"`, each fitted by the pre-registered rule of `calib_v2.py` on logits of its own precision (`fitted_on_cache` + sha256 per block). Both: T_choice 4.5, T_score 5.0, noul one-bin histogram calibration (int8 `p_cal` 0.8919, bf16 0.8649), `noul_alternative` T 0.05 kept for comparison. Top-level `temperature_by_kind`, `score_temperature_alternative`, `local_estimates_231` (incl. the composite estimate) removed. New: `precision_policy`, `precision_auto_threshold_gib` 27, `attn_implementation` "sdpa", `noul_mapping_note`, `score_temperature_note` (pre-registered dev pick 5.0 kept; the check half was looked at once on int8 and preferred 4.0; the previous revision had shipped 4.0), `nf4_note` (with its source files), `prior_work`, `max_input_tokens` 65536, `min_transformers` 5.10.1 (corrected to 5.10.0 in Round 4). `author_org` set to the agreed display name. |
| `metask_rain.py` | Precision: `METASK_RAIN_DTYPE=bf16\|int8` forces; unset = bf16 if `torch.cuda.mem_get_info` reports >= 27 GiB free, else int8; nf4/4-bit refused. The calibration block of the loaded precision is applied. noul: one-bin mapping (`P(yes) = p_cal` if `d > 0` else `1 - p_cal`, `d == 0` -> no); `METASK_RAIN_NOUL=temperature` selects the block's `noul_alternative`. `p_cal` must be in [0.8, 1) (validated at load). Pure helpers `noul_d`, `noul_one_bin`, `readout_probs` (shared with `parity_check.py`). transformers gate: model_type `gemma4_unified` must be known and version >= 5.10.1 (corrected to 5.10.0 in Round 4: the v5.10.0 tag already contains `models/gemma4_unified`); a release other than 5.17.0 warns and records `transformers_tested: false`. Import fallback (`from .base` -> `from jevbench.adapters.base`). First load exception cached; later items fail fast with it. `attn_implementation="sdpa"` pinned and the effective one recorded. Both precisions load with `device_map={"": index}`. `raw.runtime` adds `precision_selection`, `calibration_block`, `calibration_fitted_on_cache(_sha256)`, `noul_mapping`, `noul_p_cal`, `noul_alternative_T`, `mapping_applied`, `attn_implementation`, `transformers_tested`, `bitsandbytes`, `calibration_sha256`, per-type `probability_origin`; `raw.answer.d` for noul. `max_input_tokens` default 65536. Prompt, read-out, usage, `prepare()` and 422 semantics unchanged. |
| `parity_check.py` | Uses the adapter's own `load_calibration` / `readout_probs`; `--precision` (default: from the results' `raw.runtime.precision`, else int8), `--noul-mode`. Expected noul decision uses the harness thresholds (0.8 / 0.2 / abstain). Results mode reports per-type agreement, \|dp\| quantiles (p50/p90/p99/max), and classifies each disagreement as near-tie (cache margin < `--tie-tol` 0.1) or not; rc 5 only for a non-near-tie disagreement. |
| `long_input_smoke.py` | New: one forward each on synthetic items of ~16k and ~32k tokens (the longest public long_policy choice item repeated), reporting ok/status, input tokens, peak VRAM, forward seconds. |
| `selftest.sh` | Prints all package versions; runs with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; LONG-INPUT SMOKE (int8, under the same GPU lock) after the 231-item run; offline parity regenerated from the int8 cache with the current block, then results parity; final line carries `parity_rc` and `long_rc`. Unittest comparison now strips the timing ("in 0.011s") that made identical lines compare unequal. |
| `requirements-metask-rain.txt` | New: exact pins read from the test venv (`pip freeze`), with the CUDA/driver note. |
| `README-metask-rain.md` | Package layout lists the two new files; knob `METASK_RAIN_NOUL`; section "Environment and offline use"; long-input smoke result. |

### Verification (this revision)

* Local (Python 3.12): `py_compile` of `metask_rain.py`, `parity_check.py`, `long_input_smoke.py`; `bash -n selftest.sh`;
  `load_calibration()` passes on the new file; unit checks of `readout_probs` (one-bin, temperature alternative,
  score with a -1e9 label), standalone import fallback, precision selection with mocked free VRAM (30 GiB -> bf16,
  22 GiB -> int8, env nf4 refused, env bf16 honoured), version gate (5.10.0 refused, 5.10.1 accepted — the 5.10.0 refusal was wrong, see Round 4).
* Server, offline (CPU): int8 cache with the int8 block 197/231 (choice 120/139, noul 63/74 with 0 abstentions,
  score 14/18); bf16 cache with the bf16 block 195/231 (120 / 61 / 14); same decisions under the noul temperature
  alternative; no log-mass or form-set sign warnings.
* Server, GPU (`selftest.sh`, RTX 4090, 2026-10-03 20:02, out dir `out/20261003_200208`): precision auto-selected
  int8 (23.1 GiB free < 27), block int8, attn sdpa, transformers_tested true, revision_verified true; 231/231 ok,
  197/231 correct, 0 × 422; noul P(yes) takes only the values 0.1081 / 0.8919, 0 abstentions; parity vs the int8
  cache: decisions 231/231 (choice 139/139, noul 74/74, score 18/18), \|dp\| max 1e-6, 0 prompt-length mismatches,
  `parity_rc=0`; harness latency p50 0.273 s / p95 0.620 s; harness v1 top-label ECE 0.0608. LONG-INPUT SMOKE (int8):
  18,504 tokens ok, peak 16.2 GiB allocated, 8.7 s; 33,056 tokens ok, peak 20.4 GiB allocated / 22.3 GiB reserved,
  20.7 s; `long_rc=0`. Upstream unittest lines identical apart from timing (`Ran 35 tests … FAILED (errors=6)` on
  both); the script's verdict printed INSPECT only because of the timing, fixed above.

## Round 4 (2026-10-03): revision round after the frozen-package run — 21 findings (4 major on the package, 2 major on the docs, 15 minor)

Decisions of the brief were not reopened (bf16 primary / int8 fallback; one-bin noul; score T 5.0 pre-registered; `max_input_tokens` 65536; no composite numbers).

| Severity | Finding (short) | Status / change |
|---|---|---|
| major | `MIN_TRANSFORMERS = (5, 10, 1)` refuses v5.10.0, which already contains `models/gemma4_unified`; three files asserted "first tagged in v5.10.1". | **Fixed**: `MIN_TRANSFORMERS = (5, 10, 0)`; comment, `calibration.json: min_transformers`, `requirements-metask-rain.txt`, README §5, ISSUE §3, DISCLOSURE C7 all say "5.10.0 (first tag containing models/gemma4_unified; 5.10.0 and 5.10.1 both published on PyPI 2026-06-03; tested 5.17.0 only)". Re-verified this round: GitHub contents API for `src/transformers/models/gemma4_unified` → 200 at `v5.10.0`, 404 at `v5.9.1`; PyPI upload times 5.10.0 15:16:02 / 5.10.1 15:37:00 UTC. The unverifiable "commit 1423d22" reference was dropped. |
| major | `selftest.sh` parity hard-wired to the int8 cache and `--precision int8`; a bf16 run would report 6/231 false disagreements and exit 5. | **Fixed**: after the run the script reads `raw.runtime.precision` from the first ok row of `results.jsonl` and selects the cache (int8 → `logits_231_verb_gemma12b_int8.jsonl`, bf16 → `logits_231_verb_gemma12b_bf16off.jsonl`; `CACHE_INT8` / `CACHE_BF16` override; default location `$JEV/exp/`, falling back to `$PKG/caches/`); both parity lines get `--precision "$PREC"`; offline output is `parity_offline_<prec>.txt`. Header documents that the bf16 cache is numerically bf16 from a CPU-offload run. |
| major | Every item tagged `probs_source="native"` although noul under the one-bin mapping reports the constant `p_cal`; `summary.json` shows `probability_sources`. | **Fixed (mapping unchanged)**: `probs_source` is set per item — `"native"` for choice / score (and for noul under `METASK_RAIN_NOUL=temperature`), `"native_decision_one_bin_calibrated"` for noul under one_bin; also recorded in `raw.runtime.probs_source`. `calibration.json` gains a `probs_source` object; README §1 / §10, ISSUE §1 / §10 Q1, DISCLOSURE B4 say that `summary.json` will list two values. The runner treats any value other than `label_only_no_calibrated_distribution` identically, so scoring is unaffected. |
| major | README / ISSUE / DISCLOSURE listed `docs/split_231_v2.json`, `docs/calib_v2_report.md` and `caches/` as package contents; none existed in the package. | **Fixed**: `caches/logits_231_verb_gemma12b_{int8,bf16off}.jsonl` copied from the server (sha256 verified after copy: `77df7e47…0096e`, `4a1811e2…48a21`) with their `.meta.json` (the test box's model directory in `model` replaced by its basename + pinned revision; a `note` says so); `docs/split_231_v2.json` copied verbatim (check half under the key `local_sealed`, explained in README §4, ISSUE §3 and the report); `docs/calib_v2_report.md` is an edited copy of `exp/calib_v2_report.md`: a header note on the `local_sealed` key, every free-text `local_sealed` → "check half (key local_sealed)", the sentence "Composite effect was estimated immaterial (< 0.5 point)." deleted, the `Files:` footer with server paths replaced by "Full dev curves: calib_v2_results.json (available on request)". `__pycache__` directories removed. `selftest.sh` falls back to `$PKG/caches/` when `$JEV/exp` has no cache. |
| major | `<filled at freeze>` placeholders still open; texts presented the superseded 09:07 UTC run (p50 0.264 / p95 0.616, ECE 0.0703) as the self-test. | **Fixed**: ISSUE §4 (context smoke, latency), §5 harness row, DISCLOSURE C4 / C6 / D / E6 / F, README §7 / §8 now carry the frozen run `out/20261003_200208` (12:02 UTC: 231/231 ok, 197/231, 0 × 422, parity 231/231 with max \|Δp\| 1e-6, noul P(yes) ∈ {0.8919, 0.1081}, p50 0.273 s / p95 0.620 s / max 1.25 s, harness v1 ECE 0.061; smoke 18,504 tokens 16.2/17.4 GiB 8.7 s, 33,056 tokens 20.4/22.3 GiB 20.7 s), with the earlier run quoted as such. Its outputs (`summary.json`, `manifest.json`, `parity.txt`, `parity_offline_int8.txt`, `long_input_smoke.txt`, log tail) copied to `exp/submission/validate/20261003_200208_frozen/`. `calibration.json` SHA-256: that run executed `43f32dfa…52ca0`; this round's text-only edits (min_transformers, context note, probs_source, 26-option note) change it to `4682cae4d181e53abcaf67bc7a9b48b4650be95eed7ef2ca79aff658cfe64aa4` — every numeric block identical; DISCLOSURE D states both. Package commit stays `<filled when the repository is published>`. |
| minor | Shipped `selftest.sh` not executed end-to-end since the timing-strip sed. | **Re-run this round** with `LIMIT=5 SKIP_LONG=1` on the final script and adapter — result below under "Verification". |
| minor | `raw.runtime.model_path` / `calibration_path` leaked `/home/<user>/…` into `results.jsonl` and the manifest. | **Fixed** in `_runtime()`: a local model dir is recorded by basename, the calibration file by basename. The manifest `endpoint` field is written by the harness's `cli.py` from `--endpoint` and is outside the adapter; `validate/` copies are not shipped. |
| minor | `max_input_tokens` 65536 declared, only 33k run; 24 GB int8 may OOM between ~35k and 65k. | **Not changed (decision of the brief: 65536, no 422 below it)**; a 65k bf16 smoke is impossible on our 24 GB card. Disclosed instead: `calibration.json: max_input_tokens_note`, README §7 / §10, ISSUE §4 and DISCLOSURE C6 say that prompts between 33k and 65,536 tokens were not tried on the 24 GB card, may OOM there as an ordinary failed item (not a 422), and that bf16 has not been run at any length. OOM is not mapped to 422. |
| minor | The adapter's bf16 branch has never executed; the bf16 block was fitted on a CPU-offload run. | **Disclosed** in README §1 (Precision row), §5, §8 item 4; ISSUE §1 (next to "bf16 primary"); DISCLOSURE C2 — with the request that the maintainers run `parity_check.py` against the bf16off cache after their first bf16 run (expected 231/231 up to near-ties). |
| minor | `rm -rf "$WORK"` without checking WORK lies under `$JEV/submission`; `trap release EXIT` installed after the lock was taken. | **Fixed**: trailing slash stripped; `case "$WORK" in "$JEV/submission/"?*) ;; *) exit 2` before anything is removed; `release()` defined and `trap release EXIT` installed before the lock loop. |
| minor | A choice item with > 26 options was silently truncated by `zip`, then `KeyError`. | **Fixed**: `render()` raises `ValueError("choice with N options exceeds the 26-letter read-out (A..Z)")` (`MAX_CHOICE_OPTIONS = 26`); `calibration.json: choice_max_options(_note)`; DISCLOSURE B4. Behaviour on ≤ 26 options unchanged. |
| minor | `prepare()` hides the weight load from the clock; upstream documents the hook as one adapter's. | **Clarified and narrowed**: `prepare()` is a no-op when `JEVBENCH_WARM_LOAD=1` is set (the harness's own warm load then applies), otherwise it does the one-time idempotent `load()`; README §1 (Weight load row) and ISSUE §3 state it, including that the runner records the first item's `prepare_s`. |
| major (docs) | `calib_v2_report.md` as-is would reintroduce "sealed" wording, a composite remark and server paths. | **Fixed** by the edited shipped copy described above; README §4 sentence on the `local_sealed` key added; DISCLOSURE F item ticked. |
| minor | Cygnet README citation "lines 154–156" off by one. | **Fixed** → "lines 153–155" in ISSUE §2, README §0, DISCLOSURE A9. |
| minor | DISCLOSURE header METHOD-v1.5.md SHA-256 is the LF-form hash. | **Fixed**: header says the hash is the one listed in `docs/METHOD-v1.5-SHA256SUMS.txt` (LF form) and gives the CRLF checkout hash `80b41675…31fc6`. |
| minor | ISSUE §3 "no peft" (peft 0.21.0 is installed) and "CUDA-12.x wheel accepted" (never run). | **Fixed**: ISSUE §3 "peft is installed in that venv (0.21.0) but neither imported nor needed … A CUDA-12.x build of the same torch should work but has not been run by us; cu130 (driver ≥ 580) is the only tested combination"; DISCLOSURE C7 "peft not used"; README §5 and requirements comment aligned. |
| minor | Prompt-history wording "before any result existed" stronger than the record. | **Fixed**: ISSUE §1 and README §1 say "before any verbalized(-prompt) result existed for any model". |
| minor | CHANGES Round 3 opened with a dangling "Still pending" reference; Round 1 rows #2 / #9 contradicted Round 3 without a pointer. | **Fixed** above: sentence replaced by "Supersedes the Round 1 precision (#2) and transformers-gate (#9) decisions."; rows #2 and #9 carry "Superseded in Round 3" notes; the Round 3 5.10.1 mentions point to this round. |
| minor | ISSUE §3 package list omitted `long_input_smoke.py` and `requirements-metask-rain.txt`; repo URL presented as existing. | **Fixed**: list completed (also `NOTICE.md`, `CHANGES.md`, `docs/`, `caches/`); DISCLOSURE A1 says the repository is created after freeze and F carries a "posting gate" item (repo public + commit hash filled before the issue is opened). |
| minor | LICENCE-ELIGIBILITY recommended a `NOTICE.md` that did not exist; header date stale. | **Fixed**: `jevbench-rain/NOTICE.md` added (Apache-2.0 and Prohibited Use Policy URLs, no weights redistributed, Apache §6 origin statement); README §9 and the §5 layout reference it; LICENCE-ELIGIBILITY header "Checked 2026-10-02, updated 2026-10-03" and the §4 checklist item marked done. |

### Verification (Round 4)

* Local (Python 3.12): `py_compile` of `metask_rain.py`, `parity_check.py`, `long_input_smoke.py`; `bash -n selftest.sh`;
  `patch -p1 --dry-run < registration.patch` against LF-normalised pristine `cli.py` / `adapters/__init__.py` (bb05a33): both
  hunks apply. Unit checks: version tuple gate (5.9.1 refused, 5.10.0 and 5.17.0 pass); `render()` accepts 26 choice options
  and raises the explicit error at 27; `prepare()` is a no-op under `JEVBENCH_WARM_LOAD=1` and calls `load()` otherwise;
  `probs_source` per item is `native_decision_one_bin_calibrated` (noul, one_bin), `native` (choice), `native` (noul under
  `METASK_RAIN_NOUL=temperature`); `_runtime()` records `model_path` / `calibration_path` by basename (HF id kept as is);
  `readout_probs` unchanged (one-bin 0.8919 / 0.1081). Offline parity with the shipped caches and the shipped adapter:
  int8 197/231, bf16 195/231, no log-mass or form-set warnings (unchanged). `calibration.json` parses and passes
  `load_calibration()`; sha256 of the shipped caches re-verified after the copy.
* transformers gate fact re-checked live: GitHub contents API `src/transformers/models/gemma4_unified?ref=v5.10.0` → 200,
  `?ref=v5.9.1` → 404; PyPI 5.10.0 uploaded 2026-06-03 15:16:02, 5.10.1 15:37:00 UTC.
* Server, `selftest.sh` guard: `WORK=$HOME/jev/exp bash selftest.sh` exits 2 with the WORK message before any removal
  (`~/jev/exp` untouched).
* Server, GPU (`selftest.sh` as shipped, sha256 `e7aad1d1…`, `LIMIT=5 SKIP_LONG=1`, RTX 4090, 2026-10-03 20:32, out dir
  `out/20261003_203229`): upstream unittest `Ran 35 tests FAILED (errors=6)` on pristine and patched → **"patched ==
  pristine (OK)"** (the timing-strip fix verified in place); gpu.lock taken and released; precision auto-selected int8
  (23.1 GiB free < 27), block int8, sdpa, revision_verified true, transformers_tested true, `calibration_sha256`
  `4682cae4d181e53abcaf67bc7a9b48b4650be95eed7ef2ca79aff658cfe64aa4` (the shipped file); 5/5 ok, 5/5 correct, 0 × 422;
  parity step picked the int8 cache from `raw.runtime.precision`, decision agreement 5/5, |Δp| 0, 0 prompt-length
  mismatches, `parity_rc=0`, `long_rc=skipped`, script exit 0. `results.jsonl` contains no `/home/` path
  (`runtime.model_path` = `gemma-4-12b-it`, `calibration_path` = `metask_rain_calibration.json`); `probs_source`
  of the 5 (choice) items `native`, `summary.json: probability_sources ["native"]` (the two-valued listing appears on a
  run that includes noul items). `manifest.json: endpoint` still echoes `--endpoint` as given (harness `cli.py`, outside
  the adapter) — scrub it if a manifest is ever shipped; the `validate/` copies are not shipped.
* The full 231-item run with LONG-INPUT SMOKE was not repeated this round: the adapter changes do not touch the prompt,
  forward pass or read-out math (offline parity unchanged), and the previous full run (`out/20261003_200208`) remains the
  record quoted in the ISSUE / DISCLOSURE / README; its `calibration.json` differed from the shipped one in text fields only.

## Freeze run (2026-10-03 20:38 CST, validator; no code or calibration change)

* Package re-copied to the GPU box (`~/jev/submission/jevbench-rain`, previous copy kept as `jevbench-rain.prev`); `diff -rq`
  against the previous copy: identical (the Round 4 package was already in place). Every file hash equal on both sides.
* `selftest.sh` as shipped (sha256 `e7aad1d1…`), no env overrides, RTX 4090, out dir `out/20261003_freeze_203822`, log
  `~/jev/logs/submission_freeze.log` (= `submission_20261003_freeze_203822.log`, identical): upstream unittest
  `Ran 35 tests FAILED (errors=6)` on pristine and patched → "patched == pristine (OK)"; gpu.lock taken 20:38:22 CST and
  released; precision auto-selected int8 (23.1 GiB free < 27), block int8, sdpa, `revision_verified` true,
  `transformers_tested` true, `calibration_sha256` `4682cae4d181e53abcaf67bc7a9b48b4650be95eed7ef2ca79aff658cfe64aa4`
  (the shipped file). 231/231 ok, 197/231 correct (choice 120/139, noul 63/74, score 14/18; easy 48/48, original 70/72,
  hard 79/111), 0 × 422; noul P(yes) ∈ {0.1081, 0.8919}, 0 abstentions; `probs_source` native 157 /
  native_decision_one_bin_calibrated 74; harness latency p50 0.275 s / p95 0.621 s / max 1.07 s (forward p50 0.273 s);
  harness v1 top-label ECE 0.0608; Brier 0.218; paraphrase agreement 34/36; input tokens 157,046 total (max 3,942),
  output 0, charged $0. Parity vs the int8 cache: decisions 231/231 (choice 139/139, noul 74/74, score 18/18), |Δp| max
  1e-6, 0 prompt-length mismatches, `parity_rc=0`. LONG-INPUT SMOKE (int8): 18,504 tokens ok, peak 16.22 GiB allocated /
  17.40 GiB reserved, 8.7 s; 33,056 tokens ok, 20.35 / 22.32 GiB, 20.7 s; `long_rc=0`; script exit 0. `results.jsonl`
  contains no `/home/` path. Numbers identical to the 12:02 UTC run except latency noise (p50 0.273→0.275, max 1.25→1.07).
* ISSUE §5 / DISCLOSURE (preamble, C4, D, E6, checklist) / README §8 now quote this run as the frozen-package self-test,
  with the 12:02 UTC and the earlier-calibration runs kept as prior runs. Outputs (`summary.json`, `manifest.json`,
  `results.jsonl`, `ledger.jsonl`, `parity.txt`, `parity_offline_int8.txt`, `long_input_smoke.txt`, `selftest.sh.used`,
  `collect_freeze.json`, full log) under `exp/submission/validate/20261003_203822_freeze/`; hashes and commands in
  `exp/submission/FREEZE-metask-jev-rain.md`. The only remaining placeholder is the package commit hash
  (`<filled when the repository is published>`).
