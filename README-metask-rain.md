# metask-jev-rain — JevBench adapter package

**Entry:** `metask-jev-rain` · **Author display:** MeTask-Rain (same org as `metask-jev-4b`: Wayfind / metask-ai) · **Adapter id:** `metask_rain`
**Repository:** https://github.com/chenyuwang166-oss/metask-jev-rain (to be transferred to the `metask-ai` org) · **Code licence:** MIT
**Date:** 2026-10-03 · **Harness:** fstandhartinger/jevbench `main @ bb05a33` (METHOD v1.5 with its addenda)

This is a *different system* from our existing entry `metask-jev-4b` (a trained Qwen3.5-4B). Nothing here is
trained. Every number in this file is an estimate from our local re-implementation of METHOD v1.5 on the 231
public items (not an official score); we give no Speed, Cost or composite estimate and make no statement about rank.

## 0. Prior work

The read-out used here belongs to a family that already exists on the board: one forward pass of a frozen
instruct model, the probability mass of the label tokens at the answer position, a temperature. The nearest
same-base system is **Cygnet** (blockbrain, https://github.com/blockbrain-ai/cygnet-recipe): frozen
`google/gemma-4-12B-it` at the same HF revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` (Cygnet README,
read 2026-10-03: line 1 "one token per decision"; line 40 pins the same revision; lines 146–147 describe the
sum of option-letter token probabilities from vLLM top-20 logprobs, renormalised; lines 153–155 state that its
temperature T 3.4 was fitted on 241 self-generated items and that the public JevBench items were never used;
lines 172–173 credit the one-token readout to **NInfer**, https://github.com/igorls/ninfer, see Cygnet's
`CREDITS.md`). The harness's own `local_openjev` adapter and its raw-logit-control rows read direct logits the
same way.

What is ours, stated as differences and not as novelty claims: (1) a one-word *verbalized* prompt (yes/no,
digit, letter) and a read-out that sums log-mass over each label's **surface forms** (`yes`, ` yes`, …) rather
than letters only; (2) per-type calibration fitted on the public dev half (temperatures for choice and score,
a one-bin histogram calibration for noul — §4); (3) in-process `transformers` with a bf16/int8 precision policy
and per-precision calibration blocks; (4) the packaging, parity and disclosure work in this repository. **No
Cygnet or NInfer code, prompt or calibration value was used**; the overlap is the base model, its revision, and
the class of read-out.

## 1. What the system is

| | |
|---|---|
| Weights | `google/gemma-4-12B-it`, **frozen** (HF revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, pinned in `calibration.json`; verified against the HF tree: `model.safetensors` sha256 `5a84cb31…bff18d` = HF LFS oid, 23,919,549,408 bytes; `config.json` git-blob `324c841b…` = HF blob oid). No fine-tune, no LoRA, no soft prompt. |
| Prompt | One fixed instruction + the item rendered as a plain question asking for **one word** (§2). Identical for every item; no per-item adaptation. The text was written once (2026-10-02) before any verbalized result existed and first run on frozen Qwen3.5-4B/9B; it was not iterated against Gemma results. |
| Inference | **One forward pass** per decision, last-position logits only (`logits_to_keep=1`), `attn_implementation="sdpa"` pinned and recorded. No generation, no sampling, no retries. |
| Read-out | Per label: log-mass = `logsumexp` of the last-position logits over that label's single-token surface forms (§3). |
| Calibration | Per precision block in `calibration.json`: choice `P = softmax(log-mass / 4.5)`; score `P = softmax(log-mass / 5.0)`; noul **one-bin histogram calibration**: decision = sign of `d = logmass(yes) − logmass(no)`, reported `P(yes) = p_cal` if `d > 0` else `1 − p_cal`, with `p_cal` = dev-half precision of the sign rule (int8 0.8919, bf16 0.8649). §4 says plainly what this is and why. |
| Decisions vs. T | Argmax (choice, score) and the sign of `d` (noul) do not depend on the calibration; the reported probabilities, the score expected position and RPS, and whether a noul item would fall in the 0.2–0.8 abstention band do. Under the one-bin rule no item abstains. |
| Probabilities | The harness receives a distribution over `task.labels` in task order (noul: `{"no": p, "yes": p}`). `probs_source` is set per item: `"native"` for choice and score (tempered softmax of the model's own log-mass) and `"native_decision_one_bin_calibrated"` for noul under the default one-bin mapping, whose decision is the model's but whose reported probability is the calibration constant `p_cal` (§4); `summary.json` therefore lists two `probability_sources`. Under `METASK_RAIN_NOUL=temperature` every item is `"native"`. |
| Precision | **bf16 primary** when the GPU has ≥ 27 GiB free VRAM at load; **int8** (bitsandbytes LLM.int8) fallback otherwise; **nf4 refused**. The adapter selects the calibration block matching the precision it actually loaded and records both in `raw.runtime` (§6). The adapter's bf16 code path (same loader without the `BitsAndBytesConfig`) has not been executed by us on any hardware: our bf16 logits come from a separate CPU-offload run (§6, §8 item 4). |
| Weight load | `prepare()` (runner hook, before the latency clock) performs the one-time weight load only; it is idempotent, does nothing per item, and is a no-op when `JEVBENCH_WARM_LOAD=1` is set (the harness then loads before the run itself). The runner records the first item's `prepare_s`. |
| Cost | `cost_basis = "self_hosted_gpu"` (local weights, no provider tariff; price fields null). `usage = {input_tokens: len(prompt ids), output_tokens: 0}`; about 680 input tokens per decision on the public set. |
| Context | `max_input_tokens = 65536`; a longer prompt is refused with status 422 (scored wrong, not an infrastructure error). Longest public item: 3,942 tokens. |

## 2. Exact prompt

Instruction text (`SYSTEM` in `metask_rain.py`, verbatim):

```
You are a careful, decisive judge. Read the context, apply every rule, definition, threshold, exception and date it contains, and answer the question. Reply with exactly one word and nothing else.
```

User-turn body, built by `render(task)` (`state` is used as-is when it is a string, else `json.dumps(state, ensure_ascii=False)`):

```
Context:
<state>

Question: <question.instructions>

<type-specific block>

<type-specific answer line>
```

| type | type-specific block | answer line | labels |
|---|---|---|---|
| noul | `Answer yes if: <criteria.true>` ⏎ `Answer no if: <criteria.false>` (only when criteria has a true/false text) | `Answer with one word: yes or no.` | `["no", "yes"]` |
| score | `Levels:` ⏎ `0: <level 0>` ⏎ `1: <level 1>` … | `Answer with one number from 0 to K-1.` | `["0", …, "K-1"]` |
| choice | `Options:` ⏎ `A. <label>: <rubric text>` ⏎ `B. …` (letters in `task.labels` order; `: <rubric>` only when criteria gives one) | `Answer with the letter of the best option.` | `task.labels` |

Blocks are joined with blank lines. The instruction is **folded into the single user turn** (`SYSTEM + "\n\n" + body`,
no system turn, because the Gemma chat template has no system role), and the Gemma chat template is applied with
`add_generation_prompt=True, enable_thinking=False` (a `TypeError` fallback drops the `enable_thinking` kwarg on
older transformers; which branch ran is recorded per item as `raw.runtime.enable_thinking_kwarg_accepted` — the
Gemma 4 template defaults `enable_thinking` to false, so both branches give the same text today). No special
tokens are added by the tokenizer (`tokenizer.encode(prompt, add_special_tokens=False)`); the only control tokens
present are those the chat template itself writes: `<bos>`, `<|turn>` / `<turn|>`, and the empty thought channel
`<|channel>thought\n<channel|>`. The rendered prompt therefore starts
`<bos><|turn>user\nYou are a careful, decisive judge. …` and ends `…<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`.
The prompt's SHA-256 is recorded per item in `raw.runtime.prompt_sha256`.

Prompt history (disclosed because METHOD §9 keeps the ban on per-system prompt tuning): the instruction text and
`render()` were written once, on 2026-10-02, in `cache_verbalized.py`, before any verbalized result existed for any
model; the only later edits to that script added int8 loading and did not touch the prompt. The same text was
first run on frozen Qwen3.5-4B and Qwen3.5-9B, then on Gemma-4-12B-it. It was not iterated against Gemma results
on public items. The *choice of base model* (Gemma-4-12B-it over the two Qwen sizes) was made on public-item
results with this same read-out.

## 3. Read-out (surface-form sets)

The last-position logits are read at every surface form of a label; a form counts only if it is **exactly one
token** under the Gemma 4 tokenizer (the rule the calibration caches were built with). Log-mass per label is
`logsumexp` over its forms.

| type | declared forms (frozen) | effective under the Gemma 4 tokenizer |
|---|---|---|
| choice, option letter `C` | `C`, ` C`, `c`, ` c`, `C.`, `(C` | `C`, ` C`, `c`, ` c` (`C.` and `(C` are multi-token) |
| score, level `d` | `d`, ` d`, `d.` | `d` only |
| noul yes | `yes`, ` yes` (form set "lower") | both |
| noul no | `no`, ` no` | both |

**Noul form-set choice (disclosed).** The "lower" set is used instead of the full 12+12 set (`yes/Yes/YES/ yes/
 Yes/ YES/true/True/ true/ True/Y/ Y` and the no-side mirror). The pre-registered rule for each precision was: use
"lower" if the sign of `d = lse(yes) − lse(no)` is identical to the sign of the full-set aggregate on all 74 public
noul items, else fall back to the full set and say so. Result: 74/74 identical on int8 **and** on bf16 (0 flips, no
exact ties, min |d| 0.25), so "lower" is used in both blocks and the decisions are the same as with the full set.
The full sets stay in `calibration.json` for the record and for the parity check. True/false forms alone are a
different and much worse signal (13 sign flips) and are not used.

A label **none** of whose forms is a single token (score level `10` and above — Gemma splits multi-digit
numbers; a choice option beyond `Z`) keeps an empty form set and gets log-mass **−1e9**, exactly as
`cache_verbalized.py` (`… if vals else -1e9`): its probability is 0 and the distribution over `task.labels`
stays valid. Such labels are listed per item in `raw.runtime.labels_without_single_token_form`. No public item
is affected (max 5 score levels / 6 choice options); the rule exists so that an unseen sealed item can never
turn into an infrastructure error. Only if *every* label is unreadable does the adapter fail the item.

## 4. Calibration: what was fitted, where, and how it is disclosed

`calibration.json` carries **two blocks, `"int8"` and `"bf16"`**, each fitted on its own cached logits of the 231
public items (`logits_231_verb_gemma12b_int8.jsonl`, sha256 `77df7e47…0096e`; `logits_231_verb_gemma12b_bf16off.jsonl`,
sha256 `4a1811e2…48a21`). The adapter applies the block matching the precision it loaded.

**Split and rule (pre-registered, written before any number below was computed, identical for both caches).**
`split_231_v2.json`: dev 116 / check half 115 (units = paraphrase groups, balanced per type × tier; 0 groups
straddle). In `split_231_v2.json` the check half is stored under the key `local_sealed`; it is a half of the public set,
not the sealed set (`docs/calib_v2_report.md` uses the same key in its tables). Every parameter is fit on **dev only**; the check half is scored **once per cache** at the picked
values. Objective per type: maximise H = harmonic mean of (I_type, C_type) on dev under our local reconstruction
of the v1.5 formulas; ties → first grid point ascending. Choice grid 0.5…6.0 step 0.25; score grid
{1, 1.5, 2, 2.3, 3, 4, 5, 6, 8}. Full grids: `docs/calib_v2_report.md`.

| block | choice | score | noul |
|---|---|---|---|
| int8 | T 4.5 (dev H 80.8; next 80.0 at 4.25, 79.1 at 4.75) | T 5.0 (dev H 57.2; 56.9 at 6, 56.1 at 4) | one-bin, `p_cal` 0.8919 (33/37 dev; yes-side 14/14, no-side 19/23) |
| bf16 | T 4.5 (dev H 82.0; 81.7 at 4.75, 81.2 at 4.0) | T 5.0 (dev H 57.0; 56.8 at 6, 55.9 at 4) | one-bin, `p_cal` 0.8649 (32/37 dev; yes-side 13/13, no-side 19/24) |

**Noul: one-bin histogram calibration, stated plainly.** The decision is the sign of `d`. The reported
probability is **two-valued**: `P(yes) = p_cal` when `d > 0`, `1 − p_cal` otherwise, where `p_cal` is the
precision of the sign rule on the dev half. This is histogram binning (Zadrozny & Elkan 2001) with a single bin
per decision side and no free parameter beyond `p_cal`. We chose it for two reasons and we say both: (a) under
METHOD §3.1 any P(yes) in (0.2, 0.8) is an abstention counted wrong, and we wanted no item to abstain — `p_cal ≥ 0.8`
is asserted at fit time; (b) it reports as confidence exactly what the calibration set says the rule's
precision is, and under the noul calibration metric (ECE on P(yes)) a two-valued output is well calibrated
whenever the realised accuracy is close to `p_cal`. It **discards the magnitude of `d`**: an item with `d = 0.25`
and one with `d = 12` get the same probability. It was adopted after our first study showed that a two-valued map
scored much higher on our local calibration reconstruction than the smooth temperature (full-231 noul C 70.3 at
T 0.05 vs 90.3 for a 0.9/0.1 step); the one-bin form replaces the arbitrary 0.9 with the dev precision. The
smooth alternative is kept in each block as `noul_alternative` (T 0.05, `P(yes) = sigmoid(d / 0.05)`, itself
near-binary: min |d| 0.25 gives P ≥ 0.993) so the maintainers can run it instead if they prefer; its numbers are in
§4.1. The ISSUE asks the maintainers which they want.

**Score temperature (disclosed).** The pre-registered dev pick is T 5.0 on both caches and that is what the
package uses. In our first study (2026-10-02, int8, finer grid) the check half was scored once at the dev pick
5.0 and, alongside, at the then-current baseline 4.0; the check half preferred 4.0 (check-half score I/C 81.4/86.3
at 4.0 vs 78.0/82.4 at 5.0; dev 43.2/79.9 vs 44.9/78.7). An earlier draft of this package froze 4.0 for that
reason; the frozen package keeps the pre-registered 5.0 and does not re-pick on the check half. No decision
differs between the two values on any of the 18 public score items.

**How often the check half was looked at.** First study (2026-10-02, int8): once for every candidate rule family
(temperatures, position prior, shrink-to-argmax, step, affine/Platt, snap), so about a dozen scorings in total.
Second study (2026-10-03, per-precision blocks): once per cache at the picks. No value in the frozen package was
chosen on the check half; the one value that had been (T_score 4.0) was reverted.

### 4.1 Local estimates (231 public items; our reconstruction of v1.5; not official)

Equal type weights (choice / noul / score one third each), tier weights 10/20/30/40, per-item chance correction,
noul abstention band 0.2–0.8. "Full 231" is half in-sample.

| cache | subset | n | I_choice | C_choice | I_noul | C_noul | abst% | I_score | C_score | I_231 | C_axis |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| int8 (T_choice 4.5 / noul one-bin p_cal 0.8919 / T_score 5.0) | dev (fit) | 116 | 77.8 | 84.0 | 77.2 | 83.6 | 0.0 | 44.9 | 78.7 | 66.6 | 82.1 |
| int8 | check half (once) | 115 | 78.9 | 75.7 | 59.1 | 83.8 | 0.0 | 78.0 | 82.4 | 72.0 | 80.6 |
| int8 | full 231 (reference) | 231 | 78.3 | 83.7 | 68.2 | 91.9 | 0.0 | 62.4 | 86.0 | 69.6 | 87.2 |
| bf16 (T_choice 4.5 / noul one-bin p_cal 0.8649 / T_score 5.0) | dev (fit) | 116 | 77.8 | 86.7 | 72.4 | 81.0 | 0.0 | 44.9 | 78.1 | 65.1 | 81.9 |
| bf16 | check half (once) | 115 | 78.9 | 77.7 | 53.1 | 83.8 | 0.0 | 81.1 | 84.5 | 71.1 | 82.0 |
| bf16 | full 231 (reference) | 231 | 78.3 | 84.0 | 62.8 | 91.6 | 0.0 | 64.1 | 86.3 | 68.4 | 87.3 |

Per-tier I on the check half: int8 choice [easy 100 / standard 100 / hard 63.0], noul [100 / 83.3 / 36.8], score
[standard 95.0 / hard 69.5]; bf16 choice [100 / 100 / 63.0], noul [100 / 83.3 / 26.3], score [95.8 / 73.8]. Noul
I drops from dev to check half (77.2 → 59.1 int8; 72.4 → 53.1 bf16); the noul decision is independent of the
calibration, so this is the model's hard-tier noul accuracy, not a fitting effect, but with 37 items per half it
is noisy.

Noul alternative T 0.05 (maintainers only, not used): int8 dev I/C 77.2/78.3, check 59.1/62.2, full 68.2/70.3;
bf16 dev 72.4/72.9, check 53.1/56.8, full 62.8/64.9; abstention 0% everywhere.

Raw accuracy on the 231 items, int8: 197/231 (choice 120/139, noul 63/74, score 14/18); bf16: 195/231 (120 / 61 / 14).

Studied on the same dev items and **not used** (so the search space is known): a choice position prior (hurt on
dev); a shrink-to-argmax mixture for score; an affine/Platt map for noul P(yes) (no gain at zero abstention); a
"snap" variant pushing values out of the 0.2–0.8 band; the 0.9/0.1 step (superseded by the one-bin rule above).

## 5. Running with the harness

Package layout (drop-in on the upstream repo; `registration.patch` adds the import and the CLI entry the same way
`semif_direct` / `smalljev_local` are registered):

```
jevbench-rain/
  jevbench/adapters/metask_rain.py     the adapter (class MetaskRainAdapter, name "metask_rain")
  calibration.json                     model, pinned revision, two precision blocks, form sets, where fitted
  registration.patch                   unified diff for jevbench/adapters/__init__.py and jevbench/cli.py
  registration/*.orig, *.new           the pristine and edited copies the patch was made from (LF line endings)
  selftest.sh                          full 231-item run + summarize + parity check on a Linux CUDA box
  parity_check.py                      offline / results parity against the frozen logits caches
  long_input_smoke.py                  one forward each on ~16k / ~32k-token synthetic items (run by selftest.sh)
  requirements-metask-rain.txt         exact package versions of the venv everything was run in
  NOTICE.md                            base-model licence notice (Apache-2.0 + Gemma Prohibited Use Policy URLs)
  docs/split_231_v2.json               the dev / check-half split (check half under the key `local_sealed`)
  docs/calib_v2_report.md              the pre-registered rule, dev grids, check-half scorings (edited copy, §4)
  caches/logits_231_verb_gemma12b_int8.jsonl      frozen int8 logits cache  (sha256 77df7e47…0096e, in calibration.json)
  caches/logits_231_verb_gemma12b_bf16off.jsonl   frozen bf16 logits cache  (sha256 4a1811e2…48a21; CPU-offload run, numerically bf16)
  caches/*.meta.json                   the caches' run metadata (model path reduced to its basename)
  CHANGES.md                           review findings and what was changed
  README-metask-rain.md                this file
```

Install into a copy of the harness (the upstream files may be CRLF in some checkouts; the patch is LF):

```sh
cp -r /path/to/jevbench upstream-copy && cd upstream-copy
sed -i 's/\r$//' jevbench/cli.py jevbench/adapters/__init__.py
patch -p1 < /path/to/jevbench-rain/registration.patch
cp /path/to/jevbench-rain/jevbench/adapters/metask_rain.py jevbench/adapters/
cp /path/to/jevbench-rain/calibration.json jevbench/adapters/metask_rain_calibration.json   # or set METASK_RAIN_CALIBRATION
```

Run (all 231 public items) and summarize — this is what `selftest.sh` executes:

```sh
export JEVBENCH_WARM_LOAD=1          # load before the clock; the prepare() hook does the same
export METASK_RAIN_MODEL_PATH=/path/to/gemma-4-12b-it   # local weights dir (or pass --endpoint)
python -m jevbench.cli run \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --adapter metask_rain --endpoint /path/to/gemma-4-12b-it \
  --model google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --key-env '' \
  --results OUT/results.jsonl --raw-dir OUT/raw --ledger OUT/ledger.jsonl \
  --reserve-usd 0 --cap-usd 1 --cost-basis self_hosted_gpu --run-label metask-jev-rain --manifest OUT/manifest.json
python -m jevbench.cli summarize \
  --tasks datasets/public/easy.jsonl,datasets/public/original.jsonl,datasets/public/hard.jsonl \
  --results OUT/results.jsonl --ledger OUT/ledger.jsonl --public-export OUT/summary.json
```

`OUT` must be outside the repo (the runner refuses raw/results paths inside it). `--request-options` is not read by
this adapter. Knobs: `--endpoint` local model dir or HF id (`METASK_RAIN_MODEL_PATH` also works), `--revision`
(default: the pin in `calibration.json`), env `METASK_RAIN_DTYPE` (unset = policy of §6: bf16 if ≥ 27 GiB free,
else int8; `bf16` or `int8` forces one; nf4 is refused), `METASK_RAIN_NOUL` (`one_bin` default; `temperature`
runs the block's `noul_alternative` T 0.05, for comparison only), `METASK_RAIN_DEVICE` (CUDA index),
`METASK_RAIN_CALIBRATION` (path), `METASK_RAIN_STRICT_REVISION=1` (below).

**Offline image (how the evaluator's pods run).** Download the pinned snapshot into the image once —
`huggingface-cli download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --local-dir /models/gemma-4-12b-it`
— and point the adapter at it (`METASK_RAIN_MODEL_PATH=/models/gemma-4-12b-it` or `--endpoint`). The run then needs
**no network** (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` are fine) and **writes nothing** outside the harness's
`--results` / `--raw-dir` / `--ledger` / `--manifest` paths; a read-only container works. Only when no local dir
is given does the adapter fall back to loading the HF id at the pinned revision, which needs network.

Revision check: with an HF id, `revision=` is passed to `from_pretrained` (the hub enforces it). With a local dir
the adapter hashes `<dir>/config.json` and compares it with `calibration.json: config_sha256` (`478c46e8…`), plus
the size of a single-file `model.safetensors` (`23 919 549 408` bytes) when present; the outcome is recorded per
item as `raw.runtime.revision_verified` with the hashes in `raw.runtime.revision_check`. A mismatch is printed as a
warning and recorded (`revision_verified: false`); `METASK_RAIN_STRICT_REVISION=1` makes it fatal. The pins
themselves were checked against the HF tree at that revision (model.safetensors LFS sha256 and size, config.json
blob oid): the local copy is byte-identical to `707f0a3b…`.

### Environment and offline use

* **Exactly what was run** (the only tested environment): Python 3.10.12, Ubuntu 22.04.5, one RTX 4090 24 GB
  (sm_89), NVIDIA driver 580.105.08; `torch 2.14.1+cu130`, `transformers 5.17.0`, `bitsandbytes 0.50.2` (int8
  only), `accelerate 1.15.0`, `tokenizers 0.23.2`, `safetensors 0.8.0`, `huggingface_hub 1.33.0`. Exact pins for the
  whole venv: `requirements-metask-rain.txt`. `peft` is installed in that venv (0.21.0) but neither imported nor needed.
* **transformers gate:** the checkpoint's `model_type` is `gemma4_unified` (it loads as
  `Gemma4UnifiedForConditionalGeneration`). The adapter refuses a release that does not know that model type or is
  older than **5.10.0**, the first tag containing `src/transformers/models/gemma4_unified` (v5.9.x does not have it;
  5.10.0 and 5.10.1 were both published on PyPI on 2026-06-03); the message names the tested release. Any release other than 5.17.0 is accepted but untested
  and recorded per item as `raw.runtime.transformers_tested: false`. The loader passes `dtype=torch.bfloat16` and
  `attn_implementation="sdpa"` (the default the calibration caches were produced with on 5.17.0); the
  implementation actually in effect is recorded as `raw.runtime.attn_implementation`.
* **CUDA:** a cu130 torch needs a CUDA-13-capable driver (≥ 580). Other torch/CUDA builds have not been run.
* **Offline:** with `--endpoint` / `METASK_RAIN_MODEL_PATH` pointing at a local copy (see "Offline image" above) and
  the `--revision` pin checked against `config.json`, nothing is fetched (`selftest.sh` runs with
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`) and the adapter writes nothing; the model dir may be read-only.
* **Not tested on our hardware:** bf16 on a single GPU without offload (it needs ≥ 27 GiB free, more than the 4090
  has), hence no bf16 latency and no execution of the adapter's bf16 branch (the only code difference is the absent
  `BitsAndBytesConfig`; the bf16 calibration block was fitted on logits from a separate CPU-offload script). After a
  first bf16 run, `parity_check.py --results … --cache caches/logits_231_verb_gemma12b_bf16off.jsonl` should report
  decision agreement 231/231 up to near-ties (§8). GPUs other than the RTX 4090 are untested.
* The adapter can be imported standalone (relative import of `.base`, falling back to the absolute
  `jevbench.adapters.base` on `ImportError`). A load failure is cached after the first attempt and reported on every
  item as the same error, never retried per item.

## 6. VRAM and precision policy

| precision | weights | when the adapter uses it | status |
|---|---|---|---|
| **bf16** | ≈ 24 GB (12B params) + activations | **primary**: chosen when the target GPU reports **≥ 27 GiB free** at load (A100/H100 40–80 GB, RTX 6000 Ada / RTX PRO 6000 48 GB, …) | calibration block `"bf16"` fitted on bf16 logits of the 231 items. Those logits were produced on the 4090 with CPU offload (numerically bf16, peak 19.7 GB), so **no bf16 latency has been measured** and the harness self-test has not been run in bf16 (§7, §8). |
| **int8** (bitsandbytes LLM.int8) | ≈ 13 GB peak | **fallback** when less than 27 GiB is free (24 GB cards: RTX 4090 / 3090 / 5090) | calibration block `"int8"`; every measured number we have (harness self-test, latency, parity) is int8 on one RTX 4090. |
| nf4 / 4-bit | ≈ 8 GB | never | **refused by the adapter** (`METASK_RAIN_DTYPE=nf4` raises): on the 231 public items it lost ~12 I points in our first study (I_231 58.2 vs 70.0 at the same dev-picked temperatures; `verb_gemma12b_nf4_eval.log`). |

`METASK_RAIN_DTYPE=bf16|int8` overrides the free-VRAM rule. Whatever is loaded, the adapter applies the matching
calibration block and records `raw.runtime.precision`, the block applied, the attention implementation (`sdpa`) and
the library versions per item, so the row can be labelled with the precision actually measured.

**int8 vs bf16 agreement, each at its own block** (231 public items): choice argmax 135/139, noul decision 72/74,
score argmax 18/18, score round(EV) 18/18 — 225/231 overall; on the check half 68/69, 36/37, 9/9, 9/9. Flipped
items: 4 hard choice items (int8 gains one and loses one, two wrong→wrong), 2 noul items (`hard-sol-b-judge_hard-05`,
`original-adequacy-01-0`, both int8 right / bf16 wrong; max |Δd| 9.9). In our earlier temperature-only comparison
(2026-10-03, each cache at its own dev pick) I_231 was 68.9 for bf16 vs 70.0 for int8; at the frozen blocks 68.4 vs
69.6 (§4.1). So bf16 is not a lossless swap for int8 on these items: it agrees on 225/231 decisions and is about
one I point lower locally, within the noise of 74 noul items but in an unfavourable direction. We still declare
bf16 primary because it is the model's native precision and the evaluator's 12B pods have the VRAM; the int8
block exists so that a 24 GB card runs the same system with its own calibration.

## 7. Latency

One forward pass of a 12B model over a 150–4,000-token prompt. **Measured only in int8 on one RTX 4090**
(upstream harness clock around `adapter.run`, serial, 231 public items, frozen package, 2026-10-03): p50 **0.273 s**,
p95 0.620 s, max 1.25 s, raw, before any self-host adjustment (an earlier run of the same adapter at the previous
calibration measured p50 0.264 s / p95 0.616 s); latency is flat at ≈ 0.26 s for short prompts (bitsandbytes
int8 has a fixed per-call overhead). bf16 latency has **not** been measured (the bf16 logits run was CPU-offloaded
and says nothing about speed). The weight load (≈ 60–90 s for int8 quantisation) is kept out of the clock by
`JEVBENCH_WARM_LOAD=1` / the `prepare()` hook. The harness reports `latency_s`; `raw.runtime.forward_s` gives the
GPU forward alone. METHOD §5's self-host adjustment (×2 plus 0.15 s on the evaluator's servers) applies as the
evaluator decides; we report no Speed estimate.

Context: `max_input_tokens` is 65536. Smoke test on the 4090 in int8 with synthetic prompts of ≈ 16k and ≈ 32k
tokens (`long_input_smoke.py`, one forward each, 2026-10-03): 18,504 tokens ok, peak 16.2 GiB allocated / 17.4 GiB
reserved, 8.7 s; 33,056 tokens ok, peak 20.4 GiB allocated / 22.3 GiB reserved, 20.7 s. A 24 GB card in int8 is
therefore close to its limit at ~33k tokens; prompts between that and 65,536 tokens were not tried on the 4090 and
may run out of memory there (an OOM is an ordinary failed item, not a 422).

## 8. Parity with the frozen calibration caches (for the validator) and the harness self-test

The read-out and both calibration blocks were frozen on the two per-item per-form logits caches
(`cache_verbalized.py`; int8 sha256 `77df7e47…0096e`, bf16 `4a1811e2…48a21`, shipped under `caches/` with their `.meta.json`).
`parity_check.py` proves the adapter reproduces the cache matching the precision that ran:

1. *Offline* (no GPU): it imports the adapter's own `aggregate` / `tempered_softmax` / noul one-bin rule and applies
   them, with the block's form sets and parameters, to the per-form logits stored in the cache → expected per-item
   probs/decisions (int8 197/231 correct; bf16 195/231), and cross-checks the module's per-label log-mass against
   the cache's own aggregate (choice/score: |Δ| < 1e-3; noul: same sign of `d` between full and lower form sets).
   The residual |Δ| comes from float64 (adapter, `math.exp`/`math.log`) vs float32 (cache, `torch.logsumexp`)
   arithmetic and from the cache's 4-decimal rounding of the per-form logits; it is below 1e-6 in log-mass.
2. *Prompt parity* (CPU): rendering all 231 harness `Task`s through `render()` + chat template gives the same
   token count as the cache's `n_tokens` on every item (0 mismatches), and the single-token form table matches the
   cache meta.
3. *Results* (after `selftest.sh`): per item, harness `predicted` vs the offline expectation (expected 100 %
   agreement with the cache of the same precision; a run in the other precision may flip the near-tie items listed
   in §6), max |Δp|, and `usage.input_tokens` vs cache `n_tokens` (expected 0 mismatches). `selftest.sh` reads the
   precision that ran from `results.jsonl` (`raw.runtime.precision`) and picks the cache of that precision (int8 →
   `caches/logits_231_verb_gemma12b_int8.jsonl`, bf16 → `…_bf16off.jsonl`; `CACHE_INT8` / `CACHE_BF16` override).
4. *bf16, for the maintainers*: the adapter's bf16 branch has never run on our hardware. After your first bf16 run,
   the results step above against the bf16off cache is the check that the in-GPU bf16 forward reproduces the
   CPU-offload bf16 logits (expected 231/231 up to near-ties; the two caches themselves differ on 6/231, §6).

**Upstream harness self-test** (`selftest.sh`: `jevbench.cli run` + `summarize` on a patched copy of `main @ bb05a33`;
one RTX 4090). Frozen package, 2026-10-03 12:38 UTC (freeze run `out/20261003_freeze_203822`; int8 auto-selected: 23.1 GiB free < 27; block int8; sdpa): 231/231 ok, raw accuracy 0.853 (197/231; choice 120/139, noul 63/74, score 14/18; easy 48/48, original 70/72, hard 79/111), 0 refusals / 0 × 422; decisions identical to the frozen int8 cache on 231/231 (max |Δp| 1e-6, 0 prompt-length mismatches, parity_rc 0); noul P(yes) takes only the values 0.8919 / 0.1081, 0 abstentions; p50 0.275 s / p95 0.621 s / max 1.07 s raw; harness v1 top-label ECE 0.061 (10 equal-width bins; not the v1.5 calibration axis). That run executed the shipped `calibration.json` (SHA-256 `4682cae4d181e53abcaf67bc7a9b48b4650be95eed7ef2ca79aff658cfe64aa4`, recorded per item in `raw.runtime.calibration_sha256`) and the shipped adapter (`metask_rain.py` SHA-256 `b943989039858b8463b617b89d06fae2ec027e1cf012762fe4b1d1aef7682c65`). The 12:02 UTC run of the same package earlier that day (`out/20261003_200208`, `calibration.json` `43f32dfa…52ca0`, identical in every numeric block) gave the same 197/231 and the same 231/231 parity with p50 0.273 s / p95 0.620 s / max 1.25 s; an earlier run of the same adapter at the previous calibration (noul T 0.05, score T 4.0) gave the same 197/231 and p50 0.264 / p95 0.616. The upstream unit tests give the same line on the pristine and on the patched copy
(`Ran 35 tests … FAILED (errors=6)`, failures unrelated to this adapter — modules the pristine clone cannot import in a
venv without pytest). The outputs of the frozen run (`summary.json`, `manifest.json`, `parity.txt`,
`parity_offline_int8.txt`, `long_input_smoke.txt`, log tail) are kept with the submission record. `raw.runtime` records the
model directory and the calibration file by basename only; `manifest.json: endpoint` is written by the harness's `cli.py`
from the `--endpoint` argument as given.

## 9. Licence

* Weights: `google/gemma-4-12B-it`, Google DeepMind, **Apache-2.0** (model card `license: apache-2.0`,
  `license_link: https://ai.google.dev/gemma/docs/gemma_4_license`, which resolves to the Apache-2.0 text; the Gemma
  Terms of Use list Gemma 1–3 only), used subject to Google's Gemma Prohibited Use Policy
  (https://ai.google.dev/gemma/prohibited_use_policy) — see `NOTICE.md`. The HF repo is not gated. Nothing is
  redistributed here: the adapter loads the weights from HF or a local copy. No derivative weights exist for this entry.
* This adapter package (code, calibration, docs): **MIT**, same as the harness.

## 10. Disclosure items (METHOD v1.5 §7–8; leaderboard FAQ: runnable code, exact model and licence, any public-task training disclosure)

1. Frozen open weights, no training of any kind. The only fitted parameters are, per precision block: two
   temperatures (choice, score) and one noul bin value `p_cal`.
2. **Public JevBench items were used for calibration** — the dev half (116) of the 231 public items, under our local
   reconstruction of the v1.5 scorer; the other 115 were scored as a check (how often: §4). The sealed pool was
   never read; we have no access to it. METHOD §7: calibration on open items is allowed and disclosed; this is that
   disclosure. The choice of base model was also made on public-item results.
3. One fixed prompt for all items (§2), written once before any result and not iterated against Gemma results;
   no per-item adaptation; no retries; no multiple forwards; failures are recorded, never repaired.
4. `probs_source` per item: `"native"` for choice and score (next-token logit read-out at the answer position; not a
   verbalized probability) and `"native_decision_one_bin_calibrated"` for noul under the one-bin mapping, so
   `summary.json` shows both values in `probability_sources`; API flag: no (self-hosted, in-process, offline).
5. Noul probabilities are **two-valued** (one-bin histogram calibration, §4); no item abstains. The smooth
   alternative is in `calibration.json` for the maintainers.
6. Precision: bf16 primary / int8 fallback / nf4 refused; the precision actually loaded and its calibration block
   are recorded per item; all measured numbers so far are int8 on one RTX 4090.
7. `max_input_tokens = 65536` → 422 above it (no public item is near it); 16k / 32k prompts verified in int8 on the 4090, longer ones not (§7).
8. Cost basis `self_hosted_gpu`, no provider tariff; price fields null; we do not claim a cost score.
9. Relation to other entries: `metask-jev-4b` is a separate trained system by the same org (Wayfind / metask-ai);
   this entry shares no weights, prompt, calibration or code path with it.
10. Prior work: NInfer → Cygnet read-out family (§0); no Cygnet code or calibration used.
11. All local numbers are estimates from a reconstruction of the method; the official scorer decides. We give no
    Speed, Cost, composite or rank estimate.
