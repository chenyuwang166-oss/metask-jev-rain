# metask-jev-rain — JevBench adapter package

## 5. Running with the harness

Install into a copy of the harness (the upstream files may be CRLF in some checkouts; the patch is LF):

```sh
cp -r /path/to/jevbench upstream-copy && cd upstream-copy
sed -i 's/\r$//' jevbench/cli.py jevbench/adapters/__init__.py
patch -p1 < /path/to/jevbench-rain/registration.patch
cp /path/to/jevbench-rain/jevbench/adapters/metask_rain.py jevbench/adapters/
cp /path/to/jevbench-rain/calibration.json jevbench/adapters/metask_rain_calibration.json   # or set METASK_RAIN_CALIBRATION
```

Run and summarize — this is what `selftest.sh` executes:

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
(default: the pin in `calibration.json`), env `METASK_RAIN_DTYPE` (unset = bf16 if ≥ 27 GiB free,
else int8; `bf16` or `int8` forces one; nf4 is refused), `METASK_RAIN_DEVICE` (CUDA index),
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
* The adapter can be imported standalone (relative import of `.base`, falling back to the absolute
  `jevbench.adapters.base` on `ImportError`). A load failure is cached after the first attempt and reported on every
  item as the same error, never retried per item.


## 9. Licence

* Weights: `google/gemma-4-12B-it`, Google DeepMind, **Apache-2.0** (model card `license: apache-2.0`,
  `license_link: https://ai.google.dev/gemma/docs/gemma_4_license`, which resolves to the Apache-2.0 text; the Gemma
  Terms of Use list Gemma 1–3 only), used subject to Google's Gemma Prohibited Use Policy
  (https://ai.google.dev/gemma/prohibited_use_policy) — see `NOTICE.md`. The HF repo is not gated. Nothing is
  redistributed here: the adapter loads the weights from HF or a local copy. No derivative weights exist for this entry.
* This adapter package (code, calibration, docs): **MIT**, same as the harness.
