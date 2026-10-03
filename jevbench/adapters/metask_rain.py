"""metask-jev-rain: FROZEN google/gemma-4-12B-it, verbalized one-word prompt, log-mass read-out,
per-type calibration fitted per precision. In-process (native logits, no network, no generation).

System (entry "metask-jev-rain"; MeTask-Rain, same org as metask-jev-4b: Wayfind / metask-ai):
  * weights: google/gemma-4-12B-it, untouched (no fine-tune, no LoRA, no prompt tuning per item);
  * ONE prompt for every item: a fixed instruction (SYSTEM below, folded into the user turn because
    the Gemma chat template is used without a system turn, exactly as the frozen calibration caches
    were produced) + the item rendered as a plain question that asks for ONE WORD;
  * ONE forward pass per decision, logits at the last position only (logits_to_keep=1);
  * read-out: per label, log-mass = logsumexp of the last-position logits over that label's
    surface forms (yes/" yes", option letters A/" A"/a/" a", digits "0".."K-1"); only forms that
    are a single token under the Gemma tokenizer are used (same rule as the caches);
  * calibration (calibration.json, one block per precision, each fitted on logits of that precision,
    on the dev half (116 items, split_231_v2.json) of the 231 public items):
      - choice, score: P = softmax(log-mass / T_type)  (argmax unchanged);
      - noul: ONE-BIN HISTOGRAM CALIBRATION (Zadrozny & Elkan 2001, histogram binning with a single
        bin per decision side): decision = sign of d = logmass(yes) - logmass(no) (d == 0 -> no);
        P(yes) = p_cal if d > 0 else 1 - p_cal, where p_cal is the precision of that sign rule on
        the dev half. A non-decreasing step function of d, not a temperature: the magnitude of d is
        not used. The previous smooth mapping (temperature 0.05) stays in the block as
        "noul_alternative" and is selectable with METASK_RAIN_NOUL=temperature.
  * probs_source: "native" for choice and score (tempered softmax of the model's own log-mass);
    "native_decision_one_bin_calibrated" for noul under the one-bin mapping (the decision is the
    model's, the reported probability is the calibration constant), "native" for noul under
    METASK_RAIN_NOUL=temperature. summary.json therefore lists two probability_sources.
  * limits: max_input_tokens 65536 (422 above it); a choice item with more than 26 options is an
    explicit error (the read-out has one letter per option).

Prior work: the read-out belongs to the NInfer -> Cygnet family (single forward pass, label-token
probability mass at the answer position, temperature); Cygnet pins the same HF revision. Our
differences: the one-word verbalized prompt with surface-form log-mass, per-type temperatures,
in-process transformers, the one-bin noul calibration. No Cygnet code or calibration was used.

Nothing is tuned per item, nothing is retried, nothing is repaired: any failure is ok=False with
the error text, and an input over max_input_tokens is a 422 (the system refusing the input; the
runner scores it wrong and does not count it toward the stop rule).

CLI:  python -m jevbench.cli run --adapter metask_rain --endpoint <model dir or HF id> \
        --model google/gemma-4-12B-it --revision <pin> --tasks ... --results ... --raw-dir ...
Env:  METASK_RAIN_MODEL_PATH   local model dir (used when --endpoint is not given); with a local dir
                               nothing is fetched from the network and nothing is written to it
      METASK_RAIN_CALIBRATION  path to calibration.json (default: next to this file as
                               metask_rain_calibration.json, then the package root two levels up)
      METASK_RAIN_DTYPE        "bf16" | "int8" to force a precision. Unset: bf16 if the CUDA device has
                               >= 27 GiB free, else int8 (bitsandbytes LLM.int8). nf4/4-bit is refused.
      METASK_RAIN_NOUL         "one_bin" (default) | "temperature" (the noul_alternative, for comparison)
      METASK_RAIN_DEVICE       CUDA device index (default 0)
      METASK_RAIN_STRICT_REVISION  "1": a local model dir whose config.json sha256 differs from the
                               calibration pin raises instead of being recorded as unverified
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import string
import time
from pathlib import Path

try:
    from .base import DecisionResult
except ImportError:  # loaded standalone (e.g. by file path, outside the package)
    from jevbench.adapters.base import DecisionResult

# ----------------------------------------------------------------------------------------------
# Prompt and surface forms: VERBATIM from exp/cache_verbalized.py (the file the calibration caches
# logits_231_verb_gemma12b_int8.jsonl / _bf16off.jsonl were produced with). Do not edit.
# ----------------------------------------------------------------------------------------------
SYSTEM = ("You are a careful, decisive judge. Read the context, apply every rule, definition, threshold, exception and date "
          "it contains, and answer the question. Reply with exactly one word and nothing else.")
YES_FORMS = ["yes", "Yes", "YES", " yes", " Yes", " YES", "true", "True", " true", " True", "Y", " Y"]
NO_FORMS = ["no", "No", "NO", " no", " No", " NO", "false", "False", " false", " False", "N", " N"]


def letter_forms(c):
    return [c, " " + c, c.lower(), " " + c.lower(), c + ".", "(" + c]


def digit_forms(d):
    return [d, " " + d, d + "."]


PROMPT_VERSION = "verbalized-one-word-v1 (cache_verbalized.py, 2026-10-02)"
DEFAULT_MODEL_ID = "google/gemma-4-12B-it"
CALIBRATION_ENV = "METASK_RAIN_CALIBRATION"
PRECISIONS = ("bf16", "int8")
BF16_MIN_FREE_GIB = 27.0            # auto-select bf16 only with this much free VRAM on the device
NF4_REFUSAL = ("nf4 / 4-bit loading is not acceptable for this system: on the 231 public items it lost ~12 I points "
               "(local I_231 58.2 vs 70.0 for int8). Use bf16 (needs >= 27 GiB free VRAM; auto-selected) or "
               "bitsandbytes int8 (~13 GB; auto-selected otherwise), or force one with METASK_RAIN_DTYPE=bf16|int8.")
NOUL_MODES = ("one_bin", "temperature")
# First transformers release that contains models/gemma4_unified (the model_type of this checkpoint): the
# v5.10.0 tag already has src/transformers/models/gemma4_unified (v5.9.x does not); 5.10.0 and 5.10.1 were both
# published on PyPI on 2026-06-03. Only 5.17.0 has been run; any other release is accepted but recorded as
# untested in raw.runtime.
MIN_TRANSFORMERS = (5, 10, 0)
TESTED_TRANSFORMERS = "5.17.0"
REQUIRED_MODEL_TYPE = "gemma4_unified"
ATTN_IMPLEMENTATION = "sdpa"        # pinned; the default the caches were produced with on transformers 5.17.0
DEFAULT_MAX_INPUT_TOKENS = 65536
NO_FORM_LOGMASS = -1e9              # cache_verbalized.py: a label with no single-token form gets -1e9
MAX_CHOICE_OPTIONS = 26             # one option letter A..Z per option; more is an explicit error, not a silent truncation
# probs_source per item: "native" where the reported distribution is the tempered softmax of the model's own
# log-mass (choice, score, and noul under METASK_RAIN_NOUL=temperature); for noul under the default one-bin
# mapping the decision is native (sign of d) but the reported probability is the calibration constant p_cal,
# so it is labelled distinctly and shows up as such in summary.json's probability_sources.
PROBS_SOURCE_NATIVE = "native"
PROBS_SOURCE_NOUL_ONE_BIN = "native_decision_one_bin_calibrated"


def render(task):
    """Same text as cache_verbalized.render(), taking a jevbench Task instead of a dict.

    Returns (labels, text, groups): labels in the order the harness scores them, the user-turn
    text (without SYSTEM), and the surface-form group per label (full lists; single-token
    filtering happens at load time against the real tokenizer).
    """
    q = task.question
    qtype = q["type"]
    crit = q.get("criteria")
    state = task.state if isinstance(task.state, str) else json.dumps(task.state, ensure_ascii=False)
    parts = [f"Context:\n{state}\n\nQuestion: {q['instructions']}"]
    if qtype == "noul":
        labels = ["no", "yes"]
        c = crit or {}
        if c.get("true") or c.get("false"):
            parts.append(f"Answer yes if: {c.get('true', 'Yes')}\nAnswer no if: {c.get('false', 'No')}")
        parts.append("Answer with one word: yes or no.")
        groups = {"yes": YES_FORMS, "no": NO_FORMS}
    elif qtype == "score":
        labels = [str(i) for i in range(len(crit))]
        parts.append("Levels:\n" + "\n".join(f"{i}: {t}" for i, t in enumerate(crit)))
        parts.append(f"Answer with one number from 0 to {len(crit) - 1}.")
        groups = {l: digit_forms(l) for l in labels}
    else:
        labels = list(task.labels)
        if len(labels) > MAX_CHOICE_OPTIONS:
            # zip() would silently drop options beyond "Z" and the read-out would then KeyError on them
            raise ValueError(f"choice with {len(labels)} options exceeds the {MAX_CHOICE_OPTIONS}-letter read-out (A..Z)")
        rubric = crit if isinstance(crit, dict) else {}
        codes = string.ascii_uppercase[:len(labels)]
        parts.append("Options:\n" + "\n".join(f"{c}. {l}" + (f": {rubric[l]}" if rubric.get(l) else "") for c, l in zip(codes, labels)))
        parts.append("Answer with the letter of the best option.")
        groups = {l: letter_forms(c) for c, l in zip(codes, labels)}
    return labels, "\n\n".join(parts), groups


def messages_for(text):
    """Gemma: the instruction is folded into the single user turn (no system turn), as in the caches
    (`--no-system`)."""
    return [{"role": "user", "content": SYSTEM + "\n\n" + text}]


# ----------------------------------------------------------------------------------------------
# Pure read-out math (no torch): importable by the parity checker without a GPU.
# ----------------------------------------------------------------------------------------------
def logsumexp(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def aggregate(form_logits, groups_effective):
    """Per-label log-mass: logsumexp over the label's (effective, single-token) surface forms.
    form_logits: {form_string: logit}; groups_effective: {label: [form_string, ...]}.
    A label with NO single-token form gets NO_FORM_LOGMASS (-1e9), exactly as cache_verbalized.py
    (`... if vals else -1e9`): it ends with P = 0 and the distribution stays valid."""
    return {label: (logsumexp([form_logits[s] for s in forms]) if forms else NO_FORM_LOGMASS)
            for label, forms in groups_effective.items()}


def tempered_softmax(logmass, labels, T):
    """P = softmax(m / T) over `labels` (dict in that order). Numerically stable; sums to 1."""
    T = float(T)
    if not (T > 0):
        raise ValueError(f"temperature must be > 0, got {T}")
    z = [logmass[l] / T for l in labels]
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    s = sum(e)
    return {l: v / s for l, v in zip(labels, e)}


def noul_d(logmass):
    """d = logmass(yes) - logmass(no); the noul decision is yes iff d > 0."""
    return logmass["yes"] - logmass["no"]


def noul_one_bin(logmass, p_cal):
    """One-bin histogram calibration: P(yes) = p_cal if d > 0 else 1 - p_cal (d == 0 -> no)."""
    p = float(p_cal) if noul_d(logmass) > 0 else 1.0 - float(p_cal)
    return {"no": 1.0 - p, "yes": p}


def readout_probs(qtype, logmass, labels, block, noul_mode="one_bin"):
    """The adapter's whole calibration step for one item, given one precision block of calibration.json.
    Returns (probs over `labels`, description of the mapping applied)."""
    if qtype == "noul":
        if noul_mode == "one_bin":
            p = noul_one_bin(logmass, block["noul_calibration"]["p_cal"])
            return {l: p[l] for l in labels}, {"noul_mapping": "one_bin", "p_cal": float(block["noul_calibration"]["p_cal"])}
        T = float(block["noul_alternative"]["T"])
        return tempered_softmax(logmass, labels, T), {"noul_mapping": "temperature", "T": T}
    T = float(block["temperature_by_kind"][qtype])
    return tempered_softmax(logmass, labels, T), {"T": T}


def _validate_block(name, b):
    if not isinstance(b, dict):
        raise ValueError(f"calibration.json block {name!r} missing")
    t = b.get("temperature_by_kind") or {}
    for kind in ("choice", "score"):
        if not (isinstance(t.get(kind), (int, float)) and t[kind] > 0):
            raise ValueError(f"calibration.json {name}.temperature_by_kind.{kind} must be a positive number")
    if t.get("noul") != "one_bin":
        raise ValueError(f"calibration.json {name}.temperature_by_kind.noul must be 'one_bin'")
    nc = b.get("noul_calibration") or {}
    if nc.get("method") != "one_bin_histogram":
        raise ValueError(f"calibration.json {name}.noul_calibration.method must be 'one_bin_histogram'")
    p = nc.get("p_cal")
    if not (isinstance(p, (int, float)) and 0.8 <= p < 1.0):
        # p_cal >= 0.8 is what keeps every noul item outside the harness's 0.2-0.8 abstention band
        raise ValueError(f"calibration.json {name}.noul_calibration.p_cal must be in [0.8, 1): {p!r}")
    f = nc.get("forms") or {}
    if not (f.get("yes") and f.get("no") and set(f["yes"]) <= set(YES_FORMS) and set(f["no"]) <= set(NO_FORMS)):
        raise ValueError(f"calibration.json {name}.noul_calibration.forms must give non-empty yes/no lists drawn from the frozen full sets")
    alt = b.get("noul_alternative") or {}
    if not (alt.get("method") == "temperature" and isinstance(alt.get("T"), (int, float)) and alt["T"] > 0):
        raise ValueError(f"calibration.json {name}.noul_alternative must be a positive temperature")


def load_calibration(path=None):
    """calibration.json: model, revision, forms, and one block per precision ("bf16", "int8"). Required."""
    candidates = []
    if path:
        candidates.append(Path(path))
    if os.environ.get(CALIBRATION_ENV):
        candidates.append(Path(os.environ[CALIBRATION_ENV]))
    here = Path(__file__).resolve().parent
    candidates += [here / "metask_rain_calibration.json", here.parent.parent / "calibration.json"]
    for p in candidates:
        if p.is_file():
            raw = p.read_bytes()
            cal = json.loads(raw.decode("utf-8"))
            cal["_path"] = str(p)
            cal["_sha256"] = hashlib.sha256(raw).hexdigest()
            break
    else:
        raise FileNotFoundError("calibration.json not found; set " + CALIBRATION_ENV + " or place it at " +
                                " | ".join(str(c) for c in candidates))
    for k in ("model", "revision", "forms") + PRECISIONS:
        if k not in cal:
            raise ValueError(f"calibration.json missing key {k!r}")
    f = cal["forms"]
    # The declared form lists must be the frozen ones this module carries verbatim (drift guard).
    if f.get("choice_letter_templates") != ["{C}", " {C}", "{c}", " {c}", "{C}.", "({C}"]:
        raise ValueError("calibration.json choice_letter_templates differ from the frozen letter_forms()")
    if f.get("score_digit_templates") != ["{d}", " {d}", "{d}."]:
        raise ValueError("calibration.json score_digit_templates differ from the frozen digit_forms()")
    if f.get("noul_full_yes") != YES_FORMS or f.get("noul_full_no") != NO_FORMS:
        raise ValueError("calibration.json noul_full_* differ from the frozen YES_FORMS/NO_FORMS")
    for name in PRECISIONS:
        _validate_block(name, cal[name])
    return cal


def _version_tuple(v):
    out = []
    for part in str(v).split("+")[0].split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


class MetaskRainAdapter:
    name = "metask_rain"
    cost_basis = "self_hosted_gpu"  # as semif_direct: local weights, no provider tariff (cost stays null)

    def __init__(self, endpoint=None, model=None, key_env="", timeout_s=None,
                 price_input_per_m=None, price_output_per_m=None, revision=None,
                 max_input_tokens=DEFAULT_MAX_INPUT_TOKENS, calibration_path=None, **kw):
        self.calibration = load_calibration(calibration_path)
        self.endpoint = endpoint or os.environ.get("METASK_RAIN_MODEL_PATH") or self.calibration["model"]
        self.model = model or self.calibration["model"]
        self.revision = revision or self.calibration["revision"]
        self.max_input_tokens = int(max_input_tokens)
        self.price_input_per_m = price_input_per_m
        self.price_output_per_m = price_output_per_m
        mode = (os.environ.get("METASK_RAIN_NOUL") or "one_bin").strip().lower()
        if mode not in NOUL_MODES:
            raise ValueError(f"METASK_RAIN_NOUL={mode!r}: use one_bin (default) or temperature")
        self.noul_mode = mode
        self._loaded = None
        self._load_error = None      # first load exception, re-raised on every later call (no reload attempts)
        self._form_ids = {}
        self.load_s = None
        self.precision = None
        self.precision_selection = None
        self.block = None            # the calibration block matching the loaded precision
        self.noul_forms = None
        self.attn_implementation = None
        self._enable_thinking_kwarg_ok = None   # set by build_prompt: did apply_chat_template accept enable_thinking=?
        self.revision_check = {"revision_verified": None, "method": None}

    # ------------------------------------------------------------------ loading
    def _pick_precision(self, torch, device_index):
        """METASK_RAIN_DTYPE forces bf16/int8; unset -> bf16 if the device has >= 27 GiB free, else int8.
        nf4/4-bit is refused. Returns (precision, selection record)."""
        want = (os.environ.get("METASK_RAIN_DTYPE") or "").strip().lower()
        if want in ("nf4", "4bit", "int4", "fp4"):
            raise ValueError(NF4_REFUSAL)
        free_b, total_b = torch.cuda.mem_get_info(device_index)
        rec = {"free_vram_gib": round(free_b / 2**30, 2), "total_vram_gib": round(total_b / 2**30, 2),
               "bf16_min_free_gib": BF16_MIN_FREE_GIB}
        if want in PRECISIONS:
            rec["mode"] = "env METASK_RAIN_DTYPE"
            return want, rec
        if want:
            raise ValueError(f"METASK_RAIN_DTYPE={want!r}: use bf16 or int8 (or leave unset for auto). " + NF4_REFUSAL)
        rec["mode"] = "auto (free VRAM)"
        return ("bf16" if free_b / 2**30 >= BF16_MIN_FREE_GIB else "int8"), rec

    @staticmethod
    def _sha256_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def _verify_local_revision(self, src):
        """A local dir carries no HF revision: compare its config.json sha256 (and the single-file
        safetensors size, when present) with the pins in calibration.json. Mismatch is recorded in
        raw.runtime (revision_verified False); METASK_RAIN_STRICT_REVISION=1 raises instead."""
        chk = {"method": "local-dir: sha256(config.json) vs calibration.config_sha256"}
        want_cfg = self.calibration.get("config_sha256")
        cfg = Path(src) / "config.json"
        got_cfg = self._sha256_file(cfg) if cfg.is_file() else None
        chk.update({"config_sha256_expected": want_cfg, "config_sha256_local": got_cfg})
        ok = bool(want_cfg) and got_cfg == want_cfg
        want_sz = self.calibration.get("model_safetensors_bytes")
        st = Path(src) / "model.safetensors"
        if want_sz and st.is_file():
            got_sz = st.stat().st_size
            chk.update({"model_safetensors_bytes_expected": want_sz, "model_safetensors_bytes_local": got_sz})
            ok = ok and got_sz == want_sz
        chk["revision_verified"] = ok
        if not ok:
            msg = (f"metask_rain: local model dir {src} does not match the pinned revision {self.revision}: {chk}")
            if os.environ.get("METASK_RAIN_STRICT_REVISION", "").strip() == "1":
                raise RuntimeError(msg)
            print("[metask_rain] WARNING " + msg, flush=True)
        return chk

    @staticmethod
    def _check_transformers(transformers):
        """Refuse a transformers release that cannot build this checkpoint (model_type gemma4_unified)."""
        tv = _version_tuple(transformers.__version__)
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
            known = REQUIRED_MODEL_TYPE in CONFIG_MAPPING_NAMES
        except Exception:  # noqa: BLE001
            known = False
        if not known or tv < MIN_TRANSFORMERS:
            raise RuntimeError(
                f"metask_rain needs a transformers release that knows model_type {REQUIRED_MODEL_TYPE!r} "
                f"(first: {'.'.join(map(str, MIN_TRANSFORMERS))}; tested: {TESTED_TRANSFORMERS} only); found "
                f"{transformers.__version__} (gemma4_unified known: {known}). pip install transformers=={TESTED_TRANSFORMERS}")
        if transformers.__version__ != TESTED_TRANSFORMERS:
            print(f"[metask_rain] WARNING transformers {transformers.__version__} is not the tested release "
                  f"{TESTED_TRANSFORMERS}; recorded as transformers_tested=false in raw.runtime", flush=True)

    def load(self):
        if self._loaded is not None:
            return self._loaded
        if self._load_error is not None:
            # the first failure is the record; later items fail fast with the same error (no reload attempts)
            raise RuntimeError(f"cached first load error: {self._load_error}")
        try:
            return self._load()
        except Exception as e:  # noqa: BLE001
            self._load_error = f"{type(e).__name__}: {str(e)[:400]}"
            raise

    def _load(self):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("metask_rain needs a CUDA GPU (12B in bf16 or int8); no CPU path")
        self._check_transformers(transformers)
        self.device_index = int(os.environ.get("METASK_RAIN_DEVICE", "0"))
        self.device = f"cuda:{self.device_index}"
        t0 = time.perf_counter()
        self.precision, self.precision_selection = self._pick_precision(torch, self.device_index)
        self.block = self.calibration[self.precision]
        self.noul_forms = {"yes": list(self.block["noul_calibration"]["forms"]["yes"]),
                           "no": list(self.block["noul_calibration"]["forms"]["no"])}
        is_local = Path(os.path.expanduser(str(self.endpoint))).is_dir()
        src = os.path.expanduser(str(self.endpoint)) if is_local else self.endpoint
        rev = {} if is_local else {"revision": self.revision}
        if is_local:
            self.revision_check = self._verify_local_revision(src)
        else:
            self.revision_check = {"revision_verified": True, "method": "hf-hub: revision= kwarg of from_pretrained"}
        # Same load path as cache_verbalized.load_model(): bf16 weights; int8 = bitsandbytes LLM.int8.
        kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation=ATTN_IMPLEMENTATION,
                  device_map={"": self.device_index}, **rev)
        if self.precision == "int8":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(src, **kw).eval()
        tok = AutoTokenizer.from_pretrained(src, **rev)
        self.attn_implementation = getattr(model.config, "_attn_implementation", None)
        self._loaded = (model, tok)
        self.load_s = time.perf_counter() - t0
        self.torch_version = torch.__version__
        self.transformers_version = transformers.__version__
        try:
            import bitsandbytes
            self.bitsandbytes_version = bitsandbytes.__version__
        except Exception:  # noqa: BLE001
            self.bitsandbytes_version = None
        self.model_class = type(model).__name__
        # Resolve every frozen surface form once; keep single-token forms only (cache rule).
        for s in YES_FORMS + NO_FORMS + [f for c in string.ascii_uppercase[:10] for f in letter_forms(c)] \
                + [f for d in "0123456789" for f in digit_forms(d)]:
            self._fid(s)
        return self._loaded

    def prepare(self, task):
        """Runner hook (called by runner.py before the latency clock, its time recorded as prepare_s on the
        first item): the one-time weight load only, idempotent (load() returns at once when loaded), nothing
        per item. It is the same effect as the harness's own JEVBENCH_WARM_LOAD=1 (cli.py calls load() before
        the run); when that variable is set this hook does nothing, so only one mechanism is in play."""
        if os.environ.get("JEVBENCH_WARM_LOAD") == "1":
            return
        self.load()

    def _fid(self, s):
        """Token id of surface form `s` if it is exactly one token under the tokenizer, else None."""
        if s not in self._form_ids:
            tok = self._loaded[1]
            ids = tok.encode(s, add_special_tokens=False)
            self._form_ids[s] = ids[0] if len(ids) == 1 else None
        return self._form_ids[s]

    def _effective_groups(self, qtype, groups):
        if qtype == "noul":
            groups = {"yes": self.noul_forms["yes"], "no": self.noul_forms["no"]}
        eff = {}
        for label, forms in groups.items():
            # cache rule: a label with no single-token form keeps an EMPTY list -> log-mass -1e9 -> P = 0
            # (e.g. score level "10" and above: Gemma splits multi-digit numbers; choice beyond "Z").
            eff[label] = [s for s in forms if self._fid(s) is not None]
        if not any(eff.values()):
            raise ValueError(f"no label has a single-token surface form: {groups}")
        return eff

    # ------------------------------------------------------------------ one decision
    def build_prompt(self, task):
        labels, text, groups = render(task)
        tok = self._loaded[1]
        messages = messages_for(text)
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            self._enable_thinking_kwarg_ok = True
        except TypeError:
            # older transformers: the kwarg is unknown; the Gemma 4 template defaults enable_thinking to
            # false, so the text is the same today. The branch taken is recorded in raw.runtime.
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            self._enable_thinking_kwarg_ok = False
        ids = tok.encode(prompt, add_special_tokens=False)
        return labels, groups, messages, prompt, ids

    def _runtime(self):
        b = self.block or {}
        # paths are recorded by basename only: raw.runtime is copied into results.jsonl and the manifest
        ep = str(self.endpoint)
        return {"model": self.model,
                "model_path": Path(os.path.expanduser(ep)).name if Path(os.path.expanduser(ep)).is_dir() else ep,
                "revision": self.revision,
                "revision_verified": self.revision_check.get("revision_verified"),
                "revision_check": self.revision_check,
                "model_class": self.model_class, "precision": self.precision,
                "precision_selection": self.precision_selection,
                "quant": "bitsandbytes-int8" if self.precision == "int8" else "none (bf16)",
                "calibration_block": self.precision,
                "calibration_fitted_on_cache": b.get("fitted_on_cache"),
                "calibration_fitted_on_cache_sha256": b.get("fitted_on_cache_sha256"),
                "temperature_by_kind": b.get("temperature_by_kind"),
                "noul_mapping": self.noul_mode,
                "noul_p_cal": (b.get("noul_calibration") or {}).get("p_cal"),
                "noul_alternative_T": (b.get("noul_alternative") or {}).get("T"),
                "noul_forms": self.noul_forms,
                "attn_implementation": self.attn_implementation,
                "attn_implementation_requested": ATTN_IMPLEMENTATION,
                "single_token_forms_only": True,
                "system_turn": False, "enable_thinking": False,
                "enable_thinking_kwarg_accepted": self._enable_thinking_kwarg_ok,
                "max_input_tokens": self.max_input_tokens,
                "torch": self.torch_version, "transformers": self.transformers_version,
                "transformers_tested": self.transformers_version == TESTED_TRANSFORMERS,
                "bitsandbytes": self.bitsandbytes_version,
                "load_s": self.load_s,
                "calibration_path": Path(self.calibration["_path"]).name if self.calibration.get("_path") else None,
                "calibration_sha256": self.calibration.get("_sha256"),
                "calibration_fitted_on": b.get("noul_calibration", {}).get("fitted_on"),
                "prompt_version": PROMPT_VERSION}

    def run(self, task) -> DecisionResult:
        qtype = task.question["type"]
        probs_source = PROBS_SOURCE_NOUL_ONE_BIN if (qtype == "noul" and self.noul_mode == "one_bin") else PROBS_SOURCE_NATIVE
        res = DecisionResult(adapter=self.name, ok=False, probs_source=probs_source, model=self.model)
        try:
            model, tok = self.load()
        except Exception as e:  # noqa: BLE001
            res.error = f"load failed: {type(e).__name__}: {str(e)[:300]}"
            res.raw = {"runtime": {"load_error": self._load_error}}
            return res
        import torch
        t0 = time.perf_counter()
        try:
            labels, groups, messages, prompt, ids = self.build_prompt(task)
            if labels != list(task.labels):
                raise ValueError(f"rendered labels {labels} != task.labels {list(task.labels)}")
            eff = self._effective_groups(qtype, groups)
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            res.request_body = {"messages": messages, "prompt_sha256": prompt_sha, "input_tokens": len(ids),
                                "prompt_version": PROMPT_VERSION}
            n = len(ids)
            if n > self.max_input_tokens:
                # 422 semantics (as paw_local / metask_jev): the system refusing an input over its
                # declared limit; scored wrong, not an infrastructure error.
                res.latency_s = time.perf_counter() - t0
                res.status = 422
                res.error = f"422 over context limit: {n} tokens > max_input_tokens {self.max_input_tokens}"
                res.usage = {"input_tokens": n, "output_tokens": 0}
                res.raw = {"runtime": {"over_context": True, "input_tokens": n,
                                       "max_input_tokens": self.max_input_tokens, "precision": self.precision}}
                return res
            x = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                torch.cuda.synchronize(self.device)
                t1 = time.perf_counter()
                out = model(input_ids=x, use_cache=False, logits_to_keep=1)
                logits = out.logits[0, -1, :].float()
                torch.cuda.synchronize(self.device)
                fwd = time.perf_counter() - t1
            form_logits = {}
            for label, forms in eff.items():
                for s in forms:
                    form_logits[s] = float(logits[self._fid(s)])
            if not all(math.isfinite(v) for v in form_logits.values()):
                raise ValueError("non-finite logit at a surface form")
            logmass = aggregate(form_logits, eff)
            probs, mapping = readout_probs(qtype, logmass, labels, self.block, self.noul_mode)
            logz = float(torch.logsumexp(logits, 0))
            res.latency_s = time.perf_counter() - t0
            res.probs = {l: float(probs[l]) for l in task.labels}
            res.usage = {"input_tokens": n, "output_tokens": 0}
            answer = {"logmass": {k: round(v, 6) for k, v in logmass.items()},
                      "forms": {f"{l}|{s}": round(form_logits[s], 6) for l, fs in eff.items() for s in fs},
                      "logz": round(logz, 6), "probs": res.probs, "input_tokens": n,
                      "forward_seconds": fwd, "prompt_sha256": prompt_sha, "prompt_version": PROMPT_VERSION}
            if qtype == "noul":
                answer["d"] = round(noul_d(logmass), 6)
            if qtype == "noul" and self.noul_mode == "one_bin":
                origin = ("decision: sign of native last-position log-mass difference d; probability: one-bin histogram "
                          "calibration constant p_cal fitted on the dev half (P(yes)=p_cal if d>0 else 1-p_cal)")
            else:
                origin = "native last-position logits, logsumexp over surface forms, tempered softmax"
            rt = self._runtime()
            rt.update({"mapping_applied": mapping, "temperature_applied": mapping.get("T"),
                       "form_sets": eff, "labels_without_single_token_form": [l for l, fs in eff.items() if not fs],
                       "prompt_sha256": prompt_sha, "input_tokens": n,
                       "latency_s": res.latency_s, "forward_s": fwd, "probability_origin": origin,
                       "probs_source": probs_source})
            res.raw = {"answer": answer, "runtime": rt}
            res.ok = True
        except Exception as e:  # noqa: BLE001 - never repair: the failure is the record
            res.latency_s = time.perf_counter() - t0
            res.error = f"{type(e).__name__}: {str(e)[:300]}"
            if res.raw is None:
                res.raw = {"runtime": {"error_type": type(e).__name__, "precision": self.precision}}
        return res

    def reserve_estimate(self, task) -> float:
        return 0.0
