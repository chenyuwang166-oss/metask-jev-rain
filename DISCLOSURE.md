# DISCLOSURE — metask-jev-rain (2026-10-03)

| Item | Disclosure |
|---|---|
| A2. Exact model | `google/gemma-4-12B-it`, frozen. HF revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. No derivative weights. |
| A3. Licence | Base weights: Apache-2.0 (model card `license: apache-2.0`, `license_link` → Apache-2.0 text; the Gemma Terms of Use cover Gemma 1–3 only), used subject to Google's Gemma Prohibited Use Policy (https://ai.google.dev/gemma/prohibited_use_policy); HF repo not gated; nothing redistributed. Our adapter/calibration/scripts: MIT. |
| A4. Public JevBench items used during development? | **Yes — for calibration and for model choice.** |
| A5. Training on open items | **Not applicable: no training.** No fine-tuning, LoRA, soft prompts, heads or any parameter update. |
| A6. Sealed set / sealed pool | Never accessed; we have no access. No calibration job read anything but `datasets/public/*.jsonl`. |
