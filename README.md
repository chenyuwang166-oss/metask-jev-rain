# metask-jev-rain

This is MeTask's early JevBench package, recorded on **2026-10-03** and retained for reproducibility. For our current trained decision models, official results and privately deployable weights, use the 31B and 12B links below. Results for those models do not describe this historical package.

## Current trained models

Results checked on **2026-10-17** against [JevBench board v1.7.46](https://benchmarkheaven.com/jev-models) (page updated 2026-10-16). Scores are organizer-measured **Capability Scores**. Capability ranks are among **129 qualifying open-weight systems** under the official cost and latency limits; they are not composite-score ranks. The leaderboard's 'official #' label refers to its separate composite ranking.

For these full-set results, the operator measures each system on 1,500 decisions: 1,200 sealed items that submitters cannot access and 300 public items.

| Model | Capability Score | Capability rank | Weights | Service |
|---|---:|---|---|---|
| MeTask 31B (service v3) | 79.6 | 3 | [Model card](https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained) | [31B service](https://github.com/chenyuwang166-oss/metask-jev-rain-31B-trained) |
| MeTask 12B | 77.9 | 6; highest among entries labeled 12B in the system name | [Model card](https://huggingface.co/MeTask-Rain/metask-jev-rain-12B-trained) | [12B service](https://github.com/chenyuwang166-oss/metask-jev-rain-12B-trained) |


Published by MeTask under the public GitHub account `chenyuwang166-oss`. Project and contact entry point: [MeTask-Rain](https://huggingface.co/MeTask-Rain).

## Historical package and disclosure

The original model identity, training status, use of public items, local estimates, relationship to earlier entries and method attribution remain in the [original disclosure](DISCLOSURE.md). Local measurements for this package are not the official scores of the current trained models.

- [Historical technical description and run instructions](README-metask-rain.md)
- [Adapter](jevbench/adapters/metask_rain.py), [calibration](calibration.json) and [registration](registration.patch)
- [Self-test instructions](selftest.sh) and [environment pins](requirements-metask-rain.txt)

Code and calibration: [MIT](LICENSE). Base weights: Google Gemma 4 12B, Apache-2.0; the historical package's [NOTICE](NOTICE.md) and [disclosure](DISCLOSURE.md) retain the applicable attribution and use-policy statement.
