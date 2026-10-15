# metask-jev-rain

> **Status (2026-10-15):** This repository is our 2026-10-03 frozen-base package, kept for the record. Our current official JevBench results come from the trained models, measured and ranked by the benchmark organizers (not self-reported):
>
> | Model | Trained | Leaderboard | Metric | Score | As of |
> |---|---|---|---|---|---|
> | metask-jev-rain-31B-trained (service v3) | yes | JevBench open-weight leaderboard v1.6.18 | Capability Score | 79.6 (#3 among open-weight systems) | 2026-10-14 |
> | metask-jev-rain-12B-trained | yes | JevBench open-weight leaderboard v1.6.18 | Capability Score | 77.9 (highest among ranked entries labeled 12B) | 2026-10-14 |
>
> Measured by the leaderboard operator (Benchmark Heaven) on 1,500 decision items per system (1,200 sealed + 300 public).
>
> Leaderboard: https://benchmarkheaven.com/jev-models · Models: https://huggingface.co/MeTask-Rain/metask-jev-rain-31B-trained · https://huggingface.co/MeTask-Rain/metask-jev-rain-12B-trained. The numbers further below are local estimates for this 2026-10-03 package only.
>
> 说明（2026-10-15）：本仓库是 2026-10-03 的冻结底座旧包，留作记录。现行官方成绩来自训练版模型（见上表，榜方实测，非自测）；下文数字仅为本旧包当时的本地估算。榜方 Benchmark Heaven 测量，每个系统 1,500 道决策题（1,200 封存 + 300 公开）。

JevBench submission package: a **frozen** `google/gemma-4-12B-it` (HF revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, no training of any kind) read out in one forward pass with a one-word prompt, per-label surface-form log-mass aggregation, and per-type calibration fitted on the dev half of the 231 public JevBench items. In-process adapter for the upstream `jevbench` harness.

- **Long-form description, how to run, environment, precision policy, disclosure:** [`README-metask-rain.md`](README-metask-rain.md)
- **Itemised disclosure against METHOD v1.5:** [`DISCLOSURE.md`](DISCLOSURE.md)
- **Adapter:** [`jevbench/adapters/metask_rain.py`](jevbench/adapters/metask_rain.py) · **calibration:** [`calibration.json`](calibration.json) · **registration into the harness:** [`registration.patch`](registration.patch)
- **Self-test on the 231 public items (harness run + parity + long-input smoke):** [`selftest.sh`](selftest.sh)
- **Exact environment pins:** [`requirements-metask-rain.txt`](requirements-metask-rain.txt)

Author display: **MeTask-Rain** (same organisation as the existing `metask-jev-4b` row, Wayfind / metask-ai; a different system — no shared weights, prompt, calibration or code). Prior work: the read-out belongs to the NInfer → Cygnet family; see "Prior work" in `README-metask-rain.md`. All numbers in this repository are local estimates on the 231 public items; no Speed, Cost, composite or rank estimate is made.

Licence: MIT for the code and calibration files in this repository ([`LICENSE`](LICENSE)); the base weights are Google's, Apache-2.0, used subject to the Gemma Prohibited Use Policy ([`NOTICE.md`](NOTICE.md)). This repository will be transferred to the `metask-ai` organisation; the commit SHA is the pin.
