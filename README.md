# metask-jev-rain

JevBench submission package: a **frozen** `google/gemma-4-12B-it` (HF revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, no training of any kind) read out in one forward pass with a one-word prompt, per-label surface-form log-mass aggregation, and per-type calibration fitted on the dev half of the 231 public JevBench items. In-process adapter for the upstream `jevbench` harness.

- **Long-form description, how to run, environment, precision policy, disclosure:** [`README-metask-rain.md`](README-metask-rain.md)
- **Itemised disclosure against METHOD v1.5:** [`DISCLOSURE.md`](DISCLOSURE.md)
- **Adapter:** [`jevbench/adapters/metask_rain.py`](jevbench/adapters/metask_rain.py) · **calibration:** [`calibration.json`](calibration.json) · **registration into the harness:** [`registration.patch`](registration.patch)
- **Self-test on the 231 public items (harness run + parity + long-input smoke):** [`selftest.sh`](selftest.sh)
- **Exact environment pins:** [`requirements-metask-rain.txt`](requirements-metask-rain.txt)

Author display: **MeTask-Rain** (same organisation as the existing `metask-jev-4b` row, Wayfind / metask-ai; a different system — no shared weights, prompt, calibration or code). Prior work: the read-out belongs to the NInfer → Cygnet family; see "Prior work" in `README-metask-rain.md`. All numbers in this repository are local estimates on the 231 public items; no Speed, Cost, composite or rank estimate is made.

Licence: MIT for the code and calibration files in this repository ([`LICENSE`](LICENSE)); the base weights are Google's, Apache-2.0, used subject to the Gemma Prohibited Use Policy ([`NOTICE.md`](NOTICE.md)). This repository will be transferred to the `metask-ai` organisation; the commit SHA is the pin.
