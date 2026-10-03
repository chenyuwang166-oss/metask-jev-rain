# NOTICE

This package (adapter code, calibration file, scripts and documentation: MIT licence) loads the model
`google/gemma-4-12B-it` (Google DeepMind), HF revision `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`,
which is licensed under the Apache License, Version 2.0 (https://ai.google.dev/gemma/apache_2) and is
used subject to Google's Gemma Prohibited Use Policy (https://ai.google.dev/gemma/prohibited_use_policy).

No model weights, derivative weights or quantised copies are redistributed by this package: the adapter
loads the weights from the Hugging Face hub at the pinned revision or from a local copy supplied by the
operator. "Gemma" is named here only to describe the origin of the model this package runs (Apache-2.0 §6);
the entry is not a Google product.

The logits caches under `caches/` and the files under `docs/` are our own measurements on the 231 public
JevBench items (MIT, with this package).
