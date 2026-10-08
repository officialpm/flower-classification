# Results and provenance

Copyright 2026 Parth Maniar. Apache-2.0.
[GitHub](https://github.com/officialpm) | [Portfolio](https://www.parthmaniar.tech/)

Measured source version 355703905 completed October 6, 2026 in 805.4 seconds total, with 762.94 seconds recorded by the workflow. TensorFlow 2.20.0, Python 3.13.15, two Tesla T4 GPUs, MirroredStrategy with two replicas, batch size 32, seed 2026, 224-pixel images.

Training 12,753 rows; validation 3,712; test 7,382. All 104 classes appeared in both labeled splits. No dropped rows. Fine-tuned model validation macro-F1 0.8554432827, accuracy 0.8785021552, log loss 0.4665958764. Frozen head macro-F1 0.800261; equal blend 0.855260; majority floor 0.001113.

The same fine-tuned prediction artifact was submitted once October 7, 2026. Native status Complete, public macro-F1 0.86708. There was no previous submission in the current account's table. No rank or private score is claimed.

Recorded prediction SHA256: `7d6ce2e5baffff094a0c83a53492fae5ff2e8bb21653eff33338aa4f7f3b37ce`. The recorded hash was inspected in the manifest; it is not independently rehashed by this package. Artifacts are not redistributed. Portable source was syntax-checked but not retrained; hosted measurement must not be attributed to a new local run.

Selection reuses the official development split. Fine-tuning and blending are nearly tied; no statistically meaningful margin is claimed. Input distributions, rare classes and ImageNet ancestry can affect generalization. No production throughput, latency or accuracy on user photos has been measured.
