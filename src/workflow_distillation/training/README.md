# Training package

Entrypoint: [`scripts/train.py`](../../../scripts/train.py).
Review order, contracts, GPU profiles and migration procedure:
[`docs/training.md`](../../../docs/training.md).

`cli` composes validated `config`/`data`, explicit `model`/`engine` and
`checkpoints`; `activation_storage`/`memory` manage saved tensors and RAM,
`distributed` averages scaled adapter gradients, `device` isolates CUDA calls,
and `events`/`monitor`/`process` own logs and the launched worker tree.
`identity`/`compatibility` define the strict resume boundary.

No GPU allocation or historical launcher runs on import. Keep model math and
checkpoint primitives shared; add hardware choices through validated profiles.
This local implementation still needs a CUDA migration pilot before deployment.
