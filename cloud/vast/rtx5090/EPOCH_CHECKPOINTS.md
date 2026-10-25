# Current MetaCurate run: checkpoint only at epoch ends

The user changed storage policy after update65 (260 trajectories): save and
upload only at **295 and590**. An explicit graceful stop or the time budget still
writes a **local-only** recovery checkpoint. All losses/LR/throughput events
continue to be logged for every optimizer update.

Use `python -m scripts.training.train_epoch_checkpoints` with the same config,
index, selection, output, dedicated Hub repo/prefix and `--resume latest`.
`--resume` also accepts a trusted local checkpoint directory; it does not accept
`hub`. The existing output already has a verified complete checkpoint65.

The operator entrypoint overrides only checkpoint callbacks and publishes the
effective schedule in `runtime-checkpoint-policy.json` and the `checkpoint_policy`
event. The frozen trainer/config retain their original hashes and historical
save_every20 so strict checkpoint identity, optimizer/scheduler/RNG restoration
and the passing CUDA pilot stay valid. Training math is unchanged.

Epoch publication preserves Python/NumPy/Torch CPU/CUDA RNG. If an epoch upload
fails after its local commit, resuming at that boundary retries the upload before
the next batch. Hub retention pins both epoch ends. No background publisher or
checkpoint transfer to local is deployed. The existing SSH HTTPS relay remains
necessary for this host to reach Hugging Face.
