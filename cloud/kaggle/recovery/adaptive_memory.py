"""Choose parallel or staggered replica execution without allocating DDP buckets.

The estimator is calibrated for the existing Qwen3.5-4B FP16, per-layer
checkpointed, pinned-CPU input path. It is a conservative estimate, not an OOM
proof. A serial update still averages the same two rank-local scaled gradients
before unscale, clipping and one optimizer step.
"""
from __future__ import annotations

import math
from functools import wraps

GIB = 2**30
MIB = 2**20


class MemoryPolicy:
    def __init__(self, *, reserve_bytes=3 * GIB, multiplier=1.15):
        if reserve_bytes < 0 or multiplier < 1 or not math.isfinite(multiplier):
            raise ValueError("Invalid host memory safety policy")
        self.reserve_bytes = reserve_bytes
        self.multiplier = multiplier

    @staticmethod
    def _raw_activation_bytes(tokens):
        if type(tokens) is not int or not 2 <= tokens < 30000:
            raise ValueError("Expected a complete trajectory below 30000 tokens")
        # FP16 hidden width 2560. The pinned allocator rounds each input to a
        # power-of-two size class. Observed envelope: 36 blocks plus small tensors.
        # In particular, 26214 -> 26215 tokens doubles the dominant size class.
        bucket = 1 << (tokens * 2560 * 2 - 1).bit_length()
        return 36 * bucket + 64 * MIB

    def activation_bytes(self, tokens):
        return math.ceil(self._raw_activation_bytes(tokens) * self.multiplier)

    def observe(self, tokens, allocated):
        if allocated is not None and allocated > 0:
            self.multiplier = max(self.multiplier,
                                  1.15 * allocated / self._raw_activation_bytes(tokens))

    def decide(self, lengths, memory):
        if len(lengths) != 2:
            raise ValueError("Adaptive execution requires two rank-local trajectories")
        predicted = [self.activation_bytes(n) for n in lengths]
        current, limit = memory.get("current_bytes"), memory.get("limit_bytes")
        decision = {"mode": "stop", "tokens": lengths,
                    "activation_estimates_bytes": predicted,
                    "reserve_bytes": self.reserve_bytes, "limit_bytes": limit,
                    "multiplier": self.multiplier, "reason": "missing_memory_limit"}
        if current is None or limit is None or limit <= 0:
            return decision
        stat = memory.get("stat", {})
        # shmem is already INCLUDED in file; it must never be counted as
        # reclaimable disk cache. Retain dirty/writeback pages conservatively.
        reclaimable = 0
        if {"file", "shmem", "file_dirty", "file_writeback"} <= stat.keys():
            reclaimable = max(0, stat["file"] - stat["shmem"]
                              - stat["file_dirty"] - stat["file_writeback"])
        baseline = max(0, current - reclaimable)
        reserve = max(self.reserve_bytes, math.ceil(limit * 0.10))
        parallel, serial = baseline + sum(predicted), baseline + max(predicted)
        decision.update(resident_baseline_bytes=baseline, reserve_bytes=reserve,
                        predicted_parallel_bytes=parallel, predicted_serial_bytes=serial)
        if parallel + reserve <= limit:
            decision.update(mode="parallel", reason="parallel_has_headroom")
        elif serial + reserve <= limit:
            decision.update(mode="serial", reason="only_serial_has_headroom")
        else:
            decision["reason"] = "single_sample_lacks_headroom"
        return decision


def manual_replica(torch, module, *, device_ids=None, output_device=None,
                   broadcast_buffers=False, find_unused_parameters=False,
                   gradient_as_bucket_view=True, bucket_cap_mb=25):
    """Initialize trainable replicas without constructing a DDP reducer.

    Frozen base tensors come from the same verified model revision. Broadcast
    adapters explicitly, just as DDP did at construction, without advancing RNG.
    The signature accepts only the known immutable runtime's constructor options.
    """
    if broadcast_buffers or find_unused_parameters or not gradient_as_bucket_view:
        raise ValueError("Unsupported replica synchronization configuration")
    dist = torch.distributed
    if dist.get_world_size() != 2:
        raise ValueError("Manual replication requires exactly two ranks")
    if device_ids is not None and (list(device_ids) != [dist.get_rank()]
                                   or output_device != dist.get_rank()):
        raise ValueError("Replica must stay on its rank-local GPU")
    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    signature = [(n, tuple(p.shape), str(p.dtype)) for n, p in named]
    signatures = [None, None]
    dist.all_gather_object(signatures, signature)
    if not named or signatures[0] != signatures[1]:
        raise ValueError("Trainable replica parameter signatures differ")
    with torch.no_grad():
        for _, parameter in named:
            dist.broadcast(parameter, src=0)

    class Replica(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    return Replica()


class AdaptiveExecution:
    """Wrap only the decoder forward and scaler unscale boundary of this trainer.

    Rank 1 waits for rank 0's backward AND cache cleanup before its forward.
    Both modes explicitly average scaled gradients before the existing unscale.
    There is no DDP wrapper, reducer, bucket storage or implicit collective.
    """

    def __init__(self, torch, replica, scaler, parameters, *, policy, memory_reader,
                 clear_cache, emit_event, pinned_reader):
        self.torch, self.replica, self.scaler = torch, replica, scaler
        self.dist = torch.distributed
        if self.dist.get_world_size() != 2:
            raise ValueError("Adaptive execution requires exactly two DDP ranks")
        self.rank = self.dist.get_rank()
        self.parameters = parameters
        self.policy, self.memory_reader = policy, memory_reader
        self.clear_cache, self.emit = clear_cache, emit_event
        self.pinned_reader = pinned_reader
        self.original_forward, self.original_unscale = replica.forward, scaler.unscale_
        self.completed_steps, self.mode = 0, None
        self.previous_observation = None
        replica.forward, scaler.unscale_ = self.forward, self.unscale

    def forward(self, *args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Expected one input trajectory per rank")
        local = {"tokens": int(ids.shape[1]), "previous": self.previous_observation}
        gathered = [None, None]
        # Both ranks have passed the existing before-forward cache cleanup.
        self.dist.all_gather_object(gathered, local)
        decision = [None]
        if self.rank == 0:
            for value in gathered:
                if value["previous"] is not None:
                    self.policy.observe(*value["previous"])
            decision[0] = self.policy.decide([v["tokens"] for v in gathered],
                                           self.memory_reader())
        self.dist.broadcast_object_list(decision, src=0)
        self.mode = decision[0]["mode"]
        self.tokens = local["tokens"]
        self.emit({"event": "batch_execution_mode", "rank": self.rank,
                   "global_step": self.completed_steps + 1,
                   "gradient_sync": "manual_mean", "ddp_bucket_bytes": 0, **decision[0]})
        if self.mode == "stop":
            raise MemoryError("Host RAM precheck rejected this batch before forward: "
                              + decision[0]["reason"])
        if self.mode == "serial" and self.rank == 1:
            self.dist.barrier()  # Rank 0 releases us only after local backward.
        return self.original_forward(*args, **kwargs)

    def unscale(self, optimizer):
        self.previous_observation = (self.tokens, self.pinned_reader())
        if self.mode == "serial":
            self.clear_cache()
            if self.rank == 0:
                self.dist.barrier()  # Release rank 1's forward.
            self.dist.barrier()  # Both local backwards and cache cleanups finished.
        # Agree before per-parameter collectives, even for malformed graphs.
        invalid = self.torch.tensor(
            int(any(p.grad is None or p.grad.is_sparse for p in self.parameters)),
            device=self.parameters[0].device,
        )
        self.dist.all_reduce(invalid, op=self.dist.ReduceOp.MAX)
        if bool(invalid):
            raise RuntimeError("Adaptive update requires dense gradients on both ranks")
        for parameter in self.parameters:
            self.dist.all_reduce(parameter.grad)
            parameter.grad.div_(2)
        result = self.original_unscale(optimizer)
        self.completed_steps += 1
        return result


def install_adaptive_execution(torch, runtime, manager_class, emit_event):
    """Install before the immutable epoch script imports build_replica."""
    original_build, original_load = runtime.build_replica, manager_class.load
    if getattr(original_build, "_first_run_adaptive_execution", False):
        return
    required = {"model": "Qwen/Qwen3.5-4B", "compute_dtype": "float16",
                "saved_inputs": "CPU pinned", "rank": 32, "world_size": 2,
                "gradient_checkpointing": "nonreentrant", "delta_rule_block_size": 1024}
    if any(runtime.CONFIG.get(k) != v for k, v in required.items()):
        raise ValueError("Host memory calibration does not cover this runtime configuration")
    controller = None

    @wraps(original_build)
    def build(rank):
        nonlocal controller
        from host_memory_recovery import cgroup_memory, clear_host_cache

        # Keep the actual DDP class object intact: model-loading dependencies may
        # import it and use isinstance checks. Returning a plain Module from its
        # __new__ skips DDP.__init__, without invalidating those imported aliases.
        # No reducer is allocated, even transiently. This worker constructs one
        # replica on its main thread; restore the allocator before training.
        original_ddp = torch.nn.parallel.DistributedDataParallel
        had_new = "__new__" in original_ddp.__dict__
        original_new = original_ddp.__dict__.get("__new__")
        constructions = 0

        def construct(cls, module, **kwargs):
            nonlocal constructions
            if cls is not original_ddp or constructions != 0:
                raise ValueError("Unexpected additional DDP construction")
            constructions += 1
            return manual_replica(torch, module, **kwargs)

        original_ddp.__new__ = staticmethod(construct)
        try:
            result = original_build(rank)
        finally:
            if had_new:
                original_ddp.__new__ = original_new
            else:
                del original_ddp.__new__
        if constructions != 1:
            raise ValueError("Expected exactly one decoder replica construction")
        _, ddp, _, parameters, _, scaler = result
        if ddp.module.config.hidden_size != 2560:
            raise ValueError("Host memory calibration requires hidden width 2560")

        def pinned():
            stats = torch.cuda.memory.host_memory_stats()
            return stats.get("allocated_bytes.current")

        controller = AdaptiveExecution(
            torch, ddp, scaler, parameters, policy=MemoryPolicy(),
            memory_reader=cgroup_memory, pinned_reader=pinned, emit_event=emit_event,
            clear_cache=lambda: clear_host_cache(torch, "after_serial_backward", emit_event),
        )
        emit_event({"event": "manual_replica_initialized", "rank": rank,
                    "ddp_bucket_bytes": 0,
                    "trainable_parameters": sum(p.numel() for p in parameters),
                    "gradient_bytes": sum(p.numel() * p.element_size() for p in parameters),
                    "gradient_sync": "scaled SUM/world_size before unscale"})
        return result

    @wraps(original_load)
    def load(manager, *args, **kwargs):
        progress = original_load(manager, *args, **kwargs)
        if controller is not None:
            controller.completed_steps = progress["global_step"]
        return progress

    build._first_run_adaptive_execution = True
    runtime.build_replica, manager_class.load = build, load
