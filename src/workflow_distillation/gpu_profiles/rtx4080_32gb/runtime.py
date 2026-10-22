"""Opt-in execution experiments; importing this module changes no global state."""

import copy
from contextlib import nullcontext


def flash_sdpa(module, query, key, value, attention_mask, dropout=0.0,
               scaling=None, is_causal=None, **kwargs):
    """Keep repeated-KV semantics, selecting Flash SDPA on CUDA explicitly."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if kwargs.get("output_attentions") or kwargs.get("cache") is not None:
        raise ValueError("Training attention cannot use attention outputs or caches")
    repeats = query.shape[1] // key.shape[1]
    if repeats * key.shape[1] != query.shape[1]:
        raise ValueError("Query heads must be divisible by KV heads")
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    causal = bool(causal and query.shape[2] > 1 and attention_mask is None)
    context = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if query.is_cuda else nullcontext()
    with context:
        result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=dropout,
            is_causal=causal, scale=scaling,
        )
    return result.transpose(1, 2).contiguous(), None


class ActivationStorage:
    """Save forward tensors on GPU, CPU, or GPU up to a bounded byte budget.

    Checkpoint recomputation still runs. This changes only saved-tensor placement.
    The budget counts retained tensors, conservatively including repeated saves.
    """

    def __init__(self, mode, *, gpu_budget_bytes=0):
        if mode not in {"cpu", "gpu", "hybrid"} or gpu_budget_bytes < 0:
            raise ValueError("Expected cpu, gpu or hybrid and a nonnegative byte budget")
        self.mode = mode
        self.gpu_budget_bytes = gpu_budget_bytes
        self.saved_bytes = 0
        self.gpu_saved_bytes = 0
        self.cpu_saved_bytes = 0

    def pack(self, tensor):
        size = tensor.numel() * tensor.element_size()
        self.saved_bytes += size
        keep = self.mode == "gpu" or (
            self.mode == "hybrid" and self.gpu_saved_bytes + size <= self.gpu_budget_bytes
        )
        if keep or not tensor.is_cuda:
            if tensor.is_cuda:
                self.gpu_saved_bytes += size
            return tensor.device, tensor.detach()
        self.cpu_saved_bytes += size
        import torch

        # Allocate pinned storage first, like native save_on_cpu. Moving to a
        # pageable CPU tensor and then pinning would copy every activation twice.
        packed = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout,
                             device="cpu", pin_memory=True)
        packed.copy_(tensor)
        return tensor.device, packed

    @staticmethod
    def unpack(saved):
        device, tensor = saved
        return tensor.to(device, non_blocking=True)

    def __enter__(self):
        from torch.autograd.graph import saved_tensors_hooks

        self.context = saved_tensors_hooks(self.pack, self.unpack)
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)


def execution_identity(parent, parent_sha, base_identity, profile, *, launcher_sha, runtime_sha):
    """Bind an explicit execution change to one exact existing optimizer boundary."""
    if (parent.get("complete") is not True or parent.get("world_size") != 2
            or parent.get("identity") != base_identity):
        raise ValueError("Execution parent does not match the verified base training identity")
    if (set(profile) != {"attention", "activation_storage", "gpu_budget_bytes"}
            or profile["attention"] not in {"flash_repeated_kv", "t4_repeated_kv"}
            or profile["activation_storage"] not in {"cpu", "gpu", "hybrid"}
            or type(profile["gpu_budget_bytes"]) is not int
            or profile["gpu_budget_bytes"] < 0):
        raise ValueError("Unsupported RTX4080 execution profile")
    result = copy.deepcopy(base_identity)
    result["script_sha256"] = launcher_sha
    result["settings"]["config"].update(
        attention=profile["attention"], saved_inputs=profile["activation_storage"],
    )
    result["settings"]["gpu_execution"] = {
        "profile": copy.deepcopy(profile), "runtime_sha256": runtime_sha,
        "parent_manifest_sha256": parent_sha, "parent_progress": copy.deepcopy(parent["progress"]),
        "gpu": "RTX4080 with at least 30 GiB", "bitwise_equivalent": False,
        "loss_optimizer_and_sample_order_changed": False,
    }
    binding = {"parent": {"manifest_sha256": parent_sha,
                          **{k: copy.deepcopy(parent[k]) for k in
                             ("identity", "progress", "world_size")}}}
    return result, binding


def install_execution(profile):
    """Install only inside the dedicated worker process, after identity validation."""
    import torch

    import workflow_distillation.t4_attention as attention

    if profile["attention"] == "flash_repeated_kv":
        attention.repeated_kv_sdpa = flash_sdpa
    mode = profile["activation_storage"]
    if mode != "cpu":
        class ProfileActivationStorage(ActivationStorage):
            def __init__(self, **kwargs):
                super().__init__(mode, gpu_budget_bytes=profile["gpu_budget_bytes"])

        # Recovery wraps __enter__ on the context TYPE to clear inactive pinned
        # memory. A factory function would break that existing protocol.
        torch.autograd.graph.save_on_cpu = ProfileActivationStorage
