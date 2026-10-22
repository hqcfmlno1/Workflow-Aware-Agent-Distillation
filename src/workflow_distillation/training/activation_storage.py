"""Explicit saved-tensor contexts; no global torch monkey patch."""


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
        packed = torch.empty(
            tensor.size(), dtype=tensor.dtype, layout=tensor.layout, device="cpu", pin_memory=True
        )
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


def activation_context(mode, gpu_budget_bytes=0):
    if mode == "cpu":
        from torch.autograd.graph import save_on_cpu

        return save_on_cpu(pin_memory=True)
    return ActivationStorage(mode, gpu_budget_bytes=gpu_budget_bytes)
