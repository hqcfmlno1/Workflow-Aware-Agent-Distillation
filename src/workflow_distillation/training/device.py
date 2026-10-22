"""Device operations separated from objective/update logic for CPU verification."""

from contextlib import nullcontext


class DeviceRuntime:
    def __init__(self, device):
        self.device = device

    def synchronize(self):
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def clear_cache(self):
        if self.device.type == "cuda":
            import torch

            from workflow_distillation.training.memory import clear_inactive_cache

            clear_inactive_cache(torch)

    def prepare_step(self):
        if self.device.type == "cuda":
            import torch

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.synchronize()

    def autocast(self):
        if self.device.type == "cuda":
            import torch

            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def pinned_bytes(self):
        if self.device.type != "cuda":
            return None
        import torch

        return torch.cuda.memory.host_memory_stats().get("allocated_bytes.current")

    def memory_metrics(self):
        if self.device.type != "cuda":
            return {}
        import torch

        free, total = torch.cuda.mem_get_info(self.device)
        return {
            "peak_allocated": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved": torch.cuda.max_memory_reserved(self.device),
            "cuda_free": free,
            "cuda_capacity": total,
        }
