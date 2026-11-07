"""Select a base/LoRA ablation without editing or reloading checkpoint files."""

import math
from contextlib import contextmanager, nullcontext


@contextmanager
def inference_variant(model, *, mode="loaded", scale=1.0):
    if mode not in ("loaded", "base", "lora"):
        raise ValueError("adapter_mode must be loaded, base or lora")
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 0:
        raise ValueError("adapter_scale must be finite and nonnegative")
    has_adapter = bool(getattr(model, "peft_config", None))
    selected = ("lora" if has_adapter else "base") if mode == "loaded" else mode
    if selected == "lora" and not has_adapter:
        raise ValueError("This host has no LoRA adapter")
    if selected == "base" and scale != 1:
        raise ValueError("adapter_scale only applies to a LoRA request")
    snapshots = []
    try:
        if selected == "lora" and scale != 1:
            from peft.tuners.tuners_utils import BaseTunerLayer

            for module in model.modules():
                if isinstance(module, BaseTunerLayer) and hasattr(module, "scaling"):
                    snapshots.append((module, dict(module.scaling)))
                    for name in module.active_adapters:
                        module.set_scale(name, scale)
        context = model.disable_adapter() if has_adapter and selected == "base" else nullcontext()
        with context:
            yield {"adapter_mode": selected, "adapter_scale": scale if selected == "lora" else 0.0}
    finally:
        for module, original in snapshots:
            module.scaling.update(original)
