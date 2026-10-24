"""Unquantized frozen FP16 text base, FP32 LoRA and standard AdamW."""

from functools import partial
from inspect import unwrap

from workflow_distillation.optimized_qlora import blocked_delta_rule, device_plan
from workflow_distillation.training.config import LORA_TARGETS
from workflow_distillation.training.model import ModelState


def model_kwargs(config):
    return dict(
        pretrained_model_name_or_path=config.model,
        revision=config.revision,
        dtype="float16",
        device_map=device_plan(32),
        attn_implementation="sdpa",
        local_files_only=True,
    )


def build_model(config):
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.checkpoint import checkpoint
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if config.gdn_backend == "fla":
        # Explicit admission: do not silently fall back to the much slower scan
        # when a broken FLA install is present. No custom chunk wrapper on FLA.
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    else:
        kernel = unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)
        if not getattr(kernel, "_blocked_training_kernel", False):
            blocked = partial(blocked_delta_rule, kernel, block_size=1024)
            blocked._blocked_training_kernel = True
            modeling_qwen3_5.torch_chunk_gated_delta_rule = blocked
    model = Qwen3_5ForConditionalGeneration.from_pretrained(**model_kwargs(config))
    model.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        ),
    )
    decoder = model.get_base_model().model.language_model
    for layer in decoder.layers:
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if "lora_" not in name or parameter.device.type != "cuda":
                raise RuntimeError("Only text LoRA adapters may train on the local GPU")
            parameter.data = parameter.data.float()
        elif parameter.is_floating_point() and parameter.dtype != torch.float16:
            raise RuntimeError(f"Frozen base was upcast: {name}")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise RuntimeError("Quantization is forbidden for this experiment")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable adapters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
    model.train()
    return ModelState(model, decoder, model.get_output_embeddings(), parameters, optimizer, scaler)
