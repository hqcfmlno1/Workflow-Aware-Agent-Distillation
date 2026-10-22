"""Build the frozen NF4 base and FP32 adapters directly; no DDP constructor patch."""

from dataclasses import dataclass
from functools import partial

from workflow_distillation.ddp_probe import replica_device_plan
from workflow_distillation.training.config import LORA_TARGETS, MODEL, REVISION


@dataclass
class ModelState:
    model: object
    decoder: object
    head: object
    parameters: list
    optimizer: object
    scaler: object


def model_kwargs(rank):
    return dict(
        pretrained_model_name_or_path=MODEL,
        revision=REVISION,
        device_map=replica_device_plan(rank),
        attn_implementation="t4_repeated_kv",
        local_files_only=True,
    )


def build_model(rank):
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.checkpoint import checkpoint
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from workflow_distillation.optimized_qlora import blocked_delta_rule
    from workflow_distillation.t4_attention import register_t4_sdpa
    from workflow_distillation.training.distributed import initialize_replica

    register_t4_sdpa()
    # Transformers looks up this kernel globally in the GDN forward. Install
    # once during explicit model setup; never alter torch/DDP/context classes.
    kernel = modeling_qwen3_5.torch_chunk_gated_delta_rule
    if not getattr(kernel, "_blocked_training_kernel", False):
        blocked = partial(blocked_delta_rule, kernel, block_size=1024)
        blocked._blocked_training_kernel = True
        modeling_qwen3_5.torch_chunk_gated_delta_rule = blocked
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        **model_kwargs(rank),
        dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["lm_head", "model.visual"],
            llm_int8_enable_fp32_cpu_offload=True,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    if embedding.weight is not head.weight or embedding.weight.requires_grad:
        raise RuntimeError("Expected frozen tied embedding/head")
    embedding.to(dtype=torch.float16)
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        ),
    )
    model.train()
    decoder = model.get_base_model().model.language_model
    for layer in decoder.layers:
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    parameters = [p for p in decoder.parameters() if p.requires_grad]
    device = torch.device(f"cuda:{rank}")
    if not parameters or any(p.dtype != torch.float32 or p.device != device for p in parameters):
        raise RuntimeError("Expected all FP32 adapters on the local GPU")
    initialize_replica(decoder)
    optimizer = bnb.optim.PagedAdamW32bit(
        parameters,
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
    return ModelState(model, decoder, head, parameters, optimizer, scaler)
