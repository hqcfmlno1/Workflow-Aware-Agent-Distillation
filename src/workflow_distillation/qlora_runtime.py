"""The existing blocked QLoRA replica settings, shared by epoch training."""

from __future__ import annotations

from functools import partial

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CONFIG = {
    "model": MODEL, "revision": REVISION,
    "quantization": "NF4 double quantization", "compute_dtype": "float16",
    "adapter_dtype": "float32", "embedding_and_head_dtype": "float16",
    "rank": 32, "alpha": 64, "dropout": 0.05,
    "optimizer": "PagedAdamW32bit", "optimizer_state_dtype": "float32",
    "learning_rate": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
    "max_grad_norm": 1.0, "initial_gradient_scale": 8.0,
    "attention": "t4_repeated_kv", "loss_chunk_size": 64,
    "gradient_checkpointing": "nonreentrant", "saved_inputs": "CPU pinned",
    "delta_rule_block_size": 1024, "use_cache": False,
    "loss_weights": [0, 1, 0.75, 1.5, 2], "vision": "CPU offload",
    "world_size": 2, "per_gpu_batch_size": 1, "global_batch_size": 2,
    "gradient_accumulation_steps": 1, "epochs": 1, "scheduler": None,
    "ddp_find_unused_parameters": False, "ddp_broadcast_buffers": False,
    "ddp_gradient_as_bucket_view": True, "ddp_bucket_cap_mb": 25,
}


def build_replica(rank):
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.checkpoint import checkpoint
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from workflow_distillation.ddp_probe import replica_device_plan
    from workflow_distillation.optimized_qlora import blocked_delta_rule
    from workflow_distillation.t4_attention import register_t4_sdpa

    register_t4_sdpa()
    modeling_qwen3_5.torch_chunk_gated_delta_rule = partial(
        blocked_delta_rule, modeling_qwen3_5.torch_chunk_gated_delta_rule, block_size=1024,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float16,
        device_map=replica_device_plan(rank), attn_implementation="t4_repeated_kv",
        local_files_only=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["lm_head", "model.visual"],
            llm_int8_enable_fp32_cpu_offload=True,
        ),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    if embedding.weight is not head.weight or embedding.weight.requires_grad:
        raise RuntimeError("Expected frozen tied embedding/head")
    embedding.to(dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=(r"model\.language_model\.layers\.\d+\..*"
                        r"(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|"
                        r"in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)"),
    ))
    model.train()
    decoder = model.get_base_model().model.language_model
    for layer in decoder.layers:
        layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    parameters = [p for p in decoder.parameters() if p.requires_grad]
    device = torch.device(f"cuda:{rank}")
    if not parameters or any(p.dtype != torch.float32 or p.device != device for p in parameters):
        raise RuntimeError("Expected all FP32 adapters on the local GPU")
    ddp = DistributedDataParallel(
        decoder, device_ids=[rank], output_device=rank, broadcast_buffers=False,
        find_unused_parameters=False, gradient_as_bucket_view=True, bucket_cap_mb=25,
    )
    optimizer = bnb.optim.PagedAdamW32bit(
        parameters, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
    return model, ddp, head, parameters, optimizer, scaler
