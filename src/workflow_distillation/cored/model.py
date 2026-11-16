"""One frozen unquantized backbone, named independent FP32 LoRA adapters."""

from contextlib import contextmanager
from functools import partial
from inspect import unwrap

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


class Runtime:
    def __init__(self, base, config, *, count):
        if getattr(base, 'is_quantized', False):
            raise ValueError('CoReD requires an unquantized frozen base')
        base.requires_grad_(False)
        base.config.use_cache = False
        # Detached peer snapshots and checkpoint recomputation must define the same
        # deterministic policy. Diversity comes from supervision dropout, not model dropout.
        for module in base.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
            for name in ('attention_dropout', 'hidden_dropout'):
                if isinstance(getattr(module, name, None), float):
                    setattr(module, name, 0.)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.names = [f'expert{i}' for i in range(count)]
        self.model = get_peft_model(base, self._lora(config), adapter_name=self.names[0])
        for name in self.names[1:]:
            self.model.add_adapter(name, self._lora(config))
        core = self.model.get_base_model().model
        self.decoder = getattr(core, 'language_model', core)
        self.head = self.model.get_output_embeddings()
        if self.head.weight.requires_grad:
            raise ValueError('Trainable output head is unsupported')
        self.parameters = []
        for name, parameter in self.model.named_parameters():
            if 'lora_' in name:
                parameter.data = parameter.data.float()
                parameter.requires_grad_(True)
                self.parameters.append(parameter)
        if not self.parameters:
            raise ValueError('No LoRA parameters')
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate,
                                          weight_decay=config.weight_decay, foreach=False)
        self.last_input_ids = None

    @staticmethod
    def _lora(config):
        return LoraConfig(r=config.rank, lora_alpha=config.alpha, lora_dropout=0.,
                          bias='none', target_modules=list(config.target_modules),
                          task_type='CAUSAL_LM', init_lora_weights=True)

    @contextmanager
    def active(self, name):
        if name is None:
            with self.model.disable_adapter():
                yield
        else:
            self.model.set_adapter(name)
            yield

    def hidden(self, record):
        self.last_input_ids = record.input_ids
        device = self.decoder.get_input_embeddings().weight.device
        ids = torch.tensor(record.input_ids, device=device, dtype=torch.long)[None]
        return self.decoder(input_ids=ids, use_cache=False,
                            return_dict=True).last_hidden_state[0]

    @torch.no_grad()
    def snapshot(self, record, name, positions):
        self.model.eval()
        with self.active(name):
            h = self.hidden(record)
            rows = torch.tensor(positions, dtype=torch.long, device=h.device) - 1
            return h[rows].detach().cpu()

    def factors(self):
        result = []
        for name in self.names:
            result.append({path: (module.lora_A[name].weight, module.lora_B[name].weight)
                           for path, module in self.model.named_modules()
                           if hasattr(module, 'lora_A') and name in module.lora_A})
        return result

    def all_trainable(self):
        for p in self.parameters:
            p.requires_grad_(True)

    def adapter_state(self):
        return {name: {k: v.detach().cpu().clone() for k, v in
                       get_peft_model_state_dict(self.model, adapter_name=name).items()}
                for name in self.names}

    def restore_adapters(self, states):
        if set(states) != set(self.names):
            raise ValueError('Checkpoint expert names differ')
        for name, state in states.items():
            expected = get_peft_model_state_dict(self.model, adapter_name=name)
            if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
                raise ValueError('Checkpoint adapter tensor schema differs')
            set_peft_model_state_dict(self.model, state, adapter_name=name)


def load_base(path, config, device='cuda:0'):
    """Load local weights only; never download implicitly during a training stage."""
    from transformers import AutoConfig, AutoModelForCausalLM

    architecture = AutoConfig.from_pretrained(path, local_files_only=True)
    kwargs = dict(dtype=getattr(torch, config.dtype), local_files_only=True,
                  attn_implementation='sdpa')
    if architecture.model_type == 'qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5 import modeling_qwen3_5
        if config.gdn_backend == 'fla':
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
        else:
            from workflow_distillation.optimized_qlora import blocked_delta_rule
            kernel = unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)
            if not getattr(kernel, '_blocked_training_kernel', False):
                blocked = partial(blocked_delta_rule, kernel, block_size=1024)
                blocked._blocked_training_kernel = True
                modeling_qwen3_5.torch_chunk_gated_delta_rule = blocked
        # Keep the unused vision tower on CPU. The decoder and tied head share one GPU.
        kwargs['device_map'] = {'model.visual': 'cpu', 'model.language_model': device,
                                'lm_head': device}
        return Qwen3_5ForConditionalGeneration.from_pretrained(path, **kwargs)
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device)
