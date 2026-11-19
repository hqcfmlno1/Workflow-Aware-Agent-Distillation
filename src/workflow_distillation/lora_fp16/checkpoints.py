"""Compatibility import for :mod:`workflow_distillation.lora_sft.checkpoints`."""

import sys

from workflow_distillation.lora_sft import checkpoints as _implementation

sys.modules[__name__] = _implementation
