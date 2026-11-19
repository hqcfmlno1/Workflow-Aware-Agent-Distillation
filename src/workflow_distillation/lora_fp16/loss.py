"""Compatibility import for :mod:`workflow_distillation.lora_sft.loss`."""

import sys

from workflow_distillation.lora_sft import loss as _implementation

sys.modules[__name__] = _implementation
