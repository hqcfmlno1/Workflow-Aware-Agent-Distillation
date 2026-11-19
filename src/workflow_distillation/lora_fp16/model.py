"""Compatibility import for :mod:`workflow_distillation.lora_sft.model`."""

import sys

from workflow_distillation.lora_sft import model as _implementation

sys.modules[__name__] = _implementation
