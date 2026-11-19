"""Compatibility import for :mod:`workflow_distillation.lora_sft.cli`."""

import sys

from workflow_distillation.lora_sft import cli as _implementation

sys.modules[__name__] = _implementation
