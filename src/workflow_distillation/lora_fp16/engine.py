"""Compatibility import for :mod:`workflow_distillation.lora_sft.engine`."""

import sys

from workflow_distillation.lora_sft import engine as _implementation

sys.modules[__name__] = _implementation
