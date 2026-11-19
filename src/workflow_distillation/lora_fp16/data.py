"""Compatibility import for :mod:`workflow_distillation.lora_sft.data`."""

import sys

from workflow_distillation.lora_sft import data as _implementation

sys.modules[__name__] = _implementation
