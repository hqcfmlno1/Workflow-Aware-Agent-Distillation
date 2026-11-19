"""Compatibility import for :mod:`workflow_distillation.lora_sft.config`."""

import sys

from workflow_distillation.lora_sft import config as _implementation

sys.modules[__name__] = _implementation
