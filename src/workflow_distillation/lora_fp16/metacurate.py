"""Compatibility import for :mod:`workflow_distillation.lora_sft.metacurate`."""

import sys

from workflow_distillation.lora_sft import metacurate as _implementation

sys.modules[__name__] = _implementation
