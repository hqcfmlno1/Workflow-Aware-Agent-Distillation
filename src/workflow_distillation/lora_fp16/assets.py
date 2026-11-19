"""Compatibility import for :mod:`workflow_distillation.lora_sft.assets`."""

import sys

from workflow_distillation.lora_sft import assets as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
