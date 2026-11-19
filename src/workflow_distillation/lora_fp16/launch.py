"""Compatibility import for :mod:`workflow_distillation.lora_sft.launch`."""

import sys

from workflow_distillation.lora_sft import launch as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
