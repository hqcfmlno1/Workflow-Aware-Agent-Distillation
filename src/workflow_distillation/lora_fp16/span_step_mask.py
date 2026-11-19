"""Compatibility import for :mod:`workflow_distillation.span_anchor.span_step_mask`."""

import sys

from workflow_distillation.span_anchor import span_step_mask as _implementation

sys.modules[__name__] = _implementation
