"""Compatibility import for :mod:`workflow_distillation.span_anchor.span_loss`."""

import sys

from workflow_distillation.span_anchor import span_loss as _implementation

sys.modules[__name__] = _implementation
