"""Compatibility import for :mod:`workflow_distillation.span_anchor.span_scoring`."""

import sys

from workflow_distillation.span_anchor import span_scoring as _implementation

sys.modules[__name__] = _implementation
