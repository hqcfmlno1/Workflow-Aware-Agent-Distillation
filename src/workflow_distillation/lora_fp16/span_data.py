"""Compatibility import for :mod:`workflow_distillation.span_anchor.span_data`."""

import sys

from workflow_distillation.span_anchor import span_data as _implementation

sys.modules[__name__] = _implementation
