"""Compatibility import for :mod:`workflow_distillation.span_anchor.span_run`."""

import sys

from workflow_distillation.span_anchor import span_run as _implementation

sys.modules[__name__] = _implementation
