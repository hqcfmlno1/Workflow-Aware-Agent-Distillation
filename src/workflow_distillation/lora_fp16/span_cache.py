"""Compatibility import for :mod:`workflow_distillation.span_anchor.span_cache`."""

import sys

from workflow_distillation.span_anchor import span_cache as _implementation

sys.modules[__name__] = _implementation
