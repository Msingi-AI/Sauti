"""Sauti package."""

# Core utility (always expected)
from .data import get_waxal_swahili as prepare_waxal_dataset

# Optional components: import if present, but don't raise on missing modules
try:
    from .distill import Distiller
except Exception:
    Distiller = None

try:
    from .finetune import finetune_student
except Exception:
    finetune_student = None

try:
    from .inference import synthesize
except Exception:
    synthesize = None

__all__ = ["__version__"]
