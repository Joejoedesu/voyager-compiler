"""Voyager's schedule on AWS Trainium.

The shared compiler tiles, bufferizes and plans a model for one NeuronCore
(``TrainiumConfig``, priced by ``TrainiumModel``), and the translator turns
the ``model.txt`` it writes into NKI kernels without deciding anything.
An NKI instruction that does the work of several ops is a fused group of
them, matched by one of ``INSTRUCTION_PATTERNS``.
"""

from .config import LAYOUT, TRN2, TRN3, TrainiumConfig
from .model import TrainiumModel, kernel_nodes, predicted_seconds
from .patterns import INSTRUCTION_PATTERNS
from .translate import immediates, record, translate

__all__ = [
    "INSTRUCTION_PATTERNS",
    "LAYOUT",
    "TRN2",
    "TRN3",
    "TrainiumConfig",
    "TrainiumModel",
    "immediates",
    "kernel_nodes",
    "predicted_seconds",
    "record",
    "translate",
]
