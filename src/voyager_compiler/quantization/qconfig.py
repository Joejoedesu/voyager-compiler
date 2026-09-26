import copy
from collections import namedtuple
from dataclasses import asdict

from torch import nn

from voyager_compiler.quantization.quantizer import QuantizationSpec

__all__ = [
    "QConfig",
    "get_qconfig",
]


class QConfig(namedtuple("QConfig", ["activation", "weight", "error"])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like
    MinMaxObserver) or a callable that returns instances on invocation, not
    the concrete observer instances themselves. Quantization preparation
    function will instantiate observers multiple times for each of the
    layers.


    Observer classes have usually reasonable default arguments, but they can
    be overwritten with `with_args` method (that behaves like
    functools.partial)::

      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8))

    """

    def __new__(cls, activation, weight, error):
        return super().__new__(cls, activation, weight, error)


def _create_fake_quant(quantization_spec, force_scale_power_of_two):
    if quantization_spec is None:
        return nn.Identity

    spec = QuantizationSpec.from_str(quantization_spec)
    kwargs = copy.deepcopy(asdict(spec))
    return spec.observer_or_fake_quant_ctr.with_args(
        **kwargs, force_scale_power_of_two=force_scale_power_of_two
    )


def get_qconfig(activation, weight, error, force_scale_power_of_two=False):
    return QConfig(
        activation=_create_fake_quant(activation, force_scale_power_of_two),
        weight=_create_fake_quant(weight, force_scale_power_of_two),
        error=_create_fake_quant(error, force_scale_power_of_two),
    )
