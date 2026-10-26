"""INT8 policy for shared PT2E; no calibration loop, weight packing or ISA lowering."""

import torch

from voyager_compiler.extensions import QuantizationPolicy
from voyager_compiler.ir.graph import (
    PreparedDomainInputFormat as Inputs,
)
from voyager_compiler.ir.graph import (
    PreparedDomainRelation as Relation,
)
from voyager_compiler.ir.graph import (
    PreparedDomainRuleSet,
)
from voyager_compiler.ir.graph import (
    PreparedDomainTransferRule as Rule,
)
from voyager_compiler.quantization import get_default_quantizer
from voyager_compiler.quantization.calibration import (
    finalize_amax_observers,
    observe_without_fake_quant,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
)
from voyager_compiler.quantization.regions import annotate_transfer_domains

from .semantics import OPS

aten = torch.ops.aten
# Shared observers establish an INT8 boundary at average pooling and a common
# residual-add domain. Layout/view transfers use the shared graph normalization.
DOMAIN_RULES = PreparedDomainRuleSet(
    tuple(
        Rule(op, Inputs.FIRST, Relation.ALIAS)
        for op in (
            aten.relu.default,
            aten.relu_.default,
            aten.max_pool2d.default,
            aten.adaptive_avg_pool2d.default,
        )
    )
    + tuple(
        Rule(op, Inputs.PREFIX, Relation.JOIN, input_count=2)
        for op in (aten.add.Tensor, aten.add_.Tensor)
    )
)


class Int8Quantizer(XNNPACKQuantizer):
    """Retain shared per-module/type/operator/order precedence and annotations."""

    def _annotate_all_static_patterns(
        self, model, quantization_config, filter_fn=None
    ):
        spec = (
            quantization_config.input_activation
            if quantization_config
            else None
        )
        output_spec = (
            quantization_config.output_activation
            if quantization_config
            else None
        )
        annotate_transfer_domains(
            model, DOMAIN_RULES, spec, output_spec, filter_fn
        )
        return super()._annotate_all_static_patterns(
            model, quantization_config, filter_fn
        )


def default_quantizer(prepared, options):
    if any(
        n.target in (aten._int_mm.default, *OPS)
        for n in prepared.graph.graph.nodes
    ):
        return (
            None  # Integer workloads already carry their arithmetic contract.
        )
    spec = "int8,qs=per_tensor_symmetric,qmax=127,pot=1"
    common = get_default_quantizer(
        input_activation=spec, output_activation=spec, weight=spec, bias="int32"
    )
    config = common.object_type_config[aten.conv2d.default]
    return Int8Quantizer().set_global(config)


POLICY = QuantizationPolicy(
    quantizer=default_quantizer,
    prepare=observe_without_fake_quant,
    finalize=finalize_amax_observers,
    bias="int32",
    example_calibration=True,
    integer_rounding=True,
)
