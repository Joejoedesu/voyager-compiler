"""INT8 policy for shared PT2E; no calibration loop, weight packing or ISA lowering."""

from dataclasses import replace

import torch
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)

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
from voyager_compiler.quantization.fake_quantize import _FakeQuantize
from voyager_compiler.quantization.qspec import QScheme
from voyager_compiler.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
)
from voyager_compiler.quantization.regions import annotate_transfer_domains

from .semantics import OPS


class _ReLU6Observer(_FakeQuantize):
    """Fixed output units: real six is the signed INT8 saturation endpoint."""

    def __init__(self, dtype, **kwargs):
        super().__init__(dtype, **kwargs)
        self.scale.fill_(6.0 / 127)
        self.disable_observer()

    def calculate_qparams(self):
        return self.scale

    def forward(self, x):
        if not self.fake_quant_enabled:
            return x
        return (x / self.scale).round().clamp(-128, 127) * self.scale


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

    def __init__(self, *, local_domains=False, relu6_saturation=False):
        super().__init__()
        self.local_domains = local_domains
        self.relu6_saturation = relu6_saturation

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
        rules = DOMAIN_RULES
        if self.local_domains:
            # Global average may store directly in a separately calibrated
            # output domain, using the accumulator's single final multiplier.
            rules = PreparedDomainRuleSet(
                tuple(
                    r
                    for r in rules.rules
                    if r.target != aten.adaptive_avg_pool2d.default
                )
            )
        annotate_transfer_domains(model, rules, spec, output_spec, filter_fn)
        if self.local_domains:
            for node in model.graph.nodes:
                if node.target != aten.adaptive_avg_pool2d.default or (
                    filter_fn and not filter_fn(node)
                ):
                    continue
                existing = node.meta.get("quantization_annotation")
                if existing is not None and existing._annotated:
                    continue
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={node.args[0]: spec},
                    output_qspec=output_spec,
                    _annotated=True,
                )
        result = super()._annotate_all_static_patterns(
            model, quantization_config, filter_fn
        )
        if self.relu6_saturation and quantization_config:
            for node in model.graph.nodes:
                if node.target not in (
                    aten.hardtanh.default,
                    aten.hardtanh_.default,
                ) or tuple(node.args[1:]) != (0.0, 6.0):
                    continue
                if filter_fn and not filter_fn(node):
                    continue
                a = node.meta.get("quantization_annotation")
                if (
                    a is None
                    or a.output_qspec != output_spec
                    or output_spec is None
                ):
                    continue
                if (
                    output_spec.dtype != "int8"
                    or output_spec.quant_max != 127
                    or output_spec.qscheme != QScheme.PER_TENSOR_SYMMETRIC
                ):
                    continue
                a.output_qspec = replace(
                    output_spec,
                    observer_or_fake_quant_ctr=_ReLU6Observer,
                    power_2_scale=False,
                )
                source = node.args[0]
                b = source.meta.get("quantization_annotation")
                if (
                    source.target in (aten.conv2d.default, aten.linear.default)
                    and len(source.users) == 1
                    and b is not None
                    and b.output_qspec == output_spec
                    and a.input_qspec_map.get(source) == spec
                ):
                    # Quantize the activated result, not a clipped intermediate.
                    b.output_qspec = None
                    a.input_qspec_map[source] = None
        if self.local_domains and quantization_config:
            for node in model.graph.nodes:
                if (
                    node.target
                    not in (aten.conv2d.default, aten.linear.default)
                    or len(node.users) != 1
                ):
                    continue
                user = next(iter(node.users))
                if user.target not in (aten.relu.default, aten.relu_.default):
                    continue
                if filter_fn and (not filter_fn(node) or not filter_fn(user)):
                    continue
                a = node.meta.get("quantization_annotation")
                b = user.meta.get("quantization_annotation")
                if (
                    a is None
                    or b is None
                    or a.output_qspec != output_spec
                    or b.input_qspec_map.get(node) != spec
                ):
                    continue
                # Calibrate the actual fused output; retain explicit overrides.
                a.output_qspec = None
                b.input_qspec_map[node] = None
                b.output_qspec = output_spec
        return result

    def annotate(self, model):
        model = super().annotate(model)
        if not self.local_domains:
            return model
        joins = (aten.add.Tensor, aten.add_.Tensor)

        def root_spec(spec, seen=()):
            if not isinstance(spec, SharedQuantizationSpec):
                return spec
            ref = spec.edge_or_node
            if ref in seen:
                raise ValueError("Cyclic quantization sharing")
            source, user = ref if isinstance(ref, tuple) else (None, ref)
            annotation = user.meta.get("quantization_annotation")
            if annotation is None:
                return None
            value = (
                annotation.input_qspec_map.get(source)
                if source is not None
                else annotation.output_qspec
            )
            return root_spec(value, (*seen, ref))

        for node in model.graph.nodes:
            a = node.meta.get("quantization_annotation")
            if a is None:
                continue
            # Implicit sibling-edge unions can otherwise connect successive
            # residual blocks, even when the add itself disables sharing.
            a.allow_implicit_sharing = False
            for source, qspec in list(a.input_qspec_map.items()):
                b = source.meta.get("quantization_annotation")
                if b is None or qspec is None or b.output_qspec is None:
                    continue
                if root_spec(qspec) != root_spec(b.output_qspec):
                    continue
                if node.target in joins:
                    if len(source.users) == 1 and source.target in (
                        aten.conv2d.default,
                        aten.linear.default,
                    ):
                        # Produce directly in this add's units; a shared
                        # identity branch keeps its own earlier domain.
                        b.output_qspec = SharedQuantizationSpec((source, node))
                elif not isinstance(qspec, SharedQuantizationSpec):
                    a.input_qspec_map[source] = SharedQuantizationSpec(source)
        return model


def default_quantizer(prepared, options):
    if any(
        n.target in (aten._int_mm.default, *OPS)
        for n in prepared.graph.graph.nodes
    ):
        return (
            None  # Integer workloads already carry their arithmetic contract.
        )
    local_domains = options.get("gemmini_local_domains", False)
    spec = (
        f"int8,qs=per_tensor_symmetric,qmax=127,pot={0 if local_domains else 1}"
    )
    common = get_default_quantizer(
        input_activation=spec, output_activation=spec, weight=spec, bias="int32"
    )
    config = common.object_type_config[aten.conv2d.default]
    return Int8Quantizer(
        local_domains=local_domains, relu6_saturation=True
    ).set_global(config)


POLICY = QuantizationPolicy(
    quantizer=default_quantizer,
    prepare=observe_without_fake_quant,
    finalize=finalize_amax_observers,
    bias="int32",
    example_calibration=True,
    integer_rounding=True,
)
