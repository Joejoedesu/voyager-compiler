"""PT2E annotation adapter for the shared, declarative domain-transfer walk.

Targets choose which operators preserve or join domains. Existing annotations
(including an explicitly disabled region) are boundaries, never overwritten.
Arithmetic that requires an output rounding boundary must be declared explicitly.
"""

from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)

from voyager_compiler.ir.graph import QuantizationContractTransferGraph


def annotate_transfer_domains(
    graph, rules, qspec, output_qspec, filter_fn=None
):
    walk = QuantizationContractTransferGraph(graph.graph.nodes, rules)
    for node in walk.nodes:
        transfer = walk.transfer(node)
        if transfer is None or (filter_fn and not filter_fn(node)):
            continue
        existing = node.meta.get("quantization_annotation")
        if existing is not None and existing._annotated:
            continue
        inputs = transfer.inputs
        shared = SharedQuantizationSpec((inputs[0], node)) if qspec else None
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                n: qspec if n is inputs[0] else shared for n in inputs
            },
            output_qspec=shared if output_qspec == qspec else output_qspec,
            _annotated=True,
        )
