# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Select Alpha streamed calls before allocation; CMOD emission only encodes."""

import math
import re

import torch

from voyager_compiler.ir.buffer import TargetInvoke

from .layout import operand_bindings, output_buffers, region_nodes


def bits(node):
    dtype = str(node.meta.get("dtype") or node.value.dtype)
    if dtype in ("int4", "torch.int4"):
        return 4
    if dtype in ("int8", "torch.int8", "uint8", "torch.uint8"):
        return 8
    raise ValueError(
        f"{node.name}: Alpha payload has unsupported dtype {dtype}"
    )


def local_factors(
    input_shape, weight_shape, output_shape, precision, stride, config
):
    """Complete streamed C/K; choose spatial tiles fitting local memories."""
    c0, k0 = config.input_lanes(precision), config.output_lanes(precision)
    c, k = input_shape[1], output_shape[1]
    p, q = output_shape[2:]
    r, s = weight_shape[2:]
    if c % c0:
        raise ValueError("bufferized Alpha PE input must fill its input lanes")
    if k % k0 and not (precision == 8 and k == 16):
        raise ValueError(
            "bufferized Alpha PE output must fill lanes or use compact K16"
        )
    collector = config.memory_instance("output_collector").size.value // 128
    input_depth = config.memory_instance("input_buffer").size.value // 16
    for q1 in (d for d in range(min(q, collector), 0, -1) if q % d == 0):
        p_candidates = (
            (d for d in range(min(p, collector // q1), 0, -1) if p % d == 0)
            if r == s == stride == 1 and q1 == q
            else (1,)
        )
        for p1 in p_candidates:
            if ((p1 - 1) * stride + r) * ((q1 - 1) * stride + s) * (
                c // c0
            ) <= input_depth:
                return dict(
                    c1=c // c0,
                    c2=1,
                    c3=1,
                    k1=math.ceil(k / k0),
                    k2=1,
                    k3=1,
                    p0=p1,
                    p1=p1,
                    p2=p // p1,
                    p2s=1,
                    p3=1,
                    q0=q1,
                    q1=q1,
                    q2=q // q1,
                    q2s=1,
                    q3=1,
                )
    raise ValueError(
        "no Alpha spatial tile fits the input buffer and collector"
    )


def select_schedules(program, config):
    """Bind every native instruction's shape, fields and concrete buffer roles."""
    names = {
        n: f"{scope}/{n.name}"
        for scope, module in program.graph.named_modules()
        if isinstance(module, torch.fx.GraphModule)
        for n in module.graph.nodes
    }
    for call in program.graph.graph.nodes:
        bindings = operand_bindings(call)
        body = region_nodes(call)
        conv = next(
            (
                n
                for n in body
                if n.target
                in (
                    torch.ops.magnet_alpha.conv2d_int8_accumulator.default,
                    torch.ops.magnet_alpha.conv2d_vsq.default,
                )
            ),
            None,
        )
        ppu = next(
            (
                n
                for n in body
                if n.target == torch.ops.magnet_alpha.ppu_fixed_relu.default
            ),
            None,
        )
        nmp = next(
            (
                n
                for n in body
                if n.target
                == torch.ops.magnet_alpha.nmp_fixed_requantize.default
            ),
            None,
        )
        pvs = next(
            (
                n
                for n in body
                if n.target == torch.ops.magnet_alpha.pvs_quantize.default
            ),
            None,
        )
        buffers = output_buffers(call)
        fields, addresses, payloads = {}, [], []
        name = call.meta.get("scope", (call.name,))[0]

        def external(n):
            return bindings.get(n, n)

        def address(field, n, entry=0):
            addresses.append(
                dict(
                    field=field,
                    buffer=names[external(n)],
                    entry_offset=entry,
                    bits=bits(external(n)),
                )
            )

        def payload(
            filename,
            n,
            precision,
            encoding="plain",
            path=(),
            channel_slice=None,
            physical=None,
        ):
            physical = external(n) if physical is None else physical
            payloads.append(
                dict(
                    file=filename,
                    node=names[external(n)],
                    path=path,
                    bits=precision,
                    encoding=encoding,
                    channel_slice=channel_slice,
                    physical_shape=(
                        physical.meta.get("physical_shape")
                        if encoding == "plain" and channel_slice is None
                        else None
                    ),
                )
            )

        value = call.value if hasattr(call, "value") else None
        output = value[-1] if isinstance(value, (tuple, list)) else value
        if conv is not None:
            source, weight = external(conv.args[0]), external(conv.args[1])
            input_shape, weight_shape, output_shape = (
                tuple(source.value.shape),
                tuple(weight.value.shape),
                tuple(output.shape),
            )
            if len(input_shape) != 4 or input_shape[0] != 1:
                raise ValueError(f"{call.name}: Alpha requires batch-one NCHW")
            stride, padding, dilation, groups = conv.args[3:7]
            if stride[0] != stride[1] or tuple(dilation) != (1, 1):
                raise ValueError(f"{call.name}: unsupported Alpha spatial mode")
            epilogue = ppu if ppu is not None else nmp
            if epilogue is None:
                raise ValueError(f"{call.name}: missing fixed-point epilogue")
            precision = int(epilogue.kwargs["output_precision"])
            if (precision == 4) != (pvs is not None):
                raise ValueError(
                    f"{call.name}: INT4 output requires its PVS component encoding"
                )
            name = re.split(r"_weight(?:_|$)", weight.name)[0] + "_fused"
            fields.update(
                batch_size=input_shape[0],
                r=weight_shape[2],
                s=weight_shape[3],
                stride=stride[0],
                ia_precision=bits(source),
                wgt_precision=bits(weight),
                oa_precision=precision,
                oa_scale_precision=8 if pvs is not None else 0,
                padding_c=0,
                padding_k=0,
                pad_h_up=padding[0],
                pad_w_left=padding[1],
                pad_h_down=max(
                    0,
                    (output_shape[2] - 1) * stride[0]
                    + weight_shape[2]
                    - input_shape[2]
                    - padding[0],
                ),
                pad_w_right=max(
                    0,
                    (output_shape[3] - 1) * stride[1]
                    + weight_shape[3]
                    - input_shape[3]
                    - padding[1],
                ),
                input_height=input_shape[2],
                input_width=input_shape[3],
            )
            address("inputs0_baseaddr", source)
            address("outputs0_baseaddr", buffers[-1])
            payload("1_inputs.txt", source, bits(source))
            payload("1_weights.txt", weight, bits(weight))
            payload(
                "1_outputs.txt",
                call,
                precision,
                path=(2,) if pvs is not None else (),
                physical=buffers[-1],
            )
            if pvs is not None:
                payload(
                    "1_outputs_vector_scale.txt",
                    call,
                    8,
                    "activation_scale",
                    path=(1,),
                )
            if ppu is not None:
                if groups != 1 or not ppu.kwargs.get("integer_accumulator"):
                    raise ValueError(
                        f"{call.name}: invalid exact PE instruction"
                    )
                kind = "gemm_convolution"
                fields.update(
                    local_factors(
                        input_shape,
                        weight_shape,
                        output_shape,
                        bits(source),
                        stride[0],
                        config,
                    )
                )
                fields.update(
                    wgt_scale_precision=8 if bits(weight) == 4 else 0,
                    ia_scale_precision=8 if bits(source) == 4 else 0,
                    scale_length=math.ceil(output_shape[1] / 32),
                    bias_length=(
                        math.ceil(output_shape[1] / 32)
                        if ppu.args[2] is not None
                        else 0
                    ),
                    scale_factor=0,
                    bias_factor=0,
                    wt_binary_pt_loc=ppu.kwargs["primary_shift"],
                    nonlinear_scale_factor=ppu.kwargs["secondary_multiplier"],
                    nonlinear_scale_shift=ppu.kwargs["secondary_shift"],
                    padding_k=(-output_shape[1]) % 32,
                )
                address("weights0_baseaddr", weight)
                payload("1_scale.txt", ppu.args[1], 16, "unsigned")
                if ppu.args[2] is not None:
                    payload("1_bias.txt", ppu.args[2], 8)
                for field, file, encoding in (
                    (
                        "input_scale",
                        "1_inputs_vector_scale.txt",
                        "activation_scale",
                    ),
                    (
                        "weight_scale",
                        "1_weights_vector_scale.txt",
                        "weight_scale",
                    ),
                ):
                    if conv.kwargs.get(field) is not None:
                        payload(file, conv.kwargs[field], 8, encoding)
                extra = [
                    n
                    for n in body
                    if n.op == "call_function" and n not in (conv, ppu, pvs)
                ]
                if extra:
                    raise ValueError(
                        f"{call.name}: unencoded PE members {[str(n.target) for n in extra]}"
                    )
            else:
                kind = "nmp_depthwise_convolution"
                if groups != input_shape[1] or weight_shape[1] != 1:
                    raise ValueError(f"{call.name}: invalid depthwise groups")
                fields.update(
                    is_dwconv=1,
                    nmp_scale_length=0,
                    nmp_scale_factor=nmp.kwargs["scale_factor"],
                    nmp_bias_factor=nmp.kwargs["bias_factor"],
                    wt_binary_pt_loc=nmp.kwargs["binary_point"],
                    p1=output_shape[2],
                    q1=output_shape[3],
                    c1=math.ceil(input_shape[1] / 32),
                    k1=math.ceil(output_shape[1] / 32),
                    groups=groups,
                    bias_length=0,
                    nonlinear_scale_factor=1,
                    nonlinear_scale_shift=0,
                )
                address("inputs1_baseaddr", weight)
            if epilogue.kwargs.get("relu"):
                fields["is_relu"] = 1
        elif (
            call.target
            == torch.ops.magnet_alpha.channel_shuffle_buffered.default
        ):
            kind = "nmp_channel_shuffle"
            source, second, secondary, _ = call.args
            shape = tuple(output.shape)
            if call.args[3] != 2 or shape[1] % 64:
                raise ValueError(f"{call.name}: unsupported shuffle geometry")
            fields.update(
                is_shuffling=1,
                p1=shape[2],
                q1=shape[3],
                c1=source.value.shape[1] // 32,
                k1=shape[1] // 32,
                batch_size=1,
                channels_per_group=shape[1] // 2,
                groups=2,
            )
            for field, n in (
                ("inputs0_baseaddr", source),
                ("inputs1_baseaddr", second),
                ("outputs0_baseaddr", buffers[0]),
                ("outputs1_baseaddr", secondary),
            ):
                address(field, n)
            for file, n in (
                ("1_inputs.txt", source),
                ("1_inputs2.txt", second),
                ("1_outputs.txt", call),
            ):
                payload(file, n, 8)
        elif (
            call.target
            == torch.ops.magnet_alpha.channel_shuffle_copy_buffered.default
        ):
            kind = "nmp_copy"
            facts = call.meta["target_layout"]
            shape = tuple(output.shape)
            fields.update(
                is_copy=1,
                p1=shape[2],
                q1=shape[3],
                c1=1,
                k1=1,
                c2=1,
                c3=1,
                k2=1,
                k3=1,
                p2=1,
                p3=1,
                q2=1,
                q3=1,
                stride=1,
                scale_length=0,
                wgt_precision=8,
                wgt_scale_precision=0,
                ia_scale_precision=0,
                nmp_bias_factor=0,
                is_to_concat=1,
                concat_total_size_0=shape[1],
                nmp_c2=shape[1] // 32 - 1,
                nmp_k2=shape[1] // 32 - 1,
            )
            address(
                "inputs0_baseaddr",
                call.args[facts["source_output"]],
                facts["source_entry"],
            )
            address(
                "outputs0_baseaddr", call.args[0], facts["destination_entry"]
            )
            block = facts["destination_entry"] * 32
            for file in ("1_inputs.txt", "1_outputs.txt"):
                payload(file, call, 8, channel_slice=(block, block + 32))
        elif call.target in (
            torch.ops.aten.max_pool2d.default,
            torch.ops.magnet_alpha.upsample_nearest2d.default,
        ):
            source = call.args[0]
            shape = tuple(output.shape)
            input_shape = tuple(source.value.shape)
            if bits(source) != 8 or bits(call) != 8:
                raise ValueError(f"{call.name}: NMP movement requires INT8")
            fields.update(
                batch_size=1,
                c1=math.ceil(input_shape[1] / 32),
                k1=math.ceil(shape[1] / 32),
                padding_c=0,
                padding_k=0,
            )
            if call.target == torch.ops.aten.max_pool2d.default:
                kind = "nmp_max_pool"
                kernel, stride, padding, dilation = call.args[1:5]
                if (
                    any(padding)
                    or tuple(dilation) != (1, 1)
                    or stride[0] != stride[1]
                ):
                    raise ValueError(
                        f"{call.name}: maxpool requires explicit padding and symmetric stride"
                    )
                fields.update(
                    is_pool=1,
                    is_maxpool=1,
                    is_pool_1bank=int(input_shape[1] <= 16),
                    pooling_kernel=kernel[0],
                    pooling_kernel_r=kernel[0],
                    pooling_kernel_s=kernel[1],
                    pooling_stride=stride[0],
                    p_pooled=shape[2],
                    q_pooled=shape[3],
                    p1=input_shape[2],
                    q1=input_shape[3],
                    nmp_bias_factor=0,
                )
            else:
                kind = "nmp_nearest_upsample"
                factor = shape[2] // input_shape[2]
                if (
                    shape[2] != input_shape[2] * factor
                    or shape[3] != input_shape[3] * factor
                    or factor < 1
                ):
                    raise ValueError(
                        f"{call.name}: unsupported nearest upsample"
                    )
                fields.update(
                    is_upsampling=1,
                    r=factor,
                    s=factor,
                    upsample_kernel=factor,
                    p1=shape[2],
                    q1=shape[3],
                )
            address("inputs0_baseaddr", source)
            address("outputs0_baseaddr", buffers[0])
            payload("1_inputs.txt", source, 8)
            payload("1_outputs.txt", call, 8)
        else:
            continue
        if conv is None:
            fields.update(
                bias_length=0,
                nmp_scale_length=0,
                nmp_scale_factor=1,
                ia_precision=8,
                oa_precision=8,
                oa_scale_precision=0,
                wt_binary_pt_loc=0,
                nonlinear_scale_factor=1,
                nonlinear_scale_shift=0,
            )
            if kind == "nmp_copy":
                fields.pop("nmp_scale_length", None)
                fields.pop("nmp_scale_factor", None)
        call.meta["target_schedule"] = dict(
            name=name,
            kind=kind,
            fields=fields,
            addresses=addresses,
            payloads=payloads,
        )
        if "target_invoke" not in call.meta:
            call.meta["target_invoke"] = TargetInvoke(
                "magnet-alpha.whole",
                kind,
                (),
                (),
                (),
                ("nmp",),
                scheduling_owner="hardware",
            )
