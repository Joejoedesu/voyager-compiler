import functools
import logging
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchao.quantization.pt2e import FakeQuantizeBase, ObserverOrFakeQuantize

from voyager_compiler.ops import HADAMARD_SIZE, hadamard_rotate, qmap_levels
from voyager_compiler.quantization import ops
from voyager_compiler.quantization.dtypes import (
    quantize_to_minifloat,
    quantize_to_nf,
    quantize_to_posit,
)
from voyager_compiler.quantization.qspec import (
    QScheme,
    float_format,
    int_range,
    parse_codebook_dtype,
    posit_format,
)

__all__ = [
    "DirectCastFakeQuantize",
    "FusedAmaxObsFakeQuantize",
    "GroupWiseAffineFakeQuantize",
    "MXFakeQuantize",
    "_DerivedObserverOrFakeQuantize",
    "calculate_global_scale",
    "entry_levels",
    "fake_quantize_class",
    "hadamard_matrix",
]


logger = logging.getLogger(__name__)

# The random signs of the transform: Transformer Engine's fixed vector
# (``get_wgrad_sign_vector``), shared by every operand.
HADAMARD_SIGNS = (1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1)
# Stochastic rounding's seeds are drawn from [0, SEED_BOUND).
SEED_BOUND = 2**63 - 1


def hadamard_matrix(device=None) -> torch.Tensor:
    """The random Hadamard matrix ``diag(signs) @ H / 4``.

    ``H`` is Sylvester's 16x16 Hadamard matrix and ``signs`` is
    ``HADAMARD_SIGNS``; the matrix is orthogonal.

    Args:
        device: Where the matrix lives.

    Returns:
        A float32 ``HADAMARD_SIZE`` x ``HADAMARD_SIZE`` tensor.
    """
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < HADAMARD_SIZE:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)]
        )
    signs = torch.tensor(HADAMARD_SIGNS, dtype=torch.float)
    return (signs[:, None] * matrix / math.sqrt(HADAMARD_SIZE)).to(device)


@functools.lru_cache(maxsize=None)
def get_quantization_map(dtype, device=None):
    """Return the lookup table that quantizes a bfloat16 value to ``dtype``.

    Entry ``i`` is the value the bfloat16 whose bits read ``i`` rounds to,
    so quantizing is one gather.  Floats saturate: a value past the largest
    finite one, infinity included, takes it.  A lookup table's entries are
    NormalFloat's levels, snapped to its entry dtype's grid.  Each table is
    built once per dtype and device and shared by every caller, so it must
    not be modified in place.

    Args:
        dtype: ``int<N>``, ``uint<N>``, ``fp<B>_e<X>m<Y>``, ``posit<N>_<es>``
            or ``lut<I>[_to_<E>]``; None returns the identity.
        device: Where the table lives.

    Returns:
        A ``2**16``-entry bfloat16 tensor.

    Raises:
        ValueError: ``dtype`` names none of these.
    """
    values = torch.arange(2**16, dtype=torch.int16, device=device)
    values = values.view(torch.bfloat16)
    if dtype is None:
        return values

    if (bounds := int_range(dtype)) is not None:
        return torch.clamp(torch.round(values), *bounds)

    if (fmt := float_format(dtype)) is not None:
        if not fmt.signed:
            values = torch.abs(values)
        values = torch.clamp(values, -fmt.max_norm, fmt.max_norm)
        # The rounding runs in float32: bfloat16 cannot hold the halfway
        # sums it compares against.
        return quantize_to_minifloat(
            values.float(),
            fmt.mbits + 2,
            fmt.ebits,
            fmt.max_norm,
            round="even",
            saturate_normals=True,
        ).to(torch.bfloat16)

    if (posit := posit_format(dtype)) is not None:
        return quantize_to_posit(values, *posit, round_to_even=True)

    if (codebook := parse_codebook_dtype(dtype)) is not None:
        index_bits, entry = codebook
        grid = None if entry is None else entry_levels(entry, device)
        indices, levels = quantize_to_nf(values, index_bits, grid=grid)
        return levels[indices]

    raise ValueError(f"Unsupported dtype: {dtype}")


def entry_levels(dtype, device=None):
    """Return every value a lookup-table entry of ``dtype`` can hold.

    The grid is symmetric, so an integer's extra negative value is left out:
    ``int6`` spans [-31, 31], the range its ``quant_max`` gives.

    Args:
        dtype: The entry dtype, ``int<N>`` or ``fp<B>_e<X>m<Y>``.
        device: Where to build the grid.

    Returns:
        The distinct values in ascending order, as float32, with one zero.
    """
    levels = qmap_levels(get_quantization_map(dtype, device))
    return levels[levels >= -levels.max()]


def calculate_global_scale(
    input: torch.Tensor, quant_max: float, scale_max: float
) -> torch.Tensor:
    """NVFP4's per-tensor scale: ``amax(|input|) / (quant_max * scale_max)``.

    Dividing the block scales by it puts the tensor's largest one on the
    scale dtype's largest value.  An all-zero tensor gets float32's smallest
    normal value instead of zero, which its blocks then divide to nothing.

    Args:
        input: Tensor about to be quantized, or its largest magnitude.
        quant_max: Largest magnitude of the element dtype.
        scale_max: Largest value of the block scales' dtype.

    Returns:
        A 0-d tensor in ``input``'s dtype.
    """
    ratio = torch.amax(torch.abs(input)) / (quant_max * scale_max)
    return ratio.clamp_min(torch.finfo(torch.float32).tiny)


def _install(module: torch.nn.Module, name: str, value: torch.Tensor):
    """Copy ``value`` into ``module``'s buffer ``name`` if it has the
    buffer's shape and dtype, else replace the buffer with it."""
    buffer = getattr(module, name)
    if value.shape == buffer.shape and value.dtype == buffer.dtype:
        buffer.copy_(value)
    else:
        setattr(module, name, value.to(buffer.device))


def _axes(ch_axis: Union[int, List[int]]) -> List[int]:
    """A block spec's axis or axes as the list the ops take."""
    return [ch_axis] if isinstance(ch_axis, int) else list(ch_axis)


class _FreezableFlags(FakeQuantizeBase):
    """``FakeQuantizeBase`` whose enable flags are Python bools.

    The base class keeps ``fake_quant_enabled`` and ``observer_enabled`` in
    device tensors, so reading one on every call waits for the device.
    Here they are plain attributes that only the enable methods change,
    and they travel in the state dict as extra state.
    """

    def __init__(self) -> None:
        super().__init__()
        del self.fake_quant_enabled, self.observer_enabled
        self.fake_quant_enabled = True
        self.observer_enabled = True

    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled = enabled

    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled = enabled

    def get_extra_state(self):
        return {
            "fake_quant_enabled": self.fake_quant_enabled,
            "observer_enabled": self.observer_enabled,
        }

    def set_extra_state(self, state):
        self.fake_quant_enabled = state["fake_quant_enabled"]
        self.observer_enabled = state["observer_enabled"]

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # A checkpoint saved while the flags were buffers holds them so.
        if prefix + "fake_quant_enabled" in state_dict:
            state_dict[prefix + "_extra_state"] = {
                name: bool(state_dict.pop(prefix + name)[0])
                for name in ("fake_quant_enabled", "observer_enabled")
            }
        super()._load_from_state_dict(state_dict, prefix, *args)


class _FakeQuantize(_FreezableFlags):
    """The parts every fake-quant scheme shares.

    The dtype's lookup table and a ``scale``, which every scheme but
    microscaling reports.  A spec is passed whole to the class that
    implements its scheme, so each class takes the fields it uses and the
    rest fall through to ``**unused``.  A subclass fake-quantizes in
    ``forward`` and, unless it keeps none, reports its parameters from
    ``calculate_qparams``.

    Args:
        dtype: Element dtype, as a spec string names it.
        device: Where the buffers live.
        **unused: The spec's fields this scheme does not use.
    """

    qmap: torch.Tensor
    scale: torch.Tensor

    #: Buffers that take their shape from the first input; a loaded one is
    #: resized to the saved shape before its values are copied in.
    _lazily_sized = ("scale",)

    def __init__(
        self,
        dtype: str,
        device: Optional[torch.device] = None,
        **unused,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": torch.float}

        codebook = parse_codebook_dtype(dtype)
        self.is_codebook_quantization = codebook is not None
        qmap = get_quantization_map(dtype, device)
        if self.is_codebook_quantization:
            self.index_bits, self.code_dtype = codebook
            # A refit writes a codebook's table in place, so it is the
            # fake-quant's own.
            qmap = qmap.clone()
        self.register_buffer("qmap", qmap, persistent=False)
        self.register_buffer("scale", torch.tensor([1.0], **factory_kwargs))

    def extra_repr(self):
        return (
            f"fake_quant_enabled={self.fake_quant_enabled}, "
            f"observer_enabled={self.observer_enabled}, dtype={self.dtype}"
        )

    def install_qmap(self, qmap: torch.Tensor) -> None:
        """Make ``qmap`` the table the fake-quant rounds to.

        A table of the current one's shape and dtype is copied into it, so
        whatever holds the current tensor reads the new values; any other
        replaces it.  Only a codebook's fake-quant owns its table, so only
        a codebook's is installed.

        Args:
            qmap: A lookup table, or a codebook grid.
        """
        _install(self, "qmap", qmap)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for name in self._lazily_sized:
            key = prefix + name
            if key in state_dict:
                getattr(self, name).resize_(state_dict[key].shape)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class DirectCastFakeQuantize(_FakeQuantize):
    """Fake-quantize by rounding each value to the dtype, with no scale.

    For a spec with no scheme, such as ``fp8_e4m3`` or ``posit8_1``: every
    value rounds to the nearest one the dtype represents, ``scale`` stays 1
    and nothing is observed.

    Args:
        *args: Forwarded to ``_FakeQuantize``.
        **kwargs: Forwarded to ``_FakeQuantize``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.disable_observer()

    def calculate_qparams(self):
        return self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return x
        return ops.fake_quant(x, self.qmap, self.scale)


class FusedAmaxObsFakeQuantize(_FakeQuantize):
    """Fake-quantize against a scale observed from recent amaxes.

    The per-tensor and per-channel schemes keep the absolute maxima of the
    last ``amax_history_len`` inputs, one per tensor or one per channel
    along ``ch_axis``, and scale by the largest.  An input is quantized
    before its own amax joins the history, so the first call uses scale
    1.0, as Transformer Engine's delayed scaling does (whose default
    history is 1024 long).  The history and the scale are kept in the
    dtype of the tensor quantized.

    Args:
        dtype: Element dtype, as a spec string names it.
        qscheme: ``PER_TENSOR_SYMMETRIC`` or ``PER_CHANNEL_SYMMETRIC``.
        quant_max: Largest magnitude the dtype represents.
        amax_history_len: How many past amaxes the scale is taken over.
        ch_axis: The channel axis of a per-channel scale.
        power_2_scale: Round each scale up to a power of two.
        **kwargs: Forwarded to ``_FakeQuantize``.

    Raises:
        ValueError: ``qscheme`` is not per-tensor or per-channel.
    """

    amax_history: torch.Tensor

    _lazily_sized = ("scale", "amax_history")

    def __init__(
        self,
        dtype: str,
        qscheme: QScheme,
        quant_max: float,
        amax_history_len: int,
        ch_axis: Optional[int] = None,
        power_2_scale: bool = False,
        **kwargs,
    ) -> None:
        if qscheme not in (
            QScheme.PER_TENSOR_SYMMETRIC,
            QScheme.PER_CHANNEL_SYMMETRIC,
        ):
            raise ValueError(
                f"{qscheme} is fake-quantized by "
                f"{fake_quantize_class(qscheme).__name__}"
            )
        super().__init__(dtype, **kwargs)
        self.qscheme = qscheme
        self.quant_max = quant_max
        self.amax_history_len = amax_history_len
        self.ch_axis = ch_axis
        self.power_2_scale = power_2_scale
        self.is_per_channel = qscheme == QScheme.PER_CHANNEL_SYMMETRIC
        self.register_buffer(
            "amax_history",
            torch.tensor([], device=self.scale.device, dtype=torch.float),
        )

    def start_history(self, shape: Tuple[int, ...], dtype) -> None:
        """Size the history and the scale as before any input.

        Args:
            shape: Shape of one amax: ``()`` per tensor, or the channel
                shape per channel.
            dtype: Dtype of the tensor quantized.
        """
        device = self.scale.device
        self.amax_history = torch.zeros(
            (self.amax_history_len, *shape), dtype=dtype, device=device
        )
        self.scale = torch.ones(shape, dtype=dtype, device=device)

    @torch.no_grad()
    def _update_amax_scale(self, x: torch.Tensor) -> None:
        """Fold ``x``'s amax into the history and recompute ``scale``.

        The absolute maximum is taken over every axis, or over every axis
        but ``ch_axis`` when the observer is per-channel. The history and
        the scale are started on the first call, since the observed shape
        and dtype are only known once a tensor arrives.

        Args:
            x: The tensor being observed.
        """
        if self.is_per_channel:
            ch_axis = self.ch_axis % x.ndim
            dim = tuple(i for i in range(x.ndim) if i != ch_axis)
            amax_cur = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        else:
            amax_cur = torch.amax(torch.abs(x))

        if self.amax_history.numel() == 0:
            self.start_history(amax_cur.shape, x.dtype)

        amax = torch.amax(self.amax_history, dim=0)
        self.amax_history.copy_(torch.roll(self.amax_history, -1, 0))
        self.amax_history[0] = amax_cur

        sf = amax / self.quant_max
        sf = torch.where(amax > 0.0, sf, self.scale)
        sf = torch.where(torch.isfinite(amax), sf, self.scale)
        if self.power_2_scale:
            sf = torch.pow(2, torch.ceil(torch.log2(sf)))
        self.scale.copy_(sf)

    def calculate_qparams(self):
        return self.scale

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, scale={self.scale}, "
            f"qscheme={self.qscheme}, quant_max={self.quant_max}, "
            f"ch_axis={self.ch_axis}, "
            f"amax_history_len={self.amax_history_len}, "
            f"power_2_scale={self.power_2_scale}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.observer_enabled:
            self._update_amax_scale(x)
        if not self.fake_quant_enabled:
            return x
        return ops.fake_quant(x, self.qmap, self.scale)


class _BlockFakeQuantize(_FakeQuantize):
    """The parts the block schemes share.

    Blocks of ``block_size`` elements along ``ch_axis`` each take their
    parameters from their own values on every call, so nothing is observed;
    each block scale may itself be quantized to ``scale_dtype``.

    Args:
        dtype: Element dtype, as a spec string names it.
        quant_max: Largest code the blocks map onto.
        ch_axis: Axis or axes the blocks run along.
        block_size: Elements per block.
        scale_dtype: Dtype the block parameters are quantized to, or None.
        **kwargs: Forwarded to ``_FakeQuantize``.
    """

    scale_qmap: Optional[torch.Tensor]

    def __init__(
        self,
        dtype: str,
        quant_max: float,
        ch_axis: Union[int, List[int]],
        block_size: int,
        scale_dtype: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(dtype, **kwargs)
        self.quant_max = quant_max
        self.ch_axis = ch_axis
        self.block_size = block_size
        self.scale_dtype = scale_dtype
        scale_qmap = (
            get_quantization_map(scale_dtype, self.scale.device)
            if scale_dtype is not None
            else None
        )
        self.register_buffer("scale_qmap", scale_qmap, persistent=False)

    def calculate_qparams(self):
        raise NotImplementedError(
            "Block parameters are computed on every call, not kept"
        )

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, quant_max={self.quant_max}, "
            f"ch_axis={self.ch_axis}, block_size={self.block_size}, "
            f"scale_dtype={self.scale_dtype}"
        )


class MXFakeQuantize(_BlockFakeQuantize):
    """Fake-quantize in microscaling blocks.

    Each block's scale is its absolute maximum over ``quant_max``, computed
    on every call and not kept.  Stochastic rounding searches ``levels``,
    the table's (``qmap_levels``), taken whenever a table is installed.

    A random Hadamard transform rotates the input by ``hadamard_matrix``
    along ``rht_axis`` before it is quantized and the result back after.
    The rotated axis keeps ``hadamard_rotate``'s padding, so the GEMM the
    input feeds runs over the padded axis, as the hardware's does.

    An outlier filter keeps outliers out of quantization: zeroed before it,
    restored after.  An outlier is an element whose magnitude reaches the
    threshold.  The threshold is given, or calibrated while the observer is
    on as the smallest of the largest ``outlier_pct`` of the elements, the
    largest any call asked for.

    Args:
        power_2_scale: Round each block scale to a power of two.
        stochastic_rounding: Round each element to one of the two values
            around it at random, unbiased, rather than to the nearest.
        global_scale: Divide the block scales by the tensor's global scale
            (``calculate_global_scale``) before quantizing them to
            ``scale_dtype``, as NVFP4 does.
        rht_axis: Axis to rotate along, or None.
        outlier_threshold: Magnitude from which an element is an outlier.
        outlier_pct: Fraction of elements to treat as outliers instead.
        *args: Forwarded to ``_BlockFakeQuantize``.
        **kwargs: Forwarded to ``_BlockFakeQuantize``.

    Raises:
        ValueError: Both a rotation and an outlier filter, or both an
            outlier threshold and an outlier fraction, are asked for.
    """

    levels: Optional[torch.Tensor]
    rht_matrix: Optional[torch.Tensor]

    def __init__(
        self,
        *args,
        power_2_scale: bool = False,
        stochastic_rounding: bool = False,
        global_scale: bool = False,
        rht_axis: Optional[int] = None,
        outlier_threshold: Optional[float] = None,
        outlier_pct: Optional[float] = None,
        **kwargs,
    ) -> None:
        filters_outliers = (
            outlier_threshold is not None or outlier_pct is not None
        )
        if rht_axis is not None and filters_outliers:
            raise ValueError(
                "A random Hadamard transform cannot be combined with an "
                "outlier filter: the outliers would have to be zeroed "
                "between the rotation and the rounding, inside the "
                "fake-quant op"
            )
        if outlier_pct is not None and outlier_threshold is not None:
            raise ValueError(
                "Only one of outlier_pct and outlier_threshold can be set"
            )
        super().__init__(*args, **kwargs)
        device = self.scale.device
        self.outlier_pct = outlier_pct
        # The largest fraction of outliers any call has held, kept on the
        # device so that recording it never waits for it.
        self.register_buffer(
            "max_outlier_pct", torch.zeros((), device=device), persistent=False
        )
        if outlier_pct is not None:
            # Zero until calibrated, when it filters nothing.
            self.register_buffer(
                "outlier_threshold", torch.zeros((), device=device)
            )
        else:
            self.outlier_threshold = outlier_threshold
        self.rht_axis = rht_axis
        self.register_buffer(
            "rht_matrix",
            hadamard_matrix(device) if rht_axis is not None else None,
            persistent=False,
        )
        self.power_2_scale = power_2_scale
        self.stochastic_rounding = stochastic_rounding
        self.global_scale = global_scale
        # The largest block scale, which the global scale maps the tensor's
        # largest block onto.
        self.scale_max = (
            entry_levels(self.scale_dtype).max().item()
            if global_scale
            else None
        )
        self.register_buffer(
            "levels",
            qmap_levels(self.qmap) if stochastic_rounding else None,
            persistent=False,
        )

    def install_qmap(self, qmap: torch.Tensor) -> None:
        if self.stochastic_rounding and qmap.dim() > 1:
            raise NotImplementedError(
                "Stochastic rounding takes one set of levels, not a codebook "
                "grid"
            )
        super().install_qmap(qmap)
        if self.stochastic_rounding:
            _install(self, "levels", qmap_levels(self.qmap))

    def extra_repr(self):
        return (
            f"{super().extra_repr()}, rht_axis={self.rht_axis}, "
            f"power_2_scale={self.power_2_scale}, "
            f"stochastic_rounding={self.stochastic_rounding}, "
            f"global_scale={self.global_scale}"
        )

    def _outlier_mask(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return which elements reach the threshold, or None without one."""
        threshold = self.outlier_threshold
        if threshold is None:
            return None
        return (x.abs() >= threshold) & (threshold > 0)

    def _calibrate_threshold(self, x: torch.Tensor) -> None:
        """Raise the threshold to the smallest of ``x``'s largest
        ``outlier_pct`` magnitudes, if that is larger."""
        flat = x.abs().flatten()
        k = max(1, math.ceil(self.outlier_pct * flat.numel()))

        vals = torch.topk(flat, k, largest=True, sorted=False).values
        threshold = vals.min()

        # The largest threshold any calibration batch asked for, so the
        # bound holds on every batch seen.
        self.outlier_threshold.copy_(
            torch.maximum(self.outlier_threshold, threshold)
        )

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` as the fake-quant quantizes it -- rotated, or with its
        outliers zeroed -- whether or not fake-quantization is on, and
        without changing state, so a codebook fit sees the values the
        codebook will quantize."""
        if self.rht_axis is not None:
            return hadamard_rotate(x, self.rht_axis, self.rht_matrix)
        skip = self._outlier_mask(x)
        return x if skip is None else x.masked_fill(skip, 0.0)

    def forward(
        self, x: torch.Tensor, global_scale: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fake-quantize ``x``.

        A rotation the kernel fuses (``kernels.mx_fake_quant_supported``)
        runs inside ``quantization::mx_fake_quant``, its global amax taken
        by ``quantization::hadamard_amax``: the result is laid out as ``x``
        and the backward passes the gradient through unrotated.  Any other
        rotation runs here, around the op, and is differentiated.

        Args:
            x: Tensor to quantize.
            global_scale: ``x``'s global scale, computed once for every
                fake-quant of ``x`` (``_share_global_scales``), or None to
                compute it here if ``global_scale`` is on.

        Returns:
            The fake-quantized tensor.
        """
        if self.outlier_pct is not None and self.observer_enabled:
            self._calibrate_threshold(x)
        if not self.fake_quant_enabled:
            return x
        skip = self._outlier_mask(x)
        original = x
        if skip is not None:
            x = x.masked_fill(skip, 0.0)
            self.max_outlier_pct.clamp_min_(skip.float().mean())
        # Each call rounds under a fresh seed from torch's generator.
        seed = (
            torch.randint(SEED_BOUND, (), device=x.device)
            if self.stochastic_rounding
            else None
        )
        args = (
            self.levels if self.stochastic_rounding else self.qmap,
            _axes(self.ch_axis),
            self.block_size,
            self.quant_max,
            self.power_2_scale,
            self.scale_qmap,
            seed,
        )
        hadamard, axis = self.rht_matrix, self.rht_axis
        # The global scale is left out: the one computed below is in ``x``'s
        # dtype, as the kernel takes it.
        fused = (
            axis is not None
            and ops.kernels is not None
            and ops.kernels.mx_fake_quant_supported(
                x, *args, None, hadamard, axis
            )
        )
        rotate_here = axis is not None and not fused
        if rotate_here:
            x = hadamard_rotate(x, axis, hadamard)
        if self.global_scale and global_scale is None:
            if fused:
                # Detached: the scale takes no gradient, and the op has no
                # backward.
                amax = ops.hadamard_amax(x.detach(), axis, hadamard)
                global_scale = calculate_global_scale(
                    amax, self.quant_max, self.scale_max
                )
            else:
                global_scale = calculate_global_scale(
                    x, self.quant_max, self.scale_max
                )
        if not fused:
            hadamard = axis = None
        x = ops.mx_fake_quant(x, *args, global_scale, hadamard, axis)
        if rotate_here:
            x = hadamard_rotate(x, self.rht_axis, self.rht_matrix.T)
        return x if skip is None else torch.where(skip, original, x)


class GroupWiseAffineFakeQuantize(_BlockFakeQuantize):
    """Fake-quantize with a scale and a zero point per block.

    Each block maps its own ``[min, max]`` onto ``[quant_min, quant_max]``.

    Args:
        quant_min: Smallest code the blocks map onto.
        *args: Forwarded to ``_BlockFakeQuantize``.
        **kwargs: Forwarded to ``_BlockFakeQuantize``.
    """

    def __init__(self, *args, quant_min: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_min = quant_min

    def extra_repr(self):
        return f"{super().extra_repr()}, quant_min={self.quant_min}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return x
        return ops.group_wise_affine_fake_quant(
            x,
            _axes(self.ch_axis),
            self.block_size,
            self.quant_min,
            self.quant_max,
            self.scale_qmap,
        )[0]


def fake_quantize_class(qscheme):
    """Return the fake-quant class that implements ``qscheme``.

    Args:
        qscheme: A ``QScheme``, or None for a cast with no scale.

    Returns:
        The class a spec with this scheme is built with.
    """
    if qscheme is None:
        return DirectCastFakeQuantize
    if qscheme == QScheme.MICROSCALING:
        return MXFakeQuantize
    if qscheme == QScheme.GROUP_WISE_AFFINE:
        return GroupWiseAffineFakeQuantize
    return FusedAmaxObsFakeQuantize


class _DerivedObserverOrFakeQuantize(_FreezableFlags):
    r"""This observer is used to describe an observer whose quantization
    parameters are derived from other observers
    """

    def __init__(
        self,
        dtype: Optional[str],
        obs_or_fqs: List[ObserverOrFakeQuantize],
        derive_qparams_fn: Callable[
            [List[ObserverOrFakeQuantize]], Tuple[Tensor, Tensor]
        ],
    ):
        super().__init__()
        self.obs_or_fqs = obs_or_fqs
        self.derive_qparams_fn = derive_qparams_fn
        qmap = None if dtype is None else get_quantization_map(dtype)
        self.register_buffer("qmap", qmap, persistent=False)
        self.disable_observer()
        self.dtype = dtype

    def forward(self, x: Tensor) -> Tensor:
        if not self.fake_quant_enabled:
            return x
        if self.qmap is None:
            scale = self.calculate_qparams().to(x.dtype)
            return x / scale * scale
        return ops.fake_quant(x, self.qmap, self.calculate_qparams())

    def calculate_qparams(self):
        return self.derive_qparams_fn(self.obs_or_fqs)
