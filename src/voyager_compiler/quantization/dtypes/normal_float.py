import torch


def create_normal_map(offset=0.9677083, use_extra_value=True, k=4):
    try:
        from scipy.stats import norm
    except ImportError as ie:
        raise ImportError("Scipy is required for `create_normal_map`.") from ie

    num_values = 2 ** (k - 1)
    if use_extra_value:
        # one more positive value, this is an asymmetric type
        v1 = norm.ppf(torch.linspace(offset, 0.5, num_values + 1)[:-1]).tolist()
        v2 = [0]  ## we have 15 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1])).tolist()
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1]).tolist()
        v2 = [0] * 2  ## we have 14 non-zero values in this data type
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, num_values)[:-1])).tolist()

    v = v1 + v2 + v3

    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()

    assert values.numel() == 2 ** k

    return values


def quantize_to_nf(
    input: torch.Tensor,
    k: int = 4,
    use_extra_value=True,
    grid=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to NormalFloat's levels.

    Args:
        input: The tensor to quantize.
        k: Bit width; there are ``2**k`` levels.
        use_extra_value: Use the asymmetric map, which has one more positive
            level than negative ones.
        grid: Ascending values a level may take.  The levels are scaled to
            its largest value and each moved to the nearest one, which can
            merge two of them.  None keeps them in [-1, 1].

    Returns:
        The index of the level each element rounds to, and the levels.
    """
    values = create_normal_map(k=k, use_extra_value=use_extra_value)

    if grid is not None:
        values = values.to(grid) * grid.max()
        values = grid[torch.abs(values[:, None] - grid).argmin(-1)]

    values = values.to(device=input.device, dtype=input.dtype)
    input = torch.clamp(input, min=values.amin(), max=values.amax())
    indices = torch.argmin(torch.abs(values - input.unsqueeze(-1)), dim=-1)

    return indices, values
