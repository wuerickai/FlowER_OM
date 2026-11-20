import torch
from typing import Optional

def saferound_tensor(
    x: torch.Tensor,
    places: int,
    strategy: str = "difference",
    topline: Optional[float] = None
) -> torch.Tensor:
    """
    Round a tensor elementwise to `places` decimal places, adjusting
    a minimal number of entries so that the total sum is exactly preserved.

    Args:
        x (torch.Tensor): input tensor of floats.
        places (int): number of decimal places to round to.
        strategy (str): one of {"difference","largest","smallest"}:
            - "difference": pick the entries with largest fractional parts first.
            - "largest"  : pick the largest values first.
            - "smallest" : pick the smallest values first.
        topline (float, optional): if given, override the target sum
            with `topline`. Otherwise target is x.sum().

    Returns:
        torch.Tensor: same shape as `x`, rounded to `places`, but whose
        sum exactly equals the rounded(original_sum, places).
    """
    assert isinstance(places, int), "places must be integer"
    assert strategy in ("difference","largest","smallest"), f"Unknown strategy {strategy}"

    # Flatten for simplicity
    orig = x.reshape(-1).to(dtype=torch.float64)
    N = orig.numel()

    # Determine the exact sum we need to hit
    total = topline if topline is not None else orig.sum()
    if not torch.is_tensor(total):
        total = torch.tensor(total, device=x.device, dtype=torch.float64)
    else:
        total = total.to(device=x.device, dtype=torch.float64)
    scale = 10 ** places
    target_int = torch.round(total * scale).to(torch.int64)

    # Scale and take floor/ceil
    scaled = orig * scale
    low = torch.floor(scaled).to(torch.int64)   # integer floors
    sum_low = low.sum()
    residual = target_int - sum_low             # how many +1's we need
    residual_int = int(residual.item())

    if residual_int != 0:
        # Depending on strategy, create a sort key
        if strategy == "difference":
            # fractional part, descending
            frac = scaled - low.to(torch.float64)
            _, indices = torch.sort(frac, descending=True)
        elif strategy == "largest":
            # values descending
            _, indices = torch.sort(orig, descending=True)
        else:  # "smallest"
            # values ascending
            _, indices = torch.sort(orig, descending=False)

        # Pick exactly `abs(residual)` indices
        k = min(abs(residual_int), N)
        chosen = indices[:k]

        # Apply the +1 or -1
        if residual_int > 0:
            low[chosen] += 1
        else:
            # In the very rare case sum_low > target_int, we go back down
            low[chosen] -= 1

    # Convert back to float decimals
    rounded_flat = low.to(torch.float64).mul_(1.0 / scale)
    # reshape back
    return rounded_flat.view_as(x).to(dtype=x.dtype)


def saferound_tensor_batched(
    x: torch.Tensor,
    places: int,
    strategy: str = "difference",
    topline: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Batched saferound for tensors shaped (batch, ...).

    This is equivalent to applying saferound_tensor independently to each batch
    row, but uses one batched argsort/scatter instead of a Python loop over
    batch elements. Per-row rounded sums are preserved.
    """
    assert isinstance(places, int), "places must be integer"
    assert strategy in ("difference", "largest", "smallest"), f"Unknown strategy {strategy}"

    original_shape = x.shape
    original_dtype = x.dtype
    device = x.device

    batch_size = x.shape[0]
    x_flat = x.reshape(batch_size, -1).to(dtype=torch.float64)
    n_items = x_flat.shape[1]

    if topline is not None:
        total = topline.to(device=device, dtype=torch.float64)
    else:
        total = x_flat.sum(dim=1)

    scale = 10 ** places
    target_int = torch.round(total * scale).to(torch.int64)

    scaled = x_flat * scale
    low = torch.floor(scaled).to(torch.int64)

    sum_low = low.sum(dim=1)
    residual = target_int - sum_low

    if strategy == "difference":
        sort_key = scaled - low.to(torch.float64)
        descending = True
    elif strategy == "largest":
        sort_key = x_flat
        descending = True
    else:
        sort_key = x_flat
        descending = False

    indices = torch.argsort(sort_key, dim=1, descending=descending, stable=False)
    k = residual.abs().clamp_max(n_items)
    ranks = torch.arange(n_items, device=device).unsqueeze(0)
    selected_sorted = ranks < k.unsqueeze(1)
    selected = torch.zeros((batch_size, n_items), dtype=torch.bool, device=device)
    selected.scatter_(1, indices, selected_sorted)

    adjustment = torch.sign(residual).unsqueeze(1).to(low.dtype)
    low = low + selected.to(low.dtype) * adjustment

    rounded_flat = low.to(torch.float64).mul_(1.0 / scale)
    return rounded_flat.reshape(original_shape).to(dtype=original_dtype)


def redistribute_round_batched(x: torch.Tensor) -> torch.Tensor:
    """Round each matrix in a batch while preserving each matrix sum."""
    return saferound_tensor_batched(x, places=0, strategy="difference")


def custom_round_optimized(x: torch.Tensor) -> torch.Tensor:
    """
    Fast batched rounding used by beam inference.

    Falls back to exact per-item saferound if the batched path fails, preserving
    the sum invariant instead of silently returning plain torch.round.
    """
    try:
        return redistribute_round_batched(x)
    except Exception as exc:
        print(f"Batched round failed: {exc}, falling back to per-item saferound")
        flat = x.reshape(x.shape[0], -1)
        rounded = [
            saferound_tensor(row, places=0, strategy="difference").reshape_as(x[i])
            for i, row in enumerate(flat)
        ]
        return torch.stack(rounded, dim=0)
