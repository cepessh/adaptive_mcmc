import torch
import torch.nn.functional as F
from typing import Optional, List


def tridiag_matmul(
    x: torch.Tensor,
    lower: Optional[torch.Tensor] = None,
    diag: Optional[torch.Tensor] = None,
    upper: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Multiply a batch of tridiagonal matrices by x, without forming the full matrix.
    lower: (b, d‑1)  – sub‑diagonal entries
    diag:  (b, d)    – main‑diagonal entries
    upper: (b, d‑1)  – super‑diagonal entries
    x:     (b, d) or (b, n, d)   – vector(s) to multiply
    returns y = T @ x with shape (b, d) or (b, n, d)
    """
    y = torch.zeros_like(x)
    if diag is not None:
        if x.dim() == 3:
            diag = diag.unsqueeze(1)
        y = y + diag * x

    if upper is not None:
        if x.dim() == 3:
            upper = upper.unsqueeze(1)
        up_contrib = upper * x[..., 1:]               # (..., d-1)
        y = y + F.pad(up_contrib, (0, 1))             # (..., d)

    if lower is not None:
        if x.dim() == 3:
            lower = lower.unsqueeze(1)
        lo_contrib = lower * x[..., :-1]              # (..., d-1)
        y = y + F.pad(lo_contrib, (1, 0))             # (..., d)

    return y


@torch.jit.script
def bidiag_solve_jit(
    x: torch.Tensor,  # (..., n)
    diag: torch.Tensor,  # (..., n)
    upper: torch.Tensor,  # (..., n-1)
) -> torch.Tensor:
    """
    Solve C @ p = x for p, where C is upper‑bidiagonal:
      diag on the main diagonal, upper on the super‑diag.
    No in‑place writes on any Tensors.
    """
    d = x.size(-1)
    p_rev = torch.jit.annotate(List[torch.Tensor], [])

    p_last = x.select(-1, d - 1) / diag.select(-1, d - 1)
    p_rev.append(p_last)

    i = d - 2
    while i >= 0:
        xi = x.select(-1, i)
        ui = upper.select(-1, i)
        pi1 = p_rev[-1]  # p_{i+1}
        pi = (xi - ui * pi1) / diag.select(-1, i)
        p_rev.append(pi)
        i -= 1

    p_list = torch.jit.annotate(List[torch.Tensor], [])
    j = len(p_rev) - 1
    while j >= 0:
        p_list.append(p_rev[j])
        j -= 1

    return torch.stack(p_list, dim=-1)
