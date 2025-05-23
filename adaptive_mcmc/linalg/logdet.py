from typing import (
    Callable,
    List,
    Optional,
)

import torch
from torch import Tensor


def taylor_trace_estimator(matvec, dimension, batch_size, min_truncation_level=2,
                           dist=None, dist_cdf=None,
                           spectral_normalization_decay=0.99, device="cpu", dtype=torch.float32):

    if dist is None:
        p = 0.5
        dist = torch.distributions.Geometric(p)

        def dist_cdf(k):
            if k <= min_truncation_level:
                return 0
            return 1 - (1 - p) ** (k - min_truncation_level)

    truncation_level = min_truncation_level
    truncation_level += int(dist.sample((1,)).item())

    z = torch.randint(size=(batch_size, dimension), low=0, high=2, device=device, dtype=dtype) * 2 - 1
    cur_vec = z

    trace = torch.zeros_like(z, device=device, dtype=dtype)
    sign = 1

    for i in range(1, truncation_level + 1):
        cur_res = matvec(cur_vec)

        spectral_normalization = torch.clamp(
            spectral_normalization_decay
            * torch.norm(cur_vec, dim=-1, p=2) / torch.norm(cur_res, dim=-1, p=2),
            max=1,
        ).unsqueeze(-1)

        cur_vec = spectral_normalization * cur_res
        trace = trace + sign / ((1 - dist_cdf(i - 1)) * i) * cur_vec
        sign *= -1

    return torch.einsum("...i,...i->...", trace, z), cur_vec


def lanczos_tridiag(
    matmul_closure: Callable,
    max_iter: int,
    matrix_shape: torch.Size,
    dtype: torch.dtype = torch.float32,
    device: torch.device = "cpu",
    batch_shape=torch.Size(),
    init_vecs: Optional[int] = None,
    num_init_vecs: int = 1,
    tol: float = 1e-5,
):
    """Lanczos tridiagonalization without any in-place operations.
    In the rest of the code matmul transforms (b, n, d) -> (b, n, d)
    or (b, d) -> (b, d). Here however transform is (b, d, n) -> (b, d, n)
    """

    moved = matmul_closure

    def matmul(v: Tensor) -> Tensor:
        return moved(v.transpose(-1, -2)).transpose(-1, -2)

    matmul_closure = matmul

    # Initialize probing vectors
    if init_vecs is None:
        init_vecs = torch.randn(
            *batch_shape, matrix_shape[-1], num_init_vecs,
            dtype=dtype, device=device
        )
    else:
        if dtype != init_vecs.dtype or device != init_vecs.device:
            raise RuntimeError("init_vecs dtype or device mismatch.")
        if batch_shape != init_vecs.shape[:-2] or matrix_shape[-1] != init_vecs.size(-2):
            raise RuntimeError("batch_shape or matrix_shape mismatch init_vecs.")
        num_init_vecs = init_vecs.size(-1)

    # Max iterations
    num_iter = min(max_iter, matrix_shape[-1])
    dim_dim = -2

    # First Lanczos vector
    norm0 = torch.norm(init_vecs, p=2, dim=dim_dim, keepdim=True)
    q0 = init_vecs / norm0

    alpha_list: List[Tensor] = torch.jit.annotate(List[Tensor], [])
    beta_list: List[Tensor] = torch.jit.annotate(List[Tensor], [])
    q_list: List[Tensor] = torch.jit.annotate(List[Tensor], [])
    q_list.append(q0)

    # Compute initial alpha and beta
    r = matmul_closure(q0)
    alpha0 = (q0 * r).sum(dim_dim)
    alpha_list.append(alpha0)
    r = r - alpha0.unsqueeze(dim_dim) * q0
    beta0 = torch.norm(r, p=2, dim=dim_dim)

    if torch.min(beta0) <= tol:
        # build T of size 1×1
        T = alpha0.unsqueeze(-1).unsqueeze(-1)  # shape (...,1,1)
        T = T.transpose(0, 1)
        q0 = q0.transpose(0, 1)
        return q0, T

    beta_list.append(beta0)

    # First new Lanczos vector if possible
    if num_iter > 1:
        q1 = r / beta0.unsqueeze(dim_dim)
        q_list.append(q1)

    # Main Lanczos loop
    for k in range(1, num_iter):
        if k >= len(q_list):
            break
        q_prev = q_list[k - 1]
        q_curr = q_list[k]

        # Apply matrix and subtract previous beta term
        beta_prev = beta_list[k - 1].unsqueeze(dim_dim)
        r = matmul_closure(q_curr) - q_prev * beta_prev

        # Alpha
        alpha = (q_curr * r).sum(dim_dim)
        alpha_list.append(alpha)
        if k == num_iter - 1:
            break

        # Subtract current alpha component
        r = r - alpha.unsqueeze(dim_dim) * q_curr

        # Full reorthogonalization
        stacked_q = torch.stack(q_list, dim=0)
        corr = (r.unsqueeze(0) * stacked_q).sum(dim_dim, keepdim=True)
        corr = (stacked_q * corr).sum(0)
        r = r - corr

        # Normalize
        r_norm = torch.norm(r, p=2, dim=dim_dim, keepdim=True)
        r = r / r_norm

        # Beta
        beta = r_norm.squeeze(dim_dim)

        if torch.min(beta) <= tol:
            break

        beta_list.append(beta)

        # Additional reorthogonalization if needed
        inner = (stacked_q * r.unsqueeze(0)).sum(dim_dim)
        for _ in range(10):
            if not torch.any(inner > tol):
                break
            corr = (r.unsqueeze(0) * stacked_q).sum(dim_dim, keepdim=True)
            corr = (stacked_q * corr).sum(0)
            r = r - corr
            r_norm = torch.norm(r, p=2, dim=dim_dim, keepdim=True)
            r = r / r_norm
            inner = (stacked_q * r.unsqueeze(0)).sum(dim_dim)

        q_list.append(r)

    # Number of Lanczos steps
    m = len(alpha_list)

    # Stack Q vectors: shape (m+1, *batch_shape, d, num_init_vecs)
    q_stack = torch.stack(q_list[: m + 1], dim=0)
    # Reorder to (num_init_vecs, *batch_shape, d, m+1)
    perm_q = [q_stack.dim() - 1] + list(range(1, 1 + len(batch_shape))) + [q_stack.dim() - 2, 0]
    q_out = q_stack.permute(*perm_q).contiguous()

    # Build tridiagonal T
    alpha_tensor = torch.stack(alpha_list, dim=0)
    if beta_list:
        beta_tensor = torch.stack(beta_list, dim=0)
    else:
        beta_tensor = torch.empty((0, *batch_shape, num_init_vecs), dtype=dtype, device=device)

    # Prepare for diag_embed: move iterations last
    # alpha_tensor: (m, *batch_shape, num_init_vecs) -> (*batch_shape, num_init_vecs, m)
    perm_a = list(range(1, 1 + len(batch_shape))) + [alpha_tensor.dim() - 1, 0]
    a_for_diag = alpha_tensor.permute(*perm_a)
    # beta_tensor: (m-1, *batch_shape, num_init_vecs) -> (*batch_shape, num_init_vecs, m-1)
    perm_b = list(range(1, 1 + len(batch_shape))) + [beta_tensor.dim() - 1, 0]
    b_for_diag = beta_tensor.permute(*perm_b)

    # Create diagonal matrices
    main_diag = torch.diag_embed(a_for_diag, offset=0)
    off_upper = torch.diag_embed(b_for_diag, offset=1)
    off_lower = torch.diag_embed(b_for_diag, offset=-1)

    T_full = main_diag + off_upper + off_lower  # shape: (*batch_shape, num_init_vecs, m, m)

    # Reorder T to (num_init_vecs, *batch_shape, m, m)
    perm_t = [len(batch_shape), *range(0, len(batch_shape)), len(batch_shape) + 1, len(batch_shape) + 2]
    t_out = T_full.permute(*perm_t).contiguous()

    # If single init vector, drop that dim
    if num_init_vecs == 1:
        q_out = q_out.squeeze(0)
        t_out = t_out.squeeze(0)

    return q_out, t_out


def lanczos_tridiag_to_diag(t_mat):
    """
    Given a num_init_vecs x num_batch x k x k tridiagonal matrix t_mat,
    returns a num_init_vecs x num_batch x k set of eigenvalues
    and a num_init_vecs x num_batch x k x k set of eigenvectors.

    TODO: make the eigenvalue computations done in batch mode.
    """
    orig_device = t_mat.device

    if t_mat.size(-1) < 32:
        retr = torch.linalg.eigh(t_mat.cpu())
    else:
        retr = torch.linalg.eigh(t_mat)

    evals, evecs = retr
    # mask = evals.ge(0)
    # evecs = evecs * mask.type_as(evecs).unsqueeze(-2)
    # evals = evals.masked_fill_(~mask, 1)

    return evals.to(orig_device), evecs.to(orig_device)


# def lanczos_trace_estimator(matvec, dimension, batch_size,
#                             probe_vector_count=10, lanczos_steps=10,
#                             device='cpu', dtype=torch.float32, jitter=1e-6):
#     """
#     mv: (batch_size, n) -> (batch_size, n)
#     """
#     sum_trace = torch.zeros(batch_size, device=device, dtype=dtype)
# 
#     for _ in range(probe_vector_count):
#         # z = torch.randint(0, 2, (batch_size, dimension), device=device, dtype=dtype) * 2 - 1
# 
#         q_mat, t_mat = lanczos_tridiag(
#             matmul_closure=matvec,
#             max_iter=lanczos_steps,
#             dtype=dtype,
#             device=device,
#             matrix_shape=torch.Size([dimension, dimension]),
#             batch_shape=torch.Size([batch_size]),
#             # init_vecs=z.unsqueeze(-1),
#             num_init_vecs=1,
#             tol=1e-5,
#         )
# 
#         # z_norm = z.pow(2).sum(dim=-1)
#         # t_mat = t_mat + torch.eye(t_mat.shape[-1]) * jitter
#         try:
#             evals, evecs = torch.linalg.eigh(t_mat)
#         except torch._C._LinAlgError:
#             print("naninf cnt", (~t_mat.isfinite()).sum())
#             exit()
# 
#         weights = evecs[:, 0, ...].pow(2)
#         log_evals = torch.log(torch.abs(evals))
# 
#         sum_trace = sum_trace + torch.sum(weights * log_evals, dim=-1) * dimension
# 
#     return sum_trace.div(probe_vector_count)


def lanczos_trace_estimator(matvec, dimension, batch_size,
                            probe_vector_count=10, lanczos_steps=10,
                            device='cpu', dtype=torch.float32, jitter=1e-6):

    q_mat, t_mat = lanczos_tridiag(
        matmul_closure=matvec,
        max_iter=lanczos_steps,
        dtype=dtype,
        device=device,
        matrix_shape=torch.Size([dimension, dimension]),
        batch_shape=torch.Size([batch_size]),
        num_init_vecs=probe_vector_count,
        tol=1e-5,
    )

    try:
        evals, evecs = torch.linalg.eigh(t_mat)
    except torch._C._LinAlgError:
        print("naninf cnt", (~t_mat.isfinite()).sum())
        raise torch._C._LinAlgError

    # evals: (n, b, k) or (b, k)
    # evecs: (n, b, k, k) or (b, k, k)

    weights = evecs[..., 0, :].pow(2)
    log_evals = torch.log(torch.abs(evals))

    return torch.sum(weights * log_evals, dim=-1).mean(dim=0) * dimension
