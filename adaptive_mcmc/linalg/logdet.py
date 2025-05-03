import torch


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
        # cur_vec = cur_res
        trace += sign / ((1 - dist_cdf(i - 1)) * i) * cur_vec
        sign *= -1

    return torch.einsum("...i,...i->...", trace, z), cur_vec


def lanczos_tridiag(matvec, dimension, batch_size, lanczos_steps,
                    initial_vector=None,
                    device="cpu", dtype=torch.float32, eps=1e-9):
    """
    mv: (batch_size, n) -> (batch_size, n)
    """
    if initial_vector is None:
        v = torch.randint(0, 2, (batch_size, dimension), device=device, dtype=dtype) * 2 - 1
    else:
        v = initial_vector.to(device=device, dtype=dtype)

    lanczos_steps = min(dimension, lanczos_steps)

    v = torch.nn.functional.normalize(v, dim=1)
    v_prev = torch.zeros_like(v)

    alpha_list = []
    beta_list = []

    for i in range(lanczos_steps):
        w = matvec(v)

        if i > 0:
            w = w - beta_list[-1].unsqueeze(1) * v_prev

        alpha = torch.sum(v * w, dim=1)
        w = w - alpha.unsqueeze(1) * v
        beta = torch.norm(w, dim=1)

        alpha_list.append(alpha)
        beta_list.append(beta)

        if torch.min(beta) < eps:
            break

        v_next = w / beta.unsqueeze(1)
        v_prev, v = v, v_next

    alphas = torch.stack(alpha_list, dim=1)
    betas = torch.stack(beta_list[:-1], dim=1) if len(beta_list) > 1 else None

    return alphas, betas


def tridiag_eig_weights(alphas, betas, evals, eps=1e-6):
    """
    Compute w_i = q_i(1)^2 for each eigenpair of a symmetric tridiagonal T,
    without ever doing an in-place write.

    alphas : (B, k)
    betas  : (B, k-1)
    evals  : (B, k)  – eigenvalues from torch.linalg.eigvalsh(T)
    returns
    weights: (B, k)
    """
    B, k = alphas.shape
    device, dtype = alphas.device, alphas.dtype

    if k == 1:                              # single eigenvector is [1]
        return torch.ones(B, 1, device=device, dtype=dtype)

    # List to hold x_j for j=0..k-1, each tensor is (B, k)
    x_list = []

    # x0 ≡ 1 for every eigenvector
    x0 = torch.ones(B, k, device=device, dtype=dtype)
    x_list.append(x0)

    # x1 = (λ_i − α₁) / β₁   (same row repeated across columns)
    beta1 = betas[:, 0].clamp(min=eps)           # (B,)
    x1 = ((evals - alphas[:, 0:1]) / beta1[:, None])   # (B, k)
    x_list.append(x1)

    # Recurrence for j = 2..k-1 (build a NEW tensor each time)
    for j in range(1, k - 1):
        a_j  = alphas[:, j:j+1]                       # (B,1) broadcast
        lam  = evals                                  # (B,k)
        b_jm1 = betas[:, j-1:j].clamp(min=eps)        # (B,1)
        b_j   = betas[:, j:j+1].clamp(min=eps)        # (B,1)

        x_jm1 = x_list[j-1]                           # (B,k)
        x_j   = x_list[j]                             # (B,k)

        x_next = - (b_jm1 * x_jm1 + (a_j - lam) * x_j) / b_j
        x_list.append(x_next)                         # no in-place write!

    # Stack to shape (k, B, k).  axis-0 is the component index j.
    X = torch.stack(x_list, dim=0)                    # (k, B, k)

    # norms² per eigenvector:  sum over j  → shape (B, k)
    norms2 = (X * X).sum(dim=0)

    # first component squared is 1 for every eigenvector
    weights = 1.0 / norms2                           # (B, k)
    return weights


def lanczos_trace_estimator(matvec, dimension, batch_size,
                            probe_vector_count=10, lanczos_steps=10,
                            device='cpu', dtype=torch.float32):
    """
    mv: (batch_size, n) -> (batch_size, n)
    """
    sum_trace = torch.zeros(batch_size, device=device, dtype=dtype)

    for j in range(probe_vector_count):
        z = torch.randint(0, 2, (batch_size, dimension), device=device, dtype=dtype) * 2 - 1
        norm_z = torch.norm(z, dim=-1)

        alphas, betas = lanczos_tridiag(matvec=matvec,
                                        dimension=dimension,
                                        batch_size=batch_size,
                                        lanczos_steps=lanczos_steps,
                                        initial_vector=z,
                                        device=device,
                                        dtype=dtype)
        # alphas: (batch_size, k)
        # betas: (batch_size, k - 1)
        T = torch.diag_embed(alphas)
        if betas is not None:
            T = T \
                + torch.diag_embed(betas, offset=1) \
                + torch.diag_embed(betas, offset=-1)

        # evals = torch.linalg.eigvalsh(T)
        evals, evecs = torch.linalg.eigh(T)

        weights = evecs[:, 0, :]**2
        # weights = tridiag_eig_weights(alphas, betas, evals)
        log_evals = torch.log(evals)

        trace = norm_z**2 * torch.sum(weights * log_evals, dim=1)
        sum_trace += trace

    return sum_trace.div(probe_vector_count)
