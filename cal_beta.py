def _np_sample_var(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size <= 1:
        return float("nan")
    return float(x.var(ddof=1))


def _np_sample_cov(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError(f"len mismatch: {x.size} vs {y.size}")
    n = x.size
    if n <= 1:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    return float((xm * ym).sum() / (n - 1))


def _np_sample_corr(x, y, eps=1e-30):
    vx = _np_sample_var(x)
    vy = _np_sample_var(y)
    if (not np.isfinite(vx)) or (not np.isfinite(vy)):
        return float("nan")
    den = float(np.sqrt(max(vx, 0.0) * max(vy, 0.0)))
    if den < eps:
        return float("nan")
    return float(_np_sample_cov(x, y) / den)


@torch.no_grad()
def estimate_beta_on_3d_dataset(net, dataset, norm, cfg, loader_batch_size=None):
    """
    dataset: TensorDataset(theta, X, y), 且应为3D样本:
      theta [B,N,4], X [B,N,d], y [B,N,1]

    返回:
      beta_stats (基于组均值 ybar_b, gbar_b)
    """
    if loader_batch_size is None:
        # 这里 batch_size 是“组数”，不是点数
        loader_batch_size = min(256, len(dataset)) if len(dataset) > 0 else 1

    loader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=False)
    net.eval()

    ybar_all = []
    gbar_all = []

    for theta, X, y in loader:
        theta = theta.to(cfg.device)
        X = X.to(cfg.device)
        y = y.to(cfg.device)

        # 必须是3D
        if X.ndim != 3:
            raise ValueError(f"这里要求3D数据 [B,N,d]，但拿到 X.shape={X.shape}")

        theta_n, X_n = normalize(theta, X, norm)
        pred = net(theta_n, X_n)

        if pred.ndim == 2:
            pred = pred.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)

        assert pred.shape == y.shape, f"pred/y shape mismatch: {pred.shape} vs {y.shape}"

        ybar = y.mean(dim=1).squeeze(-1)    # [B]
        gbar = pred.mean(dim=1).squeeze(-1) # [B]

        ybar_all.append(ybar.detach().cpu().numpy())
        gbar_all.append(gbar.detach().cpu().numpy())

    ybar_all = np.concatenate(ybar_all, axis=0).astype(np.float64)
    gbar_all = np.concatenate(gbar_all, axis=0).astype(np.float64)

    var_y = _np_sample_var(ybar_all)
    var_g = _np_sample_var(gbar_all)
    cov_yg = _np_sample_cov(ybar_all, gbar_all)
    corr_yg = _np_sample_corr(ybar_all, gbar_all)

    if (not np.isfinite(var_g)) or abs(var_g) < 1e-30:
        beta = float("nan")
    else:
        beta = float(cov_yg / var_g)

    alpha = float(ybar_all.mean() - beta * gbar_all.mean()) if np.isfinite(beta) else float("nan")

    if np.isfinite(beta):
        resid = ybar_all - beta * gbar_all
        var_resid = _np_sample_var(resid)
    else:
        var_resid = float("nan")

    return {
        "n_groups": int(ybar_all.size),
        "mean_ybar": float(ybar_all.mean()),
        "mean_gbar": float(gbar_all.mean()),
        "var_ybar": float(var_y),
        "var_gbar": float(var_g),
        "cov_ybar_gbar": float(cov_yg),
        "corr_ybar_gbar": float(corr_yg),
        "beta_cv": float(beta),
        "alpha_ols": float(alpha),
        "var_ybar_minus_beta_gbar": float(var_resid),
        "vr_ratio_vs_ybar": float(var_resid / var_y) if np.isfinite(var_resid) and np.isfinite(var_y) and abs(var_y) > 0 else float("nan"),
    }


@torch.no_grad()
def estimate_beta_curve_by_N(net, norm, cfg, theta_fixed, N_list, B_beta=None, loader_batch_size=None):
    """
    对多个 N 估计 beta_N。每个 N 都重新生成一套 3D 数据（固定 theta）。
    """
    rows = []

    # 估计 beta 的“组数” B_beta（不是组内点数 N）
    if B_beta is None:
        # 用 cfg.dataset_size 做默认值，和你的思路一致
        B_beta = int(cfg.dataset_size)

    for N_group in N_list:
        # 生成固定 theta 的 3D 参数 [B_beta, N_group]
        r, S0, sigma, K = sample_theta(
            mode=3,
            batch_size=B_beta,   # 兼容你原接口
            B=B_beta,
            N=int(N_group),
            is_same=True,        # 固定同一个theta广播到所有组/点
            theta_same=theta_fixed,
            device=cfg.device
        )

        # 临时cfg：只改 N，generate_dataset 会调用 simulate_gbm_batch_qmc 使用 cfg.N
        cfg_beta = cfg
        old_N = getattr(cfg_beta, "N", None)
        cfg_beta.N = int(N_group)

        dataset_beta = generate_dataset(cfg_beta, theta=(r, S0, sigma, K))
        stats = estimate_beta_on_3d_dataset(
            net=net,
            dataset=dataset_beta,
            norm=norm,
            cfg=cfg_beta,
            loader_batch_size=loader_batch_size
        )

        # 恢复 cfg.N（保险）
        cfg_beta.N = old_N if old_N is not None else cfg_beta.N

        row = {
            "N": int(N_group),
            "B_beta": int(B_beta),
            **stats
        }
        rows.append(row)

        print(
            f"[Beta] N={N_group:5d} | "
            f"beta={row['beta_cv']:.8f} | "
            f"corr={row['corr_ybar_gbar']:.6f} | "
            f"vr_ratio={row['vr_ratio_vs_ybar']:.6f}"
        )

    return rows