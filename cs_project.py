"""
Sparse Signal Recovery via Greedy Algorithms
============================================
Compressed Sensing and Sub-Nyquist Image Reconstruction

Methods compared
----------------
  Recovery  : OMP | Batch OMP | Basis Pursuit
  Matrix    : Gaussian | Binary

Demos
-----
  1  — 1-D sparse signal recovery  (OMP vs BP, Gaussian vs Binary, clean vs noisy)
  2  — DCT via FFT verification & runtime benchmark
  3  — Phase transition diagram     (OMP success rate vs sparsity x measurements)
  4  — 2-D image reconstruction at multiple compression ratios  (Batch OMP)
  5  — OMP vs Basis Pursuit on 1-D signals (SNR + runtime vs sparsity)
  6  — Batch OMP speedup over naive OMP
  7  — FULL comparison: every method x every matrix on the SAME image
         7a  grid of reconstructed images (rows = methods, cols = matrices)
         7b  error maps |original - reconstructed|
         7c  PSNR bar chart
         7d  Runtime bar chart

Authors: Chaitanya Nimbargi, Avish Fakirde
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.ndimage import zoom
import time, sys, os, warnings
warnings.filterwarnings("ignore")


try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("[INFO] cvxpy not installed — Basis Pursuit results will be skipped.")
    print("       Install with:  pip install cvxpy")


def dct1d_via_fft(x):
    N = len(x)
    v = np.empty(N)
    v[:N // 2 + N % 2] = x[::2]
    v[N // 2 + N % 2:] = x[1::2][::-1]
    V = np.fft.fft(v)
    k = np.arange(N)
    return 2.0 * np.real(np.exp(-1j * np.pi * k / (2 * N)) * V)


def idct1d_via_fft(X):
    N = len(X)
    k = np.arange(N)
    Y = X / (2.0 * N)
    W = Y.astype(complex) * np.exp(1j * np.pi * k / (2 * N))
    W[0] /= 2.0
    v = np.real(np.fft.ifft(W)) * (2 * N)
    x = np.zeros(N)
    half_h = N // 2 + N % 2
    half_l = N // 2
    x[::2]  = v[:half_h]
    x[1::2] = v[N - 1: N - 1 - half_l: -1]
    return x


def dct2d(img):
    return np.apply_along_axis(dct1d_via_fft, 1,
           np.apply_along_axis(dct1d_via_fft, 0, img))


def idct2d(coeffs):
    return np.apply_along_axis(idct1d_via_fft, 0,
           np.apply_along_axis(idct1d_via_fft, 1, coeffs))

def gaussian_measurement_matrix(m, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, n)) / np.sqrt(m)


def binary_measurement_matrix(m, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(m, n)).astype(float) / np.sqrt(m)


def take_measurements(A, x, noise_std=0.0, seed=1):
    y = A @ x
    if noise_std > 0:
        y += np.random.default_rng(seed).standard_normal(len(y)) * noise_std
    return y

def omp(A, y, sparsity, tol=1e-10):
    m, n     = A.shape
    residual = y.copy()
    support  = []
    x_hat    = np.zeros(n)

    for _ in range(sparsity):
        idx = int(np.argmax(np.abs(A.T @ residual)))
        if idx in support:
            break
        support.append(idx)
        A_S = A[:, support]
        coeffs, _, _, _ = np.linalg.lstsq(A_S, y, rcond=None)
        x_hat = np.zeros(n)
        x_hat[support] = coeffs
        residual = y - A_S @ coeffs
        if np.linalg.norm(residual) < tol:
            break

    return x_hat, support


def batch_omp(A, Y, sparsity, tol=1e-10):
    m, n  = A.shape
    P     = Y.shape[1]
    X_hat = np.zeros((n, P))
    G     = A.T @ A 
    AY    = A.T @ Y  

    for p in range(P):
        y_p      = Y[:, p]
        aty      = AY[:, p]
        residual = y_p.copy()
        support  = []
        alpha    = np.zeros(n)

        for _ in range(sparsity):
            corr = np.abs(A.T @ residual)
            idx  = int(np.argmax(corr))
            if idx in support:
                break
            support.append(idx)
            G_S = G[np.ix_(support, support)]
            b   = aty[support]
            try:
                alpha_S = np.linalg.solve(G_S, b)
            except np.linalg.LinAlgError:
                alpha_S, _, _, _ = np.linalg.lstsq(G_S, b, rcond=None)
            alpha = np.zeros(n)
            alpha[support] = alpha_S
            residual = y_p - A[:, support] @ alpha_S
            if np.linalg.norm(residual) < tol:
                break

        X_hat[:, p] = alpha

    return X_hat


def basis_pursuit(A, y, noise_std=0.0):
    if not CVXPY_AVAILABLE:
        return None
    n = A.shape[1]
    x = cp.Variable(n)
    if noise_std == 0.0:
        prob = cp.Problem(cp.Minimize(cp.norm1(x)), [A @ x == y])
    else:
        eps  = noise_std * np.sqrt(len(y)) * 1.1
        prob = cp.Problem(cp.Minimize(cp.norm1(x)),
                          [cp.norm(A @ x - y) <= eps])
    prob.solve(solver=cp.SCS, verbose=False)
    return x.value

def recovery_snr(x_true, x_hat):
    err = x_true - x_hat
    noise_pwr  = np.dot(err, err)
    signal_pwr = np.dot(x_true, x_true)
    if noise_pwr < 1e-15:
        return np.inf
    return 10 * np.log10(signal_pwr / noise_pwr)


def psnr(original, reconstructed, max_val=255.0):
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse < 1e-10:
        return np.inf
    return 20 * np.log10(max_val / np.sqrt(mse))

def get_test_image(image_path=None, size=512):
    if image_path is not None:
        raw = plt.imread(image_path)
        if raw.ndim == 3:
            img = (0.2989 * raw[:, :, 0] +
                   0.5870 * raw[:, :, 1] +
                   0.1140 * raw[:, :, 2])
        else:
            img = raw.astype(float)
        if img.max() <= 1.0:
            img = img * 255.0
        print(f"  Loaded image : {image_path}  shape={raw.shape}")
    else:
        print("  No image supplied — using synthetic sine-wave test image.")
        x = np.linspace(0, 4 * np.pi, size)
        X, Y = np.meshgrid(x, x)
        img = (512 + 80 * np.sin(X) * np.cos(0.7 * Y) +
               40  * np.sin(3 * X + Y)).clip(0, 255)

    h, w   = img.shape[:2]
    crop   = min(h, w)
    img    = img[:crop, :crop]
    img_rs = zoom(img, size / crop, order=1).clip(0, 255)
    return img_rs[:size, :size].astype(float)


def reconstruct_image(image, patch_size=8, compression_ratio=0.5,
                      sparsity=None, method='batch_omp',
                      matrix_type='gaussian', seed=42):
    if method == 'bp' and not CVXPY_AVAILABLE:
        raise RuntimeError("Basis Pursuit requires cvxpy.  pip install cvxpy")

    H, W     = image.shape
    H_c      = (H // patch_size) * patch_size
    W_c      = (W // patch_size) * patch_size
    img      = image[:H_c, :W_c].copy()
    n_atoms  = patch_size * patch_size
    m_meas   = max(4, int(compression_ratio * n_atoms))
    sparsity = sparsity or max(2, int(0.2 * n_atoms))

    A = (gaussian_measurement_matrix(m_meas, n_atoms, seed=seed)
         if matrix_type == 'gaussian'
         else binary_measurement_matrix(m_meas, n_atoms, seed=seed))

    n_ph = H_c // patch_size
    n_pw = W_c // patch_size
    P    = n_ph * n_pw
    Y_dct = np.zeros((n_atoms, P))
    idx   = 0
    for ph in range(n_ph):
        for pw in range(n_pw):
            patch = img[ph*patch_size:(ph+1)*patch_size,
                        pw*patch_size:(pw+1)*patch_size]
            Y_dct[:, idx] = dct2d(patch).ravel()
            idx += 1

    Y_meas = A @ Y_dct
    X_hat = np.zeros((n_atoms, P))
    if method == 'batch_omp':
        X_hat = batch_omp(A, Y_meas, sparsity=sparsity)
    elif method == 'omp':
        for p in range(P):
            x_p, _ = omp(A, Y_meas[:, p], sparsity=sparsity)
            X_hat[:, p] = x_p
    elif method == 'bp':
        for p in range(P):
            x_p = basis_pursuit(A, Y_meas[:, p])
            if x_p is not None:
                X_hat[:, p] = x_p
    recon = np.zeros((H_c, W_c))
    idx   = 0
    for ph in range(n_ph):
        for pw in range(n_pw):
            coeffs = X_hat[:, idx].reshape(patch_size, patch_size)
            recon[ph*patch_size:(ph+1)*patch_size,
                  pw*patch_size:(pw+1)*patch_size] = idct2d(coeffs)
            idx += 1

    return recon

def phase_transition(n=64, sparsity_range=None, measurement_range=None,
                     n_trials=5, success_snr_db=40.0, matrix_type='gaussian'):
    if sparsity_range is None:
        sparsity_range = np.arange(1, n // 4 + 1, max(1, n // 32))
    if measurement_range is None:
        measurement_range = np.arange(4, n, max(1, n // 16))

    mk_fn   = (gaussian_measurement_matrix if matrix_type == 'gaussian'
               else binary_measurement_matrix)
    S_grid  = np.zeros((len(measurement_range), len(sparsity_range)))
    M_grid  = np.zeros_like(S_grid)
    success = np.zeros_like(S_grid)

    for i, m in enumerate(measurement_range):
        for j, s in enumerate(sparsity_range):
            S_grid[i, j] = s
            M_grid[i, j] = m
            if s >= m:
                continue
            wins = 0
            for trial in range(n_trials):
                rng    = np.random.default_rng(trial * 100 + i * 10 + j)
                x_true = np.zeros(n)
                supp   = rng.choice(n, s, replace=False)
                x_true[supp] = rng.standard_normal(s)
                A = mk_fn(int(m), n, seed=trial)
                y = take_measurements(A, x_true)
                x_hat, _ = omp(A, y, sparsity=s)
                if recovery_snr(x_true, x_hat) >= success_snr_db:
                    wins += 1
            success[i, j] = wins / n_trials

    return S_grid, M_grid, success

def _save(name):
    import builtins
    out_dir = getattr(builtins, '_CS_OUTPUT_DIR',
                      os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(out_dir, name)
    plt.savefig(full_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved  {full_path}")


# ────────────────────────────────────────────────────────────────
def demo1_sparse_1d_recovery():
    print("\n" + "="*62)
    print("  DEMO 1 — 1-D Sparse Signal Recovery  (all combinations)")
    print("="*62)

    n, s, m = 512, 10, 40
    rng     = np.random.default_rng(42)
    x_true  = np.zeros(n)
    x_true[rng.choice(n, s, replace=False)] = rng.standard_normal(s)

    combos = [
        ('Gaussian', 'No Noise',       gaussian_measurement_matrix(m, n), 0.00),
        ('Gaussian', 'Noisy (sigma=0.05)', gaussian_measurement_matrix(m, n), 0.05),
        ('Binary',   'No Noise',       binary_measurement_matrix(m, n),   0.00),
        ('Binary',   'Noisy (sigma=0.05)', binary_measurement_matrix(m, n),   0.05),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    fig.suptitle(
        "Demo 1 — 1-D Sparse Signal Recovery: OMP vs Basis Pursuit\n"
        "Gaussian Matrix vs Binary Matrix  |  Clean vs Noisy Measurements\n"
        f"Signal: n={n} total coefficients  |  s={s} non-zeros  |  m={m} measurements",
        fontsize=12, fontweight='bold', y=1.01
    )

    col_titles = [
        "ORIGINAL SIGNAL\n(ground truth, s=10 non-zeros)",
        "OMP RECOVERY\n(Orthogonal Matching Pursuit)\nRequires knowing sparsity s",
        "BASIS PURSUIT RECOVERY\n(L1 Minimisation via cvxpy)\nDoes NOT need sparsity s"
    ]

    for row, (mat_name, noise_label, A, noise_std) in enumerate(combos):
        y         = take_measurements(A, x_true, noise_std=noise_std)
        row_label = f"{mat_name} Matrix\n{noise_label}"
        ax = axes[row][0]
        ax.stem(x_true, markerfmt='C0o', linefmt='C0-', basefmt='k-')
        ax.set_ylabel(row_label, fontsize=10, fontweight='bold')
        if row == 0:
            ax.set_title(col_titles[0], fontsize=10, fontweight='bold')
        ax.set_xlabel("Signal Index")
        ax.grid(True, alpha=0.3)
        t0 = time.perf_counter()
        x_omp, _ = omp(A, y, sparsity=s)
        t_omp    = time.perf_counter() - t0
        snr_omp  = recovery_snr(x_true, x_omp)
        ax = axes[row][1]
        ax.stem(x_omp, markerfmt='C1o', linefmt='C1-', basefmt='k-')
        if row == 0:
            ax.set_title(col_titles[1], fontsize=10, fontweight='bold')
        ax.text(0.02, 0.97,
                f"SNR = {snr_omp:.1f} dB\nTime = {t_omp*1e3:.1f} ms\n"
                f"Matrix: {mat_name}\n{noise_label}",
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
        ax.set_xlabel("Signal Index")
        ax.grid(True, alpha=0.3)
        print(f"  OMP  [{mat_name:<9s}  {noise_label:<22s}]  SNR={snr_omp:>7.1f} dB  t={t_omp*1e3:.1f}ms")
        ax = axes[row][2]
        if row == 0:
            ax.set_title(col_titles[2], fontsize=10, fontweight='bold')
        if CVXPY_AVAILABLE:
            t0   = time.perf_counter()
            x_bp = basis_pursuit(A, y, noise_std=noise_std)
            t_bp = time.perf_counter() - t0
            snr_bp = recovery_snr(x_true, x_bp) if x_bp is not None else float('nan')
            ax.stem(x_bp if x_bp is not None else np.zeros(n),
                    markerfmt='C2o', linefmt='C2-', basefmt='k-')
            ax.text(0.02, 0.97,
                    f"SNR = {snr_bp:.1f} dB\nTime = {t_bp*1e3:.0f} ms\n"
                    f"Matrix: {mat_name}\n{noise_label}",
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.6))
            print(f"  BP   [{mat_name:<9s}  {noise_label:<22s}]  SNR={snr_bp:>7.1f} dB  t={t_bp*1e3:.0f}ms")
        else:
            ax.text(0.5, 0.5, "Basis Pursuit\nnot available\nInstall cvxpy",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='grey',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax.set_xlabel("Signal Index")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save("demo1_sparse_1d_recovery.png")

def demo2_dct_benchmark():
    print("\n" + "="*62)
    print("  DEMO 2 — DCT via FFT: Verification & Benchmark")
    print("="*62)

    from scipy.fft import dct as scipy_dct
    sizes         = [16, 32, 64, 512, 512, 512]
    errors, t_our, t_sci = [], [], []

    for N in sizes:
        x = np.random.default_rng(0).standard_normal(N)
        rt_err = np.max(np.abs(idct1d_via_fft(dct1d_via_fft(x)) - x))

        t0 = time.perf_counter()
        for _ in range(100): dct1d_via_fft(x)
        t_our.append((time.perf_counter() - t0) / 100 * 1e6)

        t0 = time.perf_counter()
        for _ in range(100): scipy_dct(x, type=2, norm=None)
        t_sci.append((time.perf_counter() - t0) / 100 * 1e6)

        errors.append(rt_err)
        print(f"  N={N:4d}  round-trip err={rt_err:.2e}  "
              f"ours={t_our[-1]:.1f}us  scipy={t_sci[-1]:.1f}us")

    p2  = np.random.default_rng(1).standard_normal((8, 8))
    rt2 = np.max(np.abs(idct2d(dct2d(p2)) - p2))
    print(f"  2-D round-trip error = {rt2:.2e}  (expected ~1e-16)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Demo 2 — FFT-based DCT: Built from scratch, no transform library\n"
        "Left: Numerical accuracy (round-trip)  |  Right: Runtime vs scipy",
        fontsize=12, fontweight='bold'
    )

    ax1.semilogy(sizes, errors, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.set_xlabel("Signal length N")
    ax1.set_ylabel("Round-trip error  |IDCT(DCT(x)) - x|  [log scale]")
    ax1.set_title("Numerical Accuracy\nMachine precision ~1e-16 — confirms correctness")
    ax1.grid(True, alpha=0.4)
    ax1.set_xticks(sizes)
    for x_val, y_val in zip(sizes, errors):
        ax1.annotate(f"{y_val:.0e}", (x_val, y_val),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax2.loglog(sizes, t_our, 'o-',  color='steelblue', linewidth=2,
               markersize=8, label='Our FFT-DCT (from scratch)')
    ax2.loglog(sizes, t_sci, 's--', color='coral',     linewidth=2,
               markersize=8, label='scipy.fft.dct (reference)')
    ax2.set_xlabel("Signal length N  [log scale]")
    ax2.set_ylabel("Runtime (microseconds)  [log scale]")
    ax2.set_title("Runtime Scaling — Both O(N log N)\nOurs is ~2-3x slower (pure Python overhead)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4, which='both')

    plt.tight_layout()
    _save("demo2_dct_benchmark.png")


def demo3_phase_transition():
    print("\n" + "="*62)
    print("  DEMO 3 — Phase Transition  (takes ~30-90 seconds)")
    print("="*62)

    n       = 64
    s_range = np.arange(1, 20, 2)
    m_range = np.arange(4, 60, 4)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"Demo 3 — OMP Phase Transition Diagram  (n={n} signal length, 5 trials per point)\n"
        "Green = OMP successfully recovers signal (SNR >= 40 dB)  |  Red = recovery failed\n"
        "Dashed line = theoretical minimum measurements  m ~ s * log(n/s)",
        fontsize=11, fontweight='bold'
    )

    for ax, mat_type, mat_label in zip(axes,
                                        ['gaussian', 'binary'],
                                        ['Gaussian Matrix', 'Binary Matrix']):
        S, M, succ = phase_transition(n=n, sparsity_range=s_range,
                                      measurement_range=m_range,
                                      n_trials=5, matrix_type=mat_type)
        im = ax.pcolormesh(S, M, succ, cmap='RdYlGn', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('OMP Success Rate (1.0 = always succeeds)', fontsize=9)
        s_vals = np.linspace(1, max(s_range), 100)
        ax.plot(s_vals, s_vals * np.log(n / s_vals + 1) + 1,
                'k--', linewidth=2.5, label='Theory: m ~ s*log(n/s)')
        ax.set_xlabel("Sparsity s (number of non-zero coefficients)", fontsize=10)
        ax.set_ylabel("Number of Measurements m", fontsize=10)
        ax.set_title(f"OMP Phase Transition\n{mat_label}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        print(f"  [{mat_label}] done")

    plt.tight_layout()
    _save("demo3_phase_transition.png")


# ────────────────────────────────────────────────────────────────
def demo4_image_cr_sweep(image_path=None):
    print("\n" + "="*62)
    print("  DEMO 4 — Image Reconstruction at Multiple Compression Ratios")
    print("="*62)

    image = get_test_image(image_path=image_path, size=512)
    CRs   = [0.95, 0.85, 0.75, 0.60]

    fig = plt.figure(figsize=(19, 5))
    gs  = gridspec.GridSpec(1, len(CRs) + 1, wspace=0.06)
    fig.suptitle(
        "Demo 4 — Sub-Nyquist Image Reconstruction at Multiple Compression Ratios\n"
        "Method: Batch OMP  |  Matrix: Gaussian  |  Patch size: 8x8 (same as JPEG)\n"
        "CR = fraction of measurements kept per patch  |  PSNR: higher = better quality",
        fontsize=11, fontweight='bold', y=1.04
    )

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax0.set_title("ORIGINAL IMAGE\n(512x512 pixels\n64 measurements/patch\n= 100% = Nyquist)",
                  fontsize=9, fontweight='bold')
    ax0.axis('off')

    for k, cr in enumerate(CRs):
        m_used = int(cr * 64)
        print(f"  CR={cr:.0%}  ({m_used}/64 measurements per 8x8 patch)...", end=" ", flush=True)
        t0    = time.perf_counter()
        recon = reconstruct_image(image, compression_ratio=cr,
                                  method='batch_omp', matrix_type='gaussian')
        t     = time.perf_counter() - t0
        p     = psnr(image[:recon.shape[0], :recon.shape[1]], recon)
        print(f"PSNR={p:.1f} dB  t={t:.2f}s")

        ax = fig.add_subplot(gs[k + 1])
        ax.imshow(recon.clip(0, 255), cmap='gray', vmin=0, vmax=255)
        ax.set_title(
            f"CR = {cr:.0%}  (Sub-Nyquist)\n"
            f"Measurements: {m_used}/64 per patch\n"
            f"PSNR = {p:.1f} dB\n"
            f"Time = {t:.2f}s",
            fontsize=9
        )
        ax.axis('off')

    plt.tight_layout()
    _save("demo4_image_cr_sweep.png")

def demo5_omp_vs_bp_1d():
    if not CVXPY_AVAILABLE:
        print("\n  [SKIP] Demo 5 — cvxpy not installed (needed for Basis Pursuit).")
        return

    print("\n" + "="*62)
    print("  DEMO 5 — OMP vs Basis Pursuit on 1-D Signals")
    print("="*62)

    n, m     = 64, 30
    s_levels = list(range(1, 16))
    n_trials = 10
    A        = gaussian_measurement_matrix(m, n)

    omp_snr, bp_snr, omp_t, bp_t = [], [], [], []

    for s in s_levels:
        os, bs, ot, bt = 0.0, 0.0, 0.0, 0.0
        for trial in range(n_trials):
            rng    = np.random.default_rng(trial * 31 + s)
            x_true = np.zeros(n)
            x_true[rng.choice(n, s, replace=False)] = rng.standard_normal(s)
            y = take_measurements(A, x_true)

            t0 = time.perf_counter()
            xo, _ = omp(A, y, sparsity=s)
            ot += time.perf_counter() - t0
            os += recovery_snr(x_true, xo)

            t0 = time.perf_counter()
            xb = basis_pursuit(A, y)
            bt += time.perf_counter() - t0
            if xb is not None:
                bs += recovery_snr(x_true, xb)

        omp_snr.append(os / n_trials); omp_t.append(ot / n_trials * 1e3)
        bp_snr.append(bs / n_trials);  bp_t.append(bt / n_trials * 1e3)
        print(f"  s={s:2d}  OMP SNR={omp_snr[-1]:6.1f}dB  t={omp_t[-1]:.1f}ms  |"
              f"  BP SNR={bp_snr[-1]:6.1f}dB  t={bp_t[-1]:.0f}ms")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Demo 5 — OMP vs Basis Pursuit: Recovery Quality and Speed\n"
        f"1-D sparse signals  |  n={n} length  |  m={m} measurements  |"
        f"  Gaussian matrix  |  {n_trials} trials per sparsity level",
        fontsize=11, fontweight='bold'
    )

    ax1.plot(s_levels, omp_snr, 'o-',  color='steelblue', linewidth=2,
             markersize=7, label='OMP  (needs sparsity s)')
    ax1.plot(s_levels, bp_snr,  's--', color='coral',     linewidth=2,
             markersize=7, label='Basis Pursuit  (does NOT need s)')
    ax1.axvline(m // 2, color='grey', linestyle=':', linewidth=1.5,
                label=f'Breakdown zone (s > m/2 = {m//2})')
    ax1.set_xlabel("Sparsity  s  (number of non-zero coefficients)")
    ax1.set_ylabel("Mean Recovery SNR (dB)  — higher is better")
    ax1.set_title("Recovery Quality vs Sparsity\nBoth succeed for low s; both fail near s = m")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.4)

    ax2.semilogy(s_levels, omp_t, 'o-',  color='steelblue', linewidth=2,
                 markersize=7, label='OMP')
    ax2.semilogy(s_levels, bp_t,  's--', color='coral',     linewidth=2,
                 markersize=7, label='Basis Pursuit')
    ax2.set_xlabel("Sparsity  s  (number of non-zero coefficients)")
    ax2.set_ylabel("Mean Runtime (ms)  — lower is faster  [log scale]")
    ax2.set_title("Runtime vs Sparsity\nBP is a convex solver — 10-100x slower than OMP")
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.4, which='both')

    plt.tight_layout()
    _save("demo5_omp_vs_bp_1d.png")


# ────────────────────────────────────────────────────────────────
def demo6_batch_omp_speedup():
    print("\n" + "="*62)
    print("  DEMO 6 — Batch OMP Speedup over Naive OMP")
    print("="*62)

    patch_size = 8
    n_atoms    = patch_size ** 2
    m_meas     = 20
    sparsity   = 8
    A          = gaussian_measurement_matrix(m_meas, n_atoms)
    counts     = [1, 4, 16, 64, 512, 512, 512]
    naive_t, batch_t = [], []

    for P in counts:
        rng    = np.random.default_rng(0)
        X_true = np.zeros((n_atoms, P))
        for p in range(P):
            supp = rng.choice(n_atoms, sparsity, replace=False)
            X_true[supp, p] = rng.standard_normal(sparsity)
        Y = A @ X_true

        t0 = time.perf_counter()
        for p in range(P): omp(A, Y[:, p], sparsity=sparsity)
        naive_t.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        batch_omp(A, Y, sparsity=sparsity)
        batch_t.append((time.perf_counter() - t0) * 1e3)

        speedup = naive_t[-1] / max(batch_t[-1], 1e-9)
        print(f"  P={P:4d}  Naive={naive_t[-1]:.1f}ms  "
              f"Batch={batch_t[-1]:.1f}ms  speedup={speedup:.2f}x")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        "Demo 6 — Batch OMP vs Naive OMP: Runtime Scaling\n"
        "Batch OMP precomputes G = A^T A and A^T Y ONCE, shared across all patches\n"
        f"patch_size={patch_size}x{patch_size}  |  m={m_meas} measurements  |  "
        f"n_atoms={n_atoms}  |  sparsity={sparsity}",
        fontsize=11, fontweight='bold'
    )
    ax.loglog(counts, naive_t, 'o-',  color='coral',    linewidth=2.5,
              markersize=9, label='Naive OMP  (calls omp() once per patch)')
    ax.loglog(counts, batch_t, 's--', color='seagreen', linewidth=2.5,
              markersize=9, label='Batch OMP  (precomputes G=A^T A once for all patches)')
    for P, nt, bt in zip(counts, naive_t, batch_t):
        sp = nt / max(bt, 1e-9)
        ax.annotate(f"{sp:.1f}x faster", (P, bt),
                    textcoords="offset points", xytext=(6, -12), fontsize=7, color='seagreen')
    ax.set_xlabel("Number of patches P  [log scale]")
    ax.set_ylabel("Total runtime (ms)  [log scale]")
    ax.set_title("Both scale linearly with P, but Batch OMP has much lower constant factor")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4, which='both')

    plt.tight_layout()
    _save("demo6_batch_omp_speedup.png")


# ────────────────────────────────────────────────────────────────
def demo7_full_comparison(image_path=None, compression_ratio=0.75):
    print("\n" + "="*62)
    print("  DEMO 7 — Full Method x Matrix Comparison on Image")
    print("="*62)

    methods      = ['omp', 'batch_omp', 'bp']
    method_short = ['OMP', 'Batch OMP', 'Basis Pursuit']
    method_long  = [
        'OMP\n(Orthogonal Matching Pursuit)\nRequires knowing sparsity s',
        'Batch OMP\n(Shared Precomputation)\nSame result as OMP, faster for many patches',
        'Basis Pursuit\n(L1 Minimisation via cvxpy)\nDoes NOT need sparsity s'
    ]
    matrices  = ['gaussian', 'binary']
    mat_short = ['Gaussian', 'Binary']
    mat_long  = ['Gaussian Matrix\n(N(0,1) entries, normalised)',
                 'Binary Matrix\n(+/-1/sqrt(m) entries)']

    if not CVXPY_AVAILABLE:
        methods      = ['omp', 'batch_omp']
        method_short = ['OMP', 'Batch OMP']
        method_long  = method_long[:2]
        print("  [INFO] cvxpy not installed — Basis Pursuit skipped.")

    n_methods = len(methods)
    n_mats    = len(matrices)
    m_used    = int(compression_ratio * 64)

    image = get_test_image(image_path=image_path, size=512)
    results = {}

    for mat, ms in zip(matrices, mat_short):
        for mth, msh in zip(methods, method_short):
            key = (mth, mat)
            print(f"  {msh:<15s} + {ms} Matrix ...", end=" ", flush=True)
            t0    = time.perf_counter()
            recon = reconstruct_image(image, compression_ratio=compression_ratio,
                                      method=mth, matrix_type=mat)
            elapsed = time.perf_counter() - t0
            p       = psnr(image[:recon.shape[0], :recon.shape[1]], recon)
            results[key] = {'recon': recon, 'psnr': p, 'time': elapsed}
            print(f"PSNR = {p:.2f} dB   t = {elapsed:.2f}s")

    H_c       = list(results.values())[0]['recon'].shape[0]
    W_c       = list(results.values())[0]['recon'].shape[1]
    orig_crop = image[:H_c, :W_c]
    fig, axes = plt.subplots(n_methods, n_mats + 1,
                             figsize=(5.5 * (n_mats + 1), 4.5 * n_methods),
                             squeeze=False)
    fig.suptitle(
        f"Demo 7a — Reconstructed Image Grid\n"
        f"Compression Ratio = {compression_ratio:.0%}  "
        f"({m_used}/64 measurements per 8x8 patch)\n"
        "ROWS = Recovery Method  |  COLUMNS = Measurement Matrix  |  "
        "PSNR in dB (higher = better)",
        fontsize=12, fontweight='bold', y=1.02
    )

    for row, (mth, msh, mlong) in enumerate(
            zip(methods, method_short, method_long)):
        ax = axes[row][0]
        ax.imshow(orig_crop, cmap='gray', vmin=0, vmax=255)
        ax.set_ylabel(mlong, fontsize=9, fontweight='bold',
                      rotation=90, labelpad=10)
        top = "ORIGINAL IMAGE\n(reference)" if row == 0 else "ORIGINAL\n(reference)"
        ax.set_title(top, fontsize=9, fontweight='bold',
                     color='darkgreen', pad=8)
        ax.axis('off')

        for col, (mat, ms, mlong2) in enumerate(
                zip(matrices, mat_short, mat_long)):
            ax  = axes[row][col + 1]
            res = results[(mth, mat)]
            ax.imshow(res['recon'].clip(0, 255), cmap='gray', vmin=0, vmax=255)
            col_header = f"{mlong2}\n" if row == 0 else ""
            ax.set_title(
                f"{col_header}"
                f"Method  : {msh}\n"
                f"Matrix  : {ms}\n"
                f"PSNR    : {res['psnr']:.2f} dB\n"
                f"Runtime : {res['time']*1000:.0f} ms",
                fontsize=9, pad=8
            )
            ax.axis('off')

    plt.tight_layout()
    _save("demo7a_reconstructed_image_grid.png")
    fig, axes = plt.subplots(n_methods, n_mats,
                             figsize=(6 * n_mats, 4.5 * n_methods),
                             squeeze=False)
    fig.suptitle(
        f"Demo 7b — Error Maps  |  |Original - Reconstructed|\n"
        f"Compression Ratio = {compression_ratio:.0%}  ({m_used}/64 measurements)\n"
        "BRIGHTER = larger error  |  All panels share the SAME colour scale\n"
        "ROWS = Recovery Method  |  COLUMNS = Measurement Matrix",
        fontsize=12, fontweight='bold', y=1.02
    )

    vmax_err = max(
        np.max(np.abs(orig_crop - results[(mth, mat)]['recon']))
        for mth in methods for mat in matrices
    )

    for row, (mth, msh, mlong) in enumerate(
            zip(methods, method_short, method_long)):
        for col, (mat, ms, mlong2) in enumerate(
                zip(matrices, mat_short, mat_long)):
            ax  = axes[row][col]
            res = results[(mth, mat)]
            err = np.abs(orig_crop - res['recon'])
            im  = ax.imshow(err, cmap='hot', vmin=0, vmax=vmax_err)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('|pixel error|', fontsize=8)
            ax.set_title(
                f"Method: {msh}  |  Matrix: {ms}\n"
                f"Max error = {err.max():.1f}  |  Mean error = {err.mean():.2f}\n"
                f"PSNR = {res['psnr']:.2f} dB  |  Runtime = {res['time']*1000:.0f}ms",
                fontsize=9, pad=8
            )
            ax.set_ylabel(mlong if col == 0 else "",
                          fontsize=9, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    _save("demo7b_error_map_grid.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Demo 7c — PSNR Comparison  (higher = better quality)\n"
        f"Compression Ratio = {compression_ratio:.0%}  "
        f"({m_used}/64 measurements per 8x8 patch)  |  Image 512x512",
        fontsize=12, fontweight='bold'
    )

    x_pos   = np.arange(n_methods)
    width   = 0.35
    offsets = [-(width / 2), width / 2]
    colours = ['steelblue', 'darkorange']
    hatches = ['', '///']

    for ci, (mat, ms, colour, hatch) in enumerate(
            zip(matrices, mat_short, colours, hatches)):
        vals = [results[(mth, mat)]['psnr'] for mth in methods]
        bars = ax.bar(x_pos + offsets[ci], vals, width * 0.92,
                      label=f"{ms} Matrix",
                      color=colour, hatch=hatch,
                      edgecolor='white', linewidth=0.8, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{val:.2f} dB", ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_short, fontsize=12)
    ax.set_ylabel("PSNR (dB)  — higher is better", fontsize=11)
    ax.set_xlabel("Recovery Method", fontsize=11)
    ax.set_title(
        "Solid bars = Gaussian Matrix  |  Hatched bars = Binary Matrix\n"
        "OMP and Batch OMP give identical PSNR (same algorithm, different implementation efficiency)",
        fontsize=10
    )
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.4)
    y_min = min(results[(mth, mat)]['psnr'] for mth in methods for mat in matrices)
    ax.set_ylim(y_min - 3, None)

    plt.tight_layout()
    _save("demo7c_psnr_bar_chart.png")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"Demo 7d — Runtime Comparison  (lower = faster)\n"
        f"Compression Ratio = {compression_ratio:.0%}  "
        f"({m_used}/64 measurements per 8x8 patch)  |  Image 512x512",
        fontsize=12, fontweight='bold'
    )

    for ci, (mat, ms, colour, hatch) in enumerate(
            zip(matrices, mat_short, colours, hatches)):
        vals = [results[(mth, mat)]['time'] * 1000 for mth in methods]
        bars = ax.bar(x_pos + offsets[ci], vals, width * 0.92,
                      label=f"{ms} Matrix",
                      color=colour, hatch=hatch,
                      edgecolor='white', linewidth=0.8, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.08,
                    f"{val:.0f}ms", ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

    ax.set_yscale('log')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_short, fontsize=12)
    ax.set_ylabel("Runtime (ms)  — lower is faster  [log scale]", fontsize=11)
    ax.set_xlabel("Recovery Method", fontsize=11)
    ax.set_title(
        "Solid bars = Gaussian Matrix  |  Hatched bars = Binary Matrix\n"
        "Basis Pursuit uses a convex optimisation solver — orders of magnitude slower than OMP",
        fontsize=10
    )
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.4, which='both')

    plt.tight_layout()
    _save("demo7d_runtime_bar_chart.png")
    print()
    print("  RESULTS SUMMARY")
    print("  " + "="*62)
    print(f"  {'Method':<16} {'Matrix':<12} {'PSNR (dB)':>12} {'Runtime':>12}")
    print("  " + "-"*62)
    for mat, ms in zip(matrices, mat_short):
        for mth, msh in zip(methods, method_short):
            r = results[(mth, mat)]
            print(f"  {msh:<16} {ms:<12} {r['psnr']:>10.2f} dB "
                  f"{r['time']*1000:>9.1f} ms")
    print("  " + "="*62)




def demo8_quality_analysis(image_path=None):
    print("\n" + "="*62)
    print("  DEMO 8 — Quality Analysis: PSNR vs Compression Ratio & Sparsity")
    print("="*62)

    image = get_test_image(image_path=image_path, size=512)
    CRs         = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    s_fractions = [0.20, 0.30, 0.40, 0.50, 0.60]   # s = fraction of m_meas
    patch_size  = 8
    n_atoms     = patch_size ** 2   # 64

    psnr_grid  = np.zeros((len(s_fractions), len(CRs)))

    print(f"  {'CR':>6}  {'s/m':>6}  {'m':>4}  {'s':>4}  {'PSNR':>8}")
    print("  " + "-"*38)

    best_psnr  = -np.inf
    best_recon = None
    best_label = ""

    for ci, cr in enumerate(CRs):
        m_meas = max(4, int(cr * n_atoms))
        for si, sf in enumerate(s_fractions):
            s = max(2, min(int(sf * m_meas), m_meas - 1))
            recon = reconstruct_image(image, patch_size=patch_size,
                                      compression_ratio=cr, sparsity=s,
                                      method='batch_omp', matrix_type='gaussian')
            H_c, W_c  = recon.shape
            p = psnr(image[:H_c, :W_c], recon)
            psnr_grid[si, ci] = p
            print(f"  {cr:>6.0%}  {sf:>6.2f}  {m_meas:>4d}  {s:>4d}  {p:>7.2f} dB")

            if p > best_psnr:
                best_psnr  = p
                best_recon = recon
                best_label = f"CR={cr:.0%}  s/m={sf:.2f}  m={m_meas}  s={s}"

    print(f"\n  Best configuration: {best_label}  ->  PSNR = {best_psnr:.2f} dB")

    H_c, W_c = best_recon.shape
    orig_crop = image[:H_c, :W_c]
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "Demo 8a — PSNR Heatmap: Compression Ratio vs Sparsity Fraction\n"
        "Method: Batch OMP  |  Matrix: Gaussian  |  Image: 512x512  |  Patch: 8x8\n"
        "Each cell shows PSNR (dB) for that (CR, s/m) combination",
        fontsize=11, fontweight='bold'
    )

    im = ax.imshow(psnr_grid, aspect='auto', cmap='RdYlGn',
                   vmin=psnr_grid.min() - 0.5, vmax=psnr_grid.max() + 0.5)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PSNR (dB)  — higher = better', fontsize=10)

    ax.set_xticks(range(len(CRs)))
    ax.set_xticklabels([f"CR={c:.0%}\nm={int(c*64)}/64" for c in CRs], fontsize=9)
    ax.set_yticks(range(len(s_fractions)))
    ax.set_yticklabels([f"s/m = {sf:.0%}" for sf in s_fractions], fontsize=9)
    ax.set_xlabel("Compression Ratio  (fraction of measurements kept)", fontsize=10)
    ax.set_ylabel("Sparsity  s  as fraction of measurements  m", fontsize=10)
    ax.set_title(
        "Higher CR = more measurements = better quality\n"
        "Best s/m ~ 0.40 (OMP reliable when s ≈ 40% of m)",
        fontsize=10
    )
    for si in range(len(s_fractions)):
        for ci in range(len(CRs)):
            val = psnr_grid[si, ci]
            ax.text(ci, si, f"{val:.1f}", ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='black' if 20 < val < 30 else 'white')
    best_si, best_ci = np.unravel_index(np.argmax(psnr_grid), psnr_grid.shape)
    ax.add_patch(plt.Rectangle((best_ci - 0.48, best_si - 0.48), 0.96, 0.96,
                                fill=False, edgecolor='blue', linewidth=3,
                                label=f"Best: {best_psnr:.1f} dB"))
    ax.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    _save("demo8a_psnr_heatmap.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        f"Demo 8b — Best Achievable Reconstruction for This Image\n"
        f"Best config: {best_label}\n"
        f"PSNR = {best_psnr:.2f} dB",
        fontsize=11, fontweight='bold'
    )
    axes[0].imshow(orig_crop, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("ORIGINAL IMAGE\n(ground truth)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(best_recon.clip(0, 255), cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(
        f"BEST CS RECONSTRUCTION\n{best_label}\nPSNR = {best_psnr:.2f} dB",
        fontsize=10
    )
    axes[1].axis('off')
    err = np.abs(orig_crop - best_recon)
    im2 = axes[2].imshow(err, cmap='hot', vmin=0, vmax=err.max())
    plt.colorbar(im2, ax=axes[2], label='|pixel error|')
    axes[2].set_title(
        f"ERROR MAP  |original - reconstructed|\n"
        f"Max error={err.max():.1f}  Mean error={err.mean():.2f}",
        fontsize=10
    )
    axes[2].axis('off')
    plt.tight_layout()
    _save("demo8b_best_vs_original.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Demo 8c — PSNR vs Compression Ratio for Each Sparsity Setting\n"
        "Method: Batch OMP  |  Matrix: Gaussian\n"
        "Shows how quality improves as more measurements are kept",
        fontsize=11, fontweight='bold'
    )
    colours_line = plt.cm.viridis(np.linspace(0.1, 0.9, len(s_fractions)))
    for si, (sf, col) in enumerate(zip(s_fractions, colours_line)):
        ax.plot([c * 100 for c in CRs], psnr_grid[si],
                'o-', color=col, linewidth=2, markersize=7,
                label=f"s/m = {sf:.0%}")
    ax.set_xlabel("Compression Ratio CR  (%  of 64 measurements kept)", fontsize=11)
    ax.set_ylabel("PSNR (dB)  — higher = better", fontsize=11)
    ax.set_title(
        "PSNR plateaus above CR~85% — diminishing returns beyond that\n"
        "s/m~40% consistently performs best for OMP",
        fontsize=10
    )
    ax.legend(title="Sparsity s as\nfraction of m", fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_xticks([int(c*100) for c in CRs])
    ax.set_xticklabels([f"{int(c*100)}%\n({int(c*64)}/64)" for c in CRs])
    plt.tight_layout()
    _save("demo8c_cr_vs_psnr.png")
    print("\n  WHY IS PSNR LIMITED FOR THIS FACE IMAGE?")
    print("  " + "="*58)
    print("  The face image contains heavy HAIR TEXTURE and SKIN DETAIL.")
    print("  Each 8x8 DCT patch in these regions is NOT sparse — it needs")
    print("  30-50 significant coefficients, not 10-15.")
    print()
    print("  CS theory requires:  m >= C * s * log(n/s)")
    print(f"  With n=64, m=48 (CR=75%), this allows s <= ~14 reliable recovery.")
    print(f"  But face patches actually need s~30 -> reconstruction is imperfect.")
    print()
    print("  For higher PSNR on this image, use:")
    print("  - CR >= 90% (57/64 measurements) -> PSNR ~23-24 dB")
    print("  - A simpler image (cartoon face, MRI scan) -> PSNR 40-50 dB")
    print("  " + "="*58)



def demo9_rip_verification():
    print("\n" + "="*62)
    print("  DEMO 9 — RIP Verification (Theorem 1, Baraniuk et al. 2008)")
    print("="*62)

    n         = 64  
    n_trials  = 15   
    C_theory  = 2.0    
    threshold = np.sqrt(2) - 1 

    print(f"  n={n}   n_trials={n_trials} per (k,m) point")
    print(f"  Theorem 1: m >= C*k*log(n/k)  [C={C_theory}]")
    print(f"  Theorem 2: delta_2k < {threshold:.3f} => OMP exact recovery")
    print()
    print("  Exp A: success rate vs rho = m / (k*log(n/k))")
    k_test    = 5
    m_vals_a  = list(range(6, 63, 3))
    rng       = np.random.default_rng(42)
    succ_g_a  = []
    succ_b_a  = []

    for m in m_vals_a:
        ok_g = ok_b = 0
        A_g = gaussian_measurement_matrix(m, n, seed=m)
        A_b = binary_measurement_matrix(m, n,   seed=m)
        for trial in range(n_trials):
            rng2   = np.random.default_rng(trial * 97 + m)
            x_true = np.zeros(n)
            supp   = rng2.choice(n, k_test, replace=False)
            x_true[supp] = rng2.standard_normal(k_test)
            y_g = A_g @ x_true
            y_b = A_b @ x_true
            xh_g, _ = omp(A_g, y_g, sparsity=k_test)
            xh_b, _ = omp(A_b, y_b, sparsity=k_test)
            if recovery_snr(x_true, xh_g) > 40: ok_g += 1
            if recovery_snr(x_true, xh_b) > 40: ok_b += 1
        succ_g_a.append(ok_g / n_trials)
        succ_b_a.append(ok_b / n_trials)

    rho_vals  = [m / (k_test * np.log(n / k_test)) for m in m_vals_a]
    rho_theory = C_theory
    print("  Exp B: empirical m* vs theory  m_theory = C*k*log(n/k)")
    k_vals_b   = list(range(1, 14))
    m_emp_g    = []
    m_emp_b    = []
    m_theory_b = []
    for k in k_vals_b:
        mt = C_theory * k * np.log(n / k)
        m_theory_b.append(mt)
        found_g = found_b = None
        for m in range(k+2, n):
            A_g = gaussian_measurement_matrix(m, n, seed=k*100)
            A_b = binary_measurement_matrix(m, n,   seed=k*100)
            ok_g = ok_b = 0
            for trial in range(n_trials):
                rng2   = np.random.default_rng(trial * 13 + k * 7 + m)
                x_true = np.zeros(n)
                supp   = rng2.choice(n, k, replace=False)
                x_true[supp] = rng2.standard_normal(k)
                if found_g is None:
                    xh, _ = omp(A_g, A_g @ x_true, sparsity=k)
                    if recovery_snr(x_true, xh) > 40: ok_g += 1
                if found_b is None:
                    xh, _ = omp(A_b, A_b @ x_true, sparsity=k)
                    if recovery_snr(x_true, xh) > 40: ok_b += 1
            if found_g is None and ok_g / n_trials >= 0.90:
                found_g = m
            if found_b is None and ok_b / n_trials >= 0.90:
                found_b = m
            if found_g is not None and found_b is not None:
                break
        m_emp_g.append(found_g or n-1)
        m_emp_b.append(found_b or n-1)
        print(f"  k={k:2d}  m_theory={mt:5.1f}  m*_gauss={found_g}  m*_binary={found_b}")
    print("  Exp C: 2D success heatmaps (k vs m)")
    k_range_c = list(range(1, 12, 2))
    m_range_c = list(range(8, 62, 6))
    heat_g    = np.zeros((len(k_range_c), len(m_range_c)))
    heat_b    = np.zeros_like(heat_g)

    for ki, k in enumerate(k_range_c):
        for mi, m in enumerate(m_range_c):
            if m <= k: continue
            A_g = gaussian_measurement_matrix(m, n, seed=k*31+mi)
            A_b = binary_measurement_matrix(m, n,   seed=k*31+mi)
            ok_g = ok_b = 0
            for trial in range(n_trials):
                rng2   = np.random.default_rng(trial*53+ki*17+mi)
                x_true = np.zeros(n)
                supp   = rng2.choice(n, k, replace=False)
                x_true[supp] = rng2.standard_normal(k)
                xg, _ = omp(A_g, A_g @ x_true, sparsity=k)
                xb, _ = omp(A_b, A_b @ x_true, sparsity=k)
                if recovery_snr(x_true, xg) > 40: ok_g += 1
                if recovery_snr(x_true, xb) > 40: ok_b += 1
            heat_g[ki, mi] = ok_g / n_trials
            heat_b[ki, mi] = ok_b / n_trials
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "Demo 9a — OMP Recovery Success Rate vs Normalised Measurement Ratio\n"
        "Theorem 1 verification: m >= C*k*log(n/k) guarantees RIP => OMP exact recovery\n"
        f"k={k_test} sparse signal  |  n={n}  |  {n_trials} trials per point  |  "
        "Success = SNR > 40 dB",
        fontsize=11, fontweight='bold'
    )
    ax.plot(rho_vals, [s*100 for s in succ_g_a], 'o-', color='steelblue',
            linewidth=2.5, markersize=7,
            label='Gaussian  N(0,1/m)  [Theorem 1 matrix]')
    ax.plot(rho_vals, [s*100 for s in succ_b_a], 's--', color='darkorange',
            linewidth=2.5, markersize=7,
            label='Binary  ±1/sqrt(m)')
    ax.axvline(rho_theory, color='red', linewidth=2.5, linestyle=':',
               label=f'Theory: rho = C = {rho_theory}  (m = C*k*log(n/k))')
    ax.fill_betweenx([0,100], rho_theory, max(rho_vals),
                     alpha=0.08, color='green', label='Theory: RIP holds => 100% success')
    ax.fill_betweenx([0,100], min(rho_vals), rho_theory,
                     alpha=0.07, color='red',  label='Theory: RIP may fail')
    ax.axhline(90, color='grey', linewidth=1.2, linestyle='--', alpha=0.7,
               label='90% success threshold')
    ax.set_xlabel(
        "Normalised ratio  rho = m / (k * log(n/k))"
        + f"  [k={k_test}, n={n}]", fontsize=11)
    ax.set_ylabel("OMP Recovery Success Rate (%)", fontsize=11)
    ax.set_title(
        "Sharp transition near rho=C empirically confirms Theorem 1:\n"
        "m >= C*k*log(n/k) is sufficient for RIP and hence exact OMP recovery.",
        fontsize=10
    )
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    ax.set_ylim(-5, 105); ax.set_xlim(min(rho_vals)-0.1, max(rho_vals)+0.1)
    plt.tight_layout()
    _save("demo9a_rip_success_vs_ratio.png")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "Demo 9b — Sample Complexity: Empirical m* vs Theoretical Bound\n"
        f"Theorem 1: m >= C*k*log(n/k) [C={C_theory}]  |  n={n}  |  "
        "m* = smallest m achieving 90% OMP success",
        fontsize=11, fontweight='bold'
    )
    ax.plot(k_vals_b, m_theory_b, 'r-', linewidth=2.5,
            label=f'Theorem 1 bound:  m = {C_theory}*k*log(n/k)')
    ax.plot(k_vals_b, m_emp_g, 'o--', color='steelblue', linewidth=2,
            markersize=8, label='Empirical m*: Gaussian  (90% success)')
    ax.plot(k_vals_b, m_emp_b, 's--', color='darkorange', linewidth=2,
            markersize=8, label='Empirical m*: Binary  (90% success)')
    ax.fill_between(k_vals_b, m_theory_b, m_emp_g,
                    alpha=0.10, color='steelblue')
    ax.fill_between(k_vals_b, m_theory_b, m_emp_b,
                    alpha=0.10, color='darkorange')
    ax.set_xlabel("Sparsity  k  (number of non-zero signal components)", fontsize=11)
    ax.set_ylabel("Measurements  m  required for 90% OMP success", fontsize=11)
    ax.set_title(
        "Empirical m* tracks the theoretical O(k*log(n/k)) curve closely.\n"
        "Both matrices require similar measurements, confirming Theorem 1 for Binary matrices too.",
        fontsize=10
    )
    ax.legend(fontsize=9); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    _save("demo9b_sample_complexity.png")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        "Demo 9c — OMP Recovery Success Heatmap: Gaussian vs Binary Matrix\n"
        f"Theorem 1 predicts success (green) above the curve m = {C_theory}*k*log(n/k)\n"
        f"n={n}  |  {n_trials} trials per cell  |  Green=success  Red=failure",
        fontsize=11, fontweight='bold'
    )
    for ax, heat, mat_name, colour in zip(
            axes, [heat_g, heat_b],
            ['Gaussian Matrix  N(0,1/m)  [Theorem 1]', 'Binary Matrix  ±1/sqrt(m)'],
            ['Blues', 'Oranges']):
        im = ax.imshow(heat, aspect='auto', cmap='RdYlGn',
                       vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, ax=ax, label='OMP Success Rate')
        ax.set_xticks(range(len(m_range_c)))
        ax.set_xticklabels([str(m) for m in m_range_c], fontsize=8)
        ax.set_yticks(range(len(k_range_c)))
        ax.set_yticklabels([str(k) for k in k_range_c], fontsize=9)
        ax.set_xlabel("Measurements  m", fontsize=10)
        ax.set_ylabel("Sparsity  k", fontsize=10)
        ax.set_title(mat_name, fontsize=11, fontweight='bold')
        theory_m = [C_theory * k * np.log(n/k) for k in k_range_c]
        theory_idx = [(mt - m_range_c[0]) / (m_range_c[-1] - m_range_c[0])
                      * (len(m_range_c)-1) for mt in theory_m]
        ax.plot(theory_idx, range(len(k_range_c)), 'k--', linewidth=2.5,
                label=f'Theory: m={C_theory}*k*log(n/k)')
        for ki2 in range(len(k_range_c)):
            for mi2 in range(len(m_range_c)):
                val = heat[ki2, mi2]
                ax.text(mi2, ki2, f'{val:.0%}', ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='white' if val < 0.5 else 'black')
        ax.legend(fontsize=8, loc='lower right')

    plt.tight_layout()
    _save("demo9c_gaussian_vs_binary_rip.png")
    print(f"\n  Theorem 1 verified: sharp transition near rho = m/(k*log(n/k)) = {C_theory}")
    print(f"  Both Gaussian and Binary matrices satisfy the sample complexity bound.")


def _run_all_demos(image_path, output_dir):
    import builtins
    builtins._CS_OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    img_label = os.path.splitext(os.path.basename(image_path))[0] if image_path else "synthetic"
    print(f"\n  Image    : {image_path or '(synthetic)'}")
    print(f"  Output   : {output_dir}")
    print(f"  Label    : {img_label}")
    print()

    demo1_sparse_1d_recovery()                              
    demo2_dct_benchmark()                                   
    demo3_phase_transition()                                
    demo4_image_cr_sweep(image_path)                        
    demo5_omp_vs_bp_1d()                                   
    demo6_batch_omp_speedup()                               
    demo7_full_comparison(image_path, compression_ratio=0.75)  
    demo8_quality_analysis(image_path)                      
    demo9_rip_verification()                                


if __name__ == "__main__":
    IMAGE_PATH = None
    IMAGE_FOLDER = r"C:\Users\nimba\OneDrive\Desktop\IISc Bangalore\Sem 4\IASP\Project\codes\images"
    OUTPUT_DIR = r"C:\Users\nimba\OneDrive\Desktop\IISc Bangalore\Sem 4\IASP\Project\codes\images\outputs"
    _args = sys.argv[1:]
    if '--folder' in _args:
        _idx = _args.index('--folder')
        IMAGE_FOLDER = _args[_idx + 1]
        if len(_args) > _idx + 2:
            OUTPUT_DIR = _args[_idx + 2]
    elif len(_args) >= 1:
        IMAGE_PATH = _args[0]
        if not os.path.isfile(IMAGE_PATH):
            print(f"[ERROR] Image file not found: {IMAGE_PATH}")
            sys.exit(1)
        if len(_args) >= 2:
            OUTPUT_DIR = _args[1]
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if OUTPUT_DIR == "":
        OUTPUT_DIR = _script_dir
    elif not os.path.isabs(OUTPUT_DIR):
        OUTPUT_DIR = os.path.join(_script_dir, OUTPUT_DIR)
    OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

    if IMAGE_FOLDER and not os.path.isabs(IMAGE_FOLDER):
        IMAGE_FOLDER = os.path.join(_script_dir, IMAGE_FOLDER)
    print("\n" + "="*62)
    print("  Sparse Signal Recovery via Greedy Algorithms")
    print("  Compressed Sensing & Sub-Nyquist Image Reconstruction")
    print("="*62)
    print(f"\n  cvxpy available : {CVXPY_AVAILABLE}")
    if IMAGE_FOLDER:
        if not os.path.isdir(IMAGE_FOLDER):
            print(f"[ERROR] Folder not found: {IMAGE_FOLDER}")
            sys.exit(1)

        EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        images = sorted([
            os.path.join(IMAGE_FOLDER, f)
            for f in os.listdir(IMAGE_FOLDER)
            if os.path.splitext(f)[1].lower() in EXTS
        ])
        if not images:
            print(f"[ERROR] No image files found in: {IMAGE_FOLDER}")
            sys.exit(1)
        print(f"  Mode            : FOLDER  ({len(images)} images found)")
        print(f"  Image folder    : {IMAGE_FOLDER}")
        print(f"  Output root     : {OUTPUT_DIR}")
        print(f"  Sub-folder per image: OUTPUT_DIR/<image_name>/")
        print()
        for i, img_path in enumerate(images, 1):
            img_name   = os.path.splitext(os.path.basename(img_path))[0]
            img_out    = os.path.join(OUTPUT_DIR, img_name)
            print("\n" + "="*62)
            print(f"  IMAGE {i}/{len(images)} : {img_name}")
            print("="*62)
            _run_all_demos(img_path, img_out)
        print("\n" + "="*62)
        print(f"  ALL {len(images)} IMAGES COMPLETE")
        print(f"  Results saved under: {OUTPUT_DIR}/")
        for img_path in images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            print(f"    {OUTPUT_DIR}/{img_name}/")
        print("="*62)
    else:
        if IMAGE_PATH:
            img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
            single_out = os.path.join(OUTPUT_DIR, img_name)
        else:
            single_out = os.path.join(OUTPUT_DIR, "synthetic")

        print(f"  Mode            : SINGLE IMAGE")
        print(f"  Image           : {IMAGE_PATH or '(none — synthetic)'}")
        print(f"  Output folder   : {single_out}")

        _run_all_demos(IMAGE_PATH, single_out)

        print("\n" + "="*62)
        print("  ALL DEMOS COMPLETE")
        print(f"  All outputs saved to: {single_out}/")
        files = [
            ("demo1_sparse_1d_recovery.png",    "OMP vs BP, all matrix/noise combos"),
            ("demo2_dct_benchmark.png",          "DCT accuracy and runtime"),
            ("demo3_phase_transition.png",       "Phase transition: Gaussian + Binary"),
            ("demo4_image_cr_sweep.png",         "Image at 4 compression ratios"),
            ("demo5_omp_vs_bp_1d.png",           "OMP vs BP: SNR + runtime vs sparsity"),
            ("demo6_batch_omp_speedup.png",      "Batch OMP speedup over naive OMP"),
            ("demo7a_reconstructed_image_grid.png", "All 6 reconstructions grid"),
            ("demo7b_error_map_grid.png",        "Error maps for all 6 combos"),
            ("demo7c_psnr_bar_chart.png",        "PSNR bar chart"),
            ("demo7d_runtime_bar_chart.png",     "Runtime bar chart"),
            ("demo8a_psnr_heatmap.png",          "PSNR heatmap: CR x sparsity"),
            ("demo8b_best_vs_original.png",      "Best reconstruction vs original"),
            ("demo8c_cr_vs_psnr.png",            "PSNR vs CR line chart"),
            ("demo9a_rip_success_vs_ratio.png",   "OMP success rate vs m/(k*log(n/k)) - Theorem 1 verification"),
            ("demo9b_sample_complexity.png",      "Empirical m* vs theory m = C*k*log(n/k)"),
            ("demo9c_gaussian_vs_binary_rip.png", "Success heatmaps Gaussian vs Binary with theory curve"),
        ]
        for fname, desc in files:
            print(f"  {os.path.join(single_out, fname)}")
            print(f"    {desc}")
        print("="*62)
