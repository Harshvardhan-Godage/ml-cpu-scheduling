"""Synthetic workload generator + feature extraction (features use only process descriptors,
i.e. information a scheduler could know/estimate from process history before scheduling)."""
import numpy as np

# type -> (cpu burst mean range, io/cpu ratio range, bursts-per-process mean range, burst distribution)
TYPES = {
    "cpu_bound":   ((30, 80), (0.05, 0.4), (2, 5),  "exp"),
    "io_bound":    ((2, 8),   (1.5, 6.0),  (8, 20), "exp"),
    "interactive": ((1, 4),   (3.0, 12.0), (10, 25), "exp"),
    "heavy_tail":  ((5, 25),  (0.3, 2.0),  (3, 10), "lognormal"),
    "mixed":       ((3, 60),  (0.1, 4.0),  (3, 15), "bimodal"),
}


def _burst(rng, dist, mean, k):
    if dist == "exp":
        b = rng.exponential(mean, k)
    elif dist == "lognormal":
        s = rng.uniform(1.0, 1.6)
        b = rng.lognormal(np.log(mean) - s * s / 2, s, k)
    else:  # bimodal: 70% short, 30% long
        short = rng.random(k) < 0.7
        b = np.where(short, rng.exponential(mean * 0.15, k), rng.exponential(mean * 2.5, k))
    return np.maximum(b, 0.5).round(2)


def generate_scenario(n, rng, wtype=None):
    wtype = wtype or rng.choice(list(TYPES))
    (c_lo, c_hi), (r_lo, r_hi), (k_lo, k_hi), dist = TYPES[wtype]
    cpu_mean, io_ratio = rng.uniform(c_lo, c_hi), rng.uniform(r_lo, r_hi)
    k_mean = rng.uniform(k_lo, k_hi)
    procs, total_cpu = [], 0.0
    for pid in range(n):
        k = max(1, rng.poisson(k_mean))
        cpu = _burst(rng, dist, cpu_mean, k)
        io = _burst(rng, "exp", cpu_mean * io_ratio, k - 1) if k > 1 else np.array([])
        procs.append({"pid": pid, "cpu": cpu.tolist(), "io": io.tolist()})
        total_cpu += cpu.sum()
    # offered load -> arrival span (rho 0.4 .. 1.5 = light .. overloaded)
    rho = rng.uniform(0.4, 1.5)
    span = total_cpu / rho
    if rng.random() < 0.25:             # batch arrival (all near t=0)
        arrivals = rng.uniform(0, span * 0.02, n)
    else:                               # Poisson-like arrivals
        arrivals = np.sort(rng.uniform(0, span, n))
    for p, a in zip(procs, arrivals):
        p["arrival"] = float(round(a, 2))
    return procs, wtype


def extract_features(procs):
    cpu = np.concatenate([p["cpu"] for p in procs])
    io = np.concatenate([p["io"] for p in procs if p["io"]] or [np.array([0.0])])
    arr = np.sort([p["arrival"] for p in procs])
    ia = np.diff(arr) if len(arr) > 1 else np.array([0.0])
    span = max(arr[-1] - arr[0], 1e-6)
    total_cpu, total_io = cpu.sum(), io.sum()
    return {
        "n_procs": len(procs),
        "mean_cpu_burst": cpu.mean(), "median_cpu_burst": np.median(cpu),
        "std_cpu_burst": cpu.std(), "cv_cpu_burst": cpu.std() / cpu.mean(),
        "p95_cpu_burst": np.percentile(cpu, 95), "max_over_mean_burst": cpu.max() / cpu.mean(),
        "short_burst_frac": float((cpu < 5).mean()),
        "mean_io_burst": io.mean(), "io_ratio": total_io / (total_io + total_cpu),
        "bursts_per_proc": len(cpu) / len(procs),
        "mean_interarrival": ia.mean(), "cv_interarrival": ia.std() / (ia.mean() + 1e-9),
        "offered_load": total_cpu / span,
    }
