"""Unwhitened sample metrics with explicit finite-sample reference controls."""
from utils.pointcloud_metrics import exact_discrete_w2_distance, energy_distance_u_statistic_samples, mmd_rbf_samples
from .posterior import sliced_w2


def distribution_metrics(generated, reference, directions, subset_count, gamma):
    # The first subset_count IID samples form a fixed, reproducible subset.
    x = generated[:subset_count].detach().cpu().numpy()
    y = reference[:subset_count].detach().cpu().numpy()
    return {
        "swd": sliced_w2(generated, reference, directions),
        "w2": exact_discrete_w2_distance(y, x, max_samples=None),
        "energy_distance": energy_distance_u_statistic_samples(y, x, max_samples=None),
        "mmd2": mmd_rbf_samples(y, x, max_samples=None, gamma=gamma, standardize_features=False, unbiased=True),
    }


def generative_plot(out):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader((out / "metrics/generative.csv").open()))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor="white")
    for ax, metric in zip(axes.flat, ["w2", "swd", "energy_distance", "mmd2"]):
        for prefix, color, label in [("", "#E07A5F", "Learned flow"), ("real_real_", "#3D405B", "Real vs real"), ("oracle_sampler_", "#81B29A", "Oracle flow")]:
            ax.plot([int(r["step"]) for r in rows], [float(r[prefix+metric]) for r in rows], "o-", color=color, label=label)
        ax.set(xlabel="Optimizer steps", ylabel=metric)
        ax.set_facecolor("white")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "plots/generative_curves.png", dpi=160)
    plt.close(fig)
