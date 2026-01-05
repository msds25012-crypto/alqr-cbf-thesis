import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------
df = pd.read_csv("toxicity_sweep_setpoint/llama8b.csv")

Q_VALUES = [.1, 1, 10, 100, 1000]
R_VALUES = [0.1, 1, 10, 100, 1000]

lambda_vals = sorted(df["lmbda"].unique())

# Compute toxicity metric
# df["tox_contr_ratio"] = df["num_tox_contr"] / 1000.0

# Aggregate duplicates
agg = df.groupby(["qf", "q", "r", "lmbda"], as_index=False).mean()


# ---------------------------------------------------------
# Function to plot one figure (3 heatmaps for r values)
# ---------------------------------------------------------

num_maps = 5
# def plot_qf_figure(metric, title_prefix, cmap="viridis"):
#     qf_values = sorted(agg["qf"].unique())

#     for qf_val in qf_values:
#         fig, axes = plt.subplots(1, num_maps, figsize=(15, 4), sharey=True)

#         for ax, r_val in zip(axes, R_VALUES):
#             sub = agg[(agg["qf"] == qf_val) & (agg["r"] == r_val)]

#             pivot = sub.pivot(index="q", columns="lmbda", values=metric).reindex(
#                 index=Q_VALUES, columns=lambda_vals
#             )

#             sns.heatmap(
#                 pivot,
#                 annot=True,
#                 cmap=cmap,
#                 fmt=".3f",
#                 vmin=agg[metric].min(),
#                 vmax=agg[metric].max(),
#                 cbar=True,
#                 ax=ax
#             )

#             ax.set_title(f"r = {r_val}")
#             ax.set_xlabel("λ")
#             ax.set_ylabel("q")

#         plt.suptitle(f"{title_prefix} — qf = {qf_val}", fontsize=16)
#         plt.tight_layout(rect=[0, 0, 1, 0.93])
#         plt.savefig(f"llama1b_heatmap_qf{qf_val}.png")

# def plot_qf_figure(metric, title_prefix, cmap="viridis"):
#     qf_values = sorted(agg["qf"].unique())
#     n_qf = len(qf_values)

#     fig, axes = plt.subplots(
#         n_qf, num_maps,
#         figsize=(4 * num_maps, 4 * n_qf),
#         sharey=True
#     )

#     # If there's only one qf, axes won't be 2D
#     if n_qf == 1:
#         axes = axes[None, :]

#     for row_idx, qf_val in enumerate(qf_values):
#         for col_idx, r_val in enumerate(R_VALUES):
#             ax = axes[row_idx, col_idx]

#             sub = agg[(agg["qf"] == qf_val) & (agg["r"] == r_val)]

#             pivot = sub.pivot(
#                 index="q",
#                 columns="lmbda",
#                 values=metric
#             ).reindex(index=Q_VALUES, columns=lambda_vals)

#             sns.heatmap(
#                 pivot,
#                 annot=True,
#                 cmap=cmap,
#                 fmt=".3f",
#                 vmin=agg[metric].min(),
#                 vmax=agg[metric].max(),
#                 cbar=(col_idx == num_maps - 1),  # one colorbar per row
#                 ax=ax
#             )

#             # Column titles (r)
#             if row_idx == 0:
#                 ax.set_title(f"r = {r_val}")

#             ax.set_xlabel("λ")
#             ax.set_ylabel("q")

#         # Row label for qf
#         for row_idx, qf_val in enumerate(qf_values):
#             y = 1 - (row_idx + 0.5) / n_qf  # center of each row in figure coords

#             fig.text(
#                 0.01, y,
#                 f"qf = {qf_val}",
#                 rotation=90,
#                 va="center",
#                 ha="left",
#                 fontsize=12,
#                 fontweight="bold"
#             )
#         # axes[row_idx, 0].annotate(
#         #     f"qf = {qf_val}",
#         #     xy=(-0.4, 0.5),
#         #     xycoords="axes fraction",
#         #     rotation=90,
#         #     va="center",
#         #     ha="center",
#         #     fontsize=12,
#         #     fontweight="bold"
#         # )

#     plt.suptitle(title_prefix, fontsize=18)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(f"llama1b_heatmap_by_qf_{metric}.png")
#     plt.close()
qfs = [1, 10, 100]
def plot_qf_figure(metric, title_prefix, cmap="viridis"):
    # qf_values = sorted(agg["qf"].unique())
    # qf_values = [qf for qf in qf_values if qf in agg["qf"].unique()]
    qf_values = sorted(
        agg.loc[agg["qf"].isin(qfs), "qf"].unique()
    )
    n_qf = len(qf_values)


    fig = plt.figure(figsize=(4 * num_maps, 4.5 * n_qf))
    subfigs = fig.subfigures(n_qf, 1, hspace=0.15)

    # Handle single-qf case
    if n_qf == 1:
        subfigs = [subfigs]

    vmin = agg[metric].min()
    vmax = agg[metric].max()

    for row_idx, (qf_val, subfig) in enumerate(zip(qf_values, subfigs)):
        subfig.suptitle(f"qf = {qf_val}", fontsize=14, va='top')

        axes = subfig.subplots(
            1, num_maps,
            sharey=True
        )

        for col_idx, (ax, r_val) in enumerate(zip(axes, R_VALUES)):
            sub = agg[(agg["qf"] == qf_val) & (agg["r"] == r_val)]

            pivot = sub.pivot(
                index="q",
                columns="lmbda",
                values=metric
            ).reindex(index=Q_VALUES, columns=lambda_vals)

            sns.heatmap(
                pivot,
                annot=True,
                cmap=cmap,
                fmt=".3f",
                vmin=vmin,
                vmax=vmax,
                cbar=(col_idx == num_maps - 1),
                ax=ax
            )

            ax.set_xlabel("λ")
            ax.set_ylabel("q")

            if row_idx == 0:
                ax.set_title(f"r = {r_val}")

    fig.suptitle(title_prefix, fontsize=18)
    fig.savefig(f"llama8b_heatmap_by_qf_{metric}.png")
    plt.close(fig)

# ---------------------------------------------------------
# 1. Toxicity heatmaps (num_tox_contr/1000)
# ---------------------------------------------------------
plot_qf_figure(
    metric="toxicity_rate",
    # metric="tox_contr_ratio",
    title_prefix="Toxicity Heatmaps (num_tox_contr/1000)",
    cmap="magma"
)

# ---------------------------------------------------------
# 2. Dist-1 heatmaps
# ---------------------------------------------------------
plot_qf_figure(
    metric="dist_1_steered",
    title_prefix="Dist-1 Steered Heatmaps",
    cmap="coolwarm"
)
