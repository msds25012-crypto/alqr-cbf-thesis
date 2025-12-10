import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Load and filter data
# ---------------------------------------------------------
df = pd.read_csv("qwen_data.csv")

Q_VALUES = [.1, 1, 10]
R_VALUES = [0.1, 1, 10]

lambda_vals = sorted(df["lambda"].unique())

# Compute toxicity metric
df["tox_contr_ratio"] = df["num_tox_contr"] / 1000.0

# Aggregate duplicates
agg = df.groupby(["qf", "q", "r", "lambda"], as_index=False).mean()


# ---------------------------------------------------------
# Function to plot one figure (3 heatmaps for r values)
# ---------------------------------------------------------
def plot_qf_figure(metric, title_prefix, cmap="viridis"):
    qf_values = sorted(agg["qf"].unique())

    for qf_val in qf_values:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

        for ax, r_val in zip(axes, R_VALUES):
            sub = agg[(agg["qf"] == qf_val) & (agg["r"] == r_val)]

            pivot = sub.pivot(index="q", columns="lambda", values=metric).reindex(
                index=Q_VALUES, columns=lambda_vals
            )

            sns.heatmap(
                pivot,
                annot=True,
                cmap=cmap,
                fmt=".3f",
                vmin=agg[metric].min(),
                vmax=agg[metric].max(),
                cbar=True,
                ax=ax
            )

            ax.set_title(f"r = {r_val}")
            ax.set_xlabel("λ")
            ax.set_ylabel("q")

        plt.suptitle(f"{title_prefix} — qf = {qf_val}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


# ---------------------------------------------------------
# 1. Toxicity heatmaps (num_tox_contr/1000)
# ---------------------------------------------------------
plot_qf_figure(
    metric="tox_contr_ratio",
    title_prefix="Toxicity Heatmaps (num_tox_contr/1000)",
    cmap="magma"
)

# ---------------------------------------------------------
# 2. Dist-1 heatmaps
# ---------------------------------------------------------
plot_qf_figure(
    metric="dist1_steered",
    title_prefix="Dist-1 Steered Heatmaps",
    cmap="coolwarm"
)
