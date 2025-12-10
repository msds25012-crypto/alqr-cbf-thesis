import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
# df = pd.read_csv("gemma_data.csv")
df = pd.read_csv("qwen_data.csv")

# Filter for qf = 0.1
df_qf = df[df["qf"] == 1]

# Define r values and metrics with colormaps
r_values = sorted(df_qf["r"].unique())
metrics = [
    ("num_tox_contr", "Toxicity Score (%)", "viridis"),
    ("dist1_steered", "Dist 1 Score", "rocket")
]

fig, axes = plt.subplots(2, len(r_values), figsize=(6 * len(r_values), 10), sharey=True)

for row_idx, (metric, metric_label, cmap) in enumerate(metrics):
    for col_idx, r_val in enumerate(r_values):
        ax = axes[row_idx, col_idx]
        df_r = df_qf[df_qf["r"] == r_val]
        
        # Pivot table with mean to handle duplicates
        heatmap_data = df_r.pivot_table(index="q", columns="lambda", values=metric, aggfunc="mean")
        
        if metric == "num_tox_contr":
            heatmap_data /= 1000
        
        # Sort rows and columns
        heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
        
        sns.heatmap(
            heatmap_data, 
            ax=ax, 
            cmap=cmap, 
            cbar=(col_idx==1),
            cbar_kws={'label': metric_label} if (col_idx==2) else None, 
            annot=True, 
            fmt=".2f"
        )
        if col_idx == 1:
            cbar = ax.collections[-1].colorbar
            cbar.set_label(metric_label, fontsize=16)
        ax.set_title(f"r = {r_val}")
        ax.set_xlabel("lambda")
        if col_idx == 0:
            ax.set_ylabel("q")
        else:
            ax.set_ylabel("")

fig.suptitle("Heatmaps for qf = 1", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()
plt.savefig("qwen_heatmap.png")
