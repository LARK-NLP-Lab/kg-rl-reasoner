import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

seed_42 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_KG_kl_0.0_seed42/layer_metrics.csv"
seed_84 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_KG_kl_0.0_seed84/layer_metrics.csv"
seed_126 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_KG_kl_0.0_seed126/layer_metrics.csv"


seed_42_1 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_KG_kl_0.1_seed42/layer_metrics.csv"
seed_84_1 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_KG_kl_0.1_seed84/layer_metrics.csv"
seed_126_1 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_KG_kl_0.1_seed126/layer_metrics.csv"

seed_42_Probsum = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_Probsum_kl_0.0_seed42/layer_metrics.csv"
seed_84_Probsum = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_Probsum_kl_0.0_seed84/layer_metrics.csv"
seed_126_Probsum = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_Probsum_kl_0.0_seed126/layer_metrics.csv"

seed_42_Probsum_1 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_Probsum_kl_0.1_seed42/layer_metrics.csv"
seed_84_Probsum_1 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_Probsum_kl_0.1_seed84/layer_metrics.csv"
seed_126_Probsum_1 = "/data/khatwans/rl/reward_modeling/3d-plot/gradients_results/Qwen7B_Probsum_kl_0.1_seed126/layer_metrics.csv"


df_42 = pd.read_csv(seed_42)
df_84 = pd.read_csv(seed_84)
df_126 = pd.read_csv(seed_126)

df_42_1 = pd.read_csv(seed_42_1)
df_84_1 = pd.read_csv(seed_84_1)
df_126_1 = pd.read_csv(seed_126_1)

df_42_Probsum = pd.read_csv(seed_42_Probsum)
df_84_Probsum = pd.read_csv(seed_84_Probsum)
df_126_Probsum = pd.read_csv(seed_126_Probsum)

df_42_Probsum_1 = pd.read_csv(seed_42_Probsum_1)
df_84_Probsum_1 = pd.read_csv(seed_84_Probsum_1)
df_126_Probsum_1 = pd.read_csv(seed_126_Probsum_1)

def generate_graph_attention_only(df, threshold):
    atten_df = df[df["layer"].str.contains("attn_q|attn_k|attn_v|attn_o", regex=True)]
    sample_1 = atten_df[atten_df["layer"].str.contains("layer_1_", regex=True)]
    num_layers = atten_df.shape[0] // sample_1.shape[0]

    per_layer = []

    for l in range(num_layers):
        layer_df = atten_df[atten_df["layer"].str.contains(f"layer_{l}_", regex=True)]
        active_df = layer_df[layer_df["energy_shift"].abs() > threshold]
        
        per_layer.append({
            "layer": l,
            "distortion": (1 - active_df['cosine']).mean(),
            "intervention_density": (layer_df["energy_shift"].abs() > threshold).mean()
        })
    
    return pd.DataFrame(per_layer)
        
for_seed_42 = generate_graph_attention_only(df_42, 1e-9)
for_seed_84 = generate_graph_attention_only(df_84, 1e-9)
for_seed_126 = generate_graph_attention_only(df_126, 1e-9)

avg_kg_0 = (for_seed_42 + for_seed_84 + for_seed_126) / 3

for_seed_42_1 = generate_graph_attention_only(df_42_1, 1e-9)
for_seed_84_1 = generate_graph_attention_only(df_84_1, 1e-9)
for_seed_126_1 = generate_graph_attention_only(df_126_1, 1e-9)

avg_kg_1 = (for_seed_42_1 + for_seed_84_1 + for_seed_126_1) / 3


for_seed_42_Probsum = generate_graph_attention_only(df_42_Probsum, 1e-9)
for_seed_84_Probsum = generate_graph_attention_only(df_84_Probsum, 1e-9)
for_seed_126_Probsum = generate_graph_attention_only(df_126_Probsum, 1e-9)


avg_probsum_0 = (for_seed_42_Probsum + for_seed_84_Probsum + for_seed_126_Probsum) / 3

for_seed_42_Probsum_1 = generate_graph_attention_only(df_42_Probsum_1, 1e-9)
for_seed_84_Probsum_1 = generate_graph_attention_only(df_84_Probsum_1, 1e-9)
for_seed_126_Probsum_1 = generate_graph_attention_only(df_126_Probsum_1, 1e-9)

avg_probsum_1 = (for_seed_42_Probsum_1 + for_seed_84_Probsum_1 + for_seed_126_Probsum_1) / 3

# -----------------------------
# Create figure
# -----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# -----------------------------
# Scatter points
# -----------------------------
ax.plot(avg_kg_0['layer'].to_numpy(), avg_kg_0['distortion'].to_numpy(), avg_kg_0['intervention_density'].to_numpy(),
           color='limegreen',
           linestyle='--',
           linewidth=2,
           alpha=0.8,
           marker='o',
           label='KG SFT \u03BB=0.0')
        
ax.plot(avg_kg_1['layer'].to_numpy(), avg_kg_1['distortion'].to_numpy(), avg_kg_1['intervention_density'].to_numpy(),
           color='dodgerblue',
           linestyle='--',
           linewidth=2,
           alpha=0.8,
           marker='o',
           label='KG SFT \u03BB=0.1')

ax.plot(avg_probsum_0['layer'].to_numpy(), avg_probsum_0['distortion'].to_numpy(), avg_probsum_0['intervention_density'].to_numpy(),
           color='salmon',
           linestyle='--',
           linewidth=2,
           alpha=0.8,
           marker='o',
           label='Task SFT \u03BB=0.0')

ax.plot(avg_probsum_1['layer'].to_numpy(), avg_probsum_1['distortion'].to_numpy(), avg_probsum_1['intervention_density'].to_numpy(),
           color='dimgray',
           linestyle='--',
           linewidth=2,
           alpha=0.8,
           marker='o',
           label='Task SFT \u03BB=0.1')


for x, y, z in zip(avg_kg_0['layer'].to_numpy(), avg_kg_0['distortion'].to_numpy(), avg_kg_0['intervention_density'].to_numpy()):
    ax.plot([x, x], [y, y], [0, z],
            color='green', alpha=0.7)

for x, y, z in zip(avg_kg_1['layer'].to_numpy(), avg_kg_1['distortion'].to_numpy(), avg_kg_1['intervention_density'].to_numpy()):
    ax.plot([x, x], [y, y], [0, z],
            color='blue', alpha=0.7)
    
for x, y, z in zip(avg_probsum_0['layer'].to_numpy(), avg_probsum_0['distortion'].to_numpy(), avg_probsum_0['intervention_density'].to_numpy()):
    ax.plot([x, x], [y, y], [0, z],
            color='red', alpha=0.7)

for x, y, z in zip(avg_probsum_1['layer'].to_numpy(), avg_probsum_1['distortion'].to_numpy(), avg_probsum_1['intervention_density'].to_numpy()):
    ax.plot([x, x], [y, y], [0, z],
            color='black', alpha=0.7)

ax.set_xlabel("Layer Depth",
              fontsize=14,
              fontweight='bold')

ax.set_ylabel("Distortion (1-cos)",
              fontsize=14,
              fontweight='bold')

ax.set_zlabel("Energy Density",
              fontsize=14,
              fontweight='bold')
ax.legend()
ax.view_init(elev=20, azim=-55)

plt.tight_layout()
plt.savefig("ablation_plot.png")
plt.show()