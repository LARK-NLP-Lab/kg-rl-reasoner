import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import gc
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

sns.set_theme(style="whitegrid", context="talk")


class ModelDivergenceAnalyzer:
    def __init__(self, checkpoint_a: str, checkpoint_b: str, tokenizer_path: str, experiment_name: str):
        self.ckpt_a = checkpoint_a
        self.ckpt_b = checkpoint_b
        self.output_dir = os.path.join(
            "gradient_and_wasserstein/divergence_results_probsum", experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # Load models with proper LoRA merging
    def load_and_merge_model(self, checkpoint_path):
        """Load model and merge LoRA adapters if present."""

        try:
            model = AutoPeftModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                trust_remote_code=True
            )
            model = model.merge_and_unload()
        except Exception as e:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map="cuda:0",
                trust_remote_code=True
            )

        model.eval()
        for p in model.parameters():
            p.requires_grad = True

        return model

    def _get_component_type(self, name):
        """Helper to classify parameters into simple categories."""
        if "self_attn" in name:
            if "q_proj" in name:
                return "attn_q"
            if "k_proj" in name:
                return "attn_k"
            if "v_proj" in name:
                return "attn_v"
            if "o_proj" in name:
                return "attn_o"
            return "attn_other"
        elif "mlp" in name:
            if "gate" in name or "up" in name:
                return "mlp_gate_up"
            if "down" in name:
                return "mlp_down"
            return "mlp_other"
        return "other"

    def _compute_gradients(self, model, text: str):
        """Compute gradients for a single example."""
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=2048).to(model.device)

        model.zero_grad()
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()

        grads = defaultdict(list)

        for name, param in model.named_parameters():
            if param.grad is None or "layers." not in name:
                continue

            parts = name.split(".")
            # print(name, parts)
            layer_idx = int(parts[parts.index("layers") + 1])
            comp_type = self._get_component_type(name)

            key = f"layer_{layer_idx}_{comp_type}"

            # make sure using the .grad not .data
            grads[key].append(param.grad.detach().cpu().float().flatten())

        return {k: torch.cat(v) for k, v in grads.items()}, outputs.loss.item()

    def _compare_gradients(self, grads_a, grads_b, store_for_batch=False, sample_id=None):
        """Compare gradients: cosine, mse, energy shift, sign agreement."""
        metrics = {}
        common_keys = set(grads_a.keys()) & set(grads_b.keys())

        for key in common_keys:
            g_a = grads_a[key]
            g_b = grads_b[key]

            g_a_norm = F.normalize(g_a.unsqueeze(0), p=2, dim=1).squeeze(0)
            g_b_norm = F.normalize(g_b.unsqueeze(0), p=2, dim=1).squeeze(0)

            # 1. Cosine similarity
            cosine = F.cosine_similarity(
                g_a_norm, g_b_norm, dim=0, eps=1e-8).item()

            # 2. MSE on normalized vectors
            mse = ((g_a_norm - g_b_norm) ** 2).mean().item()

            # 3. Energy shift (log ratio)
            energy_shift = (g_a_norm.norm().log() -
                            g_b_norm.norm().log()).item()

            # 4. Sign agreement
            sign_agreement = (g_a_norm.sign() ==
                              g_b_norm.sign()).float().mean().item()
            # print(g_a_norm.shape, g_b_norm.shape, sign_agreement)

            metrics[key] = {
                "cosine": cosine,
                "mse": mse,
                "energy_shift": energy_shift,
                "sign_agreement": sign_agreement,
            }

        return metrics
    
    def plot_combined_metrics(self, df, filename="plot_combined.png"):
            """Generate a 2x2 plot of all gradient metrics."""
            # Create a 2x2 subplot layout
            fig, axes = plt.subplots(2, 2, figsize=(24, 16))
            axes = axes.flatten()

            metrics = [
                ('cosine', "Gradient Cosine Similarity"),
                ('mse', "Gradient MSE"),
                ('energy_shift', "Gradient Energy Shift"),
                ('sign_agreement', "Gradient Sign Agreement")
            ]

            for i, (col, title) in enumerate(metrics):
                ax = axes[i]
                
                sns.lineplot(
                    data=df, x="layer_idx", y=col, hue="component", style="component",
                    markers=True, dashes=False, linewidth=2.5, markersize=9, palette="tab10",
                    errorbar=None, ax=ax, legend=(i == 0)
                )

                ax.set_title(title, pad=15, fontsize=18)
                ax.set_xlabel("Layer Index", fontsize=14)
                ax.set_ylabel(col.replace("_", " ").title(), fontsize=14)
                
                # X-Axis Ticks
                ax.set_xticks(range(int(df['layer_idx'].min()), int(df['layer_idx'].max()) + 1))
                ax.set_xlim(df['layer_idx'].min() - 0.5, df['layer_idx'].max() + 0.5)
                ax.tick_params(axis='x', labelsize=10)

            # Adjust Legend on the first subplot
            if axes[0].get_legend():
                axes[0].legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined plot: {save_path}")
            plt.close()

    def run_analysis(self, pkl_path: str, sample_n=None, batch_size=20, create_plots=True):
        """Run comprehensive gradient analysis."""

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        if sample_n:
            data = data[:sample_n]

        # Load models
        model_a = self.load_and_merge_model(self.ckpt_a)
        model_b = self.load_and_merge_model(self.ckpt_b)

        # Analysis loop
        results = []
        for idx, item in enumerate(tqdm(data, desc="Analyzing")):
            prompt = item.get('prompt')

            grads_a, loss_a = self._compute_gradients(model_a, prompt)
            grads_b, loss_b = self._compute_gradients(model_b, prompt)

            layer_metrics = self._compare_gradients(
                grads_a, grads_b, store_for_batch=True, sample_id=idx)

            # Layer-wise details
            for layer, m in layer_metrics.items():
                results.append({"sample_id": idx, "layer": layer, **m})

        # Save results
        df = pd.DataFrame(results)

        output_csv = os.path.join(self.output_dir, "layer_metrics.csv")
        df.to_csv(output_csv, index=False)

        print(f"Results saved to {output_csv}")

        # Create plots
        if create_plots:
            df[['layer_idx', 'component']] = df['layer'].str.extract(r'layer_(\d+)_(.+)')
            df['layer_idx'] = df['layer_idx'].astype(int) + 1
            df_plot = df[~df['component'].str.contains('other')].copy()
            df_plot = df_plot.sort_values(by=['layer_idx', 'component'])
            self.plot_combined_metrics(df_plot)

        del model_a, model_b
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Divergence Analysis')
    parser.add_argument('--checkpoint_a', type=str, required=True, help='Path to model A checkpoint')
    parser.add_argument('--checkpoint_b', type=str, required=True, help='Path to model B checkpoint')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name for this experiment')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data pickle file')
    parser.add_argument('--sample_n', type=int, default=20, help='Number of samples to analyze')
    parser.add_argument('--create_plots', action='store_true', default=True, help='Create plots after analysis')
    
    args = parser.parse_args()

    analyzer = ModelDivergenceAnalyzer(
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        tokenizer_path=args.tokenizer_path,
        experiment_name=args.experiment_name
    )
    
    analyzer.run_analysis(
        args.data_path,
        sample_n=args.sample_n,
        create_plots=args.create_plots
    )
