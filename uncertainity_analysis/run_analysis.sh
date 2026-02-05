#!/bin/bash

# Model paths
Qwen7B_NFT="Qwen/Qwen2.5-7B-Instruct"
Qwen8B_NFT="Qwen/Qwen3-8B"
Gemma7B_NFT="google/gemma-7b-it"

Qwen7B_TSFT="/data/chenghe6/SFT_kldiv/Diagnosis/training_results/ProbSum/Qwen-2.5-7B-Instruct/kl_0.01/checkpoint-488"
Qwen8B_TSFT="/data/chenghe6/SFT_kldiv/Diagnosis/training_results/ProbSum/Qwen3-8B/kl_0.01/checkpoint-478"
Gemma7B_TSFT="/data/chenghe6/SFT_kldiv/Diagnosis/training_results/ProbSum/gemma-7b-it/kl_0.01/checkpoint-472"

Qwen7B_MULTI_TASK_SFT_MODEL="../Probsum_DDXplus/models/Qwen7B_SFT_multi_task/"
Qwen8B_MULTI_TASK_SFT_MODEL="../Probsum_DDXplus/models/Qwen8B_SFT_multi_task/"
Gemma7B_MULTI_TASK_SFT_MODEL="../Probsum_DDXplus/models/Gemma7B_SFT_multi_task/"

Qwen7B_MULTI_TASK_GRPO_MODEL="../Probsum_DDXplus/models/GRPO_final/Qwen7B_GRPO_multi_task/"
Qwen8B_MULTI_TASK_GRPO_MODEL="../Probsum_DDXplus/models/GRPO_final/Qwen8B_GRPO_multi_task/"
Gemma7B_MULTI_TASK_GRPO_MODEL="../Probsum_DDXplus/models/GRPO/Gemma7B_GRPO_multi_task/checkpoint-50"

Qwen7B_MULTI_TASK_RMR1_MODEL="../RM-R1-distillation/probsum_ddxplus/models/grpo/Qwen7B_GRPO_multi_task/"
Qwen8B_MULTI_TASK_RMR1_MODEL="../RM-R1-distillation/probsum_ddxplus/models/grpo/Qwen8B_GRPO_multi_task/"
Gemma7B_MULTI_TASK_RMR1_MODEL="../RM-R1-distillation/probsum_ddxplus/models/grpo_final/Gemma7B_GRPO_multi_task/checkpoint-20/"

Qwen7B_TOKENIZER="Qwen/Qwen2.5-7B-Instruct"
Qwen8B_TOKENIZER="Qwen/Qwen3-8B"
Gemma7B_TOKENIZER="google/gemma-7b-it"

Qwen7B_BASE_OUTPUT_DIR="gradient_and_wasserstein/multi_task_qwen7b"
Qwen8B_BASE_OUTPUT_DIR="gradient_and_wasserstein/multi_task_qwen8b"
Gemma7B_BASE_OUTPUT_DIR="gradient_and_wasserstein/multi_task_gemma7b"

DATA_PATH="/data/chenghe6/data/SFT_data/ProbSum/test_data.pkl"
SAMPLE_SIZE=100

GRADIENT_SCRIPT="pairwise_gradient_checking.py"
WASSERSTEIN_SCRIPT="pairwise_wasserstein_distance.py"

export CUDA_VISIBLE_DEVICES=0


run_wasserstein_comparison() {
    local model_name=$1
    local base_model=$2
    local exp_name=$3
    local base_output_dir=$4
    
    # Create output directory to verify it's unique
    local output_dir="${base_output_dir}/${exp_name}.csv"
    mkdir -p "${base_output_dir}"
    echo ""
    echo "=========================================="
    echo "Running: ${exp_name}"
    echo "Model: ${model_name}"
    echo "Model Base: ${base_model}"
    echo "Output: ${output_dir}"
    echo "=========================================="
    
    python ${WASSERSTEIN_SCRIPT} \
        --model_name "${model_name}" \
        --base_model "${base_model}" \
        --output_name "${output_dir}" \
        --data_file "${DATA_PATH}"
    
    if [ $? -eq 0 ]; then
        echo "✓ ${exp_name} completed successfully"
        # echo "  Files saved to: ${output_dir}/"
        # ls -lh "${output_dir}/"
    else
        echo "✗ ${exp_name} failed"
        exit 1
    fi
}


echo ""
echo "=========================================="
echo "GROUP Wasserstein Distance Qwen7B"
echo "=========================================="

run_wasserstein_comparison "${Qwen7B_NFT}" "${Qwen7B_TSFT}" "nft_vs_tsft" "${Qwen7B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Qwen7B_NFT}" "${Qwen7B_MULTI_TASK_SFT_MODEL}" "nft_vs_sft" "${Qwen7B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Qwen7B_NFT}" "${Qwen7B_MULTI_TASK_GRPO_MODEL}" "nft_vs_grpo" "${Qwen7B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Qwen7B_NFT}" "${Qwen7B_MULTI_TASK_RMR1_MODEL}" "nft_vs_rmr1" "${Qwen7B_BASE_OUTPUT_DIR}"


echo ""
echo "=========================================="
echo "GROUP Wasserstein Distance Qwen8B"
echo "=========================================="

run_wasserstein_comparison "${Qwen8B_NFT}" "${Qwen8B_TSFT}" "nft_vs_tsft_prime" "${Qwen8B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Qwen8B_NFT}" "${Qwen8B_MULTI_TASK_SFT_MODEL}" "nft_vs_sft" "${Qwen8B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Qwen8B_NFT}" "${Qwen8B_MULTI_TASK_GRPO_MODEL}" "nft_vs_grpo" "${Qwen8B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Qwen8B_NFT}" "${Qwen8B_MULTI_TASK_RMR1_MODEL}" "nft_vs_rmr1" "${Qwen8B_BASE_OUTPUT_DIR}"


echo ""
echo "=========================================="
echo "GROUP Wasserstein Distance Gemma7B"
echo "=========================================="

run_wasserstein_comparison "${Gemma7B_NFT}" "${Gemma7B_TSFT}" "nft_vs_tsft" "${Gemma7B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Gemma7B_NFT}" "${Gemma7B_MULTI_TASK_SFT_MODEL}" "nft_vs_sft" "${Gemma7B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Gemma7B_NFT}" "${Gemma7B_MULTI_TASK_GRPO_MODEL}" "nft_vs_grpo" "${Gemma7B_BASE_OUTPUT_DIR}"
run_wasserstein_comparison "${Gemma7B_NFT}" "${Gemma7B_MULTI_TASK_RMR1_MODEL}" "nft_vs_rmr1" "${Gemma7B_BASE_OUTPUT_DIR}"

