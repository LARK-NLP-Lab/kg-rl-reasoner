import spacy
import yaml
import time
import json
import pickle
from collections import defaultdict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline, EarlyStoppingCallback
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
import re
import numpy as np
import torch
import math
from collections import Counter
import ast
import gc
import evaluate
import random
import pandas as pd
from tqdm.notebook import tqdm
import itertools
import os
from collections import deque, defaultdict
from transformers import BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel
import argparse
from peft import get_peft_model, PeftModel, PeftConfig
from transformers.trainer_utils import get_last_checkpoint


def get_quantized_models(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_nf4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config)
    return tokenizer, model_nf4

def peft_model(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", 'gate_proj'],
        bias="none",
    )

    model = get_peft_model(model, peft_config)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', type=str, default="./Qwen1.5B/")
    parser.add_argument('--model_weights', nargs='+', type=float, default=0.5)
    parser.add_argument('--output_path', type=str, default="merged_models/")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    model_weights = args.model_weights
    models = []
    tokenizers = []

    for m in args.model_names:
        model = m
        if os.path.exists(model) and "merged_models" not in model and "DogeRM" not in model:
            model = get_last_checkpoint(m)
            tempTokenizer, tempModel = get_quantized_models(model)
        else:
            tempTokenizer, tempModel = get_quantized_models(model)
            tempModel = peft_model(tempModel)

        models.append(tempModel)
        tokenizers.append(tempTokenizer)

        # print("model", model)
        # print(tempModel)
    
    for i in range(len(models)):
        for (name_seq1, param_seq1), (name_seq, param_seq) in zip(models[0].named_parameters(), models[i].named_parameters()):
            if 'embed' in name_seq or 'lm_head' in name_seq:
                continue
                
            if i == 0:
                weighted_sum_param = model_weights[i] * param_seq.data
            else:
                print("the weighted params", param_seq1.data.shape, param_seq.data.shape)
                weighted_sum_param = param_seq1.data + model_weights[i] * param_seq.data

            param_seq1.data.copy_(weighted_sum_param)

    models[0].save_pretrained(args.output_path)
    tokenizers[0].save_pretrained(args.output_path)

if __name__ == '__main__':
    main()