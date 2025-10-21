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
from peft import get_peft_model
from transformers.trainer_utils import get_last_checkpoint
from dotenv import load_dotenv


load_dotenv()

login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))
# Qwen/Qwen2.5-7B-Instruct

# torch.set_default_device("cuda:0")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-8B", help='model name')
    parser.add_argument('--output_dir', type=str, default='./models/Qwen8B_SFT', help='output directory')
    parser.add_argument('--input_dataset', type=str, default='./medQA_train.pkl', help='dataset file')
    parser.add_argument('--val_dataset', type=str, default='./medQA_test.pkl', help='val dataset file')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--eval_steps', type=int, default=500, help='steps when to run evaluation')
    args = parser.parse_args()
    return args

def formatting_prompts(textData):
    # print(textData)
    return {"prompt": f"{textData['prompt']}", "completion": f"{textData['answer']}"}


# def formatting_prompts(textData):
#     converted_sample = [
#             {"role": "user", "content": textData["prompt"].split("\nOutput:")[0]},
#             {"role": "assistant", "content": textData["answer"]},
#         ]

#     return {'messages': converted_sample}


def get_quantized_models(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_nf4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config, device_map='auto')

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
    args = parse_args()
    model_name = args.model_name
    output_folder = args.output_dir
    input_dataset = args.input_dataset
    val_dataset = args.val_dataset
    learning_rate = args.learning_rate
    eval_steps = args.eval_steps

    os.makedirs(output_folder, exist_ok=True)

    tokenizer, model = get_quantized_models(model_name)

    model = peft_model(model)

    with open(input_dataset, 'rb') as f: 
        prompt_data_train = pickle.load(f)

    with open(val_dataset, 'rb') as f: 
        prompt_data_val = pickle.load(f)

    print("lengths", len(prompt_data_train), len(prompt_data_val))
    datasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x), prompt_data_train)))
    evalDatasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x), prompt_data_val)))

    sft_config = SFTConfig(
        output_dir=f"./{output_folder}", 
        per_device_train_batch_size=1, 
        packing=True, 
        learning_rate=learning_rate, 
        num_train_epochs=2, 
        per_device_eval_batch_size=1,
        logging_dir= f'./{output_folder}/logs',
        eval_strategy="steps",
        eval_steps=eval_steps,
        dataloader_pin_memory=False,
        save_total_limit=3,
        label_names=["completion"]
        # metric_for_best_model="eval_loss"
    )

    trainer = SFTTrainer(
        model,
        args=sft_config,
        train_dataset=datasetFromList,
        eval_dataset=evalDatasetFromList
    )

    trainer.train()

    df = pd.DataFrame(trainer.state.log_history)

    df.to_csv(f'./{output_folder}/scores.csv', index=False)

if __name__ == '__main__':
    main()