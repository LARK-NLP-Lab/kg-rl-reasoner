import spacy
import yaml
import time
import json
import pickle
from collections import defaultdict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
# from sentence_transformers import SentenceTransformer
import re
import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
import math
from collections import Counter
import ast
import gc
# import util
import evaluate
import random
import pandas as pd
from tqdm.notebook import tqdm
import itertools
import os
from collections import deque, defaultdict
from transformers import BitsAndBytesConfig
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, PeftModel
import argparse
from peft import get_peft_model, PeftModel, PeftConfig
from transformers.trainer_utils import get_last_checkpoint
import time
from dotenv import load_dotenv


load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

# Qwen/Qwen2.5-7B-Instruct

# torch.set_default_device("cuda:0")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="./models/Qwen7B_SFT_d1/", help='model name')
    parser.add_argument('--output_dir', type=str, default='./models/GRPO/Qwen7B_GRPO_d1', help='output directory')
    parser.add_argument('--input_dataset', type=str, default='./grpo_data/grpo_train_d1_selected_all_paths.pkl', help='dataset file')
    parser.add_argument('--val_dataset', type=str, default='./grpo_data/grpo_train_val_d1_selected_all_paths.pkl', help='val dataset file')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate')
    args = parser.parse_args()
    return args

def formatting_prompts(textData):
    return {"prompt": f"{textData['prompt']}", "completion": f"{textData['answer']}", "answer": f"{textData['answer']}", "label_path": f"{textData['label_path'] if 'label_path' in textData else textData['answer']}"}

def get_quantized_models(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_nf4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config, device_map="auto")
    # model_nf4 = AutoModelForCausalLM.from_pretrained(model_name) 

    return tokenizer, model_nf4

def peft_model(model, model_name, is_pretrained):
    if is_pretrained:
        model = PeftModel.from_pretrained(model=model, model_id=model_name, is_trainable=True)
    else:
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
    

def reward_func(**kwargs):
    # print(kwargs.keys()) 
    if 'reasoningLabel' in kwargs.keys():
        return [1.0 if f"<CorrectPath>{gt}</CorrectPath>" in c else 0.0 for c, gt in zip(kwargs['completions'], kwargs['label_path'])]
    else:
        return [1.0 if c == gt else 0.0 for c, gt in zip(kwargs['completions'], kwargs['answer'])]

def main():
    start_time = time.time()

    args = parse_args()
    model_name = args.model_name
    output_folder = args.output_dir
    input_dataset = args.input_dataset
    val_dataset = args.val_dataset
    learning_rate = args.learning_rate
    os.makedirs(output_folder, exist_ok=True)

    is_pretrained = False

    if os.path.exists(model_name):
        model_name = get_last_checkpoint(model_name)
        print("the checkpoint fetched is", model_name)
        is_pretrained = True

    tokenizer, model = get_quantized_models(model_name)

    model = peft_model(model, model_name, is_pretrained)

    with open(input_dataset, 'rb') as f: 
        prompt_data_train = pickle.load(f)

    with open(val_dataset, 'rb') as f: 
        prompt_data_val = pickle.load(f)

    datasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x), prompt_data_train)))
    evalDatasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x), prompt_data_val)))

    # print("datasetFromList", len(prompt_data_train))

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = GRPOConfig(
        output_dir=f"./{output_folder}", 
        # per_device_train_batch_size=1, 
        learning_rate=learning_rate, 
        num_train_epochs=1, 
        # per_device_eval_batch_size=1,
        logging_dir= f'./{output_folder}/logs',
        eval_strategy="steps",
        eval_steps=10000,
        dataloader_pin_memory=False,
        save_total_limit=1
        # packing=True
    )

    trainer = GRPOTrainer(
        model,
        args=sft_config,
        train_dataset=datasetFromList,
        eval_dataset=evalDatasetFromList,
        # processing_class=tokenizer,
        reward_funcs=reward_func
    )

    trainer.train()

    df = pd.DataFrame(trainer.state.log_history)

    df.to_csv(f'./{output_folder}/scores.csv', index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


if __name__ == '__main__':
    main()