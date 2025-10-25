from COTTrainer import COTTrainer, COTDataCollator, compute_metrics_text
from COTDataset import COTDataset
import spacy
import yaml
import time
import json
import pickle
from collections import defaultdict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline, BitsAndBytesConfig, AutoModelForSeq2SeqLM, Qwen2_5_VLForConditionalGeneration, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
# from sentence_transformers import SentenceTransformer
import re
import numpy as np
import torch
from datasets import load_dataset, Dataset
import math
from collections import Counter
import ast
import gc
# import util
import evaluate
import random
from transformers import BartForConditionalGeneration, TrainingArguments, Seq2SeqTrainingArguments
import pandas as pd
from tqdm.notebook import tqdm
import itertools
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import jsonpickle
from peft import get_peft_model, PeftModel, PeftConfig, LoraConfig, TaskType
from transformers.trainer_utils import get_last_checkpoint
from transformers import pipeline
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import record
from accelerate import Accelerator
from dotenv import load_dotenv


load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))
# torch.set_default_device("cuda")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='model name')
    parser.add_argument('--output_dir', type=str, default='./models/Qwen7B_SFT_d1', help='output directory')
    parser.add_argument('--input_dataset', type=str, default='./data/train_reasoning_data_with_answer.pkl', help='dataset file')
    parser.add_argument('--val_dataset', type=str, default='./data/train_reasoning_data_val_with_answer.pkl', help='val dataset file')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--resume_training', type=int, default=0, help='if we should resume the training from checkpoint')
    parser.add_argument('--beta', type=float, default=0.1, help='beta')
    parser.add_argument('--cot_distill', type=int, default=0, help='if cotDistill')
    args = parser.parse_args()
    return args

def extract_answer_path(answerString, pattern):
    return answerString.split(f"<{pattern}>")[1].split(f"</{pattern}>")[0]

def formatting_prompts(textData):
    return {"prompt": f"{textData['prompt']}", "reasoning": f"{extract_answer_path(textData['prompt'], 'Reasoning')}", "path": f"{textData['label_path']}"}

def get_quantized_models(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_nf4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config, device_map='auto')
    #{'':Accelerator().process_index}
    return tokenizer, model_nf4

def peft_model(model):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", 'gate_proj'],
        # bias="none",
    )

    model = get_peft_model(model, peft_config)

    return model

def peft_model_resume(model, model_name):
    model = PeftModel.from_pretrained(model=model, model_id=model_name, is_trainable=True)

    return model

@record
def main():
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    args = parse_args()
    model_name = args.model_name
    output_folder = args.output_dir
    input_dataset = args.input_dataset
    val_dataset = args.val_dataset
    learning_rate = args.learning_rate
    resume_training = args.resume_training
    beta = args.beta
    cot_distill = args.cot_distill

    os.makedirs(output_folder, exist_ok=True)

    if resume_training:
        model_name = get_last_checkpoint(output_folder)

    tokenizer, model = get_quantized_models(model_name)

    if resume_training:
        model = peft_model_resume(model, model_name)
    else:
        model = peft_model(model)

    # with open(input_dataset, 'rb') as f: 
    #     prompt_data_train = pickle.load(f)

    # with open(val_dataset, 'rb') as f: 
    #     prompt_data_val = pickle.load(f)

    # print("lengths", len(prompt_data_train), len(prompt_data_val))
    # datasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x), prompt_data_train)))
    # evalDatasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x), prompt_data_val)))
    # tokenizer.pad_token_id = -100
    # model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("tokenizer pad_token_id", tokenizer.pad_token_id)

    trainDataSet = COTDataset(input_dataset, tokenizer, {
        'promptReasoning': 'reasoningLabel',
        'promptClassification': 'label_path'
    })
    evalDataSet = COTDataset(val_dataset, tokenizer, {
        'promptReasoning': 'reasoningLabel',
        'promptClassification': 'label_path'
    }, 10)

    # trainDataLoader = DataLoader(trainDataSet, shuffle=True)
    # evalDataLoader = DataLoader(evalDataSet, shuffle=True)

    # print(len(trainDataSet))
    # model = DDP(model, device_ids=[0, 1])

    sft_config = Seq2SeqTrainingArguments(
        output_dir=f"./{output_folder}", 
        per_device_train_batch_size=1, 
        # packing=True, 
        learning_rate=learning_rate, 
        num_train_epochs=2, 
        per_device_eval_batch_size=1,
        logging_dir= f'./{output_folder}/logs',
        eval_strategy="steps",
        eval_steps=250,
        dataloader_pin_memory=False,
        save_total_limit=3,
        predict_with_generate=True,
        prediction_loss_only=False,
        remove_unused_columns=False,
        # bf16='store_true',
        fp16=True,
        generation_max_length=6000
        # label_names=["completion"]
        # metric_for_best_model="eval_loss"
    )

    # model.generation_config.max_length = 3072

    trainer = COTTrainer(
        alpha_list=[0.5, 0.5],
        task_list=['promptReasoning', 'promptClassification'],
        beta=beta,
        cot_distill=bool(cot_distill),
        model=model,
        args=sft_config,
        train_dataset=trainDataSet,
        eval_dataset=evalDataSet,
        data_collator=COTDataCollator(tokenizer=tokenizer, model=model, padding='max_length', max_length=5000),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_text(tokenizer)
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    if resume_training:
        trainer.train(resume_from_checkpoint=get_last_checkpoint(output_folder))
    else:
        trainer.train()

    df = pd.DataFrame(trainer.state.log_history)

    df.to_csv(f'./{output_folder}/scores.csv', index=False)

if __name__ == '__main__':
    main()
    # mp.spawn(main, nprocs=2, join=True)
    