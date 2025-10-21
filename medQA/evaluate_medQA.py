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
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
from peft import LoraConfig, TaskType, PeftModel
import argparse
from peft import get_peft_model, PeftModel, PeftConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers import pipeline
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

torch.set_default_device("cuda")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', type=str, default="./Qwen1.5B/", help='model name')
    parser.add_argument('--data_file', type=str, default="medQA_test.pkl", help='test data')
    args = parser.parse_args()
    return args

def get_multiple_choice_answer(model, tokenizer, data_file):
    
    PROMPT_CONS = '''You are a medical assistance tool provided with medical QUESTION.
You are provided with 4 OPTIONS. You must pick the option that will answer question correct.

The OPTIONS are provided as :
A. OPTION 1
B. OPTION 2
C. OPTION 3
D. OPTION 4

Your answer needs to be a single word with your pick for the right answer.

For eg. if the options are the following:

A. OPTION 1
B. OPTION 2
C. OPTION 3
D. OPTION 4

If you think that "OPTION 2" leads to a correct diagnosis according to the provided patient records, then your output must be: "B".
Your Answer CANNOT be anything other than "A" or "B" or "C" or "D".
'''

    rouge = evaluate.load('rouge')
    answers = []
    num_correct_picks = 0
    # flag = 1
    for data in tqdm(data_file):
        prompt = PROMPT_CONS + f"\nQUESTION: {data['question']}"
        answerOption = data['answer_idx']
        prompt += "\nOptions:"
        prompt += data['prompt'].split("OPTIONS:")[1]
        
        with torch.no_grad():
            model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=1)
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred = output.split("\nAnswer:")[1]
            answers.append({
                'actual': answerOption,
                'prediction': pred
            })
            
            # m = rouge.compute(predictions=[pred], references=[data['label_path'] if 'label_path' in data.keys() else data['answer']])['rougeL']
            # flag = 0
            # for w in data['wrong_options']:
            #     if rouge.compute(predictions=[pred], references=[w])['rougeL'] > m:
            #         flag = 1
            #         break
            
            # if flag == 0:
            #     num_correct_picks += 1
            # print("all options", '\n'.join(all_answers))
            # print("prompt", prompt)
            # print("prediction", pred)
            # print("answerOption", answerOption)
            if answerOption.capitalize() == pred.strip().capitalize():
                num_correct_picks += 1
            
    return num_correct_picks / len(data_file)

def main():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    args = parse_args()
    model_names = args.model_names
    data_file = args.data_file

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    # with open(data_file, "rb") as f:
    #         data = pickle.load(f)
    
    for model in tqdm(model_names):
        if os.path.exists(model) and "merged_models" not in model:
            model = get_last_checkpoint(model)
            print("checkpoint", model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model, quantization_config=nf4_config).to("cuda")

        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        print("results", get_multiple_choice_answer(model, tokenizer, data))

        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
if __name__ == '__main__':
    main()