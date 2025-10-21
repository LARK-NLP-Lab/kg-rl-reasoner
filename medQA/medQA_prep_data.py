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
from datasets import load_dataset
from dotenv import load_dotenv


load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

# Positive (diagnosis intent)
diag_patterns = [
    r"\bwhat(?:'?s| is) the diagnosis\b",
    r"\bmost likely diagnosis\b",
    r"\bbest(?:\s+|[-])diagnosis\b",
    r"\bmost appropriate diagnosis\b",
    r"\bwhich of the following is the (?:most )?likely diagnosis\b",
    r"\bunderlying diagnosis\b",
    r"\bprimary diagnosis\b",
    r"\bdx[: ]",                        # e.g., "Dx: ..."
    # Etiology/cause ⇒ often treated as diagnostic identification on MedQA
    r"\bmost likely cause\b",
    r"\bmost likely etiology\b",
    r"\bmost likely pathogen\b",
    r"\bcausative (?:agent|organism)\b",
    r"\bcausative pathogen\b",
    r"\bmost likely organism\b",
    r"\bmost likely (?:virus|bacteria|bacterium|parasite|fungus)\b",
    r"\bmost likely (?:condition|disease)\b",
]

diag_re = re.compile("|".join(diag_patterns), flags=re.IGNORECASE)

dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

prompt = '''You are a medical assistance tool. Please review the QUESTION.
You are provided with 4 OPTIONS, you must pick the option that will answer question correct.
'''

train_data = []

for i in tqdm(dataset['train']):
    temp = {}
    match = diag_re.search(i['question'])

    if match:
        
        temp['question'] = i['question']
        temp['answer_idx'] = i['answer_idx']
        temp['answer'] = i['answer_idx'] + "." + f" {i['answer']}" 
        temp['prompt'] = prompt + f"\n{i['question']}\n"
        temp['prompt'] += f"OPTIONS:"

        for k in i['options']:
            temp['prompt'] += f"\n{k}. {i['options'][k]}"
        
        temp['prompt'] += "\nPlease pick the correct option.\nAnswer:"

        train_data.append(temp)

with open("medQA_train.pkl", "wb") as f:
    pickle.dump(train_data, f)

print("train")
print(train_data[0])
test_data = []

for i in tqdm(dataset['test']):
    temp = {}
    match = diag_re.search(i['question'])

    if match:
        temp['question'] = i['question']
        temp['answer_idx'] = i['answer_idx']
        temp['answer'] = i['answer_idx'] + "." + f" {i['answer']}" 
        temp['prompt'] = prompt + f"\n{i['question']}\n"
        temp['prompt'] += f"OPTIONS:"

        for k in i['options']:
            temp['prompt'] += f"\n{k}. {i['options'][k]}"
        
        temp['prompt'] += "\nPlease pick the correct option.\nAnswer:"
        
        test_data.append(temp)
print("test")
print(test_data[0])

with open("medQA_test.pkl", "wb") as f:
    pickle.dump(test_data, f)