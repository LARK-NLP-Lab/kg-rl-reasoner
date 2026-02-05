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
from torch.utils.data import Dataset, DataLoader
import math
from collections import Counter
import ast
import gc
# import util
import evaluate
import random
from transformers import BartForConditionalGeneration
import pandas as pd
from tqdm.notebook import tqdm
import itertools
import os
from peft import get_peft_model, PeftModel, PeftConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel
import argparse
import torch.nn.functional as F
from quickumls import *
from sklearn.metrics import f1_score
from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))
torch.set_default_device("cuda")

def minibatch_rand_projections(batchsize, dim, num_projections=1000):
    projections = torch.randn((batchsize, num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=2, keepdim=True))
    return projections

def compute_practical_moments_sw(x, y, num_projections=1000, device="cuda", degree=2.0):
    dim = x.size(2)
    batch_size = x.size(0)
    projections = minibatch_rand_projections(batch_size, dim, num_projections).to(device)
    # print("projections", projections.shape)
    xproj = x.bmm(projections.transpose(1,2)).to(device)
    yproj = y.bmm(projections.transpose(1, 2)).to(device)

    _sort = (torch.sort(xproj.transpose(1,2))[0] - torch.sort(yproj.transpose(1,2))[0]).to(device)

    _sort_pow_p_get_sum = torch.sum(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = _sort_pow_p_get_sum.mean(dim=1)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment


def main():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="../final_models/Qwen7B_SFT_d1", help='model name')
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='base model name')
    parser.add_argument('--output_name', type=str, default="./probsum_eval/Qwen7B_SFT_d1.csv", help='model output file name')
    parser.add_argument('--data_file', type=str, help='Data file')
    parser.add_argument('--num_samples', type=int, help='number of samples')
    args = parser.parse_args()


    model_name = args.model_name
    base_model = args.base_model
    output_file = args.output_name
    data_file = args.data_file
    num_samples = args.num_samples

    with open(data_file, "rb") as f:
        dataset = pickle.load(f)

    if args.num_samples:
        dataset = dataset[:args.num_samples]
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config).to("cuda:0")
    base_model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=nf4_config).to("cuda:0")

    if os.path.exists(model_name):
        model = PeftModel.from_pretrained(model, model_name)
    
    overall_sw = []

    model.eval()
    base_model.eval()

    with torch.no_grad():
        for d in tqdm(dataset):
            model_inputs = tokenizer(d['prompt'], return_tensors="pt")

            model_output = model(**model_inputs, output_hidden_states=True)
            base_model_output = base_model(**model_inputs, output_hidden_states=True)

            model_hidden_states = torch.cat(model_output.hidden_states)
            base_model_hidden_states = torch.cat(base_model_output.hidden_states)

            sw1_model_base, _ = compute_practical_moments_sw(model_hidden_states.to(torch.float32), base_model_hidden_states.to(torch.float32), num_projections=1000, device=model.device, degree=2.0)

            temp = {}

            for l in range(len(model_output.hidden_states)):
                temp[f'layer_{l + 1}'] = sw1_model_base[l].item()
            
            overall_sw.append(temp)

    overall_sw_df = pd.DataFrame(overall_sw)
    overall_sw_df.to_csv(args.output_name)

    del model
    del base_model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()