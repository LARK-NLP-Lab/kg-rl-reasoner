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
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="./models/Qwen8B_SFT_d1/", help='model name')
    parser.add_argument('--output_name', type=str, default="Qwen8B_SFT_d1", help='model output file name')
    args = parser.parse_args()
    return args


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

def peft_model(model, model_name):
    model = PeftModel.from_pretrained(model=model, model_id=model_name)

    return model


def get_prediction_labels(prediction, label):
    accepted_semtypes=['T033', 'T037', 'T046', 'T047', 'T048', 'T049', 'T184']
    matcher = QuickUMLS("../quickumls/",overlapping_criteria="score",similarity_name="cosine",threshold=0.90,accepted_semtypes=accepted_semtypes)

    output = [(ii['ngram'],ii['term'],ii['cui'], ii['semtypes']) for i in matcher.match(prediction,) for ii in i]
    output_label = [(ii['ngram'],ii['term'],ii['cui'], ii['semtypes']) for i in matcher.match(label,) for ii in i]

    output_cuis = set()
    output_label_cuis = set()

    for m in output:
        output_cuis.add(m[2])

    for m in output_label:
        output_label_cuis.add(m[2])

    return output_cuis, output_label_cuis

def main():
    args = parse_args()
    output_file_name = args.output_name
    test_df="../BioNLP2023-1A-Test.csv"
    data = pd.read_csv(test_df)
    sb=data["Subjectives"].tolist()
    ob=data["Objectives"].tolist()
    ass=data["Assessment"].tolist()
    sm=data["Summary"].tolist()
    f_id=data["File ID"].tolist()
    rouge = evaluate.load('rouge')
    note = []
    for i in range(len(sm)):
        if type(sb[i]) == str and type(ob[i]) == str and type(ass[i]) == str and type(sm[i]) == str:
            note.append((sb[i]+"\n"+ob[i]+'\n'+ass[i],sm[i], f_id[i], i))
            
    model_name = args.model_name
    
    if os.path.exists(model_name) and "merged_models" not in model_name and "DogeRM" not in model_name:
        model_name = get_last_checkpoint(model_name)

    tokenizer, model = get_quantized_models(model_name)
    # model = peft_model(model, model_name)

    predictions = []
    actual = []

    whole_rec = []

    accs = []
    prec_list= []
    recall_list = []
    # num_invalid
    print("length", len(note))
    for n in tqdm(note[:]):
        PROMPT_CONS = '''You are a medical assistance tool provided with the PROGRESS_NOTES of a patient.
You must review the PROGRESS_NOTES and return a list of possible diagnoses for the patient.
The format of your list must be:

DIAGNOSES: [List of diagnosis separated by ';']

for eg. your output can be something like DIAGNOSES: diagnosis1 ; diagnosis 2; 

Do NOT repeat any item in the DIAGNOSES.

'''
        prompt = PROMPT_CONS + f"\n\nPROGRESS_NOTES: {n[0]}" 
        # + PROMPT_JSON_FORMAT
        prompt += "\n\nREMINDER: Return ONLY a list of DIAGNOSES in SINGLE LINE. No commentary."
        prompt += "\nDIAGNOSES:"
        model_inputs = tokenizer(prompt, return_tensors="pt")
        generated_ids = model.generate(**model_inputs, max_new_tokens=30)
        # print("shape", model_inputs['input_ids'].shape, generated_ids.shape)
        output = tokenizer.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        
        predictions.append(output)
        # print("======prompt======")
        # print(prompt)
        # print("======Answer======")
        # print(n[1])
        # print("======The prediction======")
        # print(predictions[-1])
        actual.append(n[1])

        whole_rec.append({
            "prompt": prompt,
            "label": n[1],
            "prediction": predictions[-1]
        })

        notes_cui, gold_cui = get_prediction_labels(predictions[-1], n[1])

        if len(notes_cui) == 0 or len(gold_cui) == 0:
            # num_invalid += 1
            continue
        
        prec = len(gold_cui.intersection(notes_cui)) / len(notes_cui) 
        rec = len(gold_cui.intersection(notes_cui)) / len(gold_cui) 

        if prec + rec == 0:
            accs.append(0)
            prec_list.append(0)
            recall_list.append(0)
        else:
            acc = 2*(prec*rec) / (prec+rec)

            accs.append(acc)
            prec_list.append(prec)
            recall_list.append(rec)

    print("The Rouge Score\n", rouge.compute(predictions=predictions, references=actual))
    print("Mean np", np.mean(accs))
    print("Precision np", np.mean(prec_list))
    print("Recall np", np.mean(recall_list))


    # print(f"saving file {output_file_name}")

    # with open(f"./errorAnalysis/diagnosis_prediction/First10/{output_file_name}.json", "w") as json_file:
    #     json.dump(whole_rec, json_file, indent=4)
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == '__main__':
    main()