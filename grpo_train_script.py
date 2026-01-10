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
from transformers import BitsAndBytesConfig, TrainerCallback
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType, PeftModel
import argparse
from peft import get_peft_model, PeftModel, PeftConfig
from transformers.trainer_utils import get_last_checkpoint
import time
from dotenv import load_dotenv
from peft.utils import prepare_model_for_kbit_training
from quickumls import *

load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

cui_to_entity_path = "/data/yifu/data/entity_linking/cui_to_entity.pkl" 
with open(cui_to_entity_path, 'rb') as handle:
    cui_to_entity = pickle.load(handle)

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="./models/Qwen7B_SFT_d1/", help='model name')
    parser.add_argument('--output_dir', type=str, default='./models/GRPO/Qwen7B_GRPO_d1', help='output directory')
    parser.add_argument('--input_dataset', type=str, default='./grpo_data/grpo_train_d1_selected_all_paths.pkl', help='dataset file')
    parser.add_argument('--val_dataset', type=str, default='./grpo_data/grpo_train_val_d1_selected_all_paths.pkl', help='val dataset file')
    parser.add_argument('--learning_rate', type=float, default=1e-6, help='learning rate')
    parser.add_argument("--task", type=str, default="d1")
    parser.add_argument('--is_prime', type=float, default=0, help='is it primeKG training')
    parser.add_argument('--is_resume', type=int, default=0, help='should we resume')
    args = parser.parse_args()
    return args

def formatting_prompts(textData, is_prime):
    return {"prompt": f"{textData['prompt']}", "prime": str(is_prime), "answer": f"{textData['answer']}", "label_path": textData['label_path'] if 'label_path' in textData else textData['answer'], "source": textData['source'], "task": textData["task"], "prediction_type": textData['prediction_type'] if 'prediction_type' in textData else None, "wrongOptions": textData['wrong_options'] if 'wrong_options' in textData else None}

def get_quantized_models(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model_nf4 = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config, trust_remote_code=True)

    model_nf4.config.use_cache = False

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
            target_modules=["q_proj", "v_proj", "k_proj", 'o_proj'],
            bias="none",
        )

        model = get_peft_model(model, peft_config)
    return model

def print_trainable_params(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

def pattern_reward_fn(**kwargs):
    pattern = r'<Answer>.*</Answer>'
    matches = [re.findall(pattern, c) for c in kwargs['completions']]
    
    rewards = [1.0 if len(match) else 0.0 for match in matches]

    return rewards

def reward_fn_quickUMLS(extracted_ans, answer, matcher):
    pattern = r'<Answer>.*</Answer>'

    def find_entity(entity):
        entity_findings = []

        for k in cui_to_entity:
            for t in cui_to_entity[k]:
                if entity == t.capitalize():
                    entity_findings.append(k)

        return entity_findings
    
    if answer[0] == "|":
        extracted_cui = matcher.match(extracted_ans[1:] if len(extracted_ans) >= 1 and extracted_ans[0] == "|" else extracted_ans, )
        gold_cui = matcher.match(answer[1:], )

        if len(extracted_cui):
            extracted_cui = extracted_cui[0][0]['cui']
        else:
            extracted_cui = find_entity(extracted_ans[1:] if len(extracted_ans) >= 1 and extracted_ans[0] == "|" else extracted_ans)

            if len(extracted_cui):
                extracted_cui = extracted_cui[0]
            else:
                return 0
        
        if len(gold_cui):
            gold_cui = gold_cui[0][0]['cui']
        else:
            gold_cui = find_entity(answer[1:])[0]
        return extracted_cui == gold_cui
    else:
        return extracted_ans == answer


def reward_func(is_prime=False, **kwargs):

    rewards = []
    pattern = r'<Answer>.*</Answer>'
    
    accepted_semtypes=['T033','T040', 'T063', 'T037', 'T060', 'T055', 'T017', 'T069', 'T122', 'T038', 'T044', 'T130', 'T057', 'T073', 'T081', 'T185', 'T098', 'T101', 'T075', 'T184', 'T021', 'T200', 'T047', 'T022', 'T066', 'T068', 'T201', 'T089', 'T097', 'T023', 'T062', 'T070', 'T093', 'T190', 'T102', 'T001', 'T114', 'T041', 'T054', 'T049', 'T120', 'T169', 'T028', 'T131', 'T045', 'T026', 'T129', 'T092', 'T056', 'T065', 'T196', 'T123', 'T031', 'T064', 'T121', 'T058', 'T034', 'T039', 'T074', 'T018', 'T019', 'T059', 'T043', 'T104', 'T197', 'T077', 'T061', 'T099', 'T029', 'T030', 'T095', 'T020', 'T082', 'T042', 'T086', 'T032', 'T091', 'T083', 'T109', 'T046', 'T078', 'T072', 'T067', 'T090', 'T116']
    matcher = QuickUMLS("../quickumls/",overlapping_criteria="score",similarity_name="cosine",threshold=0.90,accepted_semtypes=accepted_semtypes)

    for completion, answer, task, wrongOptions, ans in zip(kwargs['completions'], kwargs['label_path'], kwargs['task'], kwargs['wrongOptions'], kwargs['answer']):

        extracted_ans = re.findall(pattern, completion)

        if len(extracted_ans) != 0:
            extracted_ans = extracted_ans[0]
        else:
            rewards.append(0)
            continue

        if task == "Multi_Path_Selection": # Covers the case of PN@10
            curr_reward = 1
            if isinstance(answer, list):
                for p in answer:
                    if p not in extracted_ans:
                        curr_reward = 0
                        break
                
                for p in wrongOptions:
                    if p in extracted_ans:
                        curr_reward = 0
                        break
        elif task == "Path_Selection": # Covers cases of P@10 and P@2
            curr_reward = 1
            # print("The extracted", extracted_ans)
            # print("The ans", ans)            
            if answer not in extracted_ans:
                curr_reward = 0

            if isinstance(wrongOptions, list):
                for p in wrongOptions:
                    if p in extracted_ans:
                        curr_reward = 0
                        break
            else:
                if wrongOptions in extracted_ans:
                    curr_reward = 0

        elif task == "NHP": # Covers NHP case
            curr_reward = reward_fn_quickUMLS(extracted_ans, ans, matcher)

        else: # PC case left
            curr_reward = 1
            # print("The extracted", extracted_ans)
            # print("The ans", ans)
            if ans not in extracted_ans:
                curr_reward = 0
    
        rewards.append(curr_reward)
        
    # if sum(rewards) > 0:
    #     print("some rewards +1 !")
    return rewards
    


class RewardEarlyStoppingCallback(TrainerCallback):
    def __init__(self, metric_key="eval_reward", patience=3, greater_is_better=True):
        super().__init__()
        self.metric_key = metric_key
        self.patience = patience
        self.best = -math.inf if greater_is_better else math.inf
        self.greater_is_better = True
        self.bad_count = 0
    
    def _get_metric(self, state, metrics):
        for entry in reversed(state.log_history):
            if self.metric_key in entry:
                current = entry[self.metric_key]
                return current
        return None

    def on_evaluate(self, args, state, control, **kwargs):
        current = None

        for entry in reversed(state.log_history):
            if self.metric_key in entry:
                current = entry[self.metric_key]
                break
        
        if current is None:
            return control
    
        improved = (current > self.best) if self.greater_is_better else (current < self.best)

        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                control.should_training_stop = True
        
        # print("The number of bad evals", self.num_bad)
        return control

def main():
    start_time = time.time()

    args = parse_args()
    model_name = args.model_name
    output_folder = args.output_dir
    input_dataset = args.input_dataset
    val_dataset = args.val_dataset
    learning_rate = args.learning_rate
    is_prime = args.is_prime == 1
    resume = args.is_resume
    # print("is_prime", is_prime)
    os.makedirs(output_folder, exist_ok=True)

    is_pretrained = False

    if os.path.exists(model_name):
        model_name = get_last_checkpoint(model_name)
        print("the checkpoint fetched is", model_name)
        is_pretrained = True

    tokenizer, model = get_quantized_models(model_name)
    # model = prepare_model_for_kbit_training(model)
    # model.get_input_embeddings().weight.requires_grad_(False)
    model = peft_model(model, model_name, is_pretrained)

    print_trainable_params(model)
    with open(input_dataset, 'rb') as f: 
        prompt_data_train = pickle.load(f)

    with open(val_dataset, 'rb') as f: 
        prompt_data_val = pickle.load(f)

    datasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x, is_prime), prompt_data_train)))
    evalDatasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x, is_prime), prompt_data_val[:250])))

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = GRPOConfig(
        output_dir=f"./{output_folder}", 
        per_device_train_batch_size=6,
        learning_rate=learning_rate, 
        num_train_epochs=1, 
        per_device_eval_batch_size=6,
        beta=0.001,
        logging_dir= f'./{output_folder}/logs',
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",

        logging_steps=125,
        dataloader_pin_memory=True,
        save_total_limit=3,
        gradient_accumulation_steps=12,
        num_generations=6,

        bf16=True
    )

    trainer = GRPOTrainer(
        model,
        args=sft_config,
        train_dataset=datasetFromList,
        eval_dataset=evalDatasetFromList,
        reward_funcs=[pattern_reward_fn, reward_func],
        callbacks=[RewardEarlyStoppingCallback(patience=2)]
    )

    if resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.model.save_pretrained(output_folder)
    tokenizer.save_pretrained(output_folder)

    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best metric:", trainer.state.best_metric)

    df = pd.DataFrame(trainer.state.log_history)

    df.to_csv(f'./{output_folder}/scores.csv', index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


if __name__ == '__main__':
    main()