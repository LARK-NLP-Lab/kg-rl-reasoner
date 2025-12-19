
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from torch.utils.data import DataLoader
# from sentence_transformers import SentenceTransformer
import re
import numpy as np
import torch
from datasets import load_dataset, Dataset
from collections import Counter
from transformers import Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
import pandas as pd
from tqdm.notebook import tqdm
import os
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
from trl import SFTConfig, SFTTrainer
import pickle
from torch.nn import KLDivLoss
from KLDiv_dataset import KLDivDataset


load_dotenv()

# torch.set_default_device('cuda')

login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))

local_rank = int(os.environ.get("LOCAL_RANK", 0))
print("local rank", local_rank)
torch.cuda.set_device(local_rank) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-7B-Instruct", help='model name')
    parser.add_argument('--output_dir', type=str, default='./models/Qwen7B_SFT_d1', help='output directory')
    parser.add_argument('--input_dataset', type=str, default='./data/train_reasoning_data_with_answer.pkl', help='dataset file')
    parser.add_argument('--val_dataset', type=str, default='./data/train_reasoning_data_val_with_answer.pkl', help='val dataset file')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--kl_lambda', type=float, default=0.05, help='beta')
    args = parser.parse_args()
    return args

def formatting_prompts(textData):
    return {"prompt": f"{textData['prompt']}", "completion": f"{textData['answer']}"}

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

class KLDivSFTTrainer(SFTTrainer):
    def __init__(self, lambda_kl=0.05, **kwargs):
        super().__init__(**kwargs)
        # self.base_model = base_model
        self.lambda_kl = lambda_kl
        self.kl_fn = KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sft_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        logits = outputs.logits

        # Compute reference logits by disabling the adapter
        with torch.no_grad(), model.module.disable_adapter():
            ref_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )
        ref_logits = ref_outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)
        base_probs = torch.softmax(ref_logits, dim=-1)

        kl_loss = self.kl_fn(log_probs, base_probs)

        loss = sft_loss + self.lambda_kl * kl_loss

        # print("sft_loss", sft_loss)
        # print("kl_loss", kl_loss)
        # print("output shape", outputs.logits.shape)

        return (loss, outputs) if return_outputs else loss



def main():
    args = parse_args()
    model_name = args.model_name
    output_folder = args.output_dir
    input_dataset = args.input_dataset
    val_dataset = args.val_dataset
    learning_rate = args.learning_rate
    kl_lambda = args.kl_lambda

    os.makedirs(output_folder, exist_ok=True)

    tokenizer, model = get_quantized_models(model_name)
    print("vocab_size", model.config.vocab_size)

    model = peft_model(model)

    with open(input_dataset, 'rb') as f: 
        prompt_data_train = pickle.load(f)

    with open(val_dataset, 'rb') as f: 
        prompt_data_val = pickle.load(f)    

    datasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x), prompt_data_train)))
    evalDatasetFromList = Dataset.from_list(list(map(lambda x: formatting_prompts(x), prompt_data_val)))

    sft_config = SFTConfig(
        output_dir=f"./{output_folder}", 
        per_device_train_batch_size=2, 
        packing=True, 
        learning_rate=learning_rate, 
        num_train_epochs=1, 
        per_device_eval_batch_size=2,
        logging_dir= f'./{output_folder}/logs',
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        dataloader_pin_memory=False,
        save_total_limit=1,
        # label_names=["completion"],
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True
    )

    trainer = KLDivSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=datasetFromList,
        eval_dataset=evalDatasetFromList,
        lambda_kl=kl_lambda,
        # compute_metrics=
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print("Best metric:", trainer.state.best_metric)

    df = pd.DataFrame(trainer.state.log_history)

    df.to_csv(f'./{output_folder}/scores.csv', index=False)

if __name__ == '__main__':
    main()