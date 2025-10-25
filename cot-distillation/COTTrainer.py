import spacy
import yaml
import time
import json
import pickle
from collections import defaultdict
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline, EarlyStoppingCallback, Trainer, Seq2SeqTrainer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu 
import re
import numpy as np
import torch
import torch.nn as nn
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
from transformers import BitsAndBytesConfig, DataCollatorForSeq2Seq, DataCollator, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType, PeftModel
import argparse
from peft import get_peft_model
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, Dict, Tuple, List
from dotenv import load_dotenv


load_dotenv()


login(token=os.getenv("HUGGING_FACE_LOGIN_TOKEN"))
 
# Reasoning
# Pick the right path
# Diagnosis
# class COTDataCollator(DataCollatorForSeq2Seq):
#     def __call__(self, features, return_tensors=None):
        
#         features_df = pd.DataFrame(features)
#         classification_features = features_df.loc[:, ~features_df.columns.isin(['reasoning_labels', 'reasoning_input_ids', 'reasoning_attention_mask'])].to_dict('records')
#         reasoning_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
#             columns={'reasoning_labels': 'labels', 'reasoning_input_ids': 'input_ids', 'reasoning_attention_mask': 'attention_mask'}).to_dict('records')
#         # print("data collator shape before")
#         # print("The length", len(classification_features))
#         # print(torch.tensor(classification_features[0]['input_ids']).shape, torch.tensor(classification_features[0]['labels']).shape)
#         # print(torch.tensor(reasoning_features[0]['input_ids']).shape, torch.tensor(reasoning_features[0]['labels']).shape)

#         classification_features = super().__call__(classification_features, return_tensors)
#         reasoning_features = super().__call__(reasoning_features, return_tensors)

#         # print("data collator shape after")
#         # print(classification_features.keys(), classification_features['labels'].shape)
#         # print(torch.tensor(reasoning_features['labels']).shape)

#         return {
#             'classification': classification_features,
#             'reasoning': reasoning_features,
#         } 
class COTDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        
        features_df = pd.DataFrame(features)
        # print(features_df)
        # classification_features = features_df.loc[:, ~features_df.columns.isin(['reasoning_labels', 'reasoning_input_ids', 'reasoning_attention_mask'])].to_dict('records')
        # reasoning_features = features_df.loc[:, ~features_df.columns.isin(['labels', 'input_ids', 'attention_mask'])].rename(
        #     columns={'reasoning_labels': 'labels', 'reasoning_input_ids': 'input_ids', 'reasoning_attention_mask': 'attention_mask'}).to_dict('records')
        # # print("data collator shape before")
        # # print("The length", len(classification_features))
        # # print(torch.tensor(classification_features[0]['input_ids']).shape, torch.tensor(classification_features[0]['labels']).shape)
        # # print(torch.tensor(reasoning_features[0]['input_ids']).shape, torch.tensor(reasoning_features[0]['labels']).shape)

        # classification_features = super().__call__(classification_features, return_tensors)
        # reasoning_features = super().__call__(reasoning_features, return_tensors)

        # print("data collator shape after")
        # print(classification_features.keys(), classification_features['labels'].shape)
        # print(torch.tensor(reasoning_features['labels']).shape)
        ans = {}

        for k in features_df.keys():
            temp_features = features_df[k]
            ans[k] = super().__call__(temp_features, return_tensors)

        # return {
        #     'classification': classification_features,
        #     'reasoning': reasoning_features,
        # } 
        return ans


def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # predictions = torch.tensor(predictions).view(-1, torch.tensor(predictions).shape[-1])
        # labels = torch.tensor(labels).view(-1, torch.tensor(labels).shape[-1])
        
        print(torch.tensor(predictions).shape, torch.tensor(labels).shape)
        predictions = np.where(predictions[0] != -100, predictions[0], tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print("decoded", torch.tensor(decoded_preds).shape, torch.tensor(decoded_labels).shape)
        # print("decoded_preds", decoded_preds)
        # print("decoded_labels", decoded_labels)
        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics

class COTTrainer(Seq2SeqTrainer):
    def __init__(self, alpha_list, task_list, beta, cot_distill, **kwargs):
        super().__init__(**kwargs)
        self.alpha_list = alpha_list
        self.task_list = task_list
        self.beta = beta
        self.cot_distill = cot_distill
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = 0
        answers = {}
        # device = next(model.parameters()).device
        # print(kwargs)
        probs = []
        ce_loss = 0
        for t in range(len(self.task_list)):
            # task_inputs = {k: v.to(device) for k, v in inputs[self.task_list[t]].items()}
            # print("model device", device, "shape is", task_inputs['input_ids'].device)
            temp = model(**inputs[self.task_list[t]])
            answers[self.task_list[t]] = temp
            loss += self.alpha_list[t] * temp.loss

            logits = (temp.logits)

            if t == 'promptClassification':
                logits = logits.detach()
            
            logit_prob = logits.softmax(dim=-1)
            logit_prob_1 = torch.max(logit_prob, dim=-2)[0]

            probs.append(logit_prob_1)
        
        if self.cot_distill:
            Loss = nn.CrossEntropyLoss()
            for i in range(1, len(probs)):
                ce_loss += Loss(probs[i - 1], probs[i])
            
            loss += self.beta * ce_loss

        return (loss, answers) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        loss = 0
        outputs = []

        for t, al in zip(self.task_list, self.alpha_list):
            temp = super().prediction_step(model, inputs[t], prediction_loss_only=False, ignore_keys=ignore_keys)
            outputs.append(temp)
            loss += al * temp[0]
       
        return (
            loss,
            [t[1] for t in outputs],
            [t[2] for t in outputs],
        )