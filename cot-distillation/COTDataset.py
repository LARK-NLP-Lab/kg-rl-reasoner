from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from transformers import AutoTokenizer

class COTDataset(Dataset):
    def __init__(self, file_name, tokenizer, taskLabelRec, num_elements=None) -> None:
        super().__init__()
        self.file_name = file_name
        self.tokenizer =  tokenizer
        self.taskLabelRec = taskLabelRec

        if num_elements == None:
            with open(self.file_name, 'rb') as f:
                files = pickle.load(f)
                
            self.num_elements = len(files)
        else:
            self.num_elements = num_elements
    
    def __len__(self):
        with open(self.file_name, 'rb') as f:
            files = pickle.load(f)

        return min(len(files), self.num_elements)

    def __getitem__(self, index):
        with open(self.file_name, 'rb') as f:
            data_rec = pickle.load(f)

        current_item = data_rec[index]
        ans = {}
        for key, value in self.taskLabelRec.items():
            # if 'Reasoning' or 'reasoning' in key:
            #     encoded_answer = self.tokenizer(current_item[key] + str(current_item[value]))
            # else:
            encoded_answer = self.tokenizer(current_item[key] + str(current_item[value]))

            encoded_input = self.tokenizer(current_item[key])
            promptLen = len(encoded_input['input_ids'])

            labels = encoded_answer['input_ids'][:]

            labels[:promptLen] = [-100] * promptLen

            encoded_answer['labels'] = labels

            ans[key] = encoded_answer
        
        return ans

        # encoded_answer_with_label_input = self.tokenizer(current_item['promptClassification'] + current_item['label_path'])
        # encoded_answer_with_label_reasoning_input = self.tokenizer(current_item['promptReasoning'] + "<think>" + current_item['reasoningLabel'] + "</think>")
        
        # encoded_answer_input = self.tokenizer(current_item['promptClassification'])
        # encoded_reasoning_input = self.tokenizer(current_item['promptReasoning'])

        # promptLenClassification = len(encoded_answer_input['input_ids'])
        # promptLenReasoning = len(encoded_reasoning_input['input_ids'])

        # # with self.tokenizer.as_target_tokenizer():
        # labels_encoded_answer_input = encoded_answer_with_label_input['input_ids'][:]
        # labels_encoded_reasoning_input = encoded_answer_with_label_reasoning_input['input_ids'][:]

        # labels_encoded_answer_input[:promptLenClassification] = [-100] * promptLenClassification
        # labels_encoded_reasoning_input[:promptLenReasoning] = [-100] * promptLenReasoning

        # encoded_answer_input['labels'] = labels_encoded_answer_input
        # encoded_reasoning_input['labels'] = labels_encoded_reasoning_input

        # return {
        #     **encoded_answer_input,
        #     "reasoning_input_ids": encoded_reasoning_input['input_ids'],
        #     "reasoning_attention_mask": encoded_reasoning_input['attention_mask'],
        #     "reasoning_labels": encoded_reasoning_input['labels'],
        # }

    

