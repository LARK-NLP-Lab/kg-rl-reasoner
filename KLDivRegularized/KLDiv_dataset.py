from torch.utils.data import DataLoader, Dataset, random_split
import pickle
from transformers import AutoTokenizer

class KLDivDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset
        self.tokenizer =  tokenizer

    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, index):
        current_item = self.raw_dataset[index]
        prompt = current_item['prompt']
        answer = current_item['completion']

        encoded_answer = self.tokenizer(prompt + answer)
        encoded_prompt = self.tokenizer(prompt)
        prompt_len = len(encoded_prompt['input_ids'])

        labels = encoded_answer['input_ids'][:]

        labels[:prompt_len] = [-100] * prompt_len

        encoded_answer['labels'] = labels

        return encoded_answer

    