import torch
from settings import device
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, input_texts, output_texts, next_texts, input_token_ids=None, output_token_ids=None, rewards=None):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.next_texts = next_texts
        self.input_token_ids = input_token_ids
        self.output_token_ids = output_token_ids
        self.rewards = rewards

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        sample = {
            "input_text": self.input_texts[idx],
            "output_text": self.output_texts[idx],
            "next_text": self.next_texts[idx],
        }
        
        if self.input_token_ids is not None:
            sample["input_token_id"] = self.input_token_ids[idx]
        if self.output_token_ids is not None:
            sample["output_token_id"] = self.output_token_ids[idx]
        if self.rewards is not None:
            sample["reward"] = self.rewards[idx]
        
        return sample

def collate_fn(batch):
    return batch

def select(array, indexes):
    if isinstance(array, torch.Tensor):
        return array[indexes]
    else:
        return [item for index, item in enumerate(array) if index in indexes]

def calculate_log_probs(model, tokenizer, input_token_ids_batch, min_input_length, output_texts_batch, output_token_ids_batch):
    if output_token_ids_batch is None:
        output_token_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in output_texts_batch]

    batch_token_ids = []
    indexes = []
    input_token_ids_lengths = []

    for i, (input_token_ids, output_token_ids) in enumerate(zip(input_token_ids_batch, output_token_ids_batch)):
        token_ids = input_token_ids + output_token_ids + [tokenizer.eos_token_id]
        if len(token_ids) <= 1024:
            batch_token_ids.append(token_ids)
            indexes.append(i)
            input_token_ids_lengths.append(len(input_token_ids))
    
    if len(batch_token_ids) == 0:
        return batch_token_ids, indexes
        
    max_length = max(len(token_ids) for token_ids in batch_token_ids)
    padded_batch_token_ids = torch.tensor([token_ids + [tokenizer.eos_token_id] * (max_length - len(token_ids)) for token_ids in batch_token_ids], dtype=torch.long, device=device)
    
    attention_mask = torch.zeros(padded_batch_token_ids.shape, device=device)
    probs_mask = torch.zeros(padded_batch_token_ids.shape, device=device)
    
    for i, attention_mask_row in enumerate(attention_mask):
        attention_mask_row[:len(batch_token_ids[i])] = 1
        probs_mask[i][input_token_ids_lengths[i]:len(batch_token_ids[i])] = 1

    logits = model(padded_batch_token_ids, attention_mask=attention_mask).logits[:, min_input_length:]
    probs = torch.softmax(logits, dim=-1)
    probs_mask = probs_mask[:, min_input_length:]
    padded_batch_token_ids = padded_batch_token_ids[:, min_input_length:]

    return torch.sum(torch.log(torch.gather(probs, dim=-1, index=padded_batch_token_ids.unsqueeze(-1)).squeeze(-1)) * probs_mask, dim=-1), indexes
