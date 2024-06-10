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
