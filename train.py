from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import argparse
from tqdm import tqdm
from sentiment import predict_sentiments
from settings import device
from utils import select, calculate_log_probs
from torch.utils.data import Dataset, DataLoader

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

def main(args):
    data = torch.load("datasets/extracted_lm_sys_chat.pth")
    input_texts = data["input_texts"]
    output_texts = data["output_texts"]
    next_texts = data["next_texts"]
    input_token_ids = data["input_token_ids"] if "input_token_ids" in data else None
    output_token_ids = data["output_token_ids"] if "output_token_ids" in data else None
    rewards = torch.tensor(data["rewards"]) if "rewards" in data else None

    dataset = ChatDataset(input_texts, output_texts, next_texts, input_token_ids, output_token_ids, rewards)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    total = len(input_texts)
    print(f"loaded {total} samples")

    model_name = "gpt2" if args.small else "gpt2-xl"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)  
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"training epoch {epoch}")

        epoch_accumulated = 0
        total_batches = 0

        for batch in tqdm(dataloader):
            input_texts_batch = batch["input_text"]
            output_texts_batch = batch["output_text"]
            next_texts_batch = batch["next_text"]
            input_token_ids_batch = batch.get("input_token_id")
            output_token_ids_batch = batch.get("output_token_id")
            rewards_batch = batch.get("reward")

            optimizer.zero_grad()
            
            if input_token_ids_batch is None:
                input_token_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in input_texts_batch]

            computated = 0
            accumulated = 0
        
            for j in range(0, len(input_token_ids_batch), args.size):
                input_token_ids_computation_batch = input_token_ids_batch[j:j + args.size]

                min_input_length = min(len(token_ids) for token_ids in input_token_ids_computation_batch)

                log_probs, indexes = calculate_log_probs(
                    model,
                    tokenizer,
                    input_token_ids_computation_batch,
                    min_input_length,
                    output_texts_batch[j:j + args.size],
                    output_token_ids_batch[j:j + args.size] if output_token_ids_batch is not None else None
                )
                
                log_probs_length = len(log_probs)

                if log_probs_length > 0:
                    computated += log_probs_length

                    max_log_probs = torch.max(log_probs)
                    log_probs = (log_probs - max_log_probs) / (-max_log_probs)

                    rewards_computation_batch = None

                    if rewards_batch is not None:
                        rewards_computation_batch = select(rewards_batch[j:j + args.size], indexes)
                    else:
                        next_texts_computation_batch = select(next_texts_batch[j:j + args.size], indexes)
                        rewards_computation_batch = predict_sentiments(next_texts_computation_batch)
                    
                    if isinstance(rewards_computation_batch, torch.Tensor):
                        rewards_computation_batch = rewards_computation_batch.to(device)
                    elif isinstance(rewards_computation_batch, list):
                        rewards_computation_batch = torch.tensor(rewards_computation_batch, device=device)
                    
                    loss = -torch.sum(torch.exp(log_probs) * rewards_computation_batch)
                    accumulated += loss.item()
                    loss.backward()

            total_batches += 1

            if computated > 0:
                optimizer.step()

                batch_accumulated = accumulated / computated
                epoch_accumulated += batch_accumulated

                tqdm.write(f"batch loss = {batch_accumulated}")

        tqdm.write(f"epoch loss = {epoch_accumulated / total_batches}")

        torch.save(
            model.state_dict(),
            "parameters/trained-model.pth"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", action="store_true", help="if true, use gpt2, else, use gpt2-xl")
    parser.add_argument("-b", "--batch", type=int, default=32, help="the batch size")
    parser.add_argument("-l", "--lr", type=float, default=1e-6, help="the optimizer learning rate")
    parser.add_argument("-z", "--size", type=int, default=2, help="the size for computation")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="the number of epochs")

    args = parser.parse_args()

    main(args)
