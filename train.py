from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import argparse
from tqdm import tqdm
from sentiment import predict_sentiments
from settings import device
from utils import select, calculate_log_probs

def main(args):
    data = torch.load("datasets/extracted_lm_sys_chat.pth")
    input_texts = data["input_texts"]
    output_texts = data["output_texts"]
    next_texts = data["next_texts"]
    input_token_ids = data["input_token_ids"] if "input_token_ids" in data else None
    output_token_ids = data["output_token_ids"] if "output_token_ids" in data else None
    rewards = torch.tensor(data["rewards"]) if "rewards" in data else None

    total = len(input_texts)
    print(f"loaded {total} samples")

    model_name = "gpt2" if args.small else "gpt2-xl"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)  
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for i in tqdm(range(0, total, args.batch)):
        input_token_ids_batch = None

        if input_token_ids is not None:
            input_token_ids_batch = input_token_ids[i:i + args.batch]
        else:
            input_texts_batch = input_texts[i:i + args.batch]
            input_token_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in input_texts_batch]
    
        min_input_length = min(len(token_ids) for token_ids in input_token_ids_batch)

        log_probs, indexes = calculate_log_probs(
            model,
            tokenizer,
            input_token_ids_batch,
            min_input_length,
            output_texts[i:i + args.batch],
            output_token_ids[i:i + args.batch] if output_token_ids is not None else None
        )
        
        if len(log_probs) > 0:
            max_log_probs = torch.max(log_probs)
            log_probs = log_probs - max_log_probs
            log_probs = log_probs / (-max_log_probs)
            probs = torch.exp(log_probs)

            rewards_batch = None

            if rewards is not None:
                rewards_batch = select(rewards[i:i + args.batch], indexes)
            else:
                next_texts_batch = select(next_texts[i:i + args.batch], indexes)
                rewards_batch = predict_sentiments(next_texts_batch)
            
            if isinstance(rewards_batch, torch.Tensor):
                rewards_batch = rewards_batch.to(device)
            elif isinstance(rewards_batch, list):
                rewards_batch = torch.tensor(rewards_batch, device=device)
            
            loss = -torch.mean(probs * rewards_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm.write(f"batch loss = {loss.item()}")

    torch.save(
        model.state_dict(),
        "parameters/trained-model.pth"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", action="store_true", help="if true, use gpt2, else, use gpt2-xl")
    parser.add_argument("-b", "--batch", type=int, default=2, help="the batch size")
    parser.add_argument("-l", "--lr", type=float, default=1e-6, help="the optimizer learning rate")

    args = parser.parse_args()

    main(args)
