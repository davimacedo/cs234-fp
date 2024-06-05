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
        optimizer.zero_grad()
        
        input_token_ids_batch = None

        if input_token_ids is not None:
            input_token_ids_batch = input_token_ids[i:i + args.batch]
        else:
            input_texts_batch = input_texts[i:i + args.batch]
            input_token_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in input_texts_batch]

        output_texts_batch = output_texts[i:i + args.batch]
        output_token_ids_batch = output_token_ids[i:i + args.batch] if output_token_ids is not None else None
        next_texts_batch = next_texts[i:i + args.batch]
        rewards_batch = rewards[i:i + args.batch] if rewards is not None else None

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
                    rewards_computation_batch.to(device)
                elif isinstance(rewards_computation_batch, list):
                    rewards_computation_batch = torch.tensor(rewards_computation_batch, device=device)
                
                loss = -torch.sum(torch.exp(log_probs) * rewards_computation_batch)
                accumulated += loss.item()
                loss.backward()

        if computated > 0:
            optimizer.step()

            tqdm.write(f"batch loss = {accumulated / computated}")

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

    args = parser.parse_args()

    main(args)
