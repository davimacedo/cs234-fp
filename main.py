from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import argparse
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_log_probs(model, tokenizer, input_token_ids_batch, min_input_length, output_texts_batch):
    output_token_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in output_texts_batch]

    batch_token_ids = []

    for input_token_ids, output_token_ids in zip(input_token_ids_batch, output_token_ids_batch):
        token_ids = input_token_ids + output_token_ids + [tokenizer.eos_token_id]
        if len(token_ids) <= 1024:
            batch_token_ids.append(token_ids)

    max_length = max(len(token_ids) for token_ids in batch_token_ids)
    padded_batch_token_ids = torch.tensor([token_ids + [tokenizer.eos_token_id] * (max_length - len(token_ids)) for token_ids in batch_token_ids], dtype=torch.long, device=device)
    
    attention_mask = torch.zeros(padded_batch_token_ids.shape)
    
    for i, attention_mask_row in enumerate(attention_mask):
        attention_mask_row[:len(batch_token_ids[i])] = 1

    logits = model(padded_batch_token_ids, attention_mask=attention_mask).logits[:, min_input_length:]
    probs = torch.softmax(logits, dim = -1)
    attention_mask = attention_mask[:, min_input_length:]
    padded_batch_token_ids = padded_batch_token_ids[:, min_input_length:]

    log_probs = torch.sum(torch.log(torch.gather(probs, dim=-1, index=padded_batch_token_ids.unsqueeze(-1)).squeeze(-1) * attention_mask), dim=-1)

def main(args):
    data = torch.load("datasets/extracted_anthropic_hh.pth")
    input_texts = data["input_texts"]
    chosen_output_texts = data["chosen_output_texts"]
    rejected_output_texts = data["rejected_output_texts"]

    total = len(input_texts)
    print(f"loaded {total} samples")

    model_name = "gpt2" if args.small else "gpt2-xl"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)  
    model.to(device)
    model.eval()

    evaluated = 0
    correct = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, total, args.batch)):
            input_texts_batch = input_texts[i:i + args.batch]
            input_token_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in input_texts_batch]
            min_input_length = min(len(token_ids) for token_ids in input_token_ids_batch)
            
            chosen_log_probs = calculate_log_probs(
                model,
                tokenizer,
                input_token_ids_batch,
                min_input_length,
                chosen_output_texts[i:i + args.batch]
            )

            rejected_log_probs = calculate_log_probs(
                model,
                tokenizer,
                input_token_ids_batch,
                min_input_length,
                rejected_output_texts[i:i + args.batch]
            )

            correct += torch.sum((chosen_log_probs > rejected_log_probs).long()).item()
            evaluated += len(input_texts_batch)
            print(f"accuracy = {correct / evaluated}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", action="store_true", help="if true, use gpt2, else, use gpt2-xl")
    parser.add_argument("-b", "--batch", type=int, default=512, help="the batch size")

    args = parser.parse_args()

    main(args)
