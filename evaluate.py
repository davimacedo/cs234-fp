from transformers import GPT2Tokenizer
import torch
import argparse
from tqdm import tqdm
from settings import device
from utils import select, calculate_log_probs
from models import get_model_name, get_model

def main(args):
    data = torch.load("datasets/extracted_anthropic_hh.pth")
    input_texts = data["input_texts"]
    chosen_output_texts = data["chosen_output_texts"]
    rejected_output_texts = data["rejected_output_texts"]
    input_token_ids = data["input_token_ids"] if "input_token_ids" in data else None
    chosen_output_token_ids = data["chosen_output_token_ids"] if "chosen_output_token_ids" in data else None
    rejected_output_token_ids = data["rejected_output_token_ids"] if "rejected_output_token_ids" in data else None

    total = len(input_texts)
    print(f"loaded {total} samples")

    model_name = get_model_name(args)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = get_model(model_name, load_state_dict=not args.original)    
    model.eval()

    evaluated = 0
    correct = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, total, args.batch)):
            input_token_ids_batch = None

            if input_token_ids is not None:
                input_token_ids_batch = input_token_ids[i:i + args.batch]
            else:
                input_texts_batch = input_texts[i:i + args.batch]
                input_token_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in input_texts_batch]
            
            min_input_length = min(len(token_ids) for token_ids in input_token_ids_batch)
            
            chosen_log_probs, chosen_indexes = calculate_log_probs(
                model,
                tokenizer,
                input_token_ids_batch,
                min_input_length,
                chosen_output_texts[i:i + args.batch],
                chosen_output_token_ids[i:i + args.batch] if chosen_output_token_ids is not None else None
            )

            rejected_log_probs, rejected_indexes = calculate_log_probs(
                model,
                tokenizer,
                select(input_token_ids_batch, chosen_indexes),
                min_input_length,
                select(rejected_output_texts[i:i + args.batch], chosen_indexes),
                select(rejected_output_token_ids[i:i + args.batch], chosen_indexes) if rejected_output_token_ids is not None else None
            )

            chosen_log_probs = select(chosen_log_probs, rejected_indexes)

            batch_correct = torch.sum((chosen_log_probs > rejected_log_probs).long()).item()
            correct += batch_correct
            batch_evaluated = len(chosen_log_probs)
            evaluated += batch_evaluated

            tqdm.write(f"batch {i} accuracy = {batch_correct / batch_evaluated}; accumulated accuracy {correct / evaluated};")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original", action="store_true", help="if true, use original model")
    parser.add_argument("-l", "--large", action="store_true", help="if true, use gpt2-xl, else, use gpt2")
    parser.add_argument("-b", "--batch", type=int, default=16, help="the batch size")

    args = parser.parse_args()

    main(args)
