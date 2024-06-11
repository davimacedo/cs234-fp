from transformers import GPT2Tokenizer
import torch
import argparse
from tqdm import tqdm
from settings import device
from models import get_model_name
from rewards_model import RewardsModel

def main(args):
    data = torch.load("datasets/extracted_anthropic_hh.pth")
    input_texts = data["input_texts"]
    chosen_output_texts = data["chosen_output_texts"]
    rejected_output_texts = data["rejected_output_texts"]
    
    total = len(input_texts)
    print(f"loaded {total} samples")

    model_name = get_model_name(args)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = RewardsModel(model_name)
    model.load_state_dict(torch.load("parameters/trained-model.pth"))
    model.to(device)
    model.eval()

    evaluated = 0
    correct = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, total, args.batch)):
            input_texts_batch = input_texts[i:i + args.batch]
            chosen_output_texts_batch = chosen_output_texts[i:i + args.batch]
            rejected_output_texts_batch = rejected_output_texts[i:i + args.batch]
            
            chosen_input_texts_batch = []
            rejected_input_texts_batch = []

            for j in range(len(input_texts_batch)):
                chosen_input_texts_batch.append(input_texts_batch[j] + chosen_output_texts_batch[j])
                rejected_input_texts_batch.append(input_texts_batch[j] + rejected_output_texts_batch[j])
            
            chosen_tokenized_input_batch = tokenizer(
                chosen_input_texts_batch,
                max_length=1024,
                truncation="only_first",
                padding=True,
                return_tensors="pt"
            )
            chosen_tokenized_input_batch.to(device)

            chosen_output_batch = model(**chosen_tokenized_input_batch)

            rejected_tokenized_input_batch = tokenizer(
                rejected_input_texts_batch,
                max_length=1024,
                truncation="only_first",
                padding=True,
                return_tensors="pt"
            )
            rejected_tokenized_input_batch.to(device)

            rejected_output_batch = model(**rejected_tokenized_input_batch)

            batch_correct = torch.sum((chosen_output_batch > rejected_output_batch).long()).item()
            correct += batch_correct
            batch_evaluated = len(chosen_output_batch)
            evaluated += batch_evaluated

            tqdm.write(f"batch {i} accuracy = {batch_correct / batch_evaluated}; accumulated accuracy {correct / evaluated};")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--original", action="store_true", help="if true, use original model")
    parser.add_argument("-l", "--large", action="store_true", help="if true, use gpt2-xl, else, use gpt2")
    parser.add_argument("-b", "--batch", type=int, default=16, help="the batch size")

    args = parser.parse_args()

    main(args)
