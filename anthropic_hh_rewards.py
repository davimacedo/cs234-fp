from datasets import load_dataset
import torch
import argparse
from transformers import GPT2Tokenizer
from tqdm import tqdm
from settings import HUMAN_PREFIX, HUMAN_PREFIX_LEN, ASSISTANT_PREFIX, ASSISTANT_PREFIX_LEN
from models import get_model_name
from anthropic_hh import extract_messages
from sentiment import predict_sentiments
from utils import select
import random

def main(args):
    dataset = load_dataset("Anthropic/hh-rlhf")
    train_dataset = dataset["train"]

    if args.amount >= 0:
        train_dataset = train_dataset.select(range(args.amount))

    input_texts = []
    output_texts = []
    next_texts = []

    print("parsing dataset...")
    
    for row in tqdm(train_dataset):
        chosen_messages = extract_messages(row["chosen"])

        input_text = None
        output_text = None
    
        for chosen_message in chosen_messages:
            role = chosen_message["role"]
            content = chosen_message["content"]

            if role == "human":
                if input_text is not None and output_text is not None:
                    input_texts.append(input_text)
                    output_texts.append(output_text)
                    next_texts.append(content)
                
                if input_text is None:
                    input_text = ""
                elif output_text is not None:
                    input_text += output_text

                input_text += HUMAN_PREFIX + content
                output_text = None
            elif role == "assistant":
                if input_text is None:
                    input_text = ""
                elif output_text is not None:
                    input_text += output_text

                input_text += ASSISTANT_PREFIX
                output_text = content

    print("calculating rewards...")

    final_input_texts = []
    final_output_texts = []
    final_next_texts = []
    rewards = []

    for i in tqdm(range(0, len(next_texts), args.batch)):
        next_texts_batch = next_texts[i:i + args.batch]
        
        sentiments = predict_sentiments(next_texts_batch)

        if args.neutral > 0 and random.random() < args.neutral:
            print("here1")
            final_input_texts += input_texts[i:i + args.batch]
            final_output_texts += output_texts[i:i + args.batch]
            final_next_texts += next_texts_batch
            rewards += sentiments.tolist()
        else:
            print("here2")
            indexes = torch.nonzero(sentiments).squeeze(-1)
            
            final_input_texts += select(input_texts[i:i + args.batch], indexes)
            final_output_texts += select(output_texts[i:i + args.batch], indexes)
            final_next_texts += select(next_texts_batch, indexes)
            rewards += sentiments[indexes].tolist()

    extracted = {
        "input_texts": final_input_texts,
        "output_texts": final_output_texts,
        "next_texts": final_next_texts,
        "rewards": rewards
    }

    if args.tokenize:
        extracted["input_token_ids"] = []
        extracted["output_token_ids"] = []

        model_name = get_model_name(args)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        print("tokenizing texts...")

        texts = list(zip(final_input_texts, final_output_texts))

        for i in tqdm(range(len(texts))):
            input_text, output_text = texts[i]
            extracted["input_token_ids"].append(tokenizer.encode(input_text, add_special_tokens=False))
            extracted["output_token_ids"].append(tokenizer.encode(output_text, add_special_tokens=False))

    torch.save(
        extracted,
        "datasets/extracted_anthropic_hh_rewards.pth"
    )

    print(f"extracted {len(input_texts)} samples from {len(train_dataset)} original samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--large", action="store_true", help="if true, use gpt2-xl, else, use gpt2")
    parser.add_argument("-t", "--tokenize", action="store_true", help="tokenize texts")
    parser.add_argument("-b", "--batch", type=int, default=256, help="the batch size")
    parser.add_argument("-n", "--neutral", type=float, default=0, help="include neutral rewards with specified rate")
    parser.add_argument("-a", "--amount", type=int, default=-1, help="the amount of rows to extract")

    args = parser.parse_args()

    main(args)
