from datasets import load_dataset
import torch
import argparse
from transformers import GPT2Tokenizer
from tqdm import tqdm
from settings import HUMAN_PREFIX, HUMAN_PREFIX_LEN, ASSISTANT_PREFIX, ASSISTANT_PREFIX_LEN
from models import get_model_name

def extract_messages(transcript):
    messages = []
    role = None

    while len(transcript) > 0:
        if role is None:
            if transcript.startswith(HUMAN_PREFIX):
                role = "human"
                transcript = transcript[HUMAN_PREFIX_LEN:]
            elif transcript.startswith(ASSISTANT_PREFIX):
                role = "assistant"
                transcript = transcript[ASSISTANT_PREFIX_LEN:]
            else:
                break
        else:
            message = {
                "role": role
            }
            messages.append(message)

            human_index = transcript.find(HUMAN_PREFIX)
            assistant_index = transcript.find(ASSISTANT_PREFIX)

            if human_index >=0 and (assistant_index < 0 or human_index < assistant_index):
                message["content"] = transcript[:human_index]
                role = "human"
                transcript = transcript[human_index + HUMAN_PREFIX_LEN:]
            elif assistant_index >=0:
                message["content"] = transcript[:assistant_index]
                role = "assistant"
                transcript = transcript[assistant_index + ASSISTANT_PREFIX_LEN:]
            else:
                message["content"] = transcript
                role = None
                break

    if role is not None:
        messages.append({
            "role": role,
            "content": ""
        })

    return messages

def main(args):
    dataset = load_dataset("Anthropic/hh-rlhf")
    train_dataset = dataset["train"]

    input_texts = []
    chosen_output_texts = []
    rejected_output_texts = []

    print("parsing dataset...")
    
    for row in tqdm(train_dataset):
        chosen_messages = extract_messages(row["chosen"])
        rejected_messages = extract_messages(row["rejected"])
        rejected_messages_length = len(rejected_messages)

        input_text = ""
    
        for i, chosen_message in enumerate(chosen_messages):
            if i >= rejected_messages_length:
                break

            rejected_message = rejected_messages[i]

            if chosen_message["role"] != rejected_message["role"]:
                break

            if chosen_message["role"] == "human":
                if chosen_message["content"] != rejected_message["content"]:
                    break

                input_text += HUMAN_PREFIX + chosen_message["content"]
            else:
                input_text += ASSISTANT_PREFIX

                if chosen_message["content"] != rejected_message["content"]:
                    input_texts.append(input_text)
                    chosen_output_texts.append(chosen_message["content"])
                    rejected_output_texts.append(rejected_message["content"])

                input_text += chosen_message["content"]

    extracted = {
        "input_texts": input_texts,
        "chosen_output_texts": chosen_output_texts,
        "rejected_output_texts": rejected_output_texts
    }

    if args.tokenize:
        extracted["input_token_ids"] = []
        extracted["chosen_output_token_ids"] = []
        extracted["rejected_output_token_ids"] = []

        model_name = get_model_name(args)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        print("tokenizing texts...")

        texts = list(zip(input_texts, chosen_output_texts, rejected_output_texts))

        for i in tqdm(range(len(texts))):
            input_text, chosen_output_text, rejected_output_text = texts[i]
            extracted["input_token_ids"].append(tokenizer.encode(input_text, add_special_tokens=False))
            extracted["chosen_output_token_ids"].append(tokenizer.encode(chosen_output_text, add_special_tokens=False))
            extracted["rejected_output_token_ids"].append(tokenizer.encode(rejected_output_text, add_special_tokens=False))

    torch.save(
        extracted,
        "datasets/extracted_anthropic_hh.pth"
    )

    print(f"extracted {len(input_texts)} samples from {len(train_dataset)} original samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--large", action="store_true", help="if true, use gpt2-xl, else, use gpt2")
    parser.add_argument("-t", "--tokenize", action="store_true", help="tokenize texts")

    args = parser.parse_args()

    main(args)
