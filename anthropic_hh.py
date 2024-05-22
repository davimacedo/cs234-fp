from datasets import load_dataset
import torch
import argparse
from transformers import GPT2Tokenizer

HUMAN_PREFIX = "\n\nHuman: "
HUMAN_PREFIX_LEN = len(HUMAN_PREFIX)
ASSISTANT_PREFIX = "\n\nAssistant: "
ASSISTANT_PREFIX_LEN = len(ASSISTANT_PREFIX)

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
    
    for row in train_dataset:
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

                input_texts.append(input_text)
                chosen_output_texts.append(chosen_message["content"])
                rejected_output_texts.append(rejected_message["content"])

                input_text += chosen_message["content"]
        
    torch.save(
        {
            "input_texts": input_texts,
            "chosen_output_texts": chosen_output_texts,
            "rejected_output_texts": rejected_output_texts
        },
        "datasets/extracted_anthropic_hh.pth"
    )

    if args.tokenize:
        model_name = "gpt2" if args.small else "gpt2-xl"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        torch.save(
            {
                "input_encodings": [tokenizer.encode(text) for text in input_texts],
                "chosen_output_encodings": [tokenizer.encode(text) for text in chosen_output_texts],
                "rejected_output_encodings": [tokenizer.encode(text) for text in rejected_output_texts],
            },
            "datasets/tokenized_anthropic_hh.pth"
        )

    print(f"saved {len(input_texts)} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", action="store_true", help="if true, use gpt2, else, use gpt2-xl")
    parser.add_argument("-t", "--tokenize", action="store_true", help="tokenize texts")

    args = parser.parse_args()

    main(args)
