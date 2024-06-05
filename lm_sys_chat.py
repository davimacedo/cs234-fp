from datasets import load_dataset
import argparse
from settings import TRANSFORMERS_ACCESS_TOKEN, HUMAN_PREFIX, HUMAN_PREFIX_LEN, ASSISTANT_PREFIX, ASSISTANT_PREFIX_LEN
from tqdm import tqdm
from sentiment import predict_sentiments
from utils import select
import torch

def main(args):
    dataset = load_dataset("lmsys/lmsys-chat-1m", token=TRANSFORMERS_ACCESS_TOKEN)
    train_dataset = dataset["train"]

    input_texts = []
    output_texts = []
    next_texts = []

    print("parsing dataset...")

    for row in tqdm(train_dataset):
        if row["language"] == "English" and row["turn"] > 1:
            conversation = row["conversation"]

            input_text = None
            output_text = None

            for conversation_turn in conversation:
                role = conversation_turn["role"]
                content = conversation_turn["content"]

                if role == "user":
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

        indexes = torch.nonzero(sentiments).squeeze(-1)

        final_next_texts += select(next_texts_batch, indexes)
        rewards += sentiments[indexes].tolist()

        indexes += i

        final_input_texts += select(input_texts, indexes)
        final_output_texts += select(output_texts, indexes)
        
    extracted = {
        "input_texts": final_input_texts,
        "output_texts": final_output_texts,
        "next_texts": final_next_texts,
        "rewards": rewards
    }

    if args.tokenize:
        extracted["input_token_ids"] = []
        extracted["output_token_ids"] = []

        model_name = "gpt2" if args.small else "gpt2-xl"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        print("tokenizing texts...")

        texts = list(zip(final_input_texts, final_output_texts))

        for i in tqdm(range(len(texts))):
            input_text, output_text = texts[i]
            extracted["input_token_ids"].append(tokenizer.encode(input_text, add_special_tokens=False))
            extracted["output_token_ids"].append(tokenizer.encode(output_text, add_special_tokens=False))

    torch.save(
        extracted,
        "datasets/extracted_lm_sys_chat.pth"
    )

    print(f"extracted {len(final_input_texts)} samples from {len(train_dataset)} original samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", action="store_true", help="if true, use gpt2, else, use gpt2-xl")
    parser.add_argument("-t", "--tokenize", action="store_true", help="tokenize texts")
    parser.add_argument("-b", "--batch", type=int, default=512, help="the batch size")

    args = parser.parse_args()

    main(args)
