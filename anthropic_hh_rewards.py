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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", type=int, default=256, help="the batch size")

    args = parser.parse_args()

    main(args)
