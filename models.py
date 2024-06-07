from transformers import GPT2LMHeadModel
from settings import device
import argparse
import torch
from peft import get_peft_model, LoraConfig

def get_model_name(args):
    return "gpt2" if args.small else "gpt2-xl"

def get_model(model_name, load_state_dict=False):
    model = GPT2LMHeadModel.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "attn.c_attn",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
            "lm_head"
        ],
    )

    model = get_peft_model(model, lora_config)

    if load_state_dict:
        model.load_state_dict(torch.load("parameters/trained-model.pth"))

    model.to(device)

    return model

def get_parameters(model):
    return model.parameters()

def main(args):
    model_name = get_model_name(args)
    model = get_model(model_name)

    print("model:")

    print(model)

    print("parameters:")

    parameters_names = sorted([name for name, _ in model.named_parameters()])

    for name in parameters_names:
        print(name)

    print("lm_head parameters:")

    lm_head_parameters_names = sorted([name for name, _ in model.lm_head.named_parameters()])

    for name in lm_head_parameters_names:
        print(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", action="store_true", help="if true, use gpt2, else, use gpt2-xl")
    
    args = parser.parse_args()

    main(args)
