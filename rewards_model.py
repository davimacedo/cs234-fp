from transformers import GPT2Tokenizer, GPT2Model
import torch.nn as nn
from peft import get_peft_model, LoraConfig
import random
from models import get_model_name
from utils import ChatDataset, collate_fn
from sentiment import predict_sentiments
import sys
from settings import device
import argparse

class RewardsModel(nn.Module):
    def __init__(self, model_name):
        super(RewardsModel, self).__init__()

        self.gpt2 = get_peft_model(
            GPT2Model.from_pretrained(model_name),
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=[
                    "attn.c_attn",
                    "attn.c_proj",
                    "mlp.c_fc",
                    "mlp.c_proj"
                ],
            )
        )
        self.linear = nn.Linear(self.gpt2.config.n_embd, 1)
    
    def forward(self, input_ids, attention_mask):
        output = self.gpt2(input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        last_indexes = attention_mask.sum(dim=1) - 1
        return self.linear(last_hidden_state[torch.arange(last_hidden_state.size(0)), last_indexes])

def main(args):
    data = None

    if args.dataset == "lm_sys_chat":
        data = torch.load("datasets/extracted_lm_sys_chat.pth")
    elif args.dataset == "anthropic_hh":
        data = torch.load("datasets/extracted_anthropic_hh_rewards.pth")
    
    input_texts = data["input_texts"]
    output_texts = data["output_texts"]
    next_texts = data["next_texts"]
    rewards = torch.tensor(data["rewards"]) if "rewards" in data else None

    total = len(input_texts)
    
    print('splitting dataset...')

    train_input_texts = []
    train_output_texts = []
    train_next_texts = []
    train_rewards = []
    validation_input_texts = []
    validation_output_texts = []
    validation_next_texts = []
    validation_rewards = []

    for i in range(total):
        if random.random() < 0.2:
            validation_input_texts.append(input_texts[i])
            validation_output_texts.append(output_texts[i])
            validation_next_texts.append(next_texts[i])
            validation_rewards.append(rewards[i])
        else:
            train_input_texts.append(input_texts[i])
            train_output_texts.append(output_texts[i])
            train_next_texts.append(next_texts[i])
            train_rewards.append(rewards[i])

    train_total = len(train_input_texts)
    train_dataset = ChatDataset(train_input_texts, train_output_texts, train_next_texts, rewards=train_rewards)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    validation_total = len(validation_input_texts)
    validation_dataset = ChatDataset(validation_input_texts, validation_output_texts, validation_next_texts, rewards=validation_rewards)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.size, collate_fn=collate_fn)
   
    print(f"loaded {total} samples: {train_total} train and {validation_total} validation")

    model_name = get_model_name(args)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = RewardsModel(model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = sys.float_info.max

    for epoch in range(args.epochs):
        print(f"training epoch {epoch}")

        epoch_accumulated = 0
        total_batches = 0

        model.train()

        for batch in tqdm(train_dataloader):
            input_texts_batch = []
            next_texts_batch = []
            rewards_batch = [] if rewards is not None else None

            for item in batch:
                input_texts_batch.append(item["input_text"] + item["output_text"])
                next_texts_batch.append(item["next_text"])

                if rewards is not None:
                    rewards_batch.append(item.get("reward"))

            optimizer.zero_grad()
            
            computated = 0
            accumulated = 0

            for j in range(0, len(input_token_ids_batch), args.size):
                input_texts_computation_batch = input_texts_batch[j:j + args.size]

                input_computation_batch = tokenizer.encode(
                    input_texts_computation_batch,
                    max_length=1024,
                    truncation="only_first",
                    return_tensors="pt"
                )
                input_computation_batch.to(device)

                output_computation_batch = model(**input_computation_batch)

                labels = None

                if rewards_batch is not None:
                    labels = torch.tensor(rewards_batch[j:j + args.size])
                else:
                    next_texts_computation_batch = next_texts_batch[j:j + args.size]
                    labels = predict_sentiments(next_texts_computation_batch)

                labels.to(device)

                loss = nn.MSELoss()(output_computation_batch, labels)
                accumulated += loss.item()
                loss.backward()
                computated += args.size
            
            total_batches += 1

            if computated > 0:
                optimizer.step()

                batch_accumulated = accumulated / computated
                epoch_accumulated += batch_accumulated

                tqdm.write(f"batch loss = {batch_accumulated}")

        print(f"epoch {epoch} loss = {epoch_accumulated / total_batches}")

        print(f"evaluating epoch {epoch}")
        
        evaluation_accumulated = 0
        evaluation_total_batches = 0

        model.eval()

        with torch.no_grad():
            for batch in tqdm(validation_dataloader):
                input_texts_batch = []
                next_texts_batch = []
                rewards_batch = [] if rewards is not None else None

                for item in batch:
                    input_texts_batch.append(item["input_text"] + item["output_text"])
                    next_texts_batch.append(item["next_text"])

                    if rewards is not None:
                        rewards_batch.append(item.get("reward"))
                
                input_computation_batch = tokenizer.encode(
                    input_texts_batch,
                    max_length=1024,
                    truncation="only_first",
                    return_tensors="pt"
                )
                input_computation_batch.to(device)

                output_computation_batch = model(**input_computation_batch)

                labels = None

                if rewards_batch is not None:
                    labels = torch.tensor(rewards_batch)
                else:
                    labels = predict_sentiments(next_texts_batch)

                labels.to(device)
                loss = nn.MSELoss()(output_computation_batch, labels)
                batch_loss = loss.item()                

                tqdm.write(f"batch loss = {batch_loss}")

                evaluation_accumulated += batch_loss
                evaluation_total_batches += 1
            
        evaluation_loss = evaluation_accumulated / evaluation_total_batches

        print(f"epoch {epoch} evaluation loss = {evaluation_loss}")

        if evaluation_loss < best_loss:
            print("saving model")

            torch.save(
                model.state_dict(),
                "parameters/trained-model.pth"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="lm_sys_chat", help="lm_sys_chat or anthropic_hh")
    parser.add_argument("-l", "--large", action="store_true", help="if true, use gpt2-xl, else, use gpt2")
    parser.add_argument("-b", "--batch", type=int, default=64, help="the batch size")
    parser.add_argument("-r", "--lr", type=float, default=1e-6, help="the optimizer learning rate")
    parser.add_argument("-z", "--size", type=int, default=4, help="the size for computation")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="the number of epochs")

    args = parser.parse_args()

    main(args)
