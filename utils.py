import torch
from settings import device

def select(array, indexes):
    if isinstance(array, torch.Tensor):
        return array[indexes]
    else:
        return [item for index, item in enumerate(array) if index in indexes]

def calculate_log_probs(model, tokenizer, input_token_ids_batch, min_input_length, output_texts_batch, output_token_ids_batch):
    if output_token_ids_batch is None:
        output_token_ids_batch = [tokenizer.encode(text, add_special_tokens=False) for text in output_texts_batch]

    batch_token_ids = []
    indexes = []
    input_token_ids_lengths = []

    for i, (input_token_ids, output_token_ids) in enumerate(zip(input_token_ids_batch, output_token_ids_batch)):
        token_ids = input_token_ids + output_token_ids + [tokenizer.eos_token_id]
        if len(token_ids) <= 1024:
            batch_token_ids.append(token_ids)
            indexes.append(i)
            input_token_ids_lengths.append(len(input_token_ids))
    
    if len(batch_token_ids) == 0:
        return batch_token_ids, indexes
        
    max_length = max(len(token_ids) for token_ids in batch_token_ids)
    padded_batch_token_ids = torch.tensor([token_ids + [tokenizer.eos_token_id] * (max_length - len(token_ids)) for token_ids in batch_token_ids], dtype=torch.long, device=device)
    
    attention_mask = torch.zeros(padded_batch_token_ids.shape, device=device)
    probs_mask = torch.zeros(padded_batch_token_ids.shape, device=device)
    
    for i, attention_mask_row in enumerate(attention_mask):
        attention_mask_row[:len(batch_token_ids[i])] = 1
        probs_mask[i][input_token_ids_lengths[i]:len(batch_token_ids[i])] = 1

    logits = model(padded_batch_token_ids, attention_mask=attention_mask).logits[:, min_input_length:]
    probs = torch.softmax(logits, dim = -1)
    probs_mask = probs_mask[:, min_input_length:]
    padded_batch_token_ids = padded_batch_token_ids[:, min_input_length:]

    return torch.sum(torch.log(torch.gather(probs, dim=-1, index=padded_batch_token_ids.unsqueeze(-1)).squeeze(-1)) * probs_mask, dim=-1), indexes
