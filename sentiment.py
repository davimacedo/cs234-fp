from transformers import AutoTokenizer, pipeline
import torch
from settings import device

MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.model_max_length = 512
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=tokenizer,
    device=device,
    truncation=True,
    padding=True
)

def predict_sentiments(texts):
    sentiments = sentiment_pipeline(texts)

    scores = []

    for sentiment in sentiments:
        if sentiment["label"] == "positive":
            scores.append(sentiment["score"])
        elif sentiment["label"] == "negative":
            scores.append(-sentiment["score"])
        else:
            scores.append(0)

    return torch.tensor(scores)

def main():
    sentiments = predict_sentiments([
        "Life is good!",
        "Life is bad!",
        "COVID is bad!",
        "COVID is good!",
        "Sentiment analysis is a ML task"
    ])

    print(sentiments)

if __name__ == "__main__":
    main()
