from decouple import config
import torch

TRANSFORMERS_ACCESS_TOKEN = config("TRANSFORMERS_ACCESS_TOKEN")

HUMAN_PREFIX = "\n\nHuman: "
HUMAN_PREFIX_LEN = len(HUMAN_PREFIX)
ASSISTANT_PREFIX = "\n\nAssistant: "
ASSISTANT_PREFIX_LEN = len(ASSISTANT_PREFIX)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
