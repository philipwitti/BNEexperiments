from tokenizers import Tokenizer
from tokenizers.models import BNE
from tokenizers.trainers import BneTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel
import datasets
from tqdm import tqdm

# Load dataset
dataset = datasets.load_dataset("JeanKaddour/minipile", split="train")

# Load tokenizer
BNE_tokenizer = Tokenizer.from_file("data/BNE_as_BPE.json")
BPE_tokenizer = Tokenizer.from_file("data/BPE_as_BPE.json")

BNE_tokenizer.decoder = ByteLevel()
BPE_tokenizer.decoder = ByteLevel()

"""
for text in tqdm(dataset["text"]):
    if BNE_tokenizer.encode(text).ids != BPE_tokenizer.encode(text).ids:
        print(text)
        print(f"BNE: {BNE_tokenizer.encode(text).ids} != BPE: {BPE_tokenizer.encode(text).ids}")
"""

text = dataset["text"][0]
print(text)
BNE_ids = BNE_tokenizer.encode(text).ids
BPE_ids = BPE_tokenizer.encode(text).ids
for i in range(len(BNE_ids)):
    print(f"{BNE_ids[i]} != {BPE_ids[i]}")