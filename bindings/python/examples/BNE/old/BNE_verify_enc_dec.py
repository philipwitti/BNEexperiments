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
tokenizer = Tokenizer.from_file("examples/BNE/old/data/bne-byte-level_wt103.json")
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevel()

for text in tqdm(dataset["text"]):
    #print(text)
    assert(tokenizer.decode(tokenizer.encode(text).ids) == text)
    #print(tokenizer.encode(text).ids)
    #print(tokenizer.encode(text).tokens)