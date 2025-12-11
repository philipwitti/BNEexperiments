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
tokenizer = Tokenizer.from_file("examples/BNE/old/data/bne_byteLevel_minipile_5_tokens_16000.json")
# tokenizer = Tokenizer.from_file("examples/BNE/old/data/bpe_byte-level_minipile.json")
# tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevel()

for text in tqdm(dataset["text"]):
    # print(len(text)) 1620
    # print(len(tokenizer.decode(tokenizer.encode(text).ids)[1:]))
    # print("====== end ======")
    if len(text) < len(tokenizer.decode(tokenizer.encode(text).ids)):
        # assert(tokenizer.decode(tokenizer.encode(text).ids)[1:] == text)
        if tokenizer.decode(tokenizer.encode(text).ids)[1:] != text:
            print("====== orig ======")
            print(text)
            print("====== enc ======")
            print(tokenizer.decode(tokenizer.encode(text).ids)[1:])
            print("====== end ======")
    else:
        # assert(tokenizer.decode(tokenizer.encode(text).ids) == text)
        if tokenizer.decode(tokenizer.encode(text).ids) != text:
            print("====== orig ======")
            print(text)
            print("====== enc ======")
            print(tokenizer.decode(tokenizer.encode(text).ids))
            print("====== end ======")
    # print(tokenizer.encode(text).ids)
    # print(tokenizer.encode(text).tokens)
