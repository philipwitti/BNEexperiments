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
tokenizerBNE = Tokenizer.from_file("examples/BNE/old/data/bne_byteLevel_minipile_5_tokens_16000.json")
tokenizerBPE = Tokenizer.from_file("examples/BNE/old/data/bpe_byte-level_minipile.json")
# tokenizer.pre_tokenizer = ByteLevel()
tokenizerBNE.decoder = ByteLevel()
tokenizerBPE.decoder = ByteLevel()
test_data = ["1620", "1368202", "58", "140353", "42565"]
test_data = [dataset["text"][736929]]

for text in test_data:
    # print(len(text)) 1620 1368202 58 140353 42565
    print(f"orig:       {text}")
    print(f"encoded:    {tokenizerBNE.encode(text).ids}")
    ids = tokenizerBNE.encode(text).ids
    ids = [5 if x == 0 else x for x in ids]
    print(f"decoded:    {tokenizerBNE.decode(tokenizerBNE.encode(text).ids)}")
    print(f"decoded:    {tokenizerBNE.decode(ids)}")
    print(f"len Comp:   Orig: {len(text)}            new:{len(tokenizerBNE.decode(tokenizerBNE.encode(text).ids))}")
