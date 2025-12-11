from tokenizers import Tokenizer
from tokenizers.models import BNE
from tokenizers.trainers import BneTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel
import datasets
import json
from tqdm import tqdm
import os

tokenizer_name = "bne_byteLevel_minipile_5_tokens_16000"
path_in = "examples/BNE/old/data/"
path_out = "examples/BNE/old/out/"
config_file = {"vocabulary_files": {}}
data_file = {}
# Load dataset
dataset = datasets.load_dataset("JeanKaddour/minipile", split="train").shuffle(seed=42).shard(num_shards=100, index=0)

# Load tokenizer
tokenizer = Tokenizer.from_file(os.path.join(path_in, tokenizer_name + ".json"))
tokenizer.decoder = ByteLevel()

# Create vocab File
with open(os.path.join(path_in, tokenizer_name + "-vocab.json"), "r") as file:
    vocab = json.load(file)

with open(os.path.join(path_out, tokenizer_name + "_vocab.txt"), "w") as file:
    file.write("\n".join([k for k, _ in sorted(vocab.items(), key=lambda item: item[1])]))

config_file["vocabulary_files"][tokenizer_name] = os.path.join(path_out, tokenizer_name + "_vocab.txt")

data_list = []
for text in tqdm(dataset["text"]):
    data_list.append(
        {
            "tokenizer_name": tokenizer_name,
            "language": "eng_Latn",
            "tokens": tokenizer.encode(text).ids,
            "text": text,
            "metadata": {"source": "minipile"},
        }
    )
data_file[tokenizer_name] = data_list

with open(os.path.join(path_out, "data_file.json"), "w") as file:
    file.write(json.dumps(data_file))

with open(os.path.join(path_out, "config_file.json"), "w") as file:
    file.write(json.dumps(config_file))
