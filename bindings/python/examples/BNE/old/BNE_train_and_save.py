from tokenizers import Tokenizer
from tokenizers.models import BNE
from tokenizers.trainers import BneTrainer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.decoders import ByteLevel
import datasets
from tqdm import tqdm

"""
# Build tokenizer
model = BNE(unk_token="[UNK]")
tokenizer = Tokenizer(model)
trainer = BneTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.pre_tokenizer = ByteLevel()
"""
# Load dataset
dataset = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train").train_test_split(
    test_size=0.75, seed=42
)

"""
# Build an iterator over this dataset
def batch_iterator():
    batch_size = 1000
    for batch in dataset["train"].iter(batch_size=batch_size):
        yield batch["text"]


print(tokenizer.pre_tokenizer.pre_tokenize)

tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(dataset["train"]))
tokenizer.save("examples/BNE/old/data/bne-byte-level_wt103_2.json")
"""

tokenizer = Tokenizer.from_file("examples/BNE/old/data/bne-byte-level_wt103.json")
tokenizer.decoder = ByteLevel()

#  model.save("data/", "bne_byte_wt103")
eval = dataset["test"]["text"]
string = " 吾輩《わがはい》は猫である。名前はまだ無い。"
cmp_string = tokenizer.decode(tokenizer.encode(string).ids)
for i in range(len(string)):
    print(f"{i} : {string[i]} == {cmp_string[i]} : {string[i] == cmp_string[i]}")

print(tokenizer.encode(string).ids)
print(tokenizer.encode(string).tokens)

"""
for text in tqdm(eval):
    #print(text)
    assert(tokenizer.decode(tokenizer.encode(text).ids) == text)
    #print(tokenizer.encode(text).ids)
    #print(tokenizer.encode(text).tokens)

# To implement: https://huggingface.co/docs/tokenizers/v0.20.3/en/api/models#tokenizers.models.Model
"""
