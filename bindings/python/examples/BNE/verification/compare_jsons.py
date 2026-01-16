import json

data_BNE = {}
with open("data/BNE_as_BPE.json", "r") as file:
    data_BNE = json.load(file)

data_BPE = {}
with open("data/BPE_as_BPE.json", "r") as file:
    data_BPE = json.load(file)

print("---- BNE ----")
print(len(data_BNE["model"]["vocab"]))
print(len(data_BNE["model"]["merges"]))

print("---- BPE ----")
print(len(data_BPE["model"]["vocab"]))
print(len(data_BPE["model"]["merges"]))

for i in range(len(data_BPE["model"]["merges"])):
    if data_BNE["model"]["merges"][i] != data_BPE["model"]["merges"][i]:
        print(f"BNE: {data_BNE['model']['merges'][i]} != BPE: {data_BPE['model']['merges'][i]}")