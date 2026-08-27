import json
import pandas as pd

path = "/Users/dilatasdelen/Desktop/HSLU/HSLU Courses/Bootcamps/Recommender systems/Project/datasets/meta_Movies_and_TV.jsonl"

records = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)
print(df.shape)
print(df.head())


