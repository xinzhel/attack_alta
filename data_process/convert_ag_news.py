import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")
args = parser.parse_args()
instances = []
with open('revised/orig_data/ag_news_'+args.split+'.csv', "r") as data_file:
    lines = data_file.readlines()
    i = 0
    for line in lines:
        line = line.strip("\n")
        if not line:
            continue
        label = line[1]
        string = line[5:-1].replace('""', ' " ')

        i += 1
        instances.append({"label": label, "text": string})
        
df = pd.DataFrame(instances)
df.to_json(f"{args.split}.json", orient = "records", lines=True)