from nltk.tree import Tree
import pandas as pd
import argparse
import os
import hashlib
import requests

instances = []

def cached_path(url: str, cache_dir: str = ".cache") -> str:
    """
    Downloads the file from the given URL if it's not already cached locally.
    Returns the local file path.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hashed filename for caching
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, url_hash)
    
    if not os.path.exists(cache_path):
        print(f"Downloading {url} to {cache_path}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(cache_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Using cached file at {cache_path}.")
    
    return cache_path


# generate command line argument to define file_path
parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="train")
args = parser.parse_args()
file_path =  f"https://allennlp.s3.amazonaws.com/datasets/sst/{args.split}.txt"
with open(cached_path(file_path), "r") as data_file:
    for line in data_file.readlines():
        line = line.strip("\n")
        if not line:
            continue
        parsed_line = Tree.fromstring(line)
        tokens = parsed_line.leaves()
        sentiment = parsed_line.label()
        text = " ".join(tokens)
        if int(sentiment) < 2:
            sentiment = 0
        elif int(sentiment) == 2:
            continue
        else:
            sentiment = 1
        instances.append({"label": sentiment, "text": text})

df = pd.DataFrame(instances)
df.to_json(f"{args.split}.json", orient = "records", lines=True)