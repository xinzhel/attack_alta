import json
from collections import Counter

def count_labels_in_jsonl(file_path):
    """
    Reads a .jsonl file with fields 'label' and 'text',
    and prints the number of examples for each label,
    along with the total number of examples.
    """
    label_counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines if any
            record = json.loads(line)
            label = record['label']
            label_counts[label] += 1

    # Print counts
    total_examples = sum(label_counts.values())
    print("Count by label:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count}")
    print(f"Total size: {total_examples}")

if __name__ == "__main__":
    for split in ["train", "test"]:
        print(f"Split: {split}")
        file_path = f"data/ag_news/{split}.jsonl"
        count_labels_in_jsonl(file_path)

