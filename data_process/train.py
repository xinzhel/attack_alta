import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

###############################################################################
# 1. Dataset & Collate Function
###############################################################################
class JsonlTextDataset(Dataset):
    """
    Reads a JSON lines file of the format:
      {"label": "3", "text": "..."}
    and creates examples of (token_ids, label).
    """
    def __init__(self, jsonl_path, vocab=None, build_vocab=False):
        super().__init__()
        self.samples = []

        # Read data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                label_str = example["label"]
                text_str = example["text"]

                # Simple whitespace tokenization
                tokens = text_str.strip().split()

                # Convert label string to int (adjust mapping if needed)
                label = int(label_str)
                self.samples.append((tokens, label))

        # Build or use existing vocabulary
        if build_vocab:
            self.vocab = {"<PAD>": 0, "<UNK>": 1}
            idx = 2
            for tokens, _ in self.samples:
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = idx
                        idx += 1
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, label = self.samples[idx]
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab["<UNK>"])
        return token_ids, label

def collate_fn(batch):
    """
    Pads each batch to the max sequence length in that batch.
    """
    all_token_ids = [x[0] for x in batch]
    all_labels = [x[1] for x in batch]
    max_len = max(len(t) for t in all_token_ids)

    padded_ids = []
    for token_ids in all_token_ids:
        needed = max_len - len(token_ids)
        padded_ids.append(token_ids + [0] * needed)  # 0 is <PAD>

    padded_ids = torch.tensor(padded_ids, dtype=torch.long)
    labels = torch.tensor(all_labels, dtype=torch.long)

    return padded_ids, labels

###############################################################################
# 2. Model Definition (LSTM-based Classifier)
###############################################################################
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # -> [batch_size, seq_len, embed_dim]
        output, (h_n, c_n) = self.lstm(embedded)
        # h_n: [num_layers, batch_size, hidden_size]
        # Take the last layer's hidden state
        final_hidden = h_n[-1]                # -> [batch_size, hidden_size]
        logits = self.classifier(final_hidden)
        return logits

###############################################################################
# 3. Training & Evaluation
###############################################################################
def train_model(train_jsonl, dev_jsonl,
                embed_dim=300,
                hidden_size=512,
                num_layers=2,
                batch_size=64,
                num_epochs=3,
                lr=1e-3,
                patience=3):
    # 3.1 Create Datasets
    train_dataset = JsonlTextDataset(train_jsonl, build_vocab=True)
    dev_dataset = JsonlTextDataset(dev_jsonl, vocab=train_dataset.vocab)

    # 3.2 Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # 3.3 Detect device (Apple MPS or GPU or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 3.4 Prepare model
    vocab_size = len(train_dataset.vocab)
    all_labels_train = [x[1] for x in train_dataset.samples]
    num_classes = len(set(all_labels_train))

    model = LSTMClassifier(vocab_size, embed_dim, hidden_size,
                           num_layers, num_classes)
    model.to(device)

    # 3.5 Set up optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0.0
    patience_counter = 0

    # 3.6 Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_ids, batch_labels in train_loader:
            batch_ids = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_ids)
            loss = criterion(logits, batch_labels)
            loss.backward()

            # Optional gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 3.7 Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_ids, batch_labels in dev_loader:
                batch_ids = batch_ids.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_ids)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

        dev_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_loss:.4f} | Dev Acc: {dev_acc:.4f}")

        # Early stopping
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete. Best Dev Accuracy:", best_dev_acc)
    # Return model, vocab, and the number of classes for further usage
    return model, train_dataset.vocab, num_classes


###############################################################################
# 4. Save & Load Model
###############################################################################
def save_model(model, vocab, path="model_checkpoint.pt"):
    """
    Saves the model state_dict and the vocab to a checkpoint file.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab
    }
    torch.save(checkpoint, path)
    print(f"Model and vocab saved to {path}.")

def load_model(path, embed_dim, hidden_size, num_layers, num_classes, device=None):
    """
    Loads the model state_dict and vocab from a checkpoint file,
    re-initializes the model, and places it on the given device.
    """
    if not device:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device)
    vocab = checkpoint["vocab"]

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model and vocab loaded from {path}. Using device: {device}")
    return model, vocab


###############################################################################
# 5. Inference (Prediction)
###############################################################################
def predict_texts(model, vocab, texts, device=None):
    """
    Given a list of raw text strings, tokenize, convert to IDs,
    run the model, and return predicted labels.
    """
    if not device:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model.eval()
    model.to(device)

    predictions = []

    for text in texts:
        tokens = text.strip().split()
        token_ids = []
        for tok in tokens:
            if tok in vocab:
                token_ids.append(vocab[tok])
            else:
                token_ids.append(vocab["<UNK>"])

        # Convert to tensor of shape [1, seq_len]
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            pred_label = torch.argmax(logits, dim=1).item()
        predictions.append(pred_label)

    return predictions


###############################################################################
# 6. Main / Example Usage
###############################################################################
if __name__ == "__main__":
    # Example usage:
    # Replace with actual local file paths
    train_path = "revised/process_data/sst2_dev.jsonl"
    dev_path   = "revised/process_data/sst2_dev.jsonl"

    # Train the model
    model, vocab, num_classes = train_model(
        train_jsonl=train_path,
        dev_jsonl=dev_path,
        embed_dim=300,
        hidden_size=512,
        num_layers=2,
        batch_size=64,
        num_epochs=3,
        lr=1e-3,
        patience=3
    )

    # Save the model
    save_model(model, vocab, path="model_checkpoint.pt")

    # Load the model
    loaded_model, loaded_vocab = load_model(
        path="model_checkpoint.pt",
        embed_dim=300,
        hidden_size=512,
        num_layers=2,
        num_classes=num_classes
    )

    # Inference on new text
    test_texts = [
        "This is a simple test sentence for prediction.",
        "Another sentence that might get a different label."
    ]
    preds = predict_texts(loaded_model, loaded_vocab, test_texts)
    print("Predictions:", preds)