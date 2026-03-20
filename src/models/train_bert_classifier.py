{"id":"81520","variant":"standard","title":"train_bert_classifier.py — Fine-Tuning BERT for Abstract Classification"}
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# -----------------------------------------
# 1. Load dataset
# -----------------------------------------
df = pd.read_csv("../../data/processed/cleaned_inflation_dataset.csv")

X = df["Abstract"]     # Use full abstract (not cleaned)
y = df["Label"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------------------
# 2. BERT Tokenizer
# -----------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class AbstractDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'labels': torch.tensor(self.labels[idx])
        }


train_dataset = AbstractDataset(X_train, y_train, tokenizer)
test_dataset = AbstractDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# -----------------------------------------
# 3. Compute class weights (VERY IMPORTANT)
# -----------------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class Weights:", class_weights)

# -----------------------------------------
# 4. Load BERT model
# -----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model.to(device)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# -----------------------------------------
# 5. Training Loop
# -----------------------------------------
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# -----------------------------------------
# 6. Evaluation
# -----------------------------------------
model.eval()
preds = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
        true_labels.extend(batch["labels"].numpy())

print("\n===== BERT CLASSIFIER REPORT =====\n")
print(classification_report(true_labels, preds))
print("Confusion Matrix:\n", confusion_matrix(true_labels, preds))
print("Accuracy:", accuracy_score(true_labels, preds))

# -----------------------------------------
# 7. Save Model
# -----------------------------------------
model.save_pretrained("../../artifacts/models/bert_classifier")
tokenizer.save_pretrained("../../artifacts/models/bert_classifier")

print("\n[INFO] Fine-tuned BERT model saved to: bert_classifier/")
