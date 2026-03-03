import os
import json
import numpy as np
import pandas as pd

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Optional plotting
import matplotlib.pyplot as plt

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# -----------------------
# Load your eval datasets
# -----------------------
def load_wisesight_test() -> pd.DataFrame:
    base = "wisesight-sentiment/kaggle-competition"
    text_path = os.path.join(base, "test.txt")
    label_path = os.path.join(base, "test_label.txt")

    with open(text_path, "r", encoding="utf-8") as f:
        texts = [x.rstrip("\n") for x in f]
    with open(label_path, "r", encoding="utf-8") as f:
        labs = [x.rstrip("\n").strip().lower() for x in f]

    ws_map = {"pos": "positive", "neu": "neutral", "neg": "negative", "q": "neutral"}

    rows = []
    for t, l in zip(texts, labs):
        if not t:
            continue
        mapped = ws_map.get(l)
        if mapped is None:
            continue
        rows.append({"text": t, "labels": LABEL2ID[mapped], "source": "wisesight_test"})

    return pd.DataFrame(rows)

# -----------------------
# Metrics helpers
# -----------------------
def confusion_matrix(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_report(y_true, y_pred):
    # returns dict with per-class + macro
    report = {}
    f1s = []
    supports = []
    for c in [0, 1, 2]:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        support = int(np.sum(y_true == c))
        report[ID2LABEL[c]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }
        f1s.append(f1)
        supports.append(support)

    report["macro_avg"] = {"f1": float(np.mean(f1s))}
    report["accuracy"] = float((y_true == y_pred).mean())
    return report


# -----------------------
# Inference
# -----------------------
def predict_proba(texts, tokenizer, model, batch_size=32, max_length=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)

    return np.vstack(all_probs)


# -----------------------
# Run evaluation
# -----------------------
MODEL_DIR = "All_Models/thai_model"  # CHANGE if needed

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Load evaluation data from each dataset
df_wise = load_wisesight_test()

# Combine for overall evaluation too
df_all = pd.concat([df_wise], ignore_index=True)

# Predict
probs = predict_proba(df_all["text"].tolist(), tokenizer, model, batch_size=32)
preds = probs.argmax(axis=1)

df_all["pred"] = preds
df_all["pred_label"] = df_all["pred"].map(ID2LABEL)
df_all["true_label"] = df_all["labels"].map(ID2LABEL)
df_all["confidence"] = probs.max(axis=1)

# Overall report
y_true = df_all["labels"].to_numpy()
y_pred = df_all["pred"].to_numpy()

overall = classification_report(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("OVERALL METRICS")
print(json.dumps(overall, indent=2))
print("\nCONFUSION MATRIX (rows=true, cols=pred)")
print(cm)

# Save artifacts
os.makedirs("final_report", exist_ok=True)
with open("final_report/overall_metrics.json", "w", encoding="utf-8") as f:
    json.dump(overall, f, indent=2)

pd.DataFrame(cm, index=["pos", "neu", "neg"], columns=["pos", "neu", "neg"]).to_csv(
    "final_report/confusion_matrix.csv"
)

# Save per-source metrics
per_source = {}
for src, dsrc in df_all.groupby("source"):
    yt = dsrc["labels"].to_numpy()
    yp = dsrc["pred"].to_numpy()
    per_source[src] = classification_report(yt, yp)

with open("final_report/per_source_metrics.json", "w", encoding="utf-8") as f:
    json.dump(per_source, f, indent=2)

# Save misclassified examples (top confidence wrong)
mis = df_all[df_all["pred"] != df_all["labels"]].copy()
mis = mis.sort_values("confidence", ascending=False).head(200)
mis.to_csv("final_report/top_misclassified.csv", index=False, encoding="utf-8-sig")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.xticks([0, 1, 2], ["positive", "neutral", "negative"], rotation=20)
plt.yticks([0, 1, 2], ["positive", "neutral", "negative"])
plt.colorbar()

for i in range(3):
    for j in range(3):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("final_report/confusion_matrix.png", dpi=200)
print("\nSaved report to: final_report/")