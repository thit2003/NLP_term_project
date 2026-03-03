import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def macro_f1_and_acc(y_true, y_pred, classes):
    """
    Macro F1 averaged over `classes` only.
    This is important for SST-2 (only positive & negative).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float((y_true == y_pred).mean())

    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp + 1e-12)
        r = tp / (tp + fn + 1e-12)
        f1 = 2 * p * r / (p + r + 1e-12)
        f1s.append(f1)

    return {"accuracy": acc, "f1_macro": float(np.mean(f1s))}


def predict(model_dir, texts, batch_size=32, max_length=128):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    return np.array(preds, dtype=int)


# -----------------------
# Load datasets
# -----------------------
def load_twitter_test(path="Data/twitter_sentiment/test.csv"):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or "," not in line:
                continue
            text, label = line.rsplit(",", 1)
            text = text.strip()
            label = label.strip().lower()
            if label in LABEL2ID and text:
                rows.append((text, LABEL2ID[label]))
    texts = [t for t, _ in rows]
    labels = np.array([y for _, y in rows], dtype=int)
    return texts, labels


def load_wisesight_test():
    base = "Data/wisesight-sentiment/kaggle-competition"
    with open(f"{base}/test.txt", "r", encoding="utf-8") as f:
        texts = [x.rstrip("\n") for x in f]
    with open(f"{base}/test_label.txt", "r", encoding="utf-8") as f:
        labs = [x.strip().lower() for x in f]

    ws_map = {"pos": "positive", "neu": "neutral", "neg": "negative", "q": "neutral"}
    x, y = [], []
    for t, l in zip(texts, labs):
        mapped = ws_map.get(l)
        if not t or mapped is None:
            continue
        x.append(t)
        y.append(LABEL2ID[mapped])
    return x, np.array(y, dtype=int)


def load_sst2_validation():
    sst2 = load_dataset("glue", "sst2")["validation"]
    x, y = [], []
    for ex in sst2:
        text = ex["sentence"]
        lab = ex["label"]  # 0=negative, 1=positive
        if lab == 1:
            x.append(text)
            y.append(LABEL2ID["positive"])
        else:
            x.append(text)
            y.append(LABEL2ID["negative"])
    return x, np.array(y, dtype=int)


# -----------------------
# A/B evaluation runner
# -----------------------
def run_ab(task_name, texts, labels, classes, model_a, model_b):
    pred_a = predict(model_a, texts)
    pred_b = predict(model_b, texts)

    m_a = macro_f1_and_acc(labels, pred_a, classes=classes)
    m_b = macro_f1_and_acc(labels, pred_b, classes=classes)

    delta = {
        "delta_f1_macro": m_b["f1_macro"] - m_a["f1_macro"],
        "delta_accuracy": m_b["accuracy"] - m_a["accuracy"],
    }

    print(f"\n=== {task_name} ===")
    print("A (monolingual):", model_a, m_a)
    print("B (multilingual):", model_b, m_b)
    print("Δ (B - A):", delta)

    return {"task": task_name, "classes": classes, "A": m_a, "B": m_b, "delta": delta}


# --------- CONFIG: set your model paths ----------
ENGLISH_MODEL_DIR = "All_Models/english_model"
THAI_MODEL_DIR = "All_Models/thai_model"
MULTI_MODEL_DIR = "All_Models/multilingual_model"

results = []

# English: Twitter (3-class)
tw_texts, tw_labels = load_twitter_test()
results.append(
    run_ab(
        "Twitter (EN, 3-class)",
        tw_texts,
        tw_labels,
        classes=[0, 1, 2],
        model_a=ENGLISH_MODEL_DIR,
        model_b=MULTI_MODEL_DIR,
    )
)

# English: SST-2 (2-class: positive & negative only)
sst_texts, sst_labels = load_sst2_validation()
results.append(
    run_ab(
        "SST-2 (EN, 2-class)",
        sst_texts,
        sst_labels,
        classes=[0, 2],  # only positive and negative
        model_a=ENGLISH_MODEL_DIR,
        model_b=MULTI_MODEL_DIR,
    )
)

# Thai: Wisesight (3-class after mapping q->neutral)
ws_texts, ws_labels = load_wisesight_test()
results.append(
    run_ab(
        "Wisesight (TH, 3-class)",
        ws_texts,
        ws_labels,
        classes=[0, 1, 2],
        model_a=THAI_MODEL_DIR,
        model_b=MULTI_MODEL_DIR,
    )
)

with open("final_report_ab_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved: final_report_ab_results.json")