#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}


def ensure_id2label(model) -> Dict[int, str]:
    cfg = getattr(model, "config", None)
    if cfg and getattr(cfg, "id2label", None):
        return {int(k): str(v).lower() for k, v in cfg.id2label.items()}
    return DEFAULT_ID2LABEL


def predict_one(text: str, model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    id2label = ensure_id2label(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    enc = tok(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())
    label = id2label.get(pred_id, f"label_{pred_id}")
    conf = float(probs[pred_id].item())
    return label, conf


def default_english_dir() -> str:
    local = Path(__file__).parent / "english_model"
    return str(local) if local.is_dir() else "./english_model"


def default_multi_dir() -> str:
    local = Path(__file__).parent / "multilingual_model"
    return str(local) if local.is_dir() else "./multilingual_model"


def main():
    p = argparse.ArgumentParser(description="Compare English vs Multilingual sentiment models")
    p.add_argument("--text", default=None, help="Input text to analyze (optional; will prompt if omitted)")
    p.add_argument("--english-model", default=os.environ.get("EN_MODEL", default_english_dir()))
    p.add_argument("--multi-model", default=os.environ.get("MULTI_MODEL", default_multi_dir()))
    args = p.parse_args()

    # If not provided, ask user
    text = args.text
    if not text:
        print("Enter text (Ctrl+C to exit):")
        text = input("> ").strip()

    en_label, en_conf = predict_one(text, args.english_model)
    ml_label, ml_conf = predict_one(text, args.multi_model)

    print(f'\nInput: "{text}"')
    print(f"* `English Model` Prediction: {en_label.capitalize()} (Confidence: {en_conf:.2f})")
    print(f"* `Multilingual Model` Prediction: {ml_label.capitalize()} (Confidence: {ml_conf:.2f})")


if __name__ == "__main__":
    main()