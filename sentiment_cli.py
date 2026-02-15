#!/usr/bin/env python3
"""
Simple CLI for Thai-English code-mixed sentiment analysis using a fine-tuned XLM-R model.

Updates in this version:
- Ensures human-readable labels (positive/neutral/negative) even when config has LABEL_0/1/2
- Allows forcing/custom label mapping via --labels JSON file
- Keeps Windows-friendly default local model directory detection
- Supports local path or Hugging Face Hub repo ID

Usage:
  python sentiment_cli.py                               # interactive prompt
  python sentiment_cli.py --text "วันนี้ server ล่ม"    # one-off prediction
  python sentiment_cli.py --model ./final_model --text "ดีมากเลย"
  python sentiment_cli.py --model D:\\Sem_2_2025\\NLP\\NLP_term_project\\final_model --text "It is not the one I order"
  python sentiment_cli.py --labels labels.json --text "mixed text"
  # labels.json example:
  # { "id2label": { "0": "positive", "1": "neutral", "2": "negative" } }
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Try to import PyThaiNLP for tokenization; if missing, fall back to simple split
try:
    from pythainlp import word_tokenize
except Exception:
    word_tokenize = None


DEFAULT_ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}
DEFAULT_LABEL2ID = {v: k for k, v in DEFAULT_ID2LABEL.items()}


def get_default_model_dir() -> str:
    """Resolve default model directory with Windows path preference.

    Priority:
    1) MODEL_ID env var
    2) Windows absolute path: D:\\Sem_2_2025\\NLP\\NLP_term_project\\final_model
    3) Local folder relative to this script: ./final_model
    """
    env_value = os.environ.get("MODEL_ID")
    if env_value:
        return env_value
    win_default = r"D:\\Sem_2_2025\\NLP\\NLP_term_project\\final_model"
    if os.path.isdir(win_default):
        return win_default
    local_dir = Path(__file__).parent / "final_model"
    if local_dir.is_dir():
        return str(local_dir)
    return "./final_model"


def load_slang_dictionary(path: Optional[str]) -> Dict[str, str]:
    """Load slang dictionary JSON mapping (token -> normalized form)."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warn] Slang dictionary not found at {p}. Continuing without normalization.")
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] Failed to load slang dictionary: {e}. Continuing without normalization.")
        return {}


def load_label_mapping(path: Optional[str]) -> Optional[Dict[int, str]]:
    """Load optional label mapping JSON file. Expected keys: id2label with int-or-str keys."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[warn] Labels file not found at {p}. Continuing with defaults or model config.")
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        id2label_raw = data.get("id2label") or data
        id2label: Dict[int, str] = {}
        for k, v in id2label_raw.items():
            try:
                kid = int(k)
            except Exception:
                # if k is already int
                kid = k if isinstance(k, int) else None
            if kid is None:
                continue
            id2label[kid] = str(v).lower()
        if not id2label:
            print(f"[warn] Labels file at {p} did not contain usable id2label mapping.")
            return None
        return id2label
    except Exception as e:
        print(f"[warn] Failed to load labels mapping: {e}.")
        return None


def detect_language(token: str) -> str:
    """Heuristic language ID: TH if any Thai Unicode char present, else EN."""
    for ch in token:
        code = ord(ch)
        if 0x0E00 <= code <= 0x0E7F:  # Thai block
            return "TH"
    return "EN"


def tokenize_text(text: str) -> list:
    """Tokenize text; prefer PyThaiNLP, else fall back to a naive split."""
    if word_tokenize:
        # newmm handles Thai without spaces
        try:
            return word_tokenize(text, engine="newmm")
        except Exception:
            pass
    # Fallback: basic whitespace split (OK for English; Thai may be suboptimal)
    return text.split()


def normalize_text(text: str, slang_dict: Dict[str, str]) -> str:
    """Normalize tokens using slang dictionary; keep original if not found."""
    if not slang_dict:
        return text
    tokens = tokenize_text(text)
    normalized = [slang_dict.get(tok, tok) for tok in tokens]
    return " ".join(normalized)


def load_model_and_tokenizer(model_id_or_path: str) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Load tokenizer and model from local directory or Hugging Face Hub ID."""
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    return tok, model


def ensure_human_labels(model, user_id2label: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    """
    Return a human-readable id2label mapping and, if necessary, patch the model.config in-memory.

    Priority:
    1) user-supplied mapping via --labels
    2) model.config.id2label (if not generic LABEL_*)
    3) DEFAULT_ID2LABEL for 3-class models
    """
    num_labels = getattr(getattr(model, "config", None), "num_labels", None)
    cfg = getattr(model, "config", None)

    # 1) user-supplied mapping
    if user_id2label:
        if cfg is not None:
            cfg.id2label = {int(k): str(v) for k, v in user_id2label.items()}
            cfg.label2id = {v: k for k, v in user_id2label.items()}
        return user_id2label

    # 2) model config mapping
    if cfg and getattr(cfg, "id2label", None):
        id2label_cfg = {int(k): str(v) for k, v in cfg.id2label.items()}
        values = [v.lower() for v in id2label_cfg.values()]
        if all(v.startswith("label_") for v in values):
            # 3) override if generic placeholders and num_labels is 3
            if num_labels == 3 or num_labels is None:
                if cfg is not None:
                    cfg.id2label = DEFAULT_ID2LABEL
                    cfg.label2id = DEFAULT_LABEL2ID
                return DEFAULT_ID2LABEL
        # Use existing config mapping
        return {k: v.lower() for k, v in id2label_cfg.items()}

    # Fallback default for typical 3-class sentiment
    if num_labels == 3 or num_labels is None:
        if cfg is not None:
            cfg.id2label = DEFAULT_ID2LABEL
            cfg.label2id = DEFAULT_LABEL2ID
        return DEFAULT_ID2LABEL

    # If an unusual number of labels, return generic but lower-case
    return {i: f"label_{i}" for i in range(num_labels or 3)}


def predict(text: str, tok, model, slang_dict: Dict[str, str], id2label: Dict[int, str]) -> Dict:
    """Run normalization + model inference; return label and confidence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Normalize
    processed = normalize_text(text, slang_dict)

    enc = tok(
        processed,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())
    label = id2label.get(pred_id, f"label_{pred_id}")
    confidence = float(probs[pred_id].item())

    # Per-class scores with human-readable keys
    scores = {id2label.get(i, f"label_{i}"): float(probs[i].item()) for i in range(probs.size(0))}

    return {
        "text": text,
        "processed": processed,
        "label": label,
        "confidence": confidence,
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Thai-English Sentiment CLI")
    parser.add_argument(
        "--model",
        default=get_default_model_dir(),
        help="Local model directory or Hugging Face Hub repo ID (default: auto-detected local ./final_model)",
    )
    parser.add_argument(
        "--slang",
        default=os.environ.get("SLANG_DICT", None),
        help="Path to slang_dictionary.json (optional)",
    )
    parser.add_argument(
        "--labels",
        default=os.environ.get("LABELS_JSON", None),
        help="Optional JSON file providing id2label mapping. Example: {\"0\":\"positive\",\"1\":\"neutral\",\"2\":\"negative\"}",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="One-off text to analyze; if omitted, enter interactive mode",
    )
    args = parser.parse_args()

    slang_dict = load_slang_dictionary(args.slang)
    user_id2label = load_label_mapping(args.labels)
    tok, model = load_model_and_tokenizer(args.model)

    # Ensure we have human-readable labels
    id2label = ensure_human_labels(model, user_id2label=user_id2label)

    if args.text:
        result = predict(args.text, tok, model, slang_dict, id2label)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # Interactive loop
    print("Thai-English Sentiment CLI (Ctrl+C to exit)")
    print(f"Model: {args.model}")
    print("Enter text:")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                continue
            result = predict(line, tok, model, slang_dict, id2label)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()