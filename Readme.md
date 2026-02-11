# Sentiment Analysis for English-Thai Code-Mixed Texts

This project delivers a practical sentiment analysis tool designed specifically for **English-Thai code-mixed text** (e.g., "I am so ดีใจ today"). [cite_start]It addresses the challenge of analyzing the "wild west" of bilingual internet slang and informal scripts found on social media platforms[cite: 54, 55, 74].

## 👥 Team Members
* [cite_start]**Thit Lwin Win Thant** (ID: 6540122) [cite: 57]
* [cite_start]**Kaung Khant Lin** (ID: 6540131) [cite: 57]
* [cite_start]**Thust Thongsricharoen** (ID: 6714508) [cite: 57]

---

## 📖 Project Overview
[cite_start]Real-world social media streams in Thailand often feature heavy language mixing, slang, and intentional misspellings[cite: 72, 73]. Standard models struggle to process this effectively. [cite_start]This project builds a lightweight pipeline that combines heuristic preprocessing tools with transformer-based fine-tuning to accurately classify sentiment[cite: 75].

### Key Objectives
* [cite_start]**Classification:** Achieve accurate sentiment labeling (**Positive, Negative, Neutral**) for mixed-language sentences[cite: 13, 79].
* [cite_start]**Identification:** Perform token-level language identification using fast heuristics (Unicode ranges) rather than heavy models[cite: 80].
* [cite_start]**Robustness:** Implement normalization for slang and misspellings to improve reliability[cite: 81].
* [cite_start]**Practicality:** Create a demo web app and a working classifier suitable for social media monitoring and customer feedback analysis[cite: 38, 43].

---

## 🧠 Model Architecture
[cite_start]We utilize **XLM-RoBERTa base (~270M parameters)** as the core model[cite: 15, 20].

**Why XLM-RoBERTa?**
* [cite_start]**Industry Standard:** It is currently the standard for this specific scenario, outperforming mBERT in Thai contexts[cite: 20, 23, 28].
* [cite_start]**Shared Vocabulary:** Uses a "SentencePiece" tokenizer that processes Thai characters and English words in the same sequence without breaking, which is critical for code-mixing[cite: 25, 26].
* [cite_start]**Cross-Lingual Transfer:** Trained on CommonCrawl data in 100+ languages; it learns that concepts like "happy" (EN) and "ดีใจ" (TH) occupy similar vector spaces[cite: 24, 27].

---

## 📂 Datasets
The model is trained on a combination of established corpora and scraped social media data:

1.  [cite_start]**Wisesight Sentiment Corpus:** 26,737 samples for monolingual Thai sentiment[cite: 3].
2.  [cite_start]**SST-2 (Stanford Sentiment Treebank):** 67,349 samples for English sentiment[cite: 4].
3.  **Code-Mixed Data (Scraped):**
    * [cite_start]**Sources:** YouTube comments from Thai tech/gaming channels (e.g., **9arm**, **Bay Riffer**) and Twitter/X hashtags[cite: 7].
    * [cite_start]**Labeling:** Uses automated "silver-labeling" via Unicode ranges and heuristics, plus a manually labeled "gold" evaluation set of ~200 sentences[cite: 8, 9].

---

## ⚙️ Methodology & Pipeline

[cite_start]The system follows a linear pipeline[cite: 42]:

1.  [cite_start]**Input:** Raw text sequence (e.g., "วันนี้ server ล่ม")[cite: 32].
2.  **Preprocessing:**
    * [cite_start]**Tokenization:** Using PyThaiNLP[cite: 86].
    * [cite_start]**Language ID (LID):** Token-level tagging using Unicode ranges (Thai script -> TH, Latin -> EN)[cite: 93].
    * [cite_start]**Normalization:** Uses a slang dictionary (`slang_dictionary.json`) and Levenshtein distance for fuzzy matching to handle misspellings[cite: 90, 91].
3.  [cite_start]**Model Inference:** Fine-tuned XLM-RoBERTa processes the normalized tokens[cite: 42].
4.  [cite_start]**Output:** Sentiment classification (Positive/Negative/Neutral) with a confidence score[cite: 34, 35].

---

## 📊 Evaluation Metrics
To prove accuracy, speed, and robustness, we track the following metrics:
* [cite_start]**Accuracy:** Macro F1 score on the gold evaluation set[cite: 45].
* [cite_start]**Error Analysis:** Confusion matrix to analyze failures, specifically when English content exceeds 50% of the sentence[cite: 46].
* [cite_start]**Speed:** Inference latency per sentence (ms)[cite: 47].
* [cite_start]**Robustness:** Word Error Rate (WER) and Character Error Rate (CER) between raw and normalized text[cite: 48].

---

## 🛠️ Usage
[cite_start]*This project results in a user-friendly web app where users enter text and the system displays the sentiment and highlighted language tags[cite: 43].*

**Input Example:**
```text
วันนี้ server ล่ม

**Output Example:**
* Sentiment: Negative