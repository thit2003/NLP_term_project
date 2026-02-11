# Sentiment Analysis for English-Thai Code-Mixed Texts

This project delivers a practical sentiment analysis tool designed specifically for **English-Thai code-mixed text** (e.g., "I am so ดีใจ today"). It addresses the challenge of analyzing the "wild west" of bilingual internet slang and informal scripts found on social media platforms.

## 👥 Team Members
* **Thit Lwin Win Thant** (ID: 6540122)
* **Kaung Khant Lin** (ID: 6540131)
* **Thust Thongsricharoen** (ID: 6714508)

---

## 📖 Project Overview
Real-world social media and comment streams often feature heavy language mixing, slang, and intentional misspellings. Standard models often struggle to process this effectively. This project builds a lightweight, practical pipeline combining heuristic preprocessing tools with transformer-based fine-tuning to accurately classify sentiment.

### Key Objectives
* **Classification:** Achieve accurate sentiment labeling (**Positive, Negative, Neutral**) for mixed-language sentences.
* **Identification:** Perform token-level language identification using fast heuristics (Unicode ranges) rather than heavy models.
* **Robustness:** Implement normalization for slang and misspellings to improve reliability.
* **Efficiency:** Create a pipeline with minimal manual work by using automated "silver labels" and a high-quality "gold set" for final evaluation.
* **Key Deliverable:** A working sentiment classifier, a demo web app, and a detailed failure analysis report.

---

## 🧠 Model Architecture
We utilize **XLM-RoBERTa base (~270M parameters)** as the core model.

**Why XLM-RoBERTa?**
* **Industry Standard:** XLM-RoBERTa (XLM-R) is currently the industry standard for this specific scenario.
* **Performance:** It consistently outperforms multilingual BERT (mBERT) because it was trained on significantly more Thai web data.
* **Shared Vocabulary:** It uses a "SentencePiece" tokenizer that doesn't rely on spaces. It can process Thai characters and English words in the same sequence without breaking.
* **Cross-Lingual Transfer:** By training on CommonCrawl data in 100+ languages, it learns that words like "happy" (EN) and "ดีใจ" (TH) occupy similar vector spaces, which is critical for code-mixed sentences.

---

## 📂 Datasets
The model is trained on a combination of established corpora and scraped social media data:

1.  **Wisesight Sentiment Corpus:** 26,737 samples for monolingual Thai sentiment.
2.  **SST-2 (Stanford Sentiment Treebank):** 67,349 samples for English sentiment.
3.  **Code-Mixed Data (Scraped) only if it is needed:**
    * **Sources:** YouTube comments from Thai tech/gaming channels (e.g., **9arm**, **Bay Riffer**) and Twitter/X hashtags where English-Thai mixing is common.
    * **Labeling:** Uses automated "silver-labeling" via Unicode ranges and heuristics, plus a manually labeled "gold" evaluation set of ~200 sentences.

---

## ⚙️ Methodology & Pipeline

The system follows a linear pipeline:

1.  **Input:** Raw text sequence (e.g., "วันนี้ server ล่ม").
2.  **Preprocessing:**
    * **Tokenization:** Using PyThaiNLP.
    * **Language ID (LID):** Token-level tagging using Unicode ranges (Thai script -> TH, Latin -> EN) to keep the pipeline lightweight.
    * **Normalization:** Uses a slang dictionary (`slang_dictionary.json`) and Levenshtein distance for fuzzy matching to handle misspellings.
3.  **Model Inference:** Fine-tuned XLM-RoBERTa processes the normalized tokens.
4.  **Output:** Sentiment classification (Positive/Negative/Neutral) with a confidence score.

---

## 📊 Evaluation Metrics
To prove accuracy, speed, and robustness, we track the following metrics:
* **Accuracy:** Macro F1 score on the gold evaluation set.
* **Error Analysis:** Confusion matrix to analyze failures, specifically when English content exceeds 50% of the sentence.
* **Speed:** Inference latency per sentence (ms).
* **Robustness:** Word Error Rate (WER) and Character Error Rate (CER) between raw and normalized text.

---

## 🛠️ Usage
*This project results in a user-friendly web app where users enter text and the system displays the sentiment.*

**Input Example:**
วันนี้ server ล่ม

**Output Example:**
Sentiment: Negative
