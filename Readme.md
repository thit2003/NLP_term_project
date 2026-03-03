# Comparative Sentiment Analysis: Monolingual vs. Multilingual Models for English and Thai

This project investigates the performance trade-offs in sentiment analysis when using dedicated monolingual models versus a unified multilingual model. By training distinct models for English and Thai, and comparing them against a combined multilingual model, we aim to understand how joint training affects language-specific accuracy.

## 👥 Team Members
* **Thit Lwin Win Thant** (ID: 6540122)
* **Kaung Khant Lin** (ID: 6540131)
* **Thust Thongsricharoen** (ID: 6714508)

---

## 📖 Project Overview
While multilingual transformers offer the convenience of a single model handling multiple languages, it is often debated whether they match the accuracy of models trained exclusively on a single language. This project builds a comparative pipeline to evaluate sentiment analysis performance across English and Thai texts. 

### Key Objectives
* **Model Training:** Develop three distinct sentiment analysis models:
    * `english-model`: Trained exclusively on English data.
    * `thai-model`: Trained exclusively on Thai data.
    * `multilingual-model`: Trained on a combined dataset of both English and Thai.
* **Comparative Evaluation:**
    * Compare the sentiment prediction of English texts using the `english-model` versus the `multilingual-model`.
    * Compare the sentiment prediction of Thai texts using the `thai-model` versus the `multilingual-model`.
* **Performance Analysis:** Determine if the multilingual model suffers from "capacity dilution" or if it benefits from cross-lingual transfer compared to its monolingual counterparts.
* **Key Deliverable:** Three trained models, a comparative evaluation report, and a demo web app showcasing side-by-side inference.

---

## 🧠 Model Architecture
To ensure a fair comparison, we utilize **XLM-RoBERTa base (~270M parameters)** as the foundational architecture for all three models. 

**Why XLM-RoBERTa?**
* **Consistent Baseline:** Using the same base architecture ensures that performance differences are due to the fine-tuning data rather than underlying model mechanics.
* **Shared Vocabulary:** Its SentencePiece tokenizer effectively handles both Thai characters and English words without needing language-specific tokenization pipelines.
* **Strong Multilingual Roots:** As an industry standard for cross-lingual tasks, it is the ideal candidate for the `multilingual_model`, while still being highly capable when fine-tuned purely as an `english_model` or `thai_model`.

---

## 📂 Datasets
The models are fine-tuned using established monolingual corpora. 

1. **English Training Data:**
   * **SST-2 (Stanford Sentiment Treebank):** 67,349 samples used to train the `english-model` and partially train the `multilingual-model`.
   * **Twitter Sentiment:** 27,481 samples used to train the `english-model` and partially train the `multilingual-model`.
2. **Thai Training Data:**
   * **Wisesight Sentiment Corpus:** 26,737 samples used to train the `thai-model` and partially train the `multilingual-model`.
3. **Evaluation Sets:**
   * Held-out test splits from both SST-2 and Wisesight.

---

## ⚙️ Methodology & Pipeline

The project follows a comparative evaluation pipeline:

1. **Data Preparation & Splitting:**
   * Standardize labels across both datasets to **Positive, Negative, and Neutral**.
   * Create strict train/validation/test splits.
2. **Independent Fine-Tuning:**
   * **Run 1:** Fine-tune XLM-R on SST-2 and twitter-sentiment to create the `english-model`.
   * **Run 2:** Fine-tune XLM-R on Wisesight to create the `thai-model`.
   * **Run 3:** Fine-tune XLM-R on combined SST-2 + twitter-sentiment + Wisesight to create the `multilingual-model`.
3. **Comparative Inference:**
   * Feed English test data into both the `english-model` and `multilingual-model`.
   * Feed Thai test data into both the `thai-model` and `multilingual-model`.
4. **Output & Routing:** The web app will include a language identification step (LID) to route user input to the correct monolingual model, while simultaneously passing it to the multilingual model for side-by-side comparison.

---

## 📊 Evaluation Metrics
We will track the following metrics to answer our core research question:
* **Primary Metric:** Macro F1 Score and Accuracy on the respective held-out test sets.
* **A/B Comparison:** Delta in F1 score between (`english-model` vs. `multilingual-model`) and (`thai-model` vs. `multilingual-model`).
* **Confusion Matrices:** To identify if the multilingual model struggles with specific sentiments (e.g., neutral sarcasm) more than the monolingual models.

---

## 🛠️ Usage
*The final deliverable is a web application where users can input text, and the system provides side-by-side sentiment predictions.*

**Input Example:**
"The new update is absolutely terrible."

**Output Example:**
* `English Model` Prediction: Negative (Confidence: 0.98)
* `Multilingual Model` Prediction: Negative (Confidence: 0.92)