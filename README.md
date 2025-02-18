# 📰 Fake News Classification Using DistilBERT 🚀

This project fine-tunes the **DistilBERT** model to classify news articles as **real** or **fake**, achieving an impressive **99.7% accuracy**. The process involves **data loading, tokenization, model training, evaluation, and prediction**.

---

## 📌 Table of Contents

- [📖 Overview](#-overview)
- [🛠️ Setup & Installation](#%EF%B8%8F-setup--installation)
- [📂 Dataset](#-dataset)
- [⚙️ Model Training Process](#%EF%B8%8F-model-training-process)
- [📊 Results](#-results)
- [🚀 Usage](#-usage)
- [👤 Author](#-author)
- [📜 License](#-license)

---

## 📖 Overview

- **Model Used:** `distilbert-base-uncased`
- **Task:** Binary classification (Fake News vs. Real News)
- **Dataset:** CSV file containing labeled news articles
- **Key Steps:**
  - Load dataset using Pandas 📊
  - Tokenize text with DistilBERT tokenizer ✍️
  - Fine-tune the model on the dataset ⚡
  - Evaluate performance using accuracy metrics 📈
- **Final Accuracy:** **99.7%** 🎯

---

## 🛠️ Setup & Installation

To use this project, first clone the repository:

```bash
git clone https://github.com/harshhmaniya/Fake-News-Classification-DistilBERT-99.7-accuracy.git
cd Fake-News-Classification-DistilBERT-99.7-accuracy
```

### Install Dependencies

Ensure that the required libraries are installed. If a `requirements.txt` file is not present, you can manually install the necessary packages:

```bash
pip install transformers pandas numpy torch scikit-learn
```

---

## 📂 Dataset

The dataset consists of **news articles** labeled as either **real** or **fake**. The data is loaded from a **CSV file** using Pandas. Before training, the **DistilBERT tokenizer** is used to process the text.

---

## ⚙️ Model Training Process

The Jupyter Notebook guides through the following steps:

1. **Data Preprocessing:**
   - Load the dataset using Pandas.
   - Tokenize text data using the **DistilBERT tokenizer**.

2. **Fine-Tuning DistilBERT:**
   - Load the pre-trained model (`distilbert-base-uncased`).
   - Train the model for **3 epochs** with optimized hyperparameters.

3. **Evaluation:**
   - Compute accuracy and performance metrics.
   - **Final Accuracy:** **99.7%** on the test set.

4. **Saving the Fine-Tuned Model:**
   - The model and tokenizer are saved for future inference.

---

## 📊 Results

- **Test Accuracy:** **99.7%** ✅
- **Evaluation Metrics:** Detailed in the notebook.
- **Performance Visualization:** Training loss graphs and accuracy plots included.

---

## 🚀 Usage

Once trained, you can use the fine-tuned model to classify new news articles.

### Load the Fine-Tuned Model

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load tokenizer and trained model (update path accordingly)
tokenizer = DistilBertTokenizer.from_pretrained('path_to_saved_tokenizer')
model = DistilBertForSequenceClassification.from_pretrained('path_to_saved_model')
```

### Make Predictions

```python
text = "Breaking news: Scientists discover water on Mars!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=1).item()

print("Fake News" if prediction == 1 else "Real News")
```

🔹 **Tip:** Adjust the model path according to where it's saved.

---

## 👤 Author

**Harsh Maniya**

🔗 [Hugging Face](https://huggingface.co/harshhmaniya)

---

## 📜 License

This project is licensed under the **MIT License**.

---

✨ Enjoy exploring and improving this project! 🚀

*Note: For detailed code and further information, refer to the [Jupyter Notebook](https://github.com/harshhmaniya/Fake-News-Classification-DistilBERT-99.7-accuracy/blob/main/real_fake.ipynb) included in this repository.* 
