# Sentiment Analysis — IMDB Review Classifier

A machine learning project that classifies movie reviews as **Positive** or **Negative** using NLP techniques. Includes a full analysis notebook and an interactive web app.

---

## Overview

- **Dataset:** IMDB — 50,000 movie reviews (balanced: 25k positive / 25k negative)
- **Approach:** Text preprocessing → TF-IDF vectorization → ML classification
- **Best model** saved as `best_model.pkl` and served via a Flask web app

---

## Project Structure

```
sentiment_classifier/
├── sentiment_analysis.ipynb   # Full analysis: EDA, training, evaluation
├── app.py                     # Flask web app for live predictions
├── best_model.pkl             # Trained model
├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
├── plot_*.png                 # Visualizations generated in the notebook
└── .gitignore
```

---

## How It Works

**Preprocessing pipeline:**
1. Strip HTML tags and entities
2. Lowercase + remove punctuation
3. Remove stopwords
4. Porter stemming

**Features:** TF-IDF (term frequency–inverse document frequency)

**Models compared:** Logistic Regression, Naive Bayes, SVM (best performer saved)

---

## Web App

Run the Flask app locally:

```bash
pip install flask nltk scikit-learn
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Paste any movie review and get an instant **Positive / Negative** prediction with a confidence score.

---

## Visualizations

| Plot | Description |
|------|-------------|
| `plot_class_distribution.png` | Label balance in the dataset |
| `plot_review_length.png` | Distribution of review lengths |
| `plot_wordcloud.png` | Most frequent words |
| `plot_ngrams.png` | Top bigrams/trigrams |
| `plot_confusion_matrices.png` | Model confusion matrices |
| `plot_roc_curves.png` | ROC curves for all models |
| `plot_model_comparison.png` | Accuracy comparison across models |

---

## Requirements

```
flask
nltk
scikit-learn
pandas
numpy
matplotlib
wordcloud
```

---

## Dataset

The IMDB dataset is not included in this repo due to its size (66 MB).  
Download it from [Kaggle — IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
