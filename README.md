# IMDB Sentiment Analysis (Text Classification)

This repository contains a machine learning–based **text classification** project that performs **sentiment analysis** on movie reviews using the **IMDB 50K Movie Reviews** dataset. The goal is to classify reviews into **positive** or **negative** sentiment using **TF-IDF** feature extraction and supervised ML models.

---

## Project Overview

Text classification is a core Natural Language Processing (NLP) task that assigns predefined labels to text. In this project, sentiment analysis is applied to IMDB reviews to automatically identify audience opinion at scale. The pipeline includes text preprocessing, feature engineering using TF-IDF, model training, evaluation, and sample predictions (“model in action”).

---

## Dataset

- **Dataset Name:** IMDB Dataset of 50K Movie Reviews  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews  
- **Size:** 50,000 English reviews  
- **Classes:** 2 (Positive / Negative)  
- **Distribution:** Balanced (25,000 positive, 25,000 negative)

---

## Models Implemented

- **Logistic Regression**
- **Support Vector Machine (SVM - LinearSVC)**

---

## Tools, Libraries, and Technologies

- **Google Colab** (development environment)
- **Python 3**
- **pandas, numpy** (data handling)
- **scikit-learn** (TF-IDF, ML models, metrics)
- **matplotlib** (plots)
- **wordcloud** (text visualisation)

---

## Methodology (Pipeline)

1. Load dataset and inspect basic statistics  
2. Split into training and testing sets  
3. Preprocess text  
   - lowercasing  
   - remove HTML tags  
   - remove punctuation/special characters  
   - remove extra spaces  
4. Feature extraction using **TF-IDF** (with tuned `max_features` and `ngram_range`)  
5. Train ML models (Logistic Regression and SVM)  
6. Evaluate using metrics and visualisations  
7. Demonstrate predictions on unseen sentences (“model in action”)

---

## Evaluation Metrics and Visualisations

- Accuracy, Precision, Recall, F1-score  
- Confusion Matrix  
- ROC Curve (Logistic Regression only)  
- Precision–Recall Curve (Logistic Regression only)  
- Learning Curves (to analyse overfitting/underfitting)  
- Top Important Words (feature weights)  
- Word Cloud (dataset overview)

> Note: ROC and Precision–Recall curves were not generated for **LinearSVC** because it does not output probabilities by default.


