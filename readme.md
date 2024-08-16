# Machine Learning Notebooks Collection

This repository contains a collection of Jupyter notebooks demonstrating various machine learning techniques and models. Each notebook is designed to provide a step-by-step guide to implementing different machine learning algorithms and techniques using Python libraries such as TensorFlow, Scikit-learn, and others.

## Table of Contents

1. [Sentiment Analysis Using LSTM](#sentiment-analysis-using-lstm)
2. [Topic Modeling with Latent Semantic Analysis](#topic-modeling-with-latent-semantic-analysis)
3. [K-Nearest Neighbors Classifier](#k-nearest-neighbors-classifier)
4. [Student Performance Analysis](#student-performance-analysis)

## Sentiment Analysis Using LSTM

**Notebook**: `SentimentAnalysisBasic.ipynb`

This notebook demonstrates how to build a sentiment analysis model using a Bidirectional LSTM network in TensorFlow.

- **Dataset**: IMDb movie reviews dataset.
- **Objective**: Classify movie reviews as positive or negative.
- **Model**:
  - Embedding Layer
  - Bidirectional LSTM Layer
  - Dense Output Layer with Sigmoid Activation
- **Training**: The model is trained for 3 epochs on the training dataset.
- **Evaluation**: Model accuracy and sample sentiment predictions are provided.

## Topic Modeling with Latent Semantic Analysis

**Notebook**: `topic_modelling.ipynb`

This notebook explores topic modeling using Latent Semantic Analysis (LSA) with Truncated SVD.

- **Data Source**: Wikipedia articles are scraped using `requests` and BeautifulSoup.
- **Objective**: Extract and analyze topics from text data.
- **Techniques**:
  - Text preprocessing (tokenization, stopword removal, text cleaning).
  - TF-IDF Vectorization.
  - Truncated SVD for dimensionality reduction.
- **Analysis**: The notebook includes steps to identify and interpret topics from the text data.

## K-Nearest Neighbors Classifier

**Notebook**: `k-nearest.ipynb`

This notebook provides a guide to implementing a K-Nearest Neighbors (KNN) classifier.

- **Datasets**:
  - Iris dataset (initially referenced).
  - `Pumpkin_Seeds_Dataset.xlsx` (used in later steps).
- **Objective**: Classify data points based on their features using the KNN algorithm.
- **Model**: KNN classifier implemented using Scikit-learn.
- **Evaluation**: Model accuracy is evaluated, and additional performance metrics may be included.

## Student Performance Analysis

**Notebook**: `Student_Performace.ipynb`

This notebook analyzes student performance data and applies a Linear Regression model.

- **Dataset**: `Student_Performance.csv`
- **Objective**: Analyze and predict student performance based on various features.
- **Data Exploration**:
  - Descriptive statistics and distribution visualizations.
- **Modeling**:
  - Linear Regression with K-Fold cross-validation.
  - Analysis of model performance and feature importance.
- **Visualization**: Data distributions and other relevant visualizations.

## Getting Started

### Prerequisites

To run these notebooks, you need to have the following libraries installed:

- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Requests
- BeautifulSoup

### Running the Notebooks

Clone the repository and navigate to the directory:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open any of the notebooks listed above to explore and run the code.

