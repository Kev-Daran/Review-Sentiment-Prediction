# Sentiment Analysis on Large-Scale Text Datasets using Traditional and Transformer-Based Models

This project explores the performance of both traditional NLP methods (TF-IDF + Logistic Regression) and modern transformer-based embeddings (DistilBERT) for sentiment classification on large-scale datasets. It emphasizes the challenges and design trade-offs of handling large data using distributed systems like PySpark and embedding-heavy models in memory-constrained environments.

---

# Recommended Environment

This project was developed and tested on:
1. Databricks Runtime 13.3 LTS ML (Apache Spark 3.4, Scala 2.12)
2. Python 3.10+
3. Transformers ≥ 4.37
   
For best results and compatibility, it's strongly recommended to run this project in the same environment or equivalent PySpark setup with MLlib support.

---

# Datasets Used

Rotten Tomatoes Movie Reviews:

~1.4M labeled reviews (positive/negative)

Link: https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews/data


Amazon Alexa Reviews:

3.1K reviews sampled for domain transfer evaluation

Link: https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews/data

---

# Models

TF-IDF + Logistic Regression:

Lightweight, efficient, works well with distributed data (PySpark)


DistilBERT Embeddings + Logistic Regression:

Provides contextual understanding but computationally intensive

---

# Reproducibility

1. Clone this repo.
2. To use the saved pipeline model:

```
from pyspark.ml import PipelineModel
model = PipelineModel.load("/dbfs/path/to/my_lr_model")
predictions = model.transform(test_data)
```

3. To use the saved pandas model:

```
import joblib
model = joblib.load("/path/to/bert_logistic_regression.pkl")
y_pred = model.predict(X_test)
```

4. To use the saved DistilBERT embeddings:

On BASH
```
unzip embeddings.zip -d embeddings/
```

Load embeddings
```
import pandas as pd

# Load Rotten Tomatoes BERT embeddings
rotten_df = pd.read_csv("embeddings/final_bert_embeddings.csv")

# Load Alexa BERT embeddings
alexa_df = pd.read_csv("embeddings/alexa_bert_embeddings.csv")
```
---

# Results

| Dataset         | Model                        | Accuracy | Precision | Recall | F1 Score |
| --------------- | ---------------------------- | -------- | --------- | ------ | -------- |
| Rotten Tomatoes | TF-IDF + Logistic Regression | 0.8499   | 0.8489    | 0.8499 | 0.8494   |
| Rotten Tomatoes | BERT + Logistic Regression   | 0.8000   | 0.8285    | 0.8840 | 0.8553   |
| Alexa Reviews   | TF-IDF + Logistic Regression | 0.7108   | 0.8835    | 0.7108 | 0.7877   |
| Alexa Reviews   | BERT + Logistic Regression   | 0.8490   | 0.9723    | 0.8602 | 0.9127   |
