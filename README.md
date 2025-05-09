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


Amazon Alexa Reviews:

3.1K reviews sampled for domain transfer evaluation

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

3. To use the sabed pandas model:

```
import joblib
model = joblib.load("/path/to/bert_logistic_regression.pkl")
y_pred = model.predict(X_test)
```


