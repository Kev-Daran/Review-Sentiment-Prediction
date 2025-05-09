# Sentiment Analysis on Large-Scale Text Datasets using Traditional and Transformer-Based Models

This project explores the performance of both traditional NLP methods (TF-IDF + Logistic Regression) and modern transformer-based embeddings (DistilBERT) for sentiment classification on large-scale datasets. It emphasizes the challenges and design trade-offs of handling large data using distributed systems like PySpark and embedding-heavy models in memory-constrained environments.

---

# Recommended Environment

This project was developed and tested on Databricks Runtime 13.3 LTS ML (Apache Spark 3.4, Scala 2.12).
For best results and compatibility, it's strongly recommended to run this project in the same environment or equivalent PySpark setup with MLlib support.

---

# Datasets Used

Rotten Tomatoes Movie Reviews
~1.4M labeled reviews (positive/negative)

Amazon Alexa Reviews
3.1K reviews sampled for domain transfer evaluation
