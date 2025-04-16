import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("legal_text_classification.csv")  # Text + Label columns
df = df.dropna(subset=["case_text", "case_outcome"])
X = df["case_text"]      # legal document content
y = df["case_outcome"]     # label/category


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train
pipeline.fit(X_train, y_train)

# Save using joblib
joblib.dump(pipeline, "legal_classifier.pkl")
print("Model saved to legal_classifier.pkl")
