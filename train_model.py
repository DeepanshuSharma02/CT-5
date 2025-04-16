import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Use ImbPipeline for SMOTE
import joblib

from custom_transformers import DenseTransformer  # ✅ Import it here

# Load dataset
df = pd.read_csv("legal_text_classification.csv")
df = df.dropna(subset=["case_text", "case_outcome"])
X = df["case_text"]
y = df["case_outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with PCA and SMOTE
pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('dense', DenseTransformer()),
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),  # Applying SMOTE
    ('pca', PCA(n_components=100)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "legal_classifier_smote.pkl")
print("Model saved to legal_classifier_smote.pkl with SMOTE applied")
