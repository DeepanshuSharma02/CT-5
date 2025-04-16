import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import joblib

# --- Optional: For scaling before PCA ---
from sklearn.preprocessing import StandardScaler

# Custom transformer to convert sparse matrix to dense
class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.toarray()

# Load dataset
df = pd.read_csv("legal_text_classification.csv")
df = df.dropna(subset=["case_text", "case_outcome"])
X = df["case_text"]
y = df["case_outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with TF-IDF -> Dense -> PCA -> LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('dense', DenseTransformer()),
    ('pca', PCA(n_components=100)),  # You can experiment with this number
    ('clf', LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, "legal_classifier.pkl")
print("Model saved to legal_classifier.pkl with PCA")
