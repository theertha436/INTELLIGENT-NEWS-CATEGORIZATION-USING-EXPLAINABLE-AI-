import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# FILE PATHS
# ==========================================

TRUE_PATH = "data/True.csv"
FAKE_PATH = "data/Fake.csv"

MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/fake_news_vectorizer.pkl"


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading real news...")

true_df = pd.read_csv(TRUE_PATH)

print("Real articles:", len(true_df))


print("Loading fake news...")

fake_df = pd.read_csv(FAKE_PATH)

print("Fake articles:", len(fake_df))


# ==========================================
# 2. ADD LABELS
# ==========================================

true_df["label"] = 0
fake_df["label"] = 1


# ==========================================
# 3. COMBINE DATA
# ==========================================

df = pd.concat(
    [true_df, fake_df],
    ignore_index=True
)


print("\nTotal articles:", len(df))


# ==========================================
# 4. CREATE TEXT FIELD
# ==========================================

# Some versions contain both title and text.
# Combining them gives the model more information.

df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

df["content"] = (
    df["title"] + " " + df["text"]
)


# ==========================================
# 5. REMOVE EMPTY ARTICLES
# ==========================================

df = df[
    df["content"].str.strip() != ""
]


print(
    "Articles after cleaning:",
    len(df)
)


# ==========================================
# 6. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates(
    subset=["content"]
)


print(
    "Articles after removing duplicates:",
    len(df)
)


# ==========================================
# 7. INPUT / OUTPUT
# ==========================================

X = df["content"]
y = df["label"]


# ==========================================
# 8. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining articles:", len(X_train))
print("Testing articles:", len(X_test))


# ==========================================
# 9. TF-IDF
# ==========================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "TF-IDF training shape:",
    X_train_tfidf.shape
)


# ==========================================
# 10. TRAIN MODEL
# ==========================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)


# ==========================================
# 11. PREDICTION
# ==========================================

predictions = model.predict(
    X_test_tfidf
)


# ==========================================
# 12. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n==========================================")
print("       FAKE NEWS MODEL PERFORMANCE")
print("==========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["REAL", "FAKE"],
        zero_division=0
    )
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)


print("\n==========================================")
print("Model saved successfully!")
print(MODEL_PATH)

print("\nVectorizer saved successfully!")
print(VECTORIZER_PATH)

print("==========================================")