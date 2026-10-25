import joblib


# Load trained model
model = joblib.load(
    "models/fake_news_model.pkl"
)

# Load TF-IDF vectorizer
vectorizer = joblib.load(
    "models/fake_news_vectorizer.pkl"
)


# -----------------------------------
# TEST ARTICLE
# -----------------------------------

article = """
The Indian government announced a new education program
to provide scholarships to students across the country.
The program will begin next academic year and is expected
to support thousands of students.
"""


# Convert article into TF-IDF features
article_vector = vectorizer.transform(
    [article]
)


# Make prediction
prediction = model.predict(
    article_vector
)[0]


# Get probabilities
probabilities = model.predict_proba(
    article_vector
)[0]


# -----------------------------------
# DISPLAY RESULT
# -----------------------------------

if prediction == 0:

    result = "LIKELY REAL"
    confidence = probabilities[0]

else:

    result = "POTENTIALLY FAKE"
    confidence = probabilities[1]


print("--------------------------------")
print("       FAKE NEWS DETECTOR")
print("--------------------------------")

print("Prediction:", result)

print(
    f"Confidence: {confidence * 100:.2f}%"
)

print("--------------------------------")