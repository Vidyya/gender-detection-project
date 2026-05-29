import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load the name dataset from the CSV file.
# The dataset has two columns: name and gender.
data = pd.read_csv("gender.csv")

# Convert the labels to lower case for consistency.
data["gender"] = data["gender"].str.lower()

# Use name text as the feature and gender as the target label.
X = data["name"]
y = data["gender"]

# Split the dataset into training and testing sets.
# This helps us measure how well the model will perform on new names.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert names into numeric feature vectors using character n-grams.
# This is useful for name-based gender prediction.
vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3))
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train a Naive Bayes classifier on the vectorized names.
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Predict on the test set and calculate accuracy.
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model training completed. Accuracy: {accuracy:.2%}")

# Save the trained model and vectorizer to disk so Flask can load them later.
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Saved model.pkl and vectorizer.pkl")
