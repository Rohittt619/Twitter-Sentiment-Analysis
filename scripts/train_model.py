import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# Load data
df = pd.read_csv("data/processed/tweets_cleaned.csv")
df.dropna(subset=["Clean_Text", "Sentiment"], inplace=True)
df = df[df["Sentiment"] != "Irrelevant"]

X = df["Clean_Text"]
y = df["Sentiment"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(X)

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB()
}

results = []

best_model = None
best_pred = None
best_acc = 0

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(f"Accuracy: {acc:.4f}")

    results.append([name, acc])

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_pred = pred

# Save model
joblib.dump(best_model, "models/sentiment_model.pkl")

# Save results
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
results_df.sort_values("Accuracy", ascending=False, inplace=True)
results_df.to_csv("outputs/model_results.csv", index=False)

print("\nBest Model:")
print(results_df.iloc[0])

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix.png")
plt.close()

print("\nDone! Files Saved Successfully.")