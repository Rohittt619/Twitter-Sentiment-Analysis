import joblib

# ==========================================
# Load Saved Model & TF-IDF Vectorizer
# ==========================================

try:
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
except FileNotFoundError:
    print("❌ Model or Vectorizer not found!")
    print("Run train_model.py first.")
    exit()

# ==========================================
# Prediction Function
# ==========================================

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    return prediction

# ==========================================
# Main Program
# ==========================================

print("=" * 55)
print("        Twitter Sentiment Analysis")
print("=" * 55)
print("Type a tweet below.")
print("Type 'exit' anytime to quit.")
print("=" * 55)

while True:

    tweet = input("\nEnter Tweet: ").strip()

    if tweet.lower() == "exit":
        print("\n👋 Thanks for using Twitter Sentiment Analysis!")
        break

    if tweet == "":
        print("⚠ Please enter a valid tweet.")
        continue

    sentiment = predict_sentiment(tweet)

    if sentiment == "Positive":
        emoji = "😊"

    elif sentiment == "Negative":
        emoji = "😡"

    elif sentiment == "Neutral":
        emoji = "😐"

    else:
        emoji = "❓"

    print("\nPrediction")
    print("-" * 30)
    print(f"Tweet     : {tweet}")
    print(f"Sentiment : {emoji} {sentiment}")
    print("-" * 30)