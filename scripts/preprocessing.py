import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/twitter_sentiment.csv")

# Keep required columns
df = df[["clean_text", "category"]]

# Remove missing values
df.dropna(inplace=True)

# Rename columns
df.rename(columns={
    "clean_text": "Clean_Text",
    "category": "Sentiment"
}, inplace=True)

# Convert labels to text
label_map = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

df["Sentiment"] = df["Sentiment"].map(label_map)

# Save processed data
df.to_csv(
    "data/processed/tweets_cleaned.csv",
    index=False
)

print(df.head())
print(df["Sentiment"].value_counts())

print("\n✅ Preprocessing Completed Successfully!")