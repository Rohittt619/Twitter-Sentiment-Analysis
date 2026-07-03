import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

# -----------------------------
# Create output folder
# -----------------------------
os.makedirs("outputs/figures", exist_ok=True)

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv("data/processed/tweets_cleaned.csv")

print(df.head())

# -----------------------------
# Sentiment Distribution
# -----------------------------
plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Sentiment",
    palette="viridis"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "outputs/figures/class_distribution.png",
    dpi=300
)

plt.close()

# -----------------------------
# Positive Words
# -----------------------------
positive_words = " ".join(
    df[df["Sentiment"]=="Positive"]["Clean_Text"]
).split()

positive_counts = Counter(positive_words).most_common(15)

pos_df = pd.DataFrame(
    positive_counts,
    columns=["Word","Count"]
)

plt.figure(figsize=(8,6))

sns.barplot(
    data=pos_df,
    x="Count",
    y="Word",
    palette="Greens_r"
)

plt.title("Top Positive Words")

plt.tight_layout()

plt.savefig(
    "outputs/figures/top_positive_words.png",
    dpi=300
)

plt.close()

# -----------------------------
# Negative Words
# -----------------------------
negative_words = " ".join(
    df[df["Sentiment"]=="Negative"]["Clean_Text"]
).split()

negative_counts = Counter(negative_words).most_common(15)

neg_df = pd.DataFrame(
    negative_counts,
    columns=["Word","Count"]
)

plt.figure(figsize=(8,6))

sns.barplot(
    data=neg_df,
    x="Count",
    y="Word",
    palette="Reds_r"
)

plt.title("Top Negative Words")

plt.tight_layout()

plt.savefig(
    "outputs/figures/top_negative_words.png",
    dpi=300
)

plt.close()

# -----------------------------
# Positive WordCloud
# -----------------------------
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(" ".join(positive_words))

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/figures/wordcloud_positive.png",
    dpi=300
)

plt.close()

# -----------------------------
# Negative WordCloud
# -----------------------------
wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(" ".join(negative_words))

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    "outputs/figures/wordcloud_negative.png",
    dpi=300
)

plt.close()

print("\n✅ EDA Completed Successfully!")
print("Figures saved inside outputs/figures/")