# 🐦 Twitter Sentiment Analysis Dashboard

An end-to-end **Natural Language Processing (NLP)** and **Machine Learning** project that classifies tweets into **Positive**, **Neutral**, or **Negative** sentiments. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and an interactive **Streamlit dashboard** for real-time sentiment prediction.

---

# 📸 Project Preview

## 🖥️ Dashboard

![Dashboard](https://raw.githubusercontent.com/Rohittt619/Twitter-Sentiment-Analysis/main/outputs/screenshots/dashboard.png)

---

## ✍️ Live Tweet Prediction

![Prediction](screenshots/prediction.png)

---

## 📊 Dataset Analytics

![Analytics](screenshots/analytics.png)

---

# 🚀 Features

- ✅ Tweet Text Preprocessing
- ✅ Natural Language Processing (NLP)
- ✅ TF-IDF Vectorization
- ✅ Multiple Machine Learning Models
- ✅ Automatic Best Model Selection
- ✅ Interactive Streamlit Dashboard
- ✅ Real-Time Sentiment Prediction
- ✅ Prediction History
- ✅ Interactive Plotly Visualizations
- ✅ Model Performance Comparison

---

# 📂 Project Structure

```text
Twitter_Sentiment_Analysis/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── twitter_sentiment.csv
│   │
│   └── processed/
│       └── tweets_cleaned.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── outputs/
│   ├── model_results.csv
│   │
│   └── figures/
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       ├── top_positive_words.png
│       ├── top_negative_words.png
│       ├── wordcloud_positive.png
│       └── wordcloud_negative.png
│
├── screenshots/
│   ├── dashboard.png
│   ├── prediction.png
│   └── analytics.png
│
└── scripts/
    ├── preprocessing.py
    ├── eda.py
    ├── train_model.py
    └── predict.py
```

---

# 📊 Dataset

The project uses a Twitter sentiment dataset containing thousands of labeled tweets.

### Sentiment Classes

| Label | Sentiment |
|--------|-----------|
| 😊 Positive | 1 |
| 😐 Neutral | 0 |
| 😡 Negative | -1 |

---

# 🧠 Project Workflow

```text
Twitter Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Text Preprocessing
       │
       ▼
TF-IDF Vectorization
       │
       ▼
Train-Test Split
       │
       ▼
Machine Learning Models
       │
       ▼
Model Evaluation
       │
       ▼
Best Model Selection
       │
       ▼
Interactive Streamlit Dashboard
```

---

# 🤖 Machine Learning Models

The following classification models were trained and evaluated:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Multinomial Naive Bayes

The best-performing model is automatically selected and saved for deployment in the Streamlit application.

---

# 📈 Exploratory Data Analysis

The project includes the following visualizations:

- 📊 Sentiment Distribution
- ☁️ Positive Word Cloud
- ☁️ Negative Word Cloud
- 🔥 Top Positive Words
- 🔥 Top Negative Words
- 📉 Confusion Matrix
- 📋 Model Performance Comparison

---

# 💻 Streamlit Dashboard

The interactive dashboard provides:

- 📝 Live Tweet Sentiment Prediction
- 📊 Dataset Overview
- 📈 Interactive Charts
- 🏆 Best Model & Accuracy
- 📜 Prediction History
- 👨‍💻 Author Information

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Plotly
- Streamlit
- Matplotlib
- WordCloud

---

# ▶️ Installation

### Clone the repository

```bash
git clone https://github.com/Rohittt619/Twitter-Sentiment-Analysis.git
```

### Navigate to the project

```bash
cd Twitter-Sentiment-Analysis
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 📊 Sample Predictions

### Input

```text
I absolutely love ChatGPT!
```

### Prediction

```text
😊 Positive
```

---

### Input

```text
Worst customer service ever.
```

### Prediction

```text
😡 Negative
```

---

### Input

```text
The meeting starts at 10 AM.
```

### Prediction

```text
😐 Neutral
```

---

# 🚀 Future Enhancements

- 🤖 BERT-based Sentiment Analysis
- 🧠 Deep Learning (LSTM)
- ☁️ Twitter API Integration
- 🐳 Docker Deployment
- 🌐 Cloud Deployment
- 📊 Explainable AI (SHAP/LIME)

---

# 👨‍💻 Author

## Rohit Rathod

**Aspiring Data Analyst | Machine Learning Enthusiast | Python Developer**

🔗 **GitHub:**  
https://github.com/Rohittt619

💼 **LinkedIn:**  
https://www.linkedin.com/in/rohit-rathod-19442a228/

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Feedback and contributions are always welcome!
