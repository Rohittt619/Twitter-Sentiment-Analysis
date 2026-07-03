import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# ======================================================
# LOAD FILES
# ======================================================

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

df = pd.read_csv("data/processed/tweets_cleaned.csv")
results = pd.read_csv("outputs/model_results.csv")

best_model = results.iloc[0]

# ======================================================
# SESSION HISTORY
# ======================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("🐦 Twitter Sentiment Analysis")

st.sidebar.markdown("---")

st.sidebar.success("### 🚀 Features")

st.sidebar.write("""
✅ Natural Language Processing

✅ Machine Learning

✅ TF-IDF Vectorizer

✅ Multiple ML Models

✅ Interactive Dashboard
""")

st.sidebar.markdown("---")

st.sidebar.info("""
### 💻 Tech Stack

- Python
- Pandas
- Scikit-Learn
- Plotly
- Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 👨‍💻 Author")

st.sidebar.write("**Rohit Rathod**")

st.sidebar.markdown(
    "[🐙 GitHub](https://github.com/Rohittt619)"
)

st.sidebar.markdown(
    "[💼 LinkedIn](https://linkedin.com/in/rohit-rathod-19442a228)"
)

# ======================================================
# TITLE
# ======================================================

st.title("🐦 Twitter Sentiment Analysis Dashboard")

st.success("🤖 AI Powered Tweet Sentiment Prediction using Machine Learning")

st.markdown("""
Analyze tweets using Machine Learning and predict whether the sentiment is:

- 😊 Positive
- 😐 Neutral
- 😡 Negative
""")

st.info("""
This dashboard predicts the sentiment of tweets using a trained
Machine Learning model based on TF-IDF features.
""")

st.divider()

# ======================================================
# METRICS
# ======================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric("🤖 Models", len(results))
m2.metric("😊 Classes", 3)
m3.metric("🏆 Best Model", best_model["Model"])
m4.metric("🎯 Accuracy", f"{best_model['Accuracy']:.2%}")

st.divider()

# ======================================================
# DATASET OVERVIEW
# ======================================================

st.subheader("📊 Dataset Overview")

c1, c2 = st.columns(2)

sentiment_count = df["Sentiment"].value_counts().reset_index()
sentiment_count.columns = ["Sentiment", "Count"]

with c1:

    fig = px.pie(
        sentiment_count,
        values="Count",
        names="Sentiment",
        hole=0.45,
        title="Sentiment Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

with c2:

    fig2 = px.bar(
        sentiment_count,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        text_auto=True,
        title="Tweets by Sentiment"
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ======================================================
# PREDICTION
# ======================================================

st.subheader("✍ Predict Tweet Sentiment")

tweet = st.text_area(
    "Enter Tweet",
    placeholder="Example: I absolutely love Cricket!"
)

if st.button(
    "🚀 Predict Sentiment",
    use_container_width=True
):

    if tweet.strip() == "":

        st.warning("⚠ Please enter a tweet.")

    else:

        vector = vectorizer.transform([tweet])

        prediction = model.predict(vector)[0]

        st.subheader("📌 Prediction Result")

        if prediction == "Positive":

            st.success("😊 Positive Sentiment")

        elif prediction == "Negative":

            st.error("😡 Negative Sentiment")

        else:

            st.info("😐 Neutral Sentiment")

        st.progress(100)

        st.toast("Prediction Completed Successfully 🎉")

        st.text_area(
            "Tweet Entered",
            tweet,
            disabled=True,
            height=120
        )

        st.session_state.history.append({
            "Tweet": tweet,
            "Prediction": prediction
        })

st.divider()

# ======================================================
# HISTORY
# ======================================================

if len(st.session_state.history) > 0:

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

st.divider()

# ======================================================
# FOOTER
# ======================================================

st.markdown(
"""
<div style='text-align:center;
padding:20px;
border-radius:10px;
background-color:#f8f9fa;'>

<h3>🐦 Twitter Sentiment Analysis Dashboard</h3>

<p>
Developed with ❤️ by <b>Rohit Rathod</b>
</p>

<p>
📊 Data Analytics | 🤖 Machine Learning | 💻 Python
</p>

<p style='color:gray;'>
© 2026 Rohit Rathod
</p>

</div>
""",
unsafe_allow_html=True
)