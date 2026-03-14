import streamlit as st
from transformers import pipeline
from groq import Groq
import os
from dotenv import load_dotenv

# Cargar modelos
@st.cache_resource
def load_models():
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True,
        max_length=512
    )
    load_dotenv()
    client = Groq(api_key=os.getenv("groq_api_key"))
    return classifier, client

classifier, client = load_models()

# Interfaz
st.title("🍽️ Restaurant Review Analyzer")
st.write("Analyze customer reviews and get actionable business insights powered by AI.")

review = st.text_area("Paste a customer review here:", height=150)

if st.button("Analyze"):
    if review:
        with st.spinner("Analyzing..."):
            result = classifier(review, truncation=True, max_length=512)
            sentiment = result[0]['label']
            score = result[0]['score']

            prompt = f"""
            A customer left this review: "{review}"
            Sentiment: {sentiment} ({score:.0%} confidence)
            In 2-3 sentences, explain why and give one specific business recommendation.
            """

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}]
            )

            col1, col2 = st.columns(2)
            with col1:
                color = "🟢" if sentiment == "POSITIVE" else "🔴"
                st.metric("Sentiment", f"{color} {sentiment}")
            with col2:
                st.metric("Confidence", f"{score:.0%}")

            st.subheader("AI Business Insight")
            st.write(response.choices[0].message.content)
    else:
        st.warning("Please enter a review first.")