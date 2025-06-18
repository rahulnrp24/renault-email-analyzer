import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Configure app
st.set_page_config(page_title="Renault-Nissan Email Analyzer", layout="wide")
nltk.download('vader_lexicon')

# Add Renault logo
st.image("https://www.renault.com/content/dam/renault/header/logo.png", width=200)

# Title
st.title("🚗 Renault-Nissan Email Sentiment Analyzer")
st.write("Upload emails (CSV) to detect negativity and clarity issues.")

# Sample data (fallback)
sample_data = {
    "Email": [
        "The test results were unsatisfactory.", 
        "Please review the specs by EOD.",
        "Great progress on the project!"
    ],
    "Sender": [
        "manager@renault.com",
        "team@nissan.com",
        "ceo@nissan.com"
    ]
}

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type="csv")
data = pd.DataFrame(sample_data)