import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# --- Configure App ---
st.set_page_config(page_title="Renault-Nissan AI Email Scanner", layout="wide")
st.image("https://www.renault.com/content/dam/renault/header/logo.png", width=200)

# --- Sentiment Scales ---
def get_smiley_scale(score):
    if score >= 80: return "😊"  # Very positive
    elif score >= 60: return "🙂"  # Positive
    elif score >= 40: return "😐"  # Neutral
    elif score >= 20: return "🙁"  # Negative
    else: return "😠"  # Very negative

def analyze_mail(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    negativity = round(sentiment['neg'] * 100, 1)
    positivity = round(sentiment['pos'] * 100, 1)
    smiley = get_smiley_scale(positivity - negativity)
    return negativity, positivity, smiley

# --- Main App ---
st.title("Renault-Nissan Email Sentiment Analyzer")

# Date range selector
date_range = st.date_input("Select analysis period:", [])

# File uploader
uploaded_file = st.file_uploader("Upload Email CSV", type=["csv"])

if uploaded_file:
    emails = pd.read_csv(uploaded_file)
    
    # Ensure required columns exist
    if all(col in emails.columns for col in ['Email', 'Sender', 'Date']):
        
        # Scale A: Individual Mail Analysis
        st.subheader("📧 Per-Mail Analysis")
        emails[['Negativity%', 'Positivity%', 'Mood']] = emails['Email'].apply(
            lambda x: pd.Series(analyze_mail(x))
        
        # Scale B: Sender-Receiver Pairs
        st.subheader("👥 Sender/Receiver Trends")
        sender_stats = emails.groupby('Sender').agg({
            'Negativity%': 'mean',
            'Positivity%': 'mean'
        }).reset_index()
        sender_stats['Mood'] = sender_stats.apply(
            lambda x: get_smiley_scale(x['Positivity%'] - x['Negativity%']), axis=1)
        
        # Scale C: Overall Period Analysis
        st.subheader("📅 Period Summary")
        overall = {
            'Avg Negativity': emails['Negativity%'].mean(),
            'Avg Positivity': emails['Positivity%'].mean(),
            'Overall Mood': get_smiley_scale(
                emails['Positivity%'].mean() - emails['Negativity%'].mean())
        }
        
        # Display all scales
        col1, col2, col3 = st.columns(3)
        with col1:
            st.dataframe(emails[['Sender', 'Email', 'Mood']].head(10))
        with col2:
            st.dataframe(sender_stats)
        with col3:
            st.metric("Overall Mood", overall['Overall Mood'])
        
        # Smiley visualization
        st.subheader("😊 Smiley Scale Guide")
        st.write("""
        | Score Range | Mood        | Smiley |
        |-------------|-------------|--------|
        | 80-100      | Very Positive | 😊     |
        | 60-79       | Positive    | 🙂     |
        | 40-59       | Neutral     | 😐     |
        | 20-39       | Negative    | 🙁     |
        | 0-19        | Very Negative | 😠     |
        """)
        
    else:
        st.error("CSV must contain 'Email', 'Sender', and 'Date' columns")
else:
    st.info("Please upload a CSV file to begin analysis")
