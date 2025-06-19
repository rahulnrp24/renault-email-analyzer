import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
# ---- Add this right after your imports ----
from PIL import Image
import requests
from io import BytesIO

def generate_suggestions(text, negativity, clarity):
    """Generates AI-powered improvement suggestions"""
    suggestions = []
    
    # Negativity fixes
    if negativity > 70:
        suggestions.append("🔴 **High Negativity**: Consider rewording strong language. "
                         f"Try: 'Please review {text.split()[0]} for improvement'")
    
    if clarity < 60:
        vague_terms = ["maybe", "later", "unsure", "perhaps"]
        found = [term for term in vague_terms if term in text.lower()]
        if found:
            suggestions.append(f"🟡 **Clarity Issue**: Replace vague terms ({', '.join(found)}) "
                             "with specific deadlines/actions")
    
    # Constructive feedback template
    if any(x in text.lower() for x in ["failed", "wrong", "bad"]):
        suggestions.append("💡 **Constructive Alternative**: "
                         "Try: 'Let's improve this together by...'")
    
    return suggestions if suggestions else ["✅ No major issues detected"]

# Your image URL (upload to Imgur/GitHub first)
LOGO_URL = "https://rntbci.in/images/rntbci-logo.svg"  # ← Replace with your image URL

# --- Configure App ---
st.image(LOGO_URL, width=200)
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


    
st.subheader(f"Analysis: {row['Sender']}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Negativity", f"{negativity}%")
        st.metric("Clarity", f"{clarity}%")
    
    with col2:
        st.metric("Suggested Action", smiley)  # Your existing smiley
    
    # NEW: Display suggestions
    suggestions = generate_suggestions(text, negativity, clarity)
    with st.expander("✏️ Improvement Suggestions", expanded=True):
        for suggestion in suggestions:
            st.write(suggestion)
    
    st.markdown("---")  # Separator
    
    with st.expander("🛠️ **Improvement Recommendations**", expanded=True):
        # Sender-Focused Fixes
        st.markdown("**✍️ Sender Should:**")
        for suggestion in suggestions['sender']:
            st.write(f"- {suggestion}")
        
        # Receiver-Focused Fixes
        st.markdown("**📩 Receiver Should:**")
        for suggestion in suggestions['receiver']:
            st.write(f"- {suggestion}")
    
    st.markdown("---")  # Visual separator

# File uploader
uploaded_file = st.file_uploader("Upload Email CSV", type=["csv"])

if uploaded_file:
    emails = pd.read_csv(uploaded_file)
    
    # Ensure required columns exist
    if all(col in emails.columns for col in ['Email', 'Sender', 'Date']):
        
        # --- Scale A: Individual Mail Analysis ---
        st.subheader("📧 Per-Mail Analysis")
        emails[['Negativity%', 'Positivity%', 'Mood']] = emails['Email'].apply(
            lambda x: pd.Series(analyze_mail(x)))  # Removed extra parenthesis
        
        # --- Scale B: Sender-Receiver Pairs ---
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
