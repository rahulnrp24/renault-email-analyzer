import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
# Add to the top of app.py
st.image("https://www.renault.com/content/dam/renault/header/logo.png", width=200)
st.markdown("<h1 style='color: #003087;'>Renault-Nissan AI Email Scanner</h1>", unsafe_allow_html=True)

# --- App Config ---
st.set_page_config(page_title="Renault Email Analyzer", layout="wide")
st.title("🚗 Renault-Nissan Email Sentiment Analyzer")
st.write("Upload emails (CSV) to detect negativity and clarity issues.")

# --- Sample Data (Fallback) ---
sample_data = {
    "Email": [
        "The test results were unsatisfactory.",
        "Please review the attached specs by EOD.",
        "Great progress on the project!"
    ],
    "Sender": [
        "manager@renault.com",
        "team@nissan.com",
        "ceo@nissan.com"
    ]
}

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
data = pd.DataFrame(sample_data)  # Default to sample data
if uploaded_file:
    data = pd.read_csv(uploaded_file)

# --- Analysis Button ---
if st.button("🔍 Analyze Emails", type="primary"):  # <-- THIS TRIGGERS EVERYTHING
    # Initialize analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Process emails
    results = []
    for _, row in data.iterrows():
        text = row["Email"]
        
        # Sentiment Analysis
        sentiment = sia.polarity_scores(text)
        negativity = round(sentiment['neg'] * 100, 1)
        
        # Clarity Analysis (Simple Example)
        vague_terms = ["maybe", "later", "unsure", "perhaps"]
        vague_count = sum(text.lower().count(term) for term in vague_terms)
        clarity = max(0, 100 - vague_count * 20)  # Deduct 20% per vague term
        
        # Action Recommendation
        if negativity > 70:
            action = "🚨 High Negativity"
            color = "#FFCCCB"  # Light red
        elif clarity < 60:
            action = "💬 Needs Clarification"
            color = "#FFF4BD"  # Light yellow
        else:
            action = "✅ OK"
            color = "#C8E6C9"  # Light green
            
        results.append({
            "Email": text,
            "Sender": row["Sender"],
            "Negativity %": negativity,
            "Clarity %": clarity,
            "Action": action,
            "Color": color
        })

    # --- Display Results ---
    st.subheader("Analysis Results")
    
    # 1. Color-Coded Table
    results_df = pd.DataFrame(results)
    st.dataframe(
        results_df.style.apply(lambda x: [f"background-color: {x['Color']}"] * len(x), axis=1),
        hide_index=True,
        column_order=["Email", "Negativity %", "Clarity %", "Action"],
        height=300
    )
    
    # 2. Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(results_df, x="Sender", y="Negativity %")
    with col2:
        st.bar_chart(results_df, x="Sender", y="Clarity %")

    # 3. Download Button
    st.download_button(
        label="📥 Download Analysis",
        data=results_df.to_csv(index=False),
        file_name="email_analysis.csv"
    )
