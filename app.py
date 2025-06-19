import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from PIL import Image
import io
import datetime

# --- Initialize NLTK ---
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# --- App Configuration ---
st.set_page_config(
    page_title="Renault-Nissan Email AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Look ---
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .st-emotion-cache-1y4p8pa { padding: 2rem; }
    .header { color: #003087; }
    .metric-card { border-radius: 10px; padding: 15px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .negative { color: #d32f2f; }
    .positive { color: #388e3c; }
    .st-bq { border-left: 5px solid #003087; }
</style>
""", unsafe_allow_html=True)

# --- Branding Header ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://www.renault.com/content/dam/renault/header/logo.png", width=120)
with col2:
    st.title("Renault-Nissan Email Sentiment Intelligence")

# --- Authentication (Demo Version) ---
password = st.sidebar.text_input("Enter Access Key:", type="password")
if password != "RNT2024":
    st.sidebar.error("Invalid credentials")
    st.stop()

# --- Navigation ---
page = st.sidebar.radio("Menu", ["📊 Dashboard", "🔍 Analyze", "📈 Reports"])

# --- Sample Data ---
SAMPLE_DATA = {
    "Date": [datetime.date.today()] * 3,
    "Sender": ["manager@renault.com", "team@nissan.com", "ceo@nissan.com"],
    "Email": [
        "The prototype failed safety tests. This needs urgent revision.",
        "Please review the latest test results by EOD Friday.",
        "Excellent progress on the EV design! Team recognition needed."
    ]
}

# --- Core Analysis Functions ---
def get_smiley_rating(score):
    ratings = {
        80: "😊 Excellent",
        60: "🙂 Good",
        40: "😐 Neutral",
        20: "🙁 Concerning",
        0: "😠 Critical"
    }
    for threshold, label in ratings.items():
        if score >= threshold:
            return label
    return "😠 Critical"

def analyze_email(text):
    scores = sia.polarity_scores(text)
    negativity = round(scores['neg'] * 100, 1)
    positivity = round(scores['pos'] * 100, 1)
    return {
        "negativity": negativity,
        "positivity": positivity,
        "overall": positivity - negativity,
        "rating": get_smiley_rating(positivity - negativity)
    }

def generate_improvements(text, analysis):
    suggestions = {"sender": [], "receiver": []}
    
    # Sender improvements
    if analysis['negativity'] > 70:
        suggestions['sender'].append(
            f"🔴 Replace negative phrasing: Try 'Opportunity to improve {text.split()[0]}'"
        )
    
    if analysis['positivity'] < 30:
        suggestions['sender'].append(
            "💡 Add positive reinforcement: Recognize what's working well"
        )
    
    # Receiver responses
    if analysis['negativity'] > 50:
        suggestions['receiver'].append(
            "📌 Response template: 'What specific changes would you like to see?'"
        )
    
    return suggestions

# --- Page: Email Analysis ---
if page == "🔍 Analyze":
    st.header("Email Sentiment Analysis")
    
    # File Uploader with Validation
    uploaded_file = st.file_uploader("Upload Email CSV", type=["csv"], 
                                   help="Requires columns: Date, Sender, Email")
    
    if uploaded_file:
        try:
            emails = pd.read_csv(uploaded_file)
            if not all(col in emails.columns for col in ['Date', 'Sender', 'Email']):
                st.error("Missing required columns. Using sample data.")
                emails = pd.DataFrame(SAMPLE_DATA)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            emails = pd.DataFrame(SAMPLE_DATA)
    else:
        emails = pd.DataFrame(SAMPLE_DATA)
        st.info("Using sample data. Upload a CSV to analyze your emails.")

    # Date Filtering
    min_date = emails['Date'].min()
    max_date = emails['Date'].max()
    date_range = st.date_input("Analysis Period:", [min_date, max_date])
    
    # Process Emails
    results = []
    for _, row in emails.iterrows():
        analysis = analyze_email(row['Email'])
        suggestions = generate_improvements(row['Email'], analysis)
        
        results.append({
            **row,
            **analysis,
            "suggestions": suggestions
        })
    
    # Display Results
    for email in results:
        with st.container():
            st.markdown(f"### ✉️ {email['Sender']}")
            
            # Metrics Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">'
                           f'<h3 class="negative">Negativity</h3>'
                           f'<h2>{email["negativity"]}%</h2></div>', 
                           unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">'
                           f'<h3 class="positive">Positivity</h3>'
                           f'<h2>{email["positivity"]}%</h2></div>', 
                           unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">'
                           f'<h3>Overall Rating</h3>'
                           f'<h2>{email["rating"]}</h2></div>', 
                           unsafe_allow_html=True)
            
            # Suggestions
            with st.expander("✏️ Improvement Recommendations", expanded=True):
                tab1, tab2 = st.tabs(["For Sender", "For Receiver"])
                
                with tab1:
                    if email['suggestions']['sender']:
                        for suggestion in email['suggestions']['sender']:
                            st.write(f"- {suggestion}")
                    else:
                        st.success("No major improvements needed!")
                
                with tab2:
                    if email['suggestions']['receiver']:
                        for suggestion in email['suggestions']['receiver']:
                            st.write(f"- {suggestion}")
                    else:
                        st.info("Standard response appropriate")
            
            st.markdown("---")

# --- Page: Dashboard ---
elif page == "📊 Dashboard":
    st.header("Executive Dashboard")
    # Add KPIs, charts, etc. (placeholder)
    st.write("Under development - coming soon!")
    
# --- Page: Reports ---
elif page == "📈 Reports":
    st.header("Custom Reports")
    # Add reporting tools (placeholder)
    st.write("Under development - coming soon!")

# --- Footer ---
st.markdown("---")
st.markdown('<div style="text-align: center; color: #666;">'
            '© 2024 Renault-Nissan Alliance | v2.0 | AI Email Analytics'
            '</div>', unsafe_allow_html=True)
