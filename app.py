import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from PIL import Image
import io
import datetime
import json
import requests
import subprocess
import os # NEW: Import the os module to access environment variables

# --- NLTK Initialization (Cached for performance) ---
@st.cache_resource
def download_nltk_data():
    """
    Downloads the VADER lexicon for NLTK sentiment analysis.
    Uses quiet=True to suppress console output if already downloaded.
    """
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK vader_lexicon: {e}. Please check your internet connection or try again.")
        st.stop()
    return SentimentIntensityAnalyzer()

sia = download_nltk_data()

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
    /* Main background color */
    .main { background-color: #f5f5f5; }
    /* Padding for the main content area */
    .st-emotion-cache-1y4p8pa { padding: 2rem; }
    /* Header color for branding */
    .header { color: #003087; }
    /* Styling for metric cards */
    .metric-card { 
        border-radius: 10px; 
        padding: 15px; 
        background: white; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        text-align: center;
        margin-bottom: 10px;
    }
    /* Specific colors for sentiment */
    .negative { color: #d32f2f; }
    .positive { color: #388e3c; }
    /* Styling for blockquotes/suggestions */
    .st-bq { border-left: 5px solid #003087; }
    /* Ensure image in header is aligned */
    .st-emotion-cache-1v0mbgd {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    /* Adjust Streamlit radio button spacing */
    .st-emotion-cache-10qgysn {
        gap: 0.5rem;
    }
    /* Center the footer text */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8em;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Branding Header ---
col1, col2 = st.columns([1, 4])
with col1:
    # Updated logo using the provided image file content URL
    # The URL 'http://googleusercontent.com/file_content/2' refers to the new logo you uploaded (image_f28bbe.png)
    st.image("http://googleusercontent.com/file_content/2", width=120, caption="Renault-Nissan Logo")
with col2:
    st.title("Renault-Nissan Email Sentiment Intelligence")
    st.markdown("An AI-powered tool for analyzing email sentiment and clarity.")

# --- Authentication (Demo Version - Not for Production) ---
password = st.sidebar.text_input("Enter Access Key:", type="password")
if password != "RNT2025":
    st.sidebar.error("Invalid credentials")
    st.stop()

# --- Navigation ---
page = st.sidebar.radio("Menu", ["📊 Dashboard", "🔍 Analyze"])

# --- Sample Data ---
SAMPLE_DATA = {
    "Date": [datetime.date.today() - datetime.timedelta(days=i) for i in range(5)],
    "Sender": ["manager@renault.com", "team@nissan.com", "ceo@nissan.com", "hr@renault.com", "supplier@external.com"],
    "Email": [
        "The prototype failed safety tests. This needs urgent revision. The results were disappointing and unclear.",
        "Please review the latest test results by EOD Friday. The report is attached.",
        "Excellent progress on the EV design! Team recognition needed. Your dedication is highly appreciated.",
        "Regarding your recent inquiry, we need more clarity on the proposed solution. It's quite vague.",
        "Your last delivery was late and incomplete. This is unacceptable."
    ]
}

sample_df = pd.DataFrame(SAMPLE_DATA)
sample_df['Date'] = pd.to_datetime(sample_df['Date'])

# --- Core Analysis Functions ---

def get_smiley_rating(score):
    if score >= 60:
        return "😊 Excellent"
    elif score >= 20:
        return "🙂 Good"
    elif score >= -20:
        return "😐 Neutral"
    elif score >= -60:
        return "🙁 Concerning"
    else:
        return "😠 Critical"

def calculate_readability(text):
    words = text.split()
    num_words = len(words)
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    
    if num_sentences == 0:
        return 100
    
    avg_words_per_sentence = num_words / num_sentences
    clarity_score = max(0, 100 - (avg_words_per_sentence * 5))
    return round(clarity_score, 1)

def analyze_email(text):
    scores = sia.polarity_scores(text)
    negativity = round(scores['neg'] * 100, 1)
    positivity = round(scores['pos'] * 100, 1)
    overall_sentiment_score = round(scores['compound'] * 100, 1)
    clarity_score = calculate_readability(text)

    return {
        "negativity": negativity,
        "positivity": positivity,
        "overall_sentiment_score": overall_sentiment_score,
        "rating": get_smiley_rating(overall_sentiment_score),
        "clarity_score": clarity_score
    }

@st.cache_data(show_spinner="Generating AI suggestions...")
def generate_llm_suggestions(email_text, sentiment_analysis):
    prompt = f"""
    Analyze the following email for its sentiment and clarity.
    Email: "{email_text}"
    
    Sentiment Analysis:
    - Negativity: {sentiment_analysis['negativity']}%
    - Positivity: {sentiment_analysis['positivity']}%
    - Overall Sentiment Score: {sentiment_analysis['overall_sentiment_score']} (Range -100 to 100)
    - Clarity Score: {sentiment_analysis['clarity_score']} (Higher is clearer, 0-100)
    
    Based on this analysis, provide concise, actionable suggestions for both the sender and the receiver
    to improve communication, especially if there's high negativity or low clarity.
    
    Format your response as a JSON object with two keys: "sender_suggestions" and "receiver_suggestions".
    Each key should contain a list of strings.
    
    Example JSON format:
    {{
        "sender_suggestions": ["Rephrase 'failed' to 'opportunity for improvement'.", "Add a clear call to action."],
        "receiver_suggestions": ["Ask for specific details on the revision needed.", "Propose a follow-up meeting."]
    }}
    """

    chatHistory = []
    chatHistory.append({"role": "user", "parts": [{"text": prompt}]})
    
    # IMPORTANT: If running locally, you MUST set an environment variable named "API_KEY"
    # with your Google Gemini API key. Example (Windows Command Prompt): set API_KEY="YOUR_KEY_HERE"
    # If running in a Canvas environment, the platform automatically provides the key if apiKey is empty.
    apiKey = os.getenv("API_KEY", "") # Reads API key from environment variable, defaults to empty string
    
    if not apiKey:
        st.warning("API Key not found. AI suggestions will not work. Please set the 'API_KEY' environment variable.")
        return {"sender_suggestions": ["API Key is missing.", "Please configure your environment variable."], "receiver_suggestions": ["API Key is missing.", "Please configure your environment variable."]}

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apiKey}"

    payload = {
        "contents": chatHistory,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "sender_suggestions": { "type": "ARRAY", "items": { "type": "STRING" } },
                    "receiver_suggestions": { "type": "ARRAY", "items": { "type": "STRING" } }
                },
                "propertyOrdering": ["sender_suggestions", "receiver_suggestions"]
            }
        }
    }

    try:
        response = requests.post(apiUrl, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        
        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            json_string = result['candidates'][0]['content']['parts'][0]['text']
            suggestions = json.loads(json_string)
            return suggestions
        else:
            st.warning("LLM response structure unexpected. Using default suggestions.")
            return {"sender_suggestions": ["Could not generate AI suggestions.", "Check API response."], "receiver_suggestions": ["Could not generate AI suggestions.", "Check API response."]}
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}. Ensure your API key is correctly configured or if running locally, check network.")
        return {"sender_suggestions": ["Error contacting AI service.", "Check network/API key."], "receiver_suggestions": ["Error contacting AI service.", "Check network/API key."]}
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON from AI response: {e}. Using default suggestions.")
        return {"sender_suggestions": ["AI response not valid JSON.", "Try again."], "receiver_suggestions": ["AI response not valid JSON.", "Try again."]}
    except Exception as e:
        st.error(f"An unexpected error occurred during AI suggestion generation: {e}. Using default suggestions.")
        return {"sender_suggestions": ["An unknown error occurred.", "Please try again."], "receiver_suggestions": ["An unknown error occurred.", "Please try again."]}


# --- Page: Email Analysis (Existing) ---
if page == "🔍 Analyze":
    st.header("Email Sentiment & Clarity Analysis")
    
    uploaded_file = st.file_uploader("Upload Email CSV", type=["csv"], 
                                     help="Requires columns: 'Date', 'Sender', 'Email'")
    
    if uploaded_file:
        try:
            emails_df = pd.read_csv(uploaded_file)
            emails_df['Date'] = pd.to_datetime(emails_df['Date'])
            if not all(col in emails_df.columns for col in ['Date', 'Sender', 'Email']):
                st.error("Uploaded CSV is missing required columns ('Date', 'Sender', 'Email'). Using sample data.")
                emails_df = sample_df
        except Exception as e:
            st.error(f"Error reading file: {str(e)}. Using sample data.")
            emails_df = sample_df
    else:
        emails_df = sample_df
        st.info("Using sample data. Upload a CSV to analyze your emails.")

    min_date_available = emails_df['Date'].min().date()
    max_date_available = emails_df['Date'].max().date()

    date_range = st.date_input(
        "Select Analysis Period:", 
        value=(min_date_available, max_date_available),
        min_value=min_date_available,
        max_value=max_date_available
    )
    
    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        filtered_emails_df = emails_df[(emails_df['Date'] >= start_date) & (emails_df['Date'] <= end_date)]
        if filtered_emails_df.empty:
            st.warning("No emails found for the selected date range. Displaying all available emails.")
            emails_to_process = emails_df
        else:
            emails_to_process = filtered_emails_df
    else:
        emails_to_process = emails_df

    st.subheader(f"Analyzing {len(emails_to_process)} Emails")

    for index, row in emails_to_process.iterrows():
        email_text = row['Email']
        analysis = analyze_email(email_text)
        
        llm_suggestions = generate_llm_suggestions(email_text, analysis)
        
        with st.container():
            st.markdown(f"### ✉️ Email from {row['Sender']} on {row['Date'].strftime('%Y-%m-%d')}")
            st.markdown(f"> *{email_text}*")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card">'
                            f'<h3 class="negative">Negativity</h3>'
                            f'<h2>{analysis["negativity"]}%</h2></div>', 
                            unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card">'
                            f'<h3 class="positive">Positivity</h3>'
                            f'<h2>{analysis["positivity"]}%</h2></div>', 
                            unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card">'
                            f'<h3>Overall Sentiment</h3>'
                            f'<h2>{analysis["rating"]}</h2></div>', 
                            unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card">'
                            f'<h3>Clarity Score</h3>'
                            f'<h2>{analysis["clarity_score"]}/100</h2></div>', 
                            unsafe_allow_html=True)
            
            with st.expander("✏️ AI-Powered Improvement Recommendations", expanded=True):
                tab1, tab2 = st.tabs(["For Sender", "For Receiver"])
                
                with tab1:
                    if llm_suggestions.get('sender_suggestions'):
                        for suggestion in llm_suggestions['sender_suggestions']:
                            st.write(f"- {suggestion}")
                    else:
                        st.info("No specific sender suggestions generated by AI for this email.")
                
                with tab2:
                    if llm_suggestions.get('receiver_suggestions'):
                        for suggestion in llm_suggestions['receiver_suggestions']:
                            st.write(f"- {suggestion}")
                    else:
                        st.info("No specific receiver suggestions generated by AI for this email.")
            
            st.markdown("---")

# --- Page: Dashboard (Existing) ---
elif page == "📊 Dashboard":
    st.header("Executive Dashboard: Overall Email Trends")
    st.write("This dashboard provides an aggregated view of email sentiment and clarity.")

    dashboard_df = sample_df.copy()

    if not dashboard_df.empty:
        dashboard_df['negativity'] = dashboard_df['Email'].apply(lambda x: analyze_email(x)['negativity'])
        dashboard_df['positivity'] = dashboard_df['Email'].apply(lambda x: analyze_email(x)['positivity'])
        dashboard_df['overall_sentiment_score'] = dashboard_df['Email'].apply(lambda x: analyze_email(x)['overall_sentiment_score'])
        dashboard_df['clarity_score'] = dashboard_df['Email'].apply(lambda x: analyze_email(x)['clarity_score'])

        avg_negativity = dashboard_df['negativity'].mean()
        avg_positivity = dashboard_df['positivity'].mean()
        avg_overall_sentiment = dashboard_df['overall_sentiment_score'].mean()
        avg_clarity = dashboard_df['clarity_score'].mean()

        st.subheader("Overall Metrics")
        col_dash1, col_dash2, col_dash3, col_dash4 = st.columns(4)
        with col_dash1:
            st.markdown(f'<div class="metric-card">'
                        f'<h3 class="negative">Avg. Negativity</h3>'
                        f'<h2>{avg_negativity:.1f}%</h2></div>', unsafe_allow_html=True)
        with col_dash2:
            st.markdown(f'<div class="metric-card">'
                        f'<h3 class="positive">Avg. Positivity</h3>'
                        f'<h2>{avg_positivity:.1f}%</h2></div>', unsafe_allow_html=True)
        with col_dash3:
            st.markdown(f'<div class="metric-card">'
                        f'<h3>Avg. Sentiment</h3>'
                        f'<h2>{get_smiley_rating(avg_overall_sentiment)}</h2></div>', unsafe_allow_html=True)
        with col_dash4:
            st.markdown(f'<div class="metric-card">'
                        f'<h3>Avg. Clarity</h3>'
                        f'<h2>{avg_clarity:.1f}/100</h2></div>', unsafe_allow_html=True)

        st.subheader("Sentiment Trend Over Time")
        daily_sentiment = dashboard_df.groupby(dashboard_df['Date'].dt.date)['overall_sentiment_score'].mean().reset_index()
        daily_sentiment.columns = ['Date', 'Average Sentiment Score']
        st.line_chart(daily_sentiment.set_index('Date'))

        st.subheader("Sentiment by Sender")
        sender_sentiment = dashboard_df.groupby('Sender')['overall_sentiment_score'].mean().reset_index()
        sender_sentiment.columns = ['Sender', 'Average Sentiment Score']
        st.bar_chart(sender_sentiment.set_index('Sender'))

        st.subheader("Clarity by Sender")
        sender_clarity = dashboard_df.groupby('Sender')['clarity_score'].mean().reset_index()
        sender_clarity.columns = ['Sender', 'Average Clarity Score']
        st.bar_chart(sender_clarity.set_index('Sender'))

    else:
        st.info("No data available to display dashboard metrics. Please upload a CSV or use sample data.")
    
# --- Page: Reports ---
elif page == "📈 Reports":
    st.header("Custom Reports & Advanced Analytics")
    st.write("This section will allow for generating custom reports based on various filters and criteria.")
    
    st.subheader("Coming Soon:")
    st.markdown("""
    - **Detailed Email Logs:** View all analyzed emails with their scores in a searchable table.
    - **Custom Filtering:** Filter by sender, receiver, date ranges, sentiment thresholds.
    - **Export Options:** Export reports to CSV or PDF.
    - **Trend Analysis:** More advanced visualizations for long-term trends.
    """)

# --- Footer ---
st.markdown("---")
st.markdown('<div class="footer">'
            '© 2025 Renault-Nissan Alliance | v2.0 | AI Email Analytics'
            '</div>', unsafe_allow_html=True)
