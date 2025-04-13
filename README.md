# 📊 YouTube Comment Analysis

Analyze YouTube comments using natural language processing (NLP) to uncover sentiment, popular keywords, spam, and user behavior trends. This tool is useful for content creators, marketers, and researchers who want to gain insights from audience feedback.

---

## 🚀 Features

- 🔍 Extract comments from YouTube videos using the YouTube Data API
- 🧹 Clean and preprocess comment text (remove stopwords, emojis, links, etc.)
- 😊 Perform sentiment analysis (Positive, Negative, Neutral)
- ☁️ Generate word clouds and keyword frequency plots
- 🧠 Optional spam detection using ML models
- 📊 Visualize data using Matplotlib, Seaborn, and Plotly

---

## 🛠️ Tech Stack

- Python 3.x
- YouTube Data API v3
- Libraries: `pandas`, `numpy`, `nltk`, `vaderSentiment`, `matplotlib`, `wordcloud`, `seaborn`, `plotly`, etc.
- Jupyter Notebook / Streamlit (optional for interface)

---

## 📦 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/youtube-comment-analysis.git
   cd youtube-comment-analysis
## 📦 Installation

```bash
git clone https://github.com/your-username/youtube-comment-analysis.git
cd youtube-comment-analysis
pip install -r requirements.txt
 Set up API Key
You will need a YouTube Data API key.

Go to the Google Cloud Console

Create a project and enable the YouTube Data API v3

Generate an API key

Create a .env file or update config.py with your key:
YOUTUBE_API_KEY=your_api_key_here



