import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
import altair as alt
import urllib.parse
from googleapiclient.discovery import build
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from wordcloud import WordCloud
from tensorflow.keras.models import load_model

# Load model and tokenizer
model1 = load_model('sentiment_model_lstm.h5')
with open('tokenizer_lstm.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

# API key
api_key = 'AIzaSyB2Ghgt1QRgUG41Uvha0yZ9TIFPU7fyInw'  

def get_video_title(video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        if "items" in response and len(response["items"]) > 0:
            return response["items"][0]["snippet"]["title"]
        else:
            return f"Video ID: {video_id}"
    except Exception as e:
        st.error(f"Error fetching video title: {e}")
        return f"Video ID: {video_id}"

def fetch_video_ids_from_playlist(playlist_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_ids = []
    next_page_token = None
    try:
        while True:
            response = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            video_ids.extend([item['contentDetails']['videoId'] for item in response['items']])
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
    except Exception as e:
        st.error(f"Error fetching playlist videos: {e}")
    return video_ids

def extract_video_id(url):
    parsed_url = urllib.parse.urlparse(url)
    if "shorts" in parsed_url.path:  # Handle Shorts URL explicitly
        return [parsed_url.path.split("/")[-1]]
    elif parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = urllib.parse.parse_qs(parsed_url.query)
        if "list" in query_params:
            playlist_id = query_params.get("list", [None])[0]
            return fetch_video_ids_from_playlist(playlist_id) if playlist_id else []
        if "v" in query_params:
            return [query_params.get("v", [None])[0]]
    elif parsed_url.hostname in ["youtu.be"]:
        return [parsed_url.path.lstrip("/")]
    return []

def video_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments_data = []
    try:
        video_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100
        ).execute()
        while video_response:
            for item in video_response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                likes = item['snippet']['topLevelComment']['snippet']['likeCount']
                timestamp = item['snippet']['topLevelComment']['snippet']['publishedAt']
                author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                comments_data.append([author, comment, likes, timestamp])
            if 'nextPageToken' in video_response:
                video_response = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    pageToken=video_response['nextPageToken']
                ).execute()
            else:
                break
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
    return comments_data

def SentimentAnalysis1(text):
    sentence = [text]
    tokenized_sentence = tokenizer1.texts_to_sequences(sentence)
    input_sequence = pad_sequences(tokenized_sentence, maxlen=32, padding='pre')
    prediction_ = model1.predict(input_sequence)
    prediction = prediction_.argmax()
    return "Negative" if prediction == 0 else "Neutral" if prediction == 1 else "Positive"

def generate_wordcloud(comments):
    if not comments:
        return None
    text = ' '.join(comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

with st.sidebar:
    select = option_menu(
        "",
        ['Home', "Sentiment Analysis"],
        icons=['house-door', 'youtube'],
        menu_icon="cast",
        default_index=0
    )

if select == 'Home':
    st.title("YouTube Comments Sentiment Analysis")

elif select == 'Sentiment Analysis':
    url = st.text_input('Enter the URL of the video, playlist, or Shorts')
    if url:
        video_ids = extract_video_id(url)
        if not video_ids:
            st.warning("No valid videos found in the URL.")
        else:
            for video_id in video_ids:
                video_title = get_video_title(video_id)
                st.subheader(f"Sentiment Analysis for Video: {video_title}")
                comments = video_comments(video_id)
                if not comments:
                    st.warning(f"No comments found for video {video_title}.")
                else:
                    df = pd.DataFrame(comments, columns=['Author', 'Comment', 'Likes', 'Timestamp'])
                    df['Sentiment'] = df['Comment'].apply(SentimentAnalysis1)
                    st.write(df[['Comment', 'Sentiment', 'Likes', 'Author', 'Timestamp']])
                    
                    chart = alt.Chart(df).mark_bar().encode(
                        x='Sentiment',
                        y='count()',
                        color='Sentiment'
                    ).properties(title='Sentiment Distribution')
                    st.altair_chart(chart, use_container_width=True)
                    
                    wordcloud = generate_wordcloud(df['Comment'].tolist())
                    if wordcloud:
                        st.image(wordcloud.to_array(), use_column_width=True)
                    else:
                        st.warning("Not enough text data to generate a word cloud.")
