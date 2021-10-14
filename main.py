import datetime as dt
import re
import pandas as pd
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier
import twint
import re
import string
import preprocessing
import nltk
from wordcloud import WordCloud
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
style.use('seaborn')

# Set page title
st.title('Twitter Sentiment Analysis')
st.write("""
# Explore different twitter hashtags!
Which one has the best popular sentiment?
""")

st.subheader('Type in a topic/movie/show/person...')

# Language
tweet_input = st.text_input('Tweet:')
selected_language = st.selectbox("Select the Tweet language", ("English", "French", "Dutch", "Spanish"))
if selected_language == 'English':
    language = "en"
    stopword_lang = "english"
elif selected_language == 'French':
    language = "fr"
    stopword_lang = "french"
elif selected_language == 'German':
    language = "de"
    stopword_lang = "german"
elif selected_language == 'Dutch':
    language = "nl"
    stopword_lang = "dutch"
elif selected_language == 'Spanish':
    language = "es"
    stopword_lang = "spanish"

# Number of tweet to scrape
number_tweets = st.slider("Tweets to scrape",100, 1000)

submit = st.button(" Let's Analyse!")
# Set up TWINT config
if submit:
    c = twint.Config()
    c.Search = tweet_input
    # Custom output format
    c.Lang = language
    c.Limit = number_tweets
    c.Pandas = True
    #c.Store_csv = True
    #c.Output = f"{tweet_input}.csv"
    twint.run.Search(c)
    #df = pd.read_csv(c.Output)
    df = twint.storage.panda.Tweets_df
    df = preprocessing.clean_df(df,language)
    #df['tweet'] = df['tweet'].apply(lambda x: preprocessing.remove_punct(x))
    #df['tweet'] = df['tweet'].apply(lambda x: preprocessing.remove_stopwords(x))
    #df['tweet'] = [','.join(map(str, l)) for l in df['tweet']]
    df['tweet'] = df['tweet'].apply(lambda x: preprocessing.convert_emoji(x))
    df['tweet'] = df['tweet'].apply(lambda x: preprocessing.remove_punct(x))
    df = preprocessing.text_blob(df)
    df['analysis'] = df['sentiment'].apply(preprocessing.getAnalysis)
    df = df.drop(['polarity'], axis = 1)

    st.title("Here are the scraped tweets")
    st.dataframe(df)

    st.subheader("Sentiment Coverage")
    st.write(df.analysis.value_counts())

    st.subheader("This plot shows the sentiment coverage per tweet")
    plt.hist(df['sentiment'], density=True, bins=30) # density=False would make counts
    plt.ylabel('tweets')
    plt.xlabel('sentiment')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # word cloud visualization
    st.subheader("This plot shows most common words in your search")
    allWords = ' '.join([twts for twts in df['tweet']])
    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    st.pyplot()

    #Word of goodbye
    st.subheader("Thank you and try again with another topic! :)")
