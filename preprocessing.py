from main import *
import pandas as pd
import string
import re
# import nltk
# nltk.download('stopwords')
from textblob import TextBlob
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
import demoji

#stopword = nltk.corpus.stopwords.words(stopword_lang)

def clean_df(df,language):
    df = df[['id', 'username', 'tweet', 'language']].copy()
    df = df[['id', 'username', 'tweet', 'language']].drop_duplicates()
    df['language'] = df['language'] == language
    df = df[df.language != False]
    df = df[['username', 'tweet']].copy()
    return df

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

#def remove_stopwords(text):
#    text = [word for word in text if word not in stopword]
#    return text

def convert_emoji(tweet):
    dict_emojis = demoji.findall(tweet)
    for emoji, emoji_word in dict_emojis.items():
        tweet = tweet.replace(emoji, f' {emoji_word}')
    return tweet

def text_blob(df):
    df['sentiment']=df['tweet'].apply(lambda x:TextBlob(x).sentiment[0])
    df['subject']=df['tweet'].apply(lambda x: TextBlob(x).sentiment[1])
    df['polarity']=df['sentiment'].apply(lambda x: 'pos' if x>=0 else 'neg')
    return df


# Create a function to compute negative (-1), neutral (0) and positive (+1) analysis
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
