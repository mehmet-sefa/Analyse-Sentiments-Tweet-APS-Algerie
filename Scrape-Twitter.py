import snscrape.modules.twitter as sntwitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import re

from mtranslate import translate
#from deep_translator import MyMemoryTranslator


maxTweets = 20000

# Creating list to append tweet data to
searched_tweets = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper('(from:APS_Algerie until:2024-01-01 since:2023-01-01)').get_items()):
    if i > maxTweets:
        break
    if tweet.lang == 'ar' or tweet.lang == 'fr':
        searched_tweets.append(tweet)

tweetlist=[]

for tweet in searched_tweets:
    translated_text = translate(tweet.rawContent, to_language='en')

    tweetlist.append([translated_text, tweet.rawContent, tweet.date, tweet.user.username, tweet.likeCount, tweet.retweetCount, tweet.viewCount, tweet.hashtags])
    #tweetlist.append([translated_text, tweet.date, tweet.user.username, tweet.likeCount, tweet.retweetCount, tweet.viewCount, tweet.hashtags])

df = pd.DataFrame(tweetlist, columns=['Content','Original','Datetime', 'Username', 'Likes', 'Retweets', 'View', 'Hashtags'])
#df = pd.DataFrame(tweetlist, columns=['Content','Datetime', 'Username', 'Likes', 'Retweets', 'View', 'Hashtags'])

def cleanTweets(text):
    text = re.sub('@[A-Za-z0-9_]+', '', text) #removes @mentions
    text = re.sub('#','',text) #removes hastag '#' symbol
    text = re.sub('RT[\s]+','',text)
    text = re.sub('https?:\/\/\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+',' ', text) 
    text = re.sub('\n',' ',text)
    return text

df["Tweet"] = df["Content"].apply(cleanTweets)

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

#prepare the last table
df1 = pd.DataFrame(df, columns=['Tweet', 'Original', 'Datetime', 'Username','Likes','Retweets', 'View', 'Hashtags', 'Sentiments'])
#df1 = pd.DataFrame(df, columns=['Tweet', 'Datetime', 'Username','Likes','Retweets', 'View', 'Hashtags', 'Sentiments'])

labels = ['Negative', 'Neutral', 'Positive']

for j in range(len(df)):
    tweet_proc = df1['Tweet'][j]

    # sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    #I think maybe this is where a possible for loop would solve my problem but I have no idea, I don't even know if I need a for loop eeeek.

    if(scores[0]> scores[1] and scores[0]> scores[2]):
        df1['Sentiments'][j]= labels[0]
    elif (scores[1]> scores[0] and scores[1]> scores[2]):
        df1['Sentiments'][j]= labels[1]
    else:
        df1['Sentiments'][j]= labels[2]

df1.to_csv("Ressources/tweets_Roberta2023.csv")
