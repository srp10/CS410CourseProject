#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 22:20:50 2022

@author: srikanthpullaihgari
"""

import re
import numpy as np
import snscrape.modules.twitter as sntwitter
import pandas as pd
from textblob import TextBlob
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
import pycld2 as cld2
import datetime
from datetime import date
from wordcloud import WordCloud
import matplotlib.pyplot as plt



def clean_tweet(tweet):
    """
    This function is used to clean tweet text by removing links, 
    special characters
    """
    if type(tweet) == np.float:
        return ""
    cln_tweet = tweet.lower() #lowercasing all letters
    cln_tweet = re.sub("'", "", cln_tweet) # to avoid removing contractions in english
    cln_tweet = re.sub("@[A-Za-z0-9_]+","", cln_tweet) #removing mentions
    cln_tweet = re.sub("#[A-Za-z0-9_]+","", cln_tweet) #removing hashtags
    cln_tweet = re.sub(r'http\S+', '', cln_tweet) #removing links
    cln_tweet = re.sub(r"www.\S+", "", cln_tweet) #removing links
    cln_tweet = re.sub('[()!?]', ' ', cln_tweet) #removing punctuations
    cln_tweet = re.sub('\[.*?\]',' ', cln_tweet) #removing punctuations
    cln_tweet = re.sub("[^a-z0-9]"," ", cln_tweet) #filtering no alpha-numeric characters
    cln_tweet = cln_tweet.split() # tokenizing text into words
    #removing stop words
    #stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    stop_words = set(stopwords.words('english'))
    cln_tweet = [w for w in cln_tweet if not w in stop_words] 
    cln_tweet = " ".join(word for word in cln_tweet)
    return cln_tweet


# POS tagger dictionary
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


# function to calculate polarity
def getPolarity(review):
    return TextBlob(review).sentiment.polarity

# function to get subjectivity
def getTextSubjectivity(txt):
    return TextBlob(txt).sentiment.subjectivity

# function to analyze the reviews
def getSentiment(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
    
    
def get_tweet_sentiment(tweet):
    '''
    This function helps to classify sentiment of passed tweet
    using textblob's sentiment method
    '''
    # create TextBlob object of passed tweet text
    analysis = TextBlob(tweet)
    
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'


# Creating list to append tweet data to
attributes_container = []
findata_container = []

# Define max no of tweets to consider for sentiment analysis. 
# Takes the most recent tweets in case of a from and to dates provided as input
no_of_tweets =1000

# specify the brand name to query on
#brand_name = "Apple"


while True:
   brand_name = input("\nEnter the name of a Brand. Please use common names \
   where possible. For Example: Type in 'Apple' and not 'Apple Inc'. Please \
   use only alphanumeric letters and remove any special chaarcters such as \
   '&' etc from the name : ")
   if not brand_name.strip().replace(" ", "").isalpha():
       print ("Invalid! Please enter a valid company name") 
   else:
        break    

#print("\nHi {}! Welcome to Brand Sentiment Analysis!".format(brand_name))

#brand_name = input("Enter the name of a Brand: ")
print(brand_name)

# from and to dates for twitter scraping
to_date = date.today() #until field in Snrscraper takes (to_date - 1)
delta = datetime.timedelta(days=15)
from_date = to_date - delta


search_Str = brand_name + " " + "since:" + date.isoformat(from_date) + " " + "until:" + date.isoformat(to_date)
#print(search_Str)

# Scraping data using TwitterSearchScraper and adding tweets to a list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_Str).get_items()):
    if i>no_of_tweets:
        break
    attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.rawContent, tweet.retweetCount])

# a dataframe to load the list of tweets
tweets_df = pd.DataFrame(attributes_container, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet", "Retweet Count"])

# empty list to store parsed tweets
tweets = []
cnt = 0
# parsing tweets one by one
for ind in tweets_df.index:
    tweet = tweets_df["Tweet"][ind]
    retweet_cnt = tweets_df["Retweet Count"][ind]
    # empty dictionary to store required params of a tweet
    parsed_tweet = {}
    cnt = cnt+1
  
    # saving text of tweet
    parsed_tweet['text'] = tweet
    
    # Cleaning the tweet
    cleaned_tweet = clean_tweet(tweet)
    
    parsed_tweet['Cleaned Tweet'] = cleaned_tweet
    
    # POS tagging
    #parsed_tweet['POS tagged'] = parsed_tweet['Cleaned Tweet'].apply(token_stop_pos)
    parsed_tweet['POS tagged'] = token_stop_pos(parsed_tweet['Cleaned Tweet'])
    
    #Obtaining Stem words - Lemmatizatiom
    parsed_tweet['Lemma'] = lemmatize(parsed_tweet['POS tagged'])
    
    
    # Language detection of the Tweet
    #print("language detection")
    #lang = get_language(cleaned_tweet)
    _, _, top_lang, detected_language = cld2.detect(cleaned_tweet,  returnVectors=True)
    #print(top_lang)
    #print(detected_language)
    #print(top_lang[0][1])
    
    #Obtaining Stem words - Lemmatizatiom
    parsed_tweet['Language'] = top_lang[0][0]
    

   
    # saving sentiment of tweet if the tweet is in English language
    if (top_lang[0][1] == 'en'):
        #parsed_tweet['sentiment'] = get_tweet_sentiment(cleaned_tweet)
        subjectivity = getTextSubjectivity(cleaned_tweet)
        parsed_tweet['Subjectivity'] = subjectivity
        polarity = getPolarity(cleaned_tweet)
        parsed_tweet['Polarity'] = polarity
        sentimnt = getSentiment(polarity)
        parsed_tweet['sentiment'] = sentimnt
        
        # appending parsed tweet to tweets list
        if retweet_cnt > 0:
            # if tweet has retweets, ensure that it is appended only once
            if parsed_tweet not in tweets:
                tweets.append(parsed_tweet)
        else:
            tweets.append(parsed_tweet)
        #   tweets.append(parsed_tweet)
        #print(parsed_tweet) 

if (len(tweets) == 0):
    print("No recent tweets for that name. Please try again later!")
else:    
    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'Positive']
    # percentage of positive tweets
    print("Positive tweets percentage: {} %".format(100*len(ptweets)/len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'Negative']
    # percentage of negative tweets
    print("Negative tweets percentage: {} %".format(100*len(ntweets)/len(tweets)))
    # percentage of neutral tweets
    print("Neutral tweets percentage: {} % \
    ".format(100*(len(tweets) -(len( ntweets )+len( ptweets)))/len(tweets)))
    
    
    
    scorecrd = []
    output_details = {}
    output_details['Positive tweets'] = len(ptweets)
    output_details['Positive tweets percentage'] = "{} %".format(100*len(ptweets)/len(tweets))
    output_details['Negative tweets'] = len(ntweets)
    output_details['Negative tweets percentage'] = "{} %".format(100*len(ntweets)/len(tweets))
    output_details['Neutral tweets'] = len(tweets) -(len( ntweets )+len( ptweets))
    output_details['Neutral tweets percentage'] = "{} %".\
    format(100*(len(tweets) -(len( ntweets )+len( ptweets)))/len(tweets))
    
    scorecrd.append(output_details)
    # create dataframe
    scorecard_marks = pd.DataFrame(scorecrd)
         
    #print(tweets_df.head(30))
    
    # create a new dataframe from the tweets list created
    fin_data = pd.DataFrame(tweets)
    
    
    # Export dataframe into a CSV - source tweet data 
    tweets_df.to_csv('output/' + brand_name + '_text-query-tweets.csv', sep=',', index=False)
    
    
    # Export dataframe into a CSV - clean and analysed tweet data 
    fin_data.to_html('output/' + brand_name + '_final_output' + '.html')
    
    # counts of positive, neutral and negative tweets
    tb_counts = fin_data.sentiment.value_counts()
    
    #scorecrd.append(tb_counts)
    
    # create dataframe
    #scorecard_marks = pd.DataFrame(scorecrd, tb_counts)
    
    # Export dataframe into a CSV - clean and analysed tweet data 
    scorecard_marks.to_html('output/' + brand_name + '_scorecard' + '.html' )
    
    
    
    print(tb_counts)
    
    # Bar Chart for Sentiment
    labels = fin_data.groupby('sentiment').count().index.values
    values = fin_data.groupby('sentiment').size().values
    plt.bar(labels, values)
    #plt.show()
    plt.savefig('output/' + brand_name + '_sentiment_bar_chart.png', dpi=100)
    
    plt.clf()
    plt.cla()
    plt.close()
    
    # Polarity & Subjectivity Chart
    for index, row in fin_data.iterrows():
        if row['sentiment'] == 'Positive':
            plt.scatter(row['Polarity'], row['Subjectivity'], color="green")
        elif row['sentiment'] == 'Negative':
            plt.scatter(row['Polarity'], row['Subjectivity'], color="red")
        elif row['sentiment'] == 'Neutral':
            plt.scatter(row['Polarity'], row['Subjectivity'], color="blue")
    plt.title('Twitter Sentiment Analysis')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    # add legend
    #plt.show()
    plt.savefig('output/' + brand_name + '_polarity_subj_plot.png', dpi=100)
    
    
    plt.clf()
    plt.cla()
    plt.close()
    
    # Creating a word cloud
    #df = pd.DataFrame(data=[tweet for tweet in tweets], columns=['text'])
    plt.figure(figsize=[80,40])
    words = ' '.join([tweet for tweet in fin_data['text']])
    wordCloud = WordCloud(width=600, height=400).generate(words)
    
    plt.imshow(wordCloud)
    #plt.show()
    plt.savefig('output/' + brand_name + '_wordcloud.png', dpi=100)
    
    plt.clf()
    plt.cla()
    plt.close()

"""
# References
https://www.freecodecamp.org/news/python-web-scraping-tutorial/
https://catriscode.com/2021/05/01/tweets-cleaning-with-python/
https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/
https://towardsdatascience.com/4-python-libraries-to-detect-english-and-non-english-language-c82ad3efd430
https://pypi.org/project/pycld2/
https://pub.towardsai.net/scraping-tweets-using-snscrape-and-building-sentiment-classifier-13811dadd11d
https://www.geeksforgeeks.org/dropdown-menus-tkinter/
https://towardsdatascience.com/twitter-sentiment-analysis-in-python-1bafebe0b566


Installation Instructions:
install snsscrape - We can install the library directly using 
$ pip install snscrape 
or with the developer version 
$ pip3 install git+https://github.com/JustAnotherArchivist/snscrape.git

Install pycld2
$ python -m pip install -U pycld2

Install wordcloud
pip install wordcloud
"""