import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings


warnings.filterwarnings('ignore')
def hashtag_extract(tweets):
    hashtags = []
    # loop words in the tweet
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags

df = pd.read_csv('Twitter Sentiments.csv')


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt

df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")


df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")
df.head()

df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))
df.head()

tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
tokenized_tweet.head()

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])

df['clean_text'] = tokenized_tweet

all_words="".join([sentence for sentence in df['clean_text']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)


ht_positive = hashtag_extract(df['clean_text'][df['label']==0])
ht_negative = hashtag_extract(df['clean_text'][df['label']==1])

ht_negative = sum(ht_negative,[])
ht_positive = sum(ht_positive,[])
print(ht_positive[:5])

freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame({'Hashtag': list(freq.keys()), 'Count': list(freq.values())})

#print(d.head())

#select top 10 hashtags

#d = d.nlargest(columns='Count', n=10)
#plt.figure(figsize=(15,9))
#sns.barplot(data=d,x='Hashtag',y='Count')
#plt.show()

freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame({'Hashtag': list(freq.keys()), 'Count': list(freq.values())})
#print(d.head())

#d = d.nlargest(columns='Count', n=10)
#plt.figure(figsize=(15,9))
#sns.barplot(data=d,x='Hashtag',y='Count')
#plt.show()

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000,stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])

from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(bow,df['label'], random_state=42,test_size=0.25)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  f1_score, accuracy_score

model = LogisticRegression()
model.fit(x_train, y_train)

pred = model.predict(x_test)
acc= accuracy_score(y_test,pred)


pred_prob = model.predict_proba(x_test)
pred = pred_prob[:,1] >= 0.3
pred = pred.astype(np.int)

acc= accuracy_score(y_test,pred)
print(acc)