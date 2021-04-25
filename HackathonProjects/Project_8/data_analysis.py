#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob
import seaborn as sns


# In[10]:


#df1 = pd.read_csv('UVdata/Fake.csv')
#df2 = pd.read_csv('UVdata/True.csv')
df1 = pd.read_csv('Fake.csv')
df2 = pd.read_csv('True.csv')
df1['label'] = 'FAKE'
df2['label'] = 'TRUE'

df = pd.concat(( df1, df2 ))
df.columns

df.label.value_counts()

df.loc[df['label'] == 'TRUE'].subject.value_counts()

df.loc[df['label'] == 'FAKE'].subject.value_counts()

def print_plot(index):
    example = df[df.index == index][['text','label']].values[0]
    if len(example) > 0:
        print(example[0])
        print('label:', example[1])


# In[11]:


print_plot(1000)


# In[ ]:


df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'] = df['text'].str.lower()

print_plot(1000)


# In[12]:


df['polarity'] = df['text'].map(lambda text: TextBlob(text).sentiment.polarity)

def text_len(x):
    if type(x) is str:
        return len(x.split())
    else:
        return 0

df['text_len'] = df['text'].apply(text_len)
nums_text = df.query('text_len > 0')['text_len']


# In[13]:


sns.histplot( df, x = 'text_len', bins = 100, kde = True )


# In[14]:


sns.histplot( df, x = 'text_len', fill = True, bins = 100, hue = 'label', kde = True  )


# In[15]:


sns.violinplot( data = df, x = 'label',  y = 'text_len', hue = 'label', pallete = 'muted' )


# In[17]:


sns.displot( data = df, x = 'polarity', hue = 'label', kind = 'kde', rug = True, rug_kws = { 'height' : 0.05, 'expand_margins' : True } )


# In[19]:


sns.displot( data = df, x = 'polarity', hue = 'subject', kind = 'kde', rug = True, rug_kws = { 'height' : 0.05, 'expand_margins' : True } )


# In[23]:


sns.violinplot( data = df, x = 'label',  y = 'polarity', hue = 'label', pallete = 'muted' )


# In[ ]:


sns.jointplot(x=df["polarity"], y=df["text_len"], kind='kde')


# In[ ]:


sns.scatterplot( data = df, x = 'polarity', y = 'text_len', hue = 'label' )


# In[20]:


print( df.groupby(['subject']).mean().sort_values('polarity', ascending=False) )

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def show_words( label ):
    common_bigram_true = get_top_n_bigram(df.loc[df['label'] == label]['text'], 20)
    for word, freq in common_bigram_true:
        print(word, freq)


# In[21]:


show_words( 'TRUE' )


# In[22]:


show_words( 'FAKE' )


# In[ ]:




