import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import pickle



def get_row(sentence,col_names):
  vectorizer = CountVectorizer(stop_words = 'english',vocabulary = col_names)
  row = vectorizer.fit_transform(sentence)
  return row

def get_prediction(models,row):
  prob = np.zeros((1,2))
    
  for m in (models):
    prediciton = m.predict_proba(row)
    prob = prob + prediciton
 
  prob = np.divide(prob,len(models))
  print(prob)
  if(prob[0,0] > prob[0,1]):
    return False
  if(prob[0,1] > prob[0,0]):
    return True