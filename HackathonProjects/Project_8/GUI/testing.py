
def testing_model(test_str):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
    from tensorflow.keras.layers import SpatialDropout1D, BatchNormalization, Input
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.text import one_hot
    from tensorflow.keras.preprocessing.text import Tokenizer

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    #Downloading stopwords 
    #Stopwords are the words in any language which does not add much meaning to a sentence.
    #They can safely be ignored without sacrificing the meaning of the sentence.
    import nltk
    import re
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    
    d = {'Total': test_str}
    X_test = pd.DataFrame(data=d)
    corpus_test=[]
    for i in range(len(X_test)):
    #     if i == 20800: continue
        input1 = re.sub('[^a-zA-Z]',' ', str(X_test.iloc[i].total)) # except a-z and A-Z, substitute all other characters with ' '
        input1 = input1.lower() # Lower case 
        input1  = input1.split() # tokenize the text
        input1= [ps.stem(word) for word in input1 if word not in stopwords.words('english')]
        text = ' '.join(input1)  # concatenating all words into a single text (list is created)#
        corpus_test.append(text) # appending text into a single corpus #
        
    #Choosing vocabulary size to be 5000 and copying data to msg for further cleaning
    voc_size = 5000
    onehot_rep_test = [one_hot(words,voc_size)for words in corpus_test]    

    #Padding Sentences to make them of same size
    embedded_docs_test = pad_sequences(onehot_rep_test,padding='pre',maxlen=25)

    test_final = np.array(embedded_docs_test)

    return test_final