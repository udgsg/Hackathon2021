
def training_model(data_path):
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

    ps = PorterStemmer()

    from wordcloud import WordCloud

    #getting data_path
    df_train = pd.read_csv(data_path)
    #df_test = pd.read_csv(data_path+'test.csv') #no label included, so unusable
    df = df_train

    #filling NULL values with empty string
    df=df.fillna('')

    #feature manipulation
    df['total'] = df['title']+' '+df['author']
    y=df['label']
    X = df.drop('label',axis=1)

    #text preprocessing
    #Downloading stopwords 
    #Stopwords are the words in any language which does not add much meaning to a sentence.
    #They can safely be ignored without sacrificing the meaning of the sentence.
    nltk.download('stopwords')


    #split test train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    corpus_train=[]
    for i in range(len(X_train)):
    #     if i == 20800: continue
        input1 = re.sub('[^a-zA-Z]',' ', str(X_train.iloc[i].total)) # except a-z and A-Z, substitute all other characters with ' '
        input1 = input1.lower() # Lower case 
        input1  = input1.split() # tokenize the text
        input1= [ps.stem(word) for word in input1 if word not in stopwords.words('english')]
        text = ' '.join(input1)  # concatenating all words into a single text (list is created)#
        corpus_train.append(text) # appending text into a single corpus #
        
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
    onehot_rep_train = [one_hot(words,voc_size)for words in corpus_train]
    onehot_rep_test = [one_hot(words,voc_size)for words in corpus_test]    

    #Padding Sentences to make them of same size
    embedded_docs = pad_sequences(onehot_rep_train,padding='pre',maxlen=25)
    embedded_docs_test = pad_sequences(onehot_rep_test,padding='pre',maxlen=25)

    #We have used embedding layers with LSTM
    model = Sequential()
    model.add(Embedding(voc_size,40,input_length=25))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    #Converting into numpy array
    X_final = np.array(embedded_docs)
    y_final = np.array(y_train)
    test_final = np.array(embedded_docs_test)

    #training model
    model.fit(X_final,y_final,epochs=1,batch_size=64)

    y_pred = model.predict_classes(test_final)

    accuracy=accuracy_score(y_test, y_pred)

    return accuracy, model