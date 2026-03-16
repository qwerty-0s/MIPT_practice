import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import math
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


def text_process(text):
    meaningless = ["a", "the", "in", "as", "at", "and"]
    no_punctuation = "". join(char for char in text if char not in string.punctuation) 
    no_upper_case = no_punctuation.lower()
    tokenized = no_upper_case.split()
    tokenized_no_meaningless = [word for word in tokenized if word not in meaningless and len(word) > 1]
    return tokenized_no_meaningless

def calculate_tf(text):
    unique_words = set()
    words_freq = {}
    n = len(text)
    for word in text:
        if word not in unique_words:
            unique_words.add(word)
            words_freq[word] = 1 
        else:
            words_freq[word]+=1
    for key in words_freq.keys():
        words_freq[key] = words_freq[key]/n
    return words_freq

def calculate_idf(texts):
    words_log = {}
    texts_with_word = {}
    total_texts = len(texts)
    for text in texts:
        unique_words = set(text)
        for word in unique_words:
            if word in unique_words:
                texts_with_word[word] = texts_with_word.get(word, 0) + 1
    
    for key in texts_with_word.keys():
        words_log[key] = math.log(total_texts/texts_with_word[key]) + 1 
        
    return words_log

def calculate_tfidf(texts):
    
    matrix_tfidf = []
    processed_texts = [text_process(text) for text in texts]     
    all_unique_words = sorted(list(set(word for doc in processed_texts for word in doc)))
    
    idf = calculate_idf(processed_texts)
    
    for text in processed_texts:
        tf = calculate_tf(text)
        words_tfidf = []
        
        for word in all_unique_words:
            if word in tf:
                words_tfidf.append(tf[word]*idf[word])
            else :
                words_tfidf.append(0)
                
        matrix_tfidf.append(words_tfidf)
    return matrix_tfidf


"""df = pd.read_csv("/home/au/python/Mipt_project/archive/spam.csv", encoding="latin-1")
print(df.columns)
print(df.iloc[0])

y = df['v1'].map({'ham': 0, 'spam': 1}).values
X_list = calculate_tfidf(df['v2'])
X = np.array(X_list) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = GaussianNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))"""