import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer #詞幹提取工具
stemmer = PorterStemmer()

def tokenize(sentence): #定義tokenize
    return nltk.word_tokenize(sentence)

def stem(word): #詞幹提取&轉成小寫的函式
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words): #詞袋模型的函式 
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words),dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


