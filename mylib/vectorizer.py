import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer

stop = stopwords.words('english')

def sgd_tokenizer(text):
    #특수기호, HTML 태그 등 제거 (단, 이모티콘은 남겨둠)
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|\^.?\^', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')
    #문장부호나 특수문자로 변환 + 소문자로 치환
    tokenized = [w for w in text.split() if w not in stop]
    #정지단어 제거 

    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=sgd_tokenizer)