'''확률적 경사하강법을 적용하여 머신러닝 수행
1. 확률적 경사하강법 버전의 로지스틱 회귀 이용.
2. 영화 리뷰 데이터는 1000개씩 머신러닝 수행.
3. 머신러닝 위한 총 리뷰 데이터는 45000개.
4. 머신러닝 결과 테스트는 나머지 5000개.
5. 머신러닝 결과는 파일로 저장.

'''

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pickle
import os
from mylib.progbar import ProgBar
from sgd_tokenizer import sgd_tokenizer

def stream_docs(path):
    with open(path, 'r') as f:
        next(f)  
        for line in f:
            text, label = line[:-3], int(line[-2])
            yield text, label #yield 이용해서 반환 --> stream_docs는 제너레이터(리턴은 함수 종료, yield는 함수 계속 존재)
            # yield 반환이기 때문에 for line in f 가 그대로 가동된다!!!!! wow. 
# 이 제너레이터는 실제 리뷰 텍스트와 그 라벨(긍정은 1, 부정은 0) 리턴. 

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)  
    #위의 text와 label들을 리스트로 만들기. !
    except StopIteration:
        return None, None
    
    return docs, y


vect = HashingVectorizer(decode_error='ignore', n_features= 2**21, tokenizer=sgd_tokenizer)
#해싱 기법을 이용하여 데이터를 독립적으로 구동. 이 함수를 이용하여 특정 벡터 구성.

clf = SGDClassifier(loss = 'log_loss', random_state=1, n_iter_no_change=1) # 로지스틱 회귀 SGD
doc_stream = stream_docs('/Users/leejimin/Desktop/refined_DeepLearning.csv')

pbar = ProgBar(45)
classes = np.array([0,1])

for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000) # 1000개의 리뷰 데이터를 리스트로 전달받기. 
    if not X_train:
        break

    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes) #부분 데이터 머신러닝 수행. 
    pbar.update()




X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)

print('정확도: %.3f' %clf.score(X_test, y_test))

curDir = os.getcwd()
dest = os.path.join(curDir, 'data', 'pklObject')
pickle.dump(clf, open(os.path.join(dest, 'SGDClassifier.pkl'), 'wb'), protocol = 4)
print('머신러닝 데이터 저장 완료 ')
# 나머지 데이터 5000개의 리뷰 데이터를 가지고 테스트해보고, 정확도를 표시한 후, 
# 머신러닝 결과를 SGDClassifier.pkl에 저장하는 방식. 


