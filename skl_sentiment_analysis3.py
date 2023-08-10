from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
import os
from time import time
import pandas as pd
from mylib.tokenizer import tokenizer, tokenizer_porter

df = pd.read_csv('/Users/leejimin/Desktop/refined_DeepLearning.csv')

X_train = df.loc[:35000, 'review'].values
y_train = df.loc[:35000, 'sentiment'].values
X_test = df.loc[15000:, 'review'].values
y_test = df.loc[15000:, 'sentiment'].values

# 35000개의 데이터로 training을 해보고, 이후 15000개로는 test(검증)

tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer)
# TfidfVectorizer()에 사용할 단어 구분을 위한 함수로 tokenizer() 적용(단어를 공백을 분리)
Ir_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(C=10.0, penalty='l2', random_state=0))])
#Pipeline : 인자로 입력된 리스트에 정의된 함수를 순차적으로 적용. 
# --> 별명이 'vect'로 된 tfidf, 별명이 'clf'로 된 LogisticRegression()을 순차적으로 실행함. 

stime = time()
print('머신러닝 시작')
Ir_tfidf.fit(X_train, y_train)  #머신러닝 수행. Ir_tfidf는 위의 'vect'와 'clf'를 순차적으로 실행함. 
print('머신러닝 종료')

''' 자연어 처리를 위한 머신러닝 수행에 있어 아래와 같은 파라미터 최적값을 찾아야 한다. 
1. 정지단어를 제거하는 것이 좋은가, 제거하지 않는 것이 좋은가? (stopwords : ex> I, me, you ... )
2. 공백으로 단어를 분리하는 것이 좋은가, 단어줄기로 단어를 분리하는 것이 좋은가?
3. 단어 빈도수를 계산할 떄, 순수한 단어 빈도수를 사용하는 것이 좋은가, 또는 tf-idf를 사용하는 것이 좋은가? 
4. 머신러닝 알고리즘의 파라미터는 어떤 값을 취해야 좋은가
 - 로지스틱 회귀의 경우, C값과 L1, L2 정규화에 대한 최적의 파라미터를 찾아내야 함. 
'''
y_pred = Ir_tfidf.predict(X_test)
print('테스트 종료: 소요시간 [%d]초' %(time()-stime))
print('정확도 : %.3f' %accuracy_score(y_test, y_pred))

#머신러닝 결과를 파일로 저장
curDir = os.getcwd()
dest = os.path.join(curDir, 'data', 'pklObject')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(Ir_tfidf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
print('머신러닝 데이터 저장 완료')

#특정 단어와 그 문서의 부정/긍정을 연관시키는 것이므로 현실적인 문맥(함축적 의미)까지 파악하기에는 한계가 있다. (품질 bad)