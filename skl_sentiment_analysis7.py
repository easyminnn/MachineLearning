import pickle
import numpy as np
from mylib.vectorizer import vect
import os


clf = pickle.load(open(os.path.join('./data/pklObject', 'SGDClassifier.pkl'), 'rb'))

label = {0:'부정적 의견', 1:'긍정적 의견'}
example = ['i think this movie is worse']
X= vect.transform(example)
print('예측: %s\n 확률 : %.3f%%' %(label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))