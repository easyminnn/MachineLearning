from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

txt1 = 'the car is expensive'
txt2 = 'the truck is cheap'
txt3 = 'the car is expensive and the truck is cheap'

count = CountVectorizer()
docs = np.array([txt1, txt2, txt3])
bag = count.fit_transform(docs)

tfidf = TfidfTransformer()
np.set_printoptions(precision=2)


print(count.vocabulary_)
print(bag.toarray())

print(tfidf.fit_transform(count.fit_transform(docs)).toarray())



'''
결괏값
[[0      1       0      1        1       1      0]
 [0      0       1      0        1       1      1]
 [1      1       1      1        2       2      1]]
 and expensive truck   car      the     is    cheap
[[0.   0.56 0.   0.56 0.43 0.43 0.  ]
 [0.   0.   0.56 0.   0.43 0.43 0.56]
 [0.4  0.31 0.31 0.31 0.48 0.48 0.31]]


'''