import re
import pandas as pd
from time import time


def preprocessor(text):
    #특수기호, HTML 태그 등 제거 (단, 이모티콘은 남겨둠)
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|\^.?\^', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-','')

    return text
 

df = pd.read_csv('/Users/leejimin/Desktop/DeepLearning.csv')

stime = time()
print('전처리 시작')
df['review'] = df['review'].apply(preprocessor)
print('전처리 완료 : 소요시간 [%d]초' %(time()-stime))

df.to_csv('/Users/leejimin/Desktop/refined_DeepLearning.csv', index = False)


df = pd.read_csv('/Users/leejimin/Desktop/refined_DeepLearning.csv')

df.head()