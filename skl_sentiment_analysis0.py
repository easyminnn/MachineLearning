import pandas as pd
import os
import numpy as np
from mylib.progbar import ProgBar

path = '/Users/leejimin/Desktop/DeepLearning/aclImdb/'

pbar = ProgBar(50000)
# 데이터 처리 진행률을 보여주기 위한 클래스(Progbar)
labels = {'pos':1, 'neg':0}
df  = pd.DataFrame()

for s in ('test', 'train'):
    for name in ('pos', 'neg'):
        subpath = '%s/%s' %(s, name)
        dirpath = path + subpath
        for file in os.listdir(dirpath):
            with open(os.path.join(dirpath, file), 'r')  as f:
                txt = f.read()
            df1 = pd.DataFrame([[txt, labels[name]]])
            df = pd.concat([df, df1], ignore_index = True)
           
            pbar.update()

df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('/Users/leejimin/Desktop/DeepLearning.csv', index = False)
