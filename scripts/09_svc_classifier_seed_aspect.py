from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from gensim.models import Word2Vec
import numpy as np


# TRAIN
# -----

df = pd.read_csv('data/processed/seed_aspect_vectors_08.csv', index_col=['SEED', 'ASP'])

# X = df[[str(i) for i in range(500)]].values.astype('float32')
X = df.drop('LABELS', axis=1).values.astype('float32')
y = df['LABELS'].values.astype('int32')


# Search for best C
svc = SVC(kernel='rbf', random_state=42, probability=True)
params = [
    {'C': [0.001, 0.01, 0.1, 1, 10]}
]
gs = GridSearchCV(svc, param_grid=params, n_jobs=-1, cv=3, verbose=1)
gs.fit(X, y)
print(f'best_score(acc.)={gs.best_score_:.2%}; best_param={gs.best_params_}')


c = gs.best_params_['C']
svc = SVC(C=c, kernel='rbf', verbose=True, random_state=42, probability=True)
svc.fit(X, y)



# todo: Refactor below
# 1. aspect lexicon
# 2. score aspect lexicon with model

# PREDICT
# -------
w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')

_x = w2vec_model['comfortable'].reshape(1,-1)

svc.predict(_x)

svc.predict_proba(_x)


# SCORE
# -----
conf_threshold = 0.7

w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')
lx_df = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
lx_words = lx_df.index.tolist() # 13297
vocabs = set(w2vec_model.wv.index2entity) # 43750
score_words = [w for w in lx_words if w in vocabs] # 5687
prediction = []

for w in score_words:
    _x = w2vec_model[w].reshape(1,-1)
    if conf_threshold:
        prob = svc.predict_proba(_x)
        if prob.max() < conf_threshold:
            print(f'Skipping {w} for {prob}')
            prediction.append(np.nan)
        else:
            prediction.append(svc.predict(_x)[0])
    else:
        prediction.append(svc.predict(_x)[0])

score_df = pd.DataFrame(np.array(prediction).reshape(-1,1), index=pd.Series(score_words, name='WORD'), columns=['DALX'])
score_df = lx_df.join(score_df)
score_df.DALX.replace(0, -1, inplace=True)
# update lexicon with dalx values
for idx,row in score_df.iterrows():
    dalx_val = row.DALX
    if not np.isnan(dalx_val):
        score_df.loc[idx] = dalx_val

score_df.loc['sharp']

score_df.to_csv('data/output/lexicon_table_dalx_07_thres0.7_C1.csv')


