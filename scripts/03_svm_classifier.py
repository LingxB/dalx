from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from gensim.models import Word2Vec
import numpy as np


# TRAIN
# -----

df = pd.read_csv('data/processed/seed_vectors.csv', index_col='SEED')

X = df[[str(i) for i in range(500)]].values.astype('float32')
y = df['LABELS'].values.astype('int32')


# Search for best C
svc = LinearSVC(random_state=42, max_iter=5000)
params = [
    {'C': [0.001, 0.01, 0.1, 1, 10]}
]
gs = GridSearchCV(svc, param_grid=params, n_jobs=-1, cv=3, verbose=1)
gs.fit(X, y)
print(f'best_score(acc.)={gs.best_score_:.2%}; best_param={gs.best_params_}')

c = gs.best_params_['C']

svc = LinearSVC(C=c, verbose=1, random_state=42, max_iter=5000)
svc.fit(X, y)


# x = X[1,:].reshape(1, -1)
#
# svc.predict(x)
#
# svc.decision_function(x)


# PREDICT
# -------
lx_df = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
lx_words = lx_df.index.tolist() # 13297

w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')
vocabs = set(w2vec_model.wv.index2entity) # 43750
score_words = [w for w in lx_words if w in vocabs] # 5687
score_vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in score_words])

prediction = svc.predict(score_vectors)
score_df = pd.DataFrame(prediction.reshape(-1,1), index=pd.Series(score_words, name='WORD'), columns=['DALX'])

# make atlx compatable
score_df = lx_df.join(score_df)
score_df.DALX.replace(0, -1, inplace=True)

for idx,row in score_df.iterrows():
    dalx_val = row.DALX
    if not np.isnan(dalx_val):
        score_df.loc[idx] = dalx_val

score_df.to_csv('data/output/lexicon_table_dalx.csv')


