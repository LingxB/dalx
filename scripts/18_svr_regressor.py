from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt


# TRAIN
# -----

df = pd.read_csv('data/processed/seed_vectors_w_neu_16.csv', index_col='SEED', dtype={'LABELS': float})

X = df[[str(i) for i in range(500)]].values.astype('float32')
y = df['LABELS'].values.astype('float32')


# Search for best C
svr = SVR(kernel='rbf', verbose=True, epsilon=0.1)
params = [
    {'C': [0.001, 0.01, 0.1, 1, 10]}
]
gs = GridSearchCV(svr, param_grid=params, n_jobs=-1, cv=3, verbose=1)
gs.fit(X, y)
print(f'best_score(acc.)={gs.best_score_:.2%}; best_param={gs.best_params_}')


c = gs.best_params_['C']
svr = SVR(C=c, kernel='rbf', verbose=True, epsilon=0.1)
svr.fit(X, y)


# PREDICT
# -------
w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')

_x = w2vec_model['sharp'].reshape(1,-1)

svr.predict(_x)



# SCORE
# -----
# conf_threshold = 0.7

w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')
lx_df = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
lx_words = lx_df.index.tolist() # 13297
vocabs = set(w2vec_model.wv.index2entity) # 43750
score_words = [w for w in lx_words if w in vocabs] # 5687
prediction = []

for w in score_words:
    _x = w2vec_model[w].reshape(1,-1)
    # if conf_threshold:
    #     prob = svr.predict_proba(_x)
    #     if prob.max() < conf_threshold:
    #         print(f'Skipping {w} for {prob}')
    #         prediction.append(np.nan)
    #     else:
    #         prediction.append(svr.predict(_x).astype(int)[0])
    # else:
    prediction.append(svr.predict(_x)[0])

score_df = pd.DataFrame(np.array(prediction).reshape(-1,1), index=pd.Series(score_words, name='WORD'), columns=['DALX'])
score_df = score_df / max(score_df.min().abs().iloc[0], score_df.max().abs().iloc[0]) # normalize to be between -1 to 1
score_df = lx_df.join(score_df)
# update lexicon with dalx values
for idx,row in score_df.iterrows():
    dalx_val = row.DALX
    if not np.isnan(dalx_val):
        score_df.loc[idx] = dalx_val

score_df.loc['sharp']

score_df.to_csv('data/output/lexicon_table_dalx_18_regress_e0.1_C10.csv')


