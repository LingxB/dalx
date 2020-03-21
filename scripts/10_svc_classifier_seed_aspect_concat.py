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


# PREDICT
# -------
w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')
alx = pd.read_csv('data/processed/lexicon_table_asp_raw_09.csv', index_col=['WORD', 'ASP'])


_w = w2vec_model['cheap'].reshape(1,-1)
_a = w2vec_model['price'].reshape(1,-1)
_x = np.concatenate([_w, _a], axis=1)

svc.predict(_x)

svc.predict_proba(_x)


# SCORE
# -----
conf_threshold = 0.7

w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')
lx_df = pd.read_csv('data/processed/lexicon_table_asp_raw_09.csv', index_col=['WORD', 'ASP'])
lx_words = lx_df.index.tolist() # 119673 (13297x9)
vocabs = set(w2vec_model.wv.index2entity) # 43750
score_words = [(w,a) for w,a in lx_words if w in vocabs] # 51183 (5687x9)
prediction = []

for w,a in score_words:
    _w = w2vec_model[w].reshape(1,-1)
    _a = w2vec_model[a].reshape(1,-1)
    _x = np.concatenate([_w, _a], axis=1)
    if conf_threshold:
        prob = svc.predict_proba(_x)
        if prob.max() < conf_threshold:
            print(f'Skipping {w},{a} for {prob}')
            prediction.append(np.nan)
        else:
            prediction.append(svc.predict(_x)[0])
    else:
        prediction.append(svc.predict(_x)[0])

score_df = pd.DataFrame(np.array(prediction).reshape(-1,1),
                        index=pd.MultiIndex.from_tuples(score_words, names=['WORD','ASP']),
                        columns=['DALX'])
score_df = lx_df.join(score_df)
score_df.DALX.replace(0, -1, inplace=True)
# update lexicon with dalx values
for idx,row in score_df.iterrows():
    dalx_val = row.DALX
    if not np.isnan(dalx_val):
        score_df.loc[idx] = dalx_val

score_df.loc['sharp']

score_df.to_csv('data/output/lexicon_table_dalx_asp_10_thres0.7_C10_concat.csv')


# for w in lx_df.index.get_level_values('WORD').unique():
#     if len(score_df.loc[w].DALX.unique()) > 1:
#         print(f'\n {w}')
#         print(score_df.loc[w])
