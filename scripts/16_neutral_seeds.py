from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from src.utils.seed_words import all_seeds_with_neutral


w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')
vocabs = set(w2vec_model.wv.index2entity) # 43750

positive_seeds, negative_seeds, neutral_seeds = all_seeds_with_neutral() # 52, 58, 50
positive_seeds = [w for w in positive_seeds if w in vocabs] # 31
negative_seeds = [w for w in negative_seeds if w in vocabs] # 34
neutral_seeds = [w for w in neutral_seeds if w in vocabs] # 35

positive_vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in positive_seeds])
negative_vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in negative_seeds])
neutral_vectors = np.concatenate([w2vec_model[w].reshape(1, -1) for w in neutral_seeds])

for idx in range(len(positive_seeds)):
    assert (w2vec_model[positive_seeds[idx]] == positive_vectors[idx]).all()

for idx in range(len(negative_seeds)):
    assert (w2vec_model[negative_seeds[idx]] == negative_vectors[idx]).all()

for idx in range(len(neutral_seeds)):
    assert (w2vec_model[neutral_seeds[idx]] == neutral_vectors[idx]).all()

train_df = pd.DataFrame(np.concatenate([positive_vectors, negative_vectors, neutral_vectors]),
                        index=pd.Index(positive_seeds+negative_seeds+neutral_seeds, name='SEED')
                        )
pol_map = {}

train_df['LABELS'] = ['+1'] * len(positive_seeds) + ['-1'] * len(negative_seeds) + ['00'] * len(neutral_seeds)

assert (train_df[list(range(500))].values == np.concatenate([positive_vectors, negative_vectors, neutral_vectors])).all()
assert train_df.loc[train_df.LABELS=='+1'].index.tolist() == positive_seeds
assert train_df.loc[train_df.LABELS=='-1'].index.tolist() == negative_seeds
assert train_df.loc[train_df.LABELS=='00'].index.tolist() == neutral_seeds

train_df = train_df.sample(frac=1, random_state=42)

train_df.to_csv('data/processed/seed_vectors_w_neu_16.csv')
