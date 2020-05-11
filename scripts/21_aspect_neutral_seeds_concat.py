from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from src.utils.seed_words import all_seeds_with_neutral
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 500)


w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')
vocabs = set(w2vec_model.wv.index2entity) # 43750

train = pd.read_csv('data/SemEval15_laptop/train.csv')
test = pd.read_csv('data/SemEval15_laptop/test.csv')
aspects = sorted(set(train.ASP.unique().tolist() + test.ASP.unique().tolist()))
aspects = [w for w in aspects if w in vocabs]

positive_seeds, negative_seeds, neutral_seeds = all_seeds_with_neutral() # 52, 58, 35
positive_seeds = [w for w in positive_seeds if w in vocabs] # 31
negative_seeds = [w for w in negative_seeds if w in vocabs] # 34
neutral_seeds = [w for w in neutral_seeds if w in vocabs] # 35

seeds = positive_seeds+negative_seeds+neutral_seeds

positive_vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in positive_seeds])
negative_vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in negative_seeds])
neutral_vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in neutral_seeds])
seed_vectors = pd.DataFrame(np.concatenate([positive_vectors, negative_vectors, neutral_vectors]),
                        index=pd.Index(positive_seeds+negative_seeds+neutral_seeds, name='SEED')
                        )

aspect_vectors = np.concatenate([w2vec_model[a].reshape(1,-1) for a in aspects])
aspect_vectors = pd.DataFrame(aspect_vectors, index=pd.Index(aspects, name='ASP'))


_df = pd.DataFrame(index=pd.MultiIndex.from_product([seeds, aspects], names=['SEED', 'ASP'])).reset_index()

pol_map = {w:'+1' for w in positive_seeds}
pol_map.update({w:'-1' for w in negative_seeds})
pol_map.update({w: '00' for w in neutral_seeds})
_df['LABELS'] = [pol_map[w] for w in _df.SEED]

_df = pd.merge(_df, seed_vectors.reset_index(), how='left', on='SEED')
train_df = pd.merge(_df, aspect_vectors.reset_index(), how='left', on='ASP', suffixes=('_s', '_a')).set_index(['SEED', 'ASP'])


train_df = train_df.sample(frac=1, random_state=42)

train_df.to_csv('data/processed/seed_aspect_vectors_21.csv')

