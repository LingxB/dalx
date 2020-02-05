from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from src.utils.seed_words import amazon_seeds


w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')

positive_seeds, negative_seeds = amazon_seeds()
# remove OVW
negative_seeds = [w for w in negative_seeds if w != 'banal']

positive_vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in positive_seeds])
negative_vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in negative_seeds])


for idx in range(len(positive_seeds)):
    assert (w2vec_model[positive_seeds[idx]] == positive_vectors[idx]).all()

for idx in range(len(negative_seeds)):
    assert (w2vec_model[negative_seeds[idx]] == negative_vectors[idx]).all()


train_df = pd.DataFrame(np.concatenate([positive_vectors, negative_vectors]),
                        index=pd.Index(positive_seeds+negative_seeds, name='SEED')
                        )
train_df['LABELS'] = [0 if w in negative_seeds else 1 for w in positive_seeds+negative_seeds]

assert (train_df[list(range(500))].values == np.concatenate([positive_vectors, negative_vectors])).all()

train_df = train_df.sample(frac=1, random_state=42)

train_df.to_csv('data/processed/seed_vectors.csv')