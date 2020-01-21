from gensim.models import Word2Vec
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


w2vec_model = Word2Vec.load('data/raw/amazon/Electronics.bin')

# Use vocab ordered by frequency
vocabs = w2vec_model.wv.index2entity # [w for w in w2vec_model.wv.vocab]

vectors = np.concatenate([w2vec_model[w].reshape(1,-1) for w in vocabs])

assert all(vectors[vocabs.index('man')] == w2vec_model['man'])


pca = PCA(n_components=2, random_state=42)

vectors_2d = pca.fit_transform(vectors)

wvdf = pd.DataFrame(data=vectors_2d, index=pd.Index(vocabs, name='WORD'), columns=['X', 'Y'])

wvdf.to_parquet('data/processed/amazon_2d.parquet')







df = pd.read_parquet('data/processed/amazon_2d.parquet')

pd.testing.assert_frame_equal(df, wvdf)


