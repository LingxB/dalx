import sys
sys.path.append('P:\Projects\dalx')
import pandas as pd
import streamlit as st
from src.utils.seed_words import amazon_seeds
from src.utils.st_utils import set_polarity, plot_chart


'''
# Word vector explorer

Amazon review word embeddings with PCA to 2d vectors

'''

@st.cache
def load_vectors(path, top_words=5000):
    df = pd.read_parquet(path).reset_index()
    return df.iloc[:top_words,:]



# START
# -----

wvdf = load_vectors('data\processed/amazon_2d.parquet')
positive_seeds, negative_seeds = amazon_seeds()

f'''
## Exploring seed words

**Positive seeds**: {positive_seeds} 

**Negative seeds**: {negative_seeds}
'''

wvdf = set_polarity(wvdf, positive_seeds, negative_seeds)

if st.checkbox('Show dataframe with seed words'):
    wvdf

plot_chart(wvdf)


'''
---
'''

wvdf = load_vectors('data\processed/amazon_2d.parquet')

lx = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
lx['POLARITY'] = lx.mean(axis=1)
lx_words = lx[lx.POLARITY!=0].POLARITY.apply(lambda x: '+' if x > 0 else '-')
positive_words = lx_words[lx_words=='+'].index.tolist()
negative_words = lx_words[lx_words=='-'].index.tolist()

wvdf = set_polarity(wvdf, positive_words, negative_words)

f'''
## Exploring lexicon

**Positive words**: {positive_words[500:520]} ...

**Negative words**: {negative_words[500:520]} ...
'''

if st.checkbox('Show dataframe with original lexicon words'):
    wvdf

plot_chart(wvdf)


'''
---
'''

wvdf = load_vectors('data\processed/amazon_2d.parquet')
score_df = pd.read_csv('data/output/lexicon_table_dalx.csv', index_col='WORD')
pos = score_df.loc[score_df.DALX==1].index.tolist()
neg = score_df.loc[score_df.DALX==0].index.tolist()

wvdf = set_polarity(wvdf, pos, neg)

f'''
## Exploring domain adapted lexicon

**Positive words**: {pos[:20]} ...

**Negative words**: {neg[:20]} ...
'''

if st.checkbox('Show dataframe with domain adapted words'):
    wvdf

plot_chart(wvdf)









# '''
# # ALT plots example
# '''
#
# source = data.cars()
#
# chart = alt.Chart(source).mark_circle(size=60).encode(
#     x='Horsepower',
#     y='Miles_per_Gallon',
#     color='Origin',
#     tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
# ).interactive()
#
# st.write(chart)



#
# w2vec_model = gensim.models.Word2Vec.load('data/raw/amazon/Electronics.bin')
#
#
# w2vec_model.most_similar(positive=['woman', 'king'], negative=['man'])
#
#
# w2vec_model.most_similar(positive=['woman', 'keyboard'], negative=['man'])

