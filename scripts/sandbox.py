import sys
sys.path.append('P:\Projects\dalx')
import pandas as pd
import streamlit as st
import altair as alt
from src.utils.seed_words import amazon_seeds


'''
# Word vector explorer

Streamlit is awessome!
'''

@st.cache
def load_vectors(path, top_words=5000):
    df = pd.read_parquet(path).reset_index()
    return df.iloc[:top_words,:]

def set_polarity(df, positive=None, negative=None):
    _df = df.copy()
    _df['POLARITY'] = 'unk'
    if positive:
        _df['POLARITY'].loc[_df['WORD'].isin(positive)] = '+'
    if negative:
        _df['POLARITY'].loc[_df['WORD'].isin(negative)] = '-'
    return _df

def plot_chart(df):
    wv_chart = alt.Chart(df).mark_circle().encode(
        x='X',
        y='Y',
        color=alt.Color(
            'POLARITY', scale=alt.Scale(
                domain=['unk', '+', '-'],
                range=['#d6d6d6', '#1f77b4', '#ff7f0e']

            )
        ),
        tooltip=['WORD', 'POLARITY']
    ).interactive()

    st.write(wv_chart)


wvdf = load_vectors('data\processed/amazon_2d.parquet')
positive_seeds, negative_seeds = amazon_seeds()

f'''
## Exploring seed words

**Positive seeds**: {positive_seeds} 

**Negative seeds**: {negative_seeds}
'''

wvdf = set_polarity(wvdf, positive_seeds, negative_seeds)

if st.checkbox('Show dataframe with lexicon words'):
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

if st.checkbox('Show dataframe'):
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

