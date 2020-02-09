import sys
sys.path.append('P:\Projects\dalx')
import pandas as pd
import streamlit as st
import altair as alt
from src.utils.st_utils import set_polarity, plot_chart


wvdf = pd.read_parquet('data\processed/amazon_2d.parquet')
atlx = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
dalx = pd.read_csv('data/output/lexicon_table_dalx.csv', index_col='WORD')

vec_df = wvdf.loc[atlx.index].dropna()



"""
Original ATLX lexicon
"""

atlx['POLARITY'] = atlx.mean(axis=1)
lx_words = atlx[atlx.POLARITY!=0].POLARITY.apply(lambda x: '+' if x > 0 else '-')
positive_words = lx_words[lx_words=='+'].index.tolist()
negative_words = lx_words[lx_words=='-'].index.tolist()

_wvdf = set_polarity(vec_df.reset_index(), positive_words, negative_words)

plot_chart(_wvdf)


"""
Domain adapted lexicon
"""
pos = dalx.loc[dalx.DALX==1].index.tolist()
neg = dalx.loc[dalx.DALX==-1].index.tolist()

_wvdf = set_polarity(vec_df.reset_index(), pos, neg)
plot_chart(_wvdf)
