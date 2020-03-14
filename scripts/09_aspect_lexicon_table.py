import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 500)
from src.utils.seed_words import semeval15_aspects


lx = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
aspects = semeval15_aspects()


alx = pd.DataFrame(index=pd.MultiIndex.from_product([lx.index.tolist(), aspects], names=['WORD', 'ASP'])).reset_index()

alx = pd.merge(alx, lx, how='left', on='WORD').set_index(['WORD', 'ASP'])

alx.to_csv('data/processed/lexicon_table_asp_raw_09.csv')