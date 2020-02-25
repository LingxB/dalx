import pandas as pd
import numpy as np
from src.utils import read_yaml, load_semeval15_laptop, search_keyword
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 500)


corpus = load_semeval15_laptop('data/SemEval15_laptop/train.csv', 'data/SemEval15_laptop/test.csv')

vocab = read_yaml('data/SemEval15_laptop/glove_symdict.yml')

atlx = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
atlx['ATLX'] = atlx.mean(axis=1).apply(lambda p: -1 if p < 0 else (0 if p==0 else 1))

dalx = pd.read_csv('data/output/lexicon_table_dalx_03.csv', index_col='WORD')

diff_df = atlx.join(dalx[['DALX']])
diff_df['INVOCAB'] = diff_df.index.isin(vocab)



# Words in S15 vocab
diff_df.INVOCAB.value_counts() # 839


# Words that changed polarity by SVM
diff_df.loc[(diff_df.DALX.notna()) & (diff_df.ATLX != diff_df.DALX)] # 1962

# Words in S15 vocab but not changed polarity by SVM
diff_df.loc[(diff_df.DALX.notna()) & (diff_df.ATLX == diff_df.DALX) & (diff_df.INVOCAB == True)] # 530

# Words in S15 vocab and changed polarity by SVM
diff_df.loc[(diff_df.DALX.notna()) & (diff_df.ATLX != diff_df.DALX) & (diff_df.INVOCAB == True)] # 304


# A lot of words become negative
#  ATLX        DALX
#  1    137    103
#  0    110    NaN
# -1     57    201

search_keyword(corpus, 'air')
search_keyword(corpus, 'better')
search_keyword(corpus, 'big')
search_keyword(corpus, 'bright')
search_keyword(corpus, 'brightness')

search_keyword(corpus, 'comfortable')
search_keyword(corpus, 'delay')
search_keyword(corpus, 'fire')
search_keyword(corpus, 'flash')
search_keyword(corpus, 'giant')
search_keyword(corpus, 'heck')
search_keyword(corpus, 'joke')
search_keyword(corpus, 'sensitive')
search_keyword(corpus, 'sharp')


