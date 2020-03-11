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

dalx = pd.read_csv('data/output/lexicon_table_dalx_07_thres0.7_C10.csv', index_col='WORD')

diff_df = atlx.join(dalx[['DALX']])
diff_df['INVOCAB'] = diff_df.index.isin(vocab)



# Words in S15 vocab
diff_df.INVOCAB.value_counts() # 839


# Words that changed polarity by SVM                                 # 03   05_C10 05_C10_T.7
diff_df.loc[(diff_df.DALX.notna()) & (diff_df.ATLX != diff_df.DALX)] # 1962 2058   308

# Words in S15 vocab but not changed polarity by SVM                                               03  05_C10 05_C10_T 05_C1_T
diff_df.loc[(diff_df.DALX.notna()) & (diff_df.ATLX == diff_df.DALX) & (diff_df.INVOCAB == True)] # 530 534    326      131

# Words in S15 vocab and changed polarity by SVM                                                 # 03  05_C10 05_C10_T 05_C1_T
diff_df.loc[(diff_df.DALX.notna()) & (diff_df.ATLX != diff_df.DALX) & (diff_df.INVOCAB == True)] # 304 300    109      20



# Some intresting words in the corpus and changed polarity in dalx_07_thres0.7_C10
#                 MPQA  OPENER        OL     VADER  ATLX  DALX  INVOCAB
# WORD
# addicted     -1.0000 -1.0000 -1.000000 -1.000000    -1   1.0     True
# affect        0.0000  0.0000  0.000000  0.000000     0  -1.0     True
# alright       0.2500  0.2500  0.250000  0.250000     1  -1.0     True
# cheaper       1.0000  1.0000  1.000000  1.000000     1  -1.0     True
# darker       -1.0000 -1.0000 -1.000000 -1.000000    -1   1.0     True
# discontinued -1.0000 -1.0000 -1.000000 -1.000000    -1   1.0     True
# drive         1.0000  1.0000  1.000000  1.000000     1  -1.0     True
# duty          0.0000  0.0000  0.000000  0.000000     0   1.0     True
# enemies      -1.0000 -1.0000 -1.000000 -0.550000    -1   1.0     True
# envy         -1.0000 -1.0000  1.000000 -0.275000    -1   1.0     True
# extremely    -1.0000 -1.0000 -1.000000 -1.000000    -1   1.0     True
# fancy         1.0000  1.0000  1.000000  1.000000     1  -1.0     True
# giant         0.0000  0.0000  0.000000  0.000000     0  -1.0     True
# hesitate     -0.2750 -0.2750 -0.275000 -0.275000    -1   1.0     True
# high          0.0000  0.0000  0.000000  0.000000     0   1.0     True
# killer       -1.0000 -1.0000 -1.000000 -0.825000    -1   1.0     True
# lol           0.5875  0.5875  0.587500  0.587500     1  -1.0     True
# repair        1.0000  1.0000  1.000000  1.000000     1  -1.0     True
# sensitive     1.0000  1.0000  1.000000  1.000000     1  -1.0     True
# sharp        -1.0000 -1.0000  1.000000 -0.333333    -1   1.0     True
# yes           1.0000  1.0000  0.808333  0.425000     1  -1.0     True


search_keyword(corpus, 'addicted')
search_keyword(corpus, 'cheaper')
search_keyword(corpus, 'darker')
search_keyword(corpus, 'discontinued')
search_keyword(corpus, 'envy')
search_keyword(corpus, 'extremely')
search_keyword(corpus, 'fancy')
search_keyword(corpus, 'giant')
search_keyword(corpus, 'hesitate')
search_keyword(corpus, 'killer')
search_keyword(corpus, 'repair')
search_keyword(corpus, 'sensitive')
search_keyword(corpus, 'sharp')
search_keyword(corpus, 'yes')






































# 03 - 304 total A lot of words become negative
#       ATLX   DALX
#  1    137    103
#  0    110    NaN
# -1     57    201

# 05 - 300 total
#       ATLX   DALX
# -1    127    100
#  0    110    NaN
#  1     63    200


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


