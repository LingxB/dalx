import sys
sys.path.append('P:\Projects\dalx')
import pandas as pd
import numpy as np
import streamlit as st
from src.utils import read_yaml, load_semeval15_laptop, search_keyword
import logging.config
pd.set_option('display.width', 1000)
import time


logging.config.dictConfig(read_yaml('logger_configs.yml'))
logger = logging.getLogger()
logger.info('----- Start annotator -----')

"""# SemEval15 Laptop Lexicon Annotator"""

@st.cache
def load_corpus_and_general_lexicon():
    train = 'data/SemEval15_laptop/train.csv'
    test = 'data/SemEval15_laptop/test.csv'
    logger.info(f'Loading SemEval15 Laptop corpus from: {train} {test}')
    corpus = load_semeval15_laptop(train, test)
    g_lx = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
    return corpus, g_lx

def load_annotation_lexicon(path):
    logger.info(f'Loading lexicon from: {path}')
    a_df = pd.read_csv(path, index_col='WORD')
    n_notes = a_df.ANNOTATION.notna().sum()
    total_words = len(a_df)
    logger.info(f'{n_notes}/{total_words} words has been annotated.')
    st.info(f'Data loaded, {n_notes}/{total_words} words has been annotated.')
    return a_df, n_notes, total_words

def display_search_results(s_df):
    for i,row in s_df[['SENT']].drop_duplicates().iterrows():
        # st.write(f"CLS *{row.CLS}* - ASP *{row.ASP}* - TRAIN *{row.IS_TRAIN}*")
        st.write(f"{row.SENT.replace(w, f'**{w}**')}")

def update_lexicon(a_df, w, user_pol, out_path):
    _df = a_df.copy()
    _df.loc[w, 'ANNOTATION'] = user_pol
    logger.info(f'Annotated {w} as {user_pol}')
    _df.to_csv(out_path)
    logger.info(f'Saved annotation to {out_path}')
    return _df

def next_row(a_df):
    for idx,(w,row) in enumerate(a_df.iterrows()):
        if np.isnan(row.ANNOTATION):
            return idx, w, row


corpus, g_lx = load_corpus_and_general_lexicon()
working_lexicon = 'data/lx_annotator/s15_annotation_out.csv'

# load
a_df, n_notes, total_words = load_annotation_lexicon(working_lexicon)

idx, w, row = next_row(a_df)

# annotate
msg = f"## **{w}** {idx + 1}/{total_words}"
logger.info(msg)
st.write(msg)

f"### Word in corpus"
s_df = search_keyword(corpus, w).drop('EA', axis=1)
st.write(s_df)
display_search_results(s_df)

f"### Polarity in general lexicon"
st.write(g_lx.loc[w].to_frame().transpose())

f"### Your choice"
user_pol = st.radio('Polarity', [-1, 0, 1])


if st.button('SAVE'):
    # write
    update_lexicon(a_df, w, user_pol, working_lexicon)
    st.success('Annotation updated')



# a_df, n_notes, total_words = load_annotation_lexicon(working_lexicon)





# # loop starts here
# idx = 0; w = 'ability'; row = a_df.iloc[0,:]
#
# msg = f"## **{w}** {idx+1}/{total_words}"
# logger.info(msg); st.write(msg)
#
# f"### Word in corpus"
# s_df = search_keyword(corpus, w).drop('EA', axis=1)
# st.write(s_df)
# display_search_results(s_df)
#
# f"### Polarity in general lexicon"
# st.write(g_lx.loc[w].to_frame().transpose())
#
# f"### Your choice"
# user_pol = st.radio('Polarity', [-1, 0, 1])
#
# if st.button('NEXT'):
#     out_df = update_lexicon(a_df, w, user_pol, working_lexicon)


# for idx,(w,row) in enumerate(a_df.iterrows()):
#
#     if idx == 0:
#         msg = f"## **{w}** {idx+1}/{total_words}"
#         logger.info(msg); st.write(msg)
#
#         f"### Word in corpus"
#         s_df = search_keyword(corpus, w).drop('EA', axis=1)
#         st.write(s_df)
#         display_search_results(s_df)
#
#         f"### Polarity in general lexicon"
#         st.write(g_lx.loc[w].to_frame().transpose())
#
#         f"### Your choice"
#         user_pol = st.radio('Polarity', [-1, 0, 1])

    # button = st.button('NEXT')
    # while not button:
    #     time.sleep(1)
    # else:
    #     break


    # if st.button('NEXT'):
    #     out_df = update_lexicon(a_df, w, user_pol, working_lexicon)
    #     idx += 1
    # else:
    #     time.sleep(1)





# atlx = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')
# vocab = read_yaml('data/SemEval15_laptop/glove_symdict.yml')
# atlx['IN_VOCAB'] = atlx.index.isin(vocab)
# annotate_df = atlx.loc[atlx.IN_VOCAB].copy()
# annotate_df['ANNOTATION'] = np.nan
# annotate_df.to_csv('data/lx_annotator/s15_annotation_raw.csv')

