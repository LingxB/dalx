import pandas as pd
pd.set_option('display.width', 1000)



gold_lx = pd.read_csv('data/lx_annotator/s15_annotation_out.csv', index_col='WORD')
# raw_lx = pd.read_csv('data/processed/lexicon_table_v2.csv', index_col='WORD')

gold_lx = gold_lx.drop('IN_VOCAB', axis=1)

gold_lx['RAW_AVG'] = gold_lx[['MPQA', 'OPENER', 'OL', 'VADER']].mean(axis=1)


pure_annotated = gold_lx[['ANNOTATION']].copy()
pure_annotated['A1'] = pure_annotated.ANNOTATION
pure_annotated['A2'] = pure_annotated.ANNOTATION


avg_annotated = gold_lx[['ANNOTATION', 'RAW_AVG']].mean(axis=1).to_frame('AVG_ANNOTATION')
avg_annotated['AA1'] = avg_annotated.AVG_ANNOTATION
avg_annotated['AA2'] = avg_annotated.AVG_ANNOTATION




pure_annotated.to_csv('data/output/lexicon_table_dalx_15_gold.csv')
avg_annotated.to_csv('data/output/lexicon_table_dalx_15_gold_avg.csv')
gold_lx.to_csv('data/output/lexicon_table_dalx_15_out_ref.csv')
