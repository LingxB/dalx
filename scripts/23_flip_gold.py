import pandas as pd
import random
random.seed(42)


def random_flip(samples):
    flipped = []
    for s in samples:
        if s == 1:
            p = random.choice([-1, 0])
        elif s == 0:
            p = random.choice([-1, 1])
        elif s == -1:
            p = random.choice([0, 1])
        else:
            raise ValueError
        flipped.append(p)
    return flipped

def sample_and_replace(df, frac):
    _df = df.copy()
    samples = _df.sample(frac=frac, random_state=42)
    flipped = random_flip(samples['ANNOTATION'])
    samples['FLIPPED'] = flipped
    for w,f in zip(samples.index, flipped):
        _df.loc[w] = [f] * 3
    return _df, samples


gold = pd.read_csv('data/lx_annotator/lexicon_table_dalx_15_gold_v2.csv', index_col='WORD')


for i in range(11):
    frac = float(i/10)
    muddy, sampled = sample_and_replace(gold, frac)
    muddy.to_csv(f'data/output/muddy/muddy_gold_{frac}.csv')




