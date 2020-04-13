import yaml
import pandas as pd


def save_yaml(dict, path, mode='w'):
    with open(path, mode) as outfile:
        yaml.dump(dict, outfile, default_flow_style=False)

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as ymlfile:
        yml = yaml.load(ymlfile)
    return yml

def load_semeval15_laptop(train, test):
    _train = pd.read_csv(train)
    _train['IS_TRAIN'] = True
    _test = pd.read_csv(test)
    _test['IS_TRAIN'] = False
    return pd.concat([_train, _test], ignore_index=True)

def load_semeval14_restruant(train, test):
    return load_semeval15_laptop(train, test)

def search_keyword(df, key: str, col: str = 'SENT'):
    in_sent = df[col].apply(lambda s: True if key in s.split() else False)
    return df.loc[in_sent]