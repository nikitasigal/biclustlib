import os
from os.path import dirname, join

import pandas as pd


def load_tavazoie() -> pd.DataFrame:
    return pd.read_csv(join(dirname(__file__), 'tavazoie.csv'), index_col='gene_id')


def load_prelic() -> pd.DataFrame:
    return pd.read_csv(join(dirname(__file__), 'prelic.csv'), index_col='gene_id')


def load_jaskowiak() -> list[pd.DataFrame]:
    return [pd.read_csv(file, index_col='gene_id') for file in
            os.listdir(join(dirname(__file__), 'jaskowiak'))]
