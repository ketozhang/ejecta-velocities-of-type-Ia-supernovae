from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
from astropy.io import ascii

PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / 'data'


def import_wang13(fname='Wang13_ts1.txt'):
    df = pd.read_csv(DATA_PATH/fname, sep='\s+', index_col='SN',
                     usecols=['SN', 'v_siII', 'v_siII_err'])
    df['v_siII'] = np.around(df['v_siII'] * 10**4, -2)  # change to units of km/s
    df['v_siII_err'] = df['v_siII_err']
    df = clean_sn_data(df)
    df.index.name = 'Wang13'
    return df


def import_folatelli13(fname='Folatelli13_t3.txt'):
    df = ascii.read(DATA_PATH / fname, format='cds').to_pandas()
    df = df[['Name', 'SiII-6355', 'e_SiII-6355']]
    df = df.rename(columns={'Name': 'SN',
                            'SiII-6355': 'v_siII',
                            'e_SiII-6355': 'v_siII_err'}
                   ).set_index('SN')
    df['v_siII'] = np.around(df['v_siII'])
    df = clean_sn_data(df)
    df.index.name = 'Folatelli13'
    return df


def import_foley11(fname='Foley11_t1.txt'):
    df = pd.read_csv(DATA_PATH/fname, sep='\t', index_col='SN')
    df = df.rename(columns={
        'V0SiII': 'v_siII',
        'e_V0SiII': 'v_siII_err'
    })
    df['v_siII'] = np.around(-df['v_siII'] * 10**3, -1) # change to units of km/s
    df['v_siII_err'] = df['v_siII_err'] * 10**3
    df = clean_sn_data(df)
    df.index.name = 'Foley11'
    return df


def import_zheng18(fname='Zheng18_t1.txt'):
    df = pd.read_csv(DATA_PATH/fname, sep='\t',
                     usecols=['SN', 'v_siII', 'v_siII_err'], index_col='SN', comment='#')
    df['v_siII'] = np.around(df['v_siII'] * 10**3, -2)  # change to units of km/s
    df['v_siII_err'] = df['v_siII_err'] * 10**3
    df = clean_sn_data(df)
    df.index.name = 'Zheng18'
    return df


def import_kaepora(fname='kaepora_v1.db'):
    con = sqlite3.connect(DATA_PATH/fname)
    query = "SELECT * FROM Events;"
    df = pd.read_sql(query, con)[['SN', 'V_at_max', 'V_err']].set_index('SN')
    colname = ['v_siII', 'v_siII_err']
    df = df.rename(columns={
        'V_at_max': 'v_siII',
        'V_err': 'v_siII_err'
    })
    df['v_siII'] = -df['v_siII'] * 10**3
    df['v_siII_err'] = df['v_siII_err'] * 10**3
    df = clean_sn_data(df)
    df.index.name = 'kaepora'
    return df

def clean_sn_data(df):
    df = (df
          .loc[~df['v_siII'].isnull(), :]  # Remove null data
          # Filter 5000 < v <= 20000
          .loc[(df['v_siII'] > 5000) & (df['v_siII'] <= 20000)]
          )
    df.index = map(
        lambda x: x.lower().replace(' ', '').replace('sn', ''), df.index
    )
    return df


def import_all_data(merge=True):
    wang13 = import_wang13()
    folatelli13 = import_folatelli13()
    foley11 = import_foley11()
    zheng18 = import_zheng18()
    kaepora = import_kaepora()

    if merge:
        return pd.concat([wang13, folatelli13, foley11, zheng18, kaepora])
    else:
        return [wang13, folatelli13, foley11, zheng18, kaepora]


if __name__ == '__main__':
    from itertools import cycle
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style='whitegrid', color_codes=True)
    colors = cycle(['r', 'b', 'g', 'orange', 'purple', 'black'])
    shapes = cycle(['o', '+', 'd', '^', 's'])

    ts = import_all_data(merge=False)
    sns_nokaepora = set()
    sns = set()
    plt.figure(figsize=(25,5))
    for t in ts:
        t['source'] = t.index.name
        sns.update(t.index)
        if t.index.name != 'kaepora':
            sns_nokaepora.update(t.index)

    for t in ts:
        plt.scatter(t.index, t['v_siII'], label=t.index.name, marker=next(shapes), color=next(colors))
        print(f"{t.index.name} Size: {len(t)}")

    print(f"Total Dataset Size (w/o kaepora): {len(sns_nokaepora)}")
    print(f"Total Dataset Size: {len(sns)}")

    # df = import_all_data()
    # with open('sn_v_siII2.txt', 'w') as f:
    #     pd.options.display.max_rows = len(df)
    #     f.write(df[['source', 'v_siII', 'v_siII_err']].__repr__())
    # with open('sn_v_siII_pivoted2.txt', 'w') as f:
    #     f.write(df.pivot(columns='source').__repr__())
    # print(pd.read_csv('sn_v_siII.txt', sep='\s+').head())
    # print(pd.read_csv('sn_v_siII_pivoted.txt', sep='\s+').head())

    # plt.xticks(rotation=90)
    # plt.legend()
    # plt.show()