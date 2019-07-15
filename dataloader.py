import numpy as np
import pandas as pd
import sqlite3

def import_sn_data(fpath='sn_data.txt'):
    df = pd.read_csv(fpath, sep='\s+', index_col=0, na_values=['...', 'NaN'])
    df = clean_sn_data(df)
    return df


def clean_sn_data(df):
    df = (df
          .iloc[:,[9,10]]
          .loc[~df['v_siII'].isnull(), :]
          .loc[(df['v_siII'] > 0.7) & (df['v_siII'] < 1.8)]
         )
    df['v_siII'] = df['v_siII'] * 10 # change to units of 10^3 km/s
    return df

def import_all_data():
     sn_data = import_sn_data()

     con = sqlite3.connect('kaepora_v1.db')
     query = "SELECT * FROM Events;"
     kaepora = pd.read_sql(query, con).set_index('SN')
     col_slc = [
     #     'RA', 'DEC', 'M_b_cfa', 'M_b_cfa_err', 'Bmag_salt', 'e_Bmag_salt', 'Bmag_salt2', 'e_Bmag_salt2',
     'V_at_max', 'V_err'
     ]
     colname = [
     #     'RA', 'Dec', 'Bmag_CFA', 'Bmag_CFA_err', 'Bmag_SALT', 'Bmag_SALT_err', 'Bmag_SALT2', 'Bmag_SALT2_err',
     'v_siII', 'v_siII_err'
     ]
     kaepora = kaepora[col_slc].rename({old: new for old, new in zip(col_slc, colname)}, axis=1)
     kaepora = kaepora.loc[kaepora['v_siII'].notna()]
     kaepora['v_siII'] = -kaepora['v_siII']

     return pd.concat([sn_data, kaepora])