from pathlib import Path
import numpy as np
import pandas as pd
import sqlite3
from astropy.io import ascii

PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data"

###
# SNe OBSERVATIONAL DATA
###
def import_wang13(fname="Wang13_ts1.txt"):
    df = pd.read_csv(
        DATA_PATH / fname,
        sep="\s+",
        index_col="SN",
        usecols=["SN", "v_siII", "v_siII_err"],
    )
    df["v_siII"] = np.around(df["v_siII"] * 10 ** 4, -2)  # change to units of km/s
    df["v_siII_err"] = df["v_siII_err"]
    df = clean_sn_data(df)
    df.index.name = "Wang13"
    return df


def import_folatelli13(fname="Folatelli13_t3.txt"):
    df = ascii.read(DATA_PATH / fname, format="cds").to_pandas()
    df = df[["Name", "SiII-6355", "e_SiII-6355"]]
    df = df.rename(
        columns={"Name": "SN", "SiII-6355": "v_siII", "e_SiII-6355": "v_siII_err"}
    ).set_index("SN")
    df["v_siII"] = np.around(df["v_siII"])
    df = clean_sn_data(df)
    df.index.name = "Folatelli13"
    return df


def import_foley11(fname="Foley11_t1.txt"):
    df = pd.read_csv(DATA_PATH / fname, sep="\t", index_col="SN")
    df = df.rename(columns={"V0SiII": "v_siII", "e_V0SiII": "v_siII_err"})
    df["v_siII"] = np.around(-df["v_siII"] * 10 ** 3, -1)  # change to units of km/s
    df["v_siII_err"] = df["v_siII_err"] * 10 ** 3
    df = clean_sn_data(df)
    df.index.name = "Foley11"
    return df


def import_zheng18(fname="Zheng18_t1.txt"):
    df = pd.read_csv(
        DATA_PATH / fname,
        sep="\t",
        usecols=["SN", "v_siII", "v_siII_err"],
        index_col="SN",
        comment="#",
    )
    df["v_siII"] = np.around(df["v_siII"] * 10 ** 3, -2)  # change to units of km/s
    df["v_siII_err"] = df["v_siII_err"] * 10 ** 3
    df = clean_sn_data(df)
    df.index.name = "Zheng18"
    return df


def import_kaepora(fname="kaepora_v1.db"):
    con = sqlite3.connect(str(DATA_PATH / fname))
    query = "SELECT * FROM Events;"
    df = pd.read_sql(query, con)[["SN", "V_at_max", "V_err", "Redshift"]].set_index(
        "SN"
    )
    df = df.rename(
        columns={"V_at_max": "v_siII", "V_err": "v_siII_err", "Redshift": "z"}
    )
    df["v_siII"] = -df["v_siII"] * 10 ** 3
    df["v_siII_err"] = df["v_siII_err"] * 10 ** 3
    df = clean_sn_data(df)
    df.index.name = "kaepora"
    return df


def clean_sn_data(df):
    # Remove null data
    df = df.loc[~df["v_siII"].isnull(), :]

    # Fix SN name to be follow sn<yyyy><abcd>
    df.index = map(lambda x: x.lower().replace(" ", "").replace("sn", ""), df.index)
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


###
# SNe Model Data
###


def import_townsley19(fname="Townsley19.txt"):
    df = pd.read_csv(
        DATA_PATH / fname, sep="\t", usecols=["cos_theta", "v_siII", "v_siII_err"],
    )
    df["viewing_angle"] = np.rad2deg(np.arccos(df["cos_theta"]))
    df["v_siII"] = -df["v_siII"]
    df = df.drop(columns=["cos_theta"])
    df.index.name = "Townsley19"
    return df


def import_kasen07(fname="Kasen07.txt"):
    df = pd.read_csv(DATA_PATH / fname, sep="\s+")
    df["v_siII"] = -df["v_siII"]
    df.index.name = "Townsley07"
    return df


if __name__ == "__main__":
    from itertools import cycle

    ts = import_all_data(merge=False)
    kaepora = import_kaepora()
    sns_nokaepora = set()
    sns_kaepora_overlap = set()
    sns = set()
    for t in ts:
        sns.update(t.index)

        if t.index.name != "kaepora":
            sns_nokaepora.update(t.index)
            sns_kaepora_overlap.update(set(t.index).intersection(set(kaepora.index)))

    for t in ts:
        print(f"{t.index.name} Size: {len(t)}")

    print(f"Total Dataset Size (w/o kaepora): {len(sns_nokaepora)}")
    print(f"Total Dataset Size: {len(sns)}")
    print(f"KAEPORA Overlap Size: {len(sns_kaepora_overlap)}")
