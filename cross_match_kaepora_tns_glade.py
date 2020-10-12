#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from dataloader import import_kaepora


# In[2]:


# Kaepora subsample for paper
kaepora = import_kaepora()
kaepora.info()


# In[3]:


# TNS
tns = pd.read_csv("data/tns_sn_before2016aqv.csv", usecols=range(8))

# Clean SN Name
tns["Name"] = (tns["Name"]
 .str.lower() # Use lowercase SN names
 .str.replace("sn", "") # Remove "sn" prefix
 .str.replace("\s", "") # Replace spaces with whitespace
 .str.strip() # Remove leading and trailing characters.
)

# Clean host name
tns["Host Name"] = (tns["Host Name"]
                    .str.upper() # Use uppercase galaxy names
                    .str.replace("\s",  "")
                    .str.strip()
                   )
tns.info()


# In[4]:


# GLADE
# File is very large, keep only what's necessary
glade_cols = ["PGC", "GWGC name", "HyperLEDA name", "2MASS name", "SDSS-DR12 name", "flag1", "RA", "dec", "dist", "dist_err", "z", "B", "B_err", "B_Abs", "J", "J_err", "H", "H_err", "K", "K_err", "flag2", "flag3"]
glade = pd.read_csv("data/GLADE_2.4.txt", sep="\s+", 
                    names=glade_cols,
                    usecols=["GWGC name", "HyperLEDA name", "2MASS name", "SDSS-DR12 name", "z", "B", "B_err", "B_Abs", "dist", "dist_err"]
                   )

# Removing rows where galaxy name aren't recorded
# Cast galaxy name columns as string because some of them are read as numerics
glade = (glade
         .dropna(subset=["GWGC name", "HyperLEDA name", "2MASS name"], how="all")
         .astype({"GWGC name": str, "HyperLEDA name": str, "2MASS name": str, "SDSS-DR12 name": str})
        )

glade[["GWGC name", "HyperLEDA name", "2MASS name"]] = glade[["GWGC name", "HyperLEDA name", "2MASS name"]].apply(
    lambda s: ((s
                .str.upper()  # Use uppercase galaxy names
                .str.replace("\s", "")
                .str.strip())
               ))

# glade = glade[
# #     (glade["GWGC name"].isin(tns["Host Name"].unique())) & 
#     (~glade["GWGC name"].isnull())
# ]

glade.info()


# In[5]:


# Merge kaepora and TNS on SN name
kaepora_tns = kaepora.merge(tns, how="inner", left_index=True, right_index=False, right_on="Name", suffixes=("_kaepora", "_tns"))
kaepora_tns.info()


# In[6]:


# Merge kaepora+TNS with GLADE on galaxy name
kaepora_tns_glade = kaepora_tns.merge(glade, how="left", left_on="Host Name", right_on="HyperLEDA name", suffixes=("_kaepora", "_glade"))
kaepora_tns_glade.info()


# In[7]:


# Save to file
(kaepora_tns_glade
    .rename(columns={"Host Redshift": "z_tns", "Name": "sn"})
    .sort_values(["HyperLEDA name", "sn"])
    [["sn", "Host Name", "HyperLEDA name", "v_siII", "v_siII_err",
        "z_kaepora", "z_glade", "z_tns", "B", "B_err", "dist", "dist_err", "RA", "DEC"]]
 ).to_csv("kaepora_tns_glade.csv", index=False)


# In[12]:


# Finding NGC<1-3 numbers>
matches = tns["Host Name"].str.match("NGC\d{1,3}$", na=False)
tns["Host Name"][matches]


# In[32]:


# Padding zeroes to NGC<1-3numbers> to NGC<4 numbers>

(tns["Host Name"][matches]
 .str.extract("(\d+)", expand=False) # Extract the digits
 .apply(lambda ngc_number: f"NGC{int(ngc_number):04d}") # Pad until 4 digits
)

