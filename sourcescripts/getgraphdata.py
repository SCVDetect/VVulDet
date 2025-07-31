
import numpy as np
import os

import utils.utills as imp 
import utils.nodeedgesdata as xtra
import utils.preprocessdata as prep

NUM_JOBS = 1
JOB_ARRAY_NUMBER = 0 


df = prep.Dataset()

df = df.iloc[::-1]
splits = np.array_split(df, NUM_JOBS)


def preprocess(row):
    """
    df = Dataset()
    row = df.iloc[180189] 
    row = df.iloc[177860]  
    preprocess(row)
    """
    savedir_before = imp.get_dir(imp.processed_dir() / row["dataset"] / "before")
    savedir_after = imp.get_dir(imp.processed_dir() / row["dataset"] / "after")
    
    savedir_description_CVE = imp.get_dir(imp.processed_dir() / row['dataset'] / "CVEdescription")
    savedir_description_CWE = imp.get_dir(imp.processed_dir() / row['dataset'] / "CWEdescription")
    savedir_sample_func = imp.get_dir(imp.processed_dir() / row['dataset'] / "CWE_Samples")

    fpath1 = savedir_before / f"{row['id']}.java"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.java"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])
            
    fpath3 = savedir_description_CVE / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath3}.txt") :
        with open(fpath3, 'w') as f:
            f.write(row['CVE_vuldescription'])     
              
    fpath4 = savedir_description_CWE / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath4}.txt") : 
        with open(fpath4, 'w') as f:
            f.write(row['CWE_vuldescription'])
            
    fpath5 = savedir_sample_func / f"{row['id']}.txt"
    if not os.path.exists(f"{fpath5}.txt"): 
        with open(fpath5, 'w') as f:
            f.write(row['CWE_Sample'])
 
    if not os.path.exists(f"{fpath1}.edges.json"):
        xtra.full_run_joern(fpath1, verbose=3)

    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        xtra.full_run_joern(fpath2, verbose=3)
    
        

if __name__ == "__main__":
    imp.dfmp(splits[JOB_ARRAY_NUMBER], preprocess, ordr=False, workers=8)
