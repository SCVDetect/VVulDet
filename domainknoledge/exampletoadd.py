# the code example to add domain in the bigvul dataset
import pandas as pd

pathdata = "."
domain = pd.read_csv(f"{pathdata}/sample_domain_data.csv")
domain.head()

# select only C/C++ reference functions
samplec = domain[(domain['Language'] == ' C ') | (domain['Language'] == ' C# ') | (domain['Language'] == ' C++ ')]

#  read bigvul dataset
bpath = "./datasets/Java"
bivul = pd.read_csv(f"{bpath}/MSR_data_cleaned.csv")

bivul['id'] = bivul.index
bivul['commit_ID'] = bivul['commit_id']
bivul['CVE-ID'] = bivul['CVE ID'] 
bivul['CWE-ID'] = bivul['CWE ID']

def mergedf(df1, df2, on_column: str, hhow = 'left'):
    df = pd.merge(df1, df2, on = on_column, how = hhow)
    df = df[~df.duplicated(subset='id', keep = "last")]
    return df

# download the Mitre 1000.csv desciption from Mitre at: https://cwe.mitre.org/data/csv/1000.csv.zip
bpath = "./datasets/Java"
df1000 = pd.read_csv(f"{bpath}/1000.csv")

df1000['CWE-ID'] = df1000.index
lts = df1000['CWE-ID'].tolist()
listsc = []
for i in lts:
    listsc.append(f"CWE-{i}")
    
df1000['CWE-ID'] = listsc
ddf = pd.merge(bivul, df1000, on='CWE-ID', how='left')
ddf['Description_Mitre']  = ddf['Description']
ddf['Domain_decsriptions'] = 'I could not find CVE descripption'
ddf['P Language'] = "C/C++"

bigvul = pd.merge(samplec, ddf, on='CWE-ID', how='left')
bigvul['Sample Code'] = bigvul['Sample_code']
bigvul['diff_lines'] = ''
bigvul['id'] = bigvul.index

cols = ['id', 'commit_ID','CVE-ID','CWE-ID','project','func_before','func_after','diff_lines','vul',
        'Domain_decsriptions','Description_Mitre','P Language', 'Sample Code']

bigvul = bigvul[cols]
bigvul = bigvul.dropna(subset=['func_before'])

bigvul.to_csv(f"{bpath}/bigvul_domain.csv", index = False)

# A similar process can be used to add the domain information in other datasets and also add other domain information, such as CVE and reference vulnerable functions.