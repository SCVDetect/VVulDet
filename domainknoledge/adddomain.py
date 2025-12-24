import pandas as pd
import numpy as np
import random
import re
import json
import os
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def extract_cve_description(cve_id, cve_base_path):
    """
    Extract CVE description from JSON files in the CVE v5 format directory structure
    """
    if not cve_id or pd.isna(cve_id):
        return "CVE description not available"
    
    cve_str = str(cve_id).strip()
    
    # Check if it's a valid CVE format
    if not cve_str.startswith('CVE-'):
        return f"Invalid CVE format: {cve_str}"
    
    try:
        parts = cve_str.split('-')
        if len(parts) != 3:
            return f"Invalid CVE format: {cve_str}"
        
        year = parts[1]
        cve_number = int(parts[2])
        
        # Determine the directory structure based on CVE number
        if cve_number < 1000:
            dir_name = "0xxx"
        elif cve_number < 10000:
            dir_name = "1xxx"
        elif cve_number < 20000:
            dir_name = "10xxx"
        elif cve_number < 30000:
            dir_name = "20xxx"
        elif cve_number < 40000:
            dir_name = "30xxx"
        elif cve_number < 50000:
            dir_name = "40xxx"
        elif cve_number < 60000:
            dir_name = "50xxx"
        elif cve_number < 70000:
            dir_name = "60xxx"
        elif cve_number < 80000:
            dir_name = "70xxx"
        elif cve_number < 90000:
            dir_name = "80xxx"
        elif cve_number < 100000:
            dir_name = "90xxx"
        elif cve_number < 1000000:
            dir_name = "1000xxx"
        else:
            # For very large CVE numbers
            dir_name = f"{cve_number // 1000}xxx"
        
        # Construct the file path
        json_file_path = os.path.join(cve_base_path, year, dir_name, f"{cve_str}.json")
        
        # Try to read the JSON file
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as f:
                cve_data = json.load(f)
            
            # Extract description from the JSON structure
            # Try multiple possible paths for description
            descriptions = []
            
            # Try path: containers.cna.descriptions
            if 'containers' in cve_data:
                if 'cna' in cve_data['containers']:
                    if 'descriptions' in cve_data['containers']['cna']:
                        for desc in cve_data['containers']['cna']['descriptions']:
                            if desc.get('lang') == 'en' and 'value' in desc:
                                descriptions.append(desc['value'])
                    # Also try 'description' (singular)
                    elif 'description' in cve_data['containers']['cna']:
                        if isinstance(cve_data['containers']['cna']['description'], list):
                            for desc in cve_data['containers']['cna']['description']:
                                if desc.get('lang') == 'en' and 'value' in desc:
                                    descriptions.append(desc['value'])
                        elif isinstance(cve_data['containers']['cna']['description'], dict):
                            if cve_data['containers']['cna']['description'].get('lang') == 'en':
                                descriptions.append(cve_data['containers']['cna']['description'].get('value', ''))
            
            # Try path: cveMetadata
            if not descriptions and 'cveMetadata' in cve_data:
                if 'description' in cve_data['cveMetadata']:
                    if isinstance(cve_data['cveMetadata']['description'], list):
                        for desc in cve_data['cveMetadata']['description']:
                            if desc.get('lang') == 'en' and 'value' in desc:
                                descriptions.append(desc['value'])
            
            # Try path: descriptions (top level)
            if not descriptions and 'descriptions' in cve_data:
                for desc in cve_data['descriptions']:
                    if desc.get('lang') == 'en' and 'value' in desc:
                        descriptions.append(desc['value'])
            
            # If we found descriptions, return the first English one
            if descriptions:
                # Clean up the description
                description = descriptions[0]
                # Remove excessive whitespace
                description = ' '.join(description.split())
                # Limit length to avoid overly long descriptions
                if len(description) > 500:
                    description = description[:497] + "..."
                return description
            else:
                return "No English description found in CVE JSON"
        
        else:
            # Try alternative path structure (without xxx dirs)
            alt_json_file_path = os.path.join(cve_base_path, year, f"{cve_str}.json")
            if os.path.exists(alt_json_file_path):
                with open(alt_json_file_path, 'r', encoding='utf-8') as f:
                    cve_data = json.load(f)
                
                # Extract description using same logic as above
                if 'containers' in cve_data and 'cna' in cve_data['containers']:
                    if 'descriptions' in cve_data['containers']['cna']:
                        for desc in cve_data['containers']['cna']['descriptions']:
                            if desc.get('lang') == 'en' and 'value' in desc:
                                return ' '.join(desc['value'].split())[:500]
            
            return f"CVE JSON file not found: {cve_str}"
    
    except json.JSONDecodeError:
        return f"Invalid JSON format for {cve_str}"
    except Exception as e:
        return f"Error extracting CVE description: {str(e)[:100]}"

# Set paths
cve_base_path = "/home/wp3/Domain-Kn/VVulDet/domainknoledge/cvelistV5-main/cves"
pathdata = "."

print(f"Reading domain data from: {pathdata}")
try:
    domain = pd.read_csv(f"{pathdata}/sample_domain_data.csv")
    print(f"Domain data columns: {list(domain.columns)}")
except FileNotFoundError:
    print(f"ERROR: Domain data file not found at {pathdata}/sample_domain_data.csv")
    print("Please check the file path and try again.")
    exit(1)

print(f"\nTotal domain samples: {len(domain)}")
print(f"Unique CWE-IDs in domain: {domain['CWE-ID'].nunique() if 'CWE-ID' in domain.columns else 'CWE-ID column not found'}")

# Clean language
if 'Language' in domain.columns:
    domain['Language'] = domain['Language'].str.strip()
    print("\nAvailable languages in domain data:")
    print(domain['Language'].value_counts())
else:
    print("WARNING: 'Language' column not found in domain data")
    domain['Language'] = 'Unknown'

# Filter for C/C++/C# languages
if 'Language' in domain.columns:
    samplec = domain[domain['Language'].isin(['C', 'C#', 'C++', ' C ', ' C# ', ' C++ '])]
else:
    samplec = domain.copy()
    samplec['Language'] = 'C/C++'

print(f"\nFiltered domain samples (C/C++/C#): {len(samplec)}")

# Remove duplicates
if 'CWE-ID' in samplec.columns:
    samplec = samplec.drop_duplicates(subset='CWE-ID', keep='first')
    print(f"After removing duplicate CWE-IDs: {len(samplec)}")
else:
    print("WARNING: 'CWE-ID' column not found, cannot remove duplicates")

# Read bigvul dataset
bpath = "/home/wp3/DataProcessing" 
try:
    bivul = pd.read_csv(f"{bpath}/preprocessed_bigvul_dataset.csv")
    print(f"BigVul columns: {list(bivul.columns)}")
except FileNotFoundError:
    print(f"ERROR: BigVul dataset not found at {bpath}/preprocessed_bigvul_dataset.csv")
    exit(1)
    
if 'id' not in bivul.columns:
    bivul['id'] = bivul.index

if 'commit_id' in bivul.columns:
    bivul['commit_ID'] = bivul['commit_id']
elif 'commit_ID' in bivul.columns:
    pass 
else:
    bivul['commit_ID'] = f"commit_{bivul['id']}"

if 'CVE ID' in bivul.columns:
    bivul['CVE-ID'] = bivul['CVE ID']
elif 'CVE-ID' in bivul.columns:
    pass 
else:
    bivul['CVE-ID'] = f"CVE-placeholder-{bivul['id']}"

if 'CWE ID' in bivul.columns:
    bivul['CWE-ID'] = bivul['CWE ID']
elif 'CWE-ID' in bivul.columns:
    pass  
else:
    cwe_found = False
    for col in bivul.columns:
        if 'cwe' in col.lower() or 'CWE' in col:
            bivul['CWE-ID'] = bivul[col]
            cwe_found = True
            break
    if not cwe_found:
        bivul['CWE-ID'] = f"CWE-placeholder-{bivul['id']}"

# Clean CWE-ID
def clean_cwe_id(cwe_id):
    if pd.isna(cwe_id):
        return None
    cwe_str = str(cwe_id).strip()
    if cwe_str.startswith('CWE-'):
        return cwe_str
    elif cwe_str.isdigit():
        return f"CWE-{cwe_str}"
    elif 'CWE' in cwe_str.upper():
        match = re.search(r'CWE[-\s]*(\d+)', cwe_str.upper())
        if match:
            return f"CWE-{match.group(1)}"
        return cwe_str
    else:
        return cwe_str

if 'CWE-ID' in bivul.columns:
    bivul['CWE-ID'] = bivul['CWE-ID'].apply(clean_cwe_id)
else:
    bivul['CWE-ID'] = f"CWE-placeholder-{bivul['id']}"

if 'CWE-ID' in samplec.columns:
    samplec['CWE-ID'] = samplec['CWE-ID'].apply(clean_cwe_id)
else:
    samplec['CWE-ID'] = f"DOMAIN-CWE-{samplec.index}"

print(f"\nBigVul dataset size: {len(bivul)}")
print(f"Unique CWE-IDs in BigVul: {bivul['CWE-ID'].nunique()}")
print(f"Unique CVE-IDs in BigVul: {bivul['CVE-ID'].nunique()}")

# Extract CVE descriptions for all unique CVE-IDs in BigVul
print("\nExtracting CVE descriptions from JSON files...")
unique_cves = bivul['CVE-ID'].dropna().unique()
cve_descriptions_cache = {}

# First, let's test if we can access the CVE directory
print(f"CVE base path: {cve_base_path}")
if os.path.exists(cve_base_path):
    print(f"CVE directory exists: {os.path.exists(cve_base_path)}")
    print(f"Number of unique CVE-IDs to process: {len(unique_cves)}")
    
    # Process CVE descriptions with progress
    for i, cve_id in enumerate(unique_cves):
        if i % 100 == 0:
            print(f"Processed {i}/{len(unique_cves)} CVE-IDs...")
        
        if cve_id and pd.notna(cve_id) and str(cve_id).startswith('CVE-'):
            description = extract_cve_description(cve_id, cve_base_path)
            cve_descriptions_cache[cve_id] = description
else:
    print(f"WARNING: CVE directory not found at {cve_base_path}")
    print("Using placeholder CVE descriptions instead...")

# Try to load MITRE 1000.csv for descriptions
try:
    mitre_path = "./1000.csv"
    print(f"\nAttempting to load MITRE descriptions from: {mitre_path}")
    df1000 = pd.read_csv(mitre_path)
    
    if 'CWE-ID' not in df1000.columns:
        df1000['CWE-ID'] = df1000.index
    df1000['CWE-ID'] = df1000['CWE-ID'].apply(lambda x: f"CWE-{x}" if str(x).isdigit() else clean_cwe_id(x))
    
    if 'Description' not in df1000.columns:
        desc_cols = [col for col in df1000.columns if 'desc' in col.lower() or 'name' in col.lower()]
        if desc_cols:
            df1000['Description'] = df1000[desc_cols[0]]
        else:
            df1000['Description'] = "No description available"
    
    print(f"MITRE data loaded successfully: {len(df1000)} entries")
    
    # Merge BigVul with MITRE descriptions
    print("Merging BigVul with MITRE descriptions...")
    ddf = pd.merge(bivul, df1000[['CWE-ID', 'Description']], on='CWE-ID', how='left')
    ddf['Description_Mitre'] = ddf['Description'].fillna("CWE description not available in MITRE database")
    
except Exception as e:
    print(f"WARNING: Could not load MITRE descriptions: {e}")
    print("Using placeholder descriptions instead...")
    ddf = bivul.copy()
    ddf['Description_Mitre'] = "CWE description not available (MITRE data missing)"

# Set programming language
if 'P Language' not in ddf.columns:
    ddf['P Language'] = "C/C++"

print(f"\nDataset after MITRE merge: {len(ddf)} rows")

correct_matches = []
random_assignments = []
language_usage = defaultdict(list)

used_domain_samples = set()

def merge_with_fallback(bigvul_df, domain_df):
    """Merge datasets with fallback to random domain samples when no match found"""
    result_rows = []
    
    if 'Language' not in domain_df.columns:
        domain_df['Language'] = 'C/C++'
    domain_df['Language'] = domain_df['Language'].str.strip()
    
    domain_by_lang = defaultdict(list)
    for idx, row in domain_df.iterrows():
        lang = row['Language']
        domain_by_lang[lang].append(idx)
    
    available_by_lang = domain_by_lang.copy()
    
    if 'CWE-ID' not in domain_df.columns:
        domain_df['CWE-ID'] = [f"DOMAIN-{i}" for i in range(len(domain_df))]
    
    if 'Sample_code' not in domain_df.columns and 'Sample Code' in domain_df.columns:
        domain_df['Sample_code'] = domain_df['Sample Code']
    elif 'Sample_code' not in domain_df.columns:
        domain_df['Sample_code'] = ["// Sample code not available"] * len(domain_df)
    
    for idx, bigvul_row in bigvul_df.iterrows():
        cwe_id = bigvul_row.get('CWE-ID', None)
        cve_id = bigvul_row.get('CVE-ID', None)
        
        domain_match = None
        domain_idx = None
        
        if cwe_id and cwe_id != 'None' and pd.notna(cwe_id):
            if 'CWE-ID' in domain_df.columns:
                exact_matches = domain_df[domain_df['CWE-ID'] == cwe_id]
                if not exact_matches.empty:
                    domain_match = exact_matches.iloc[0]
                    domain_idx = exact_matches.index[0]
                    correct_matches.append(cwe_id)
        
        if domain_match is None:
            target_lang = "C"
            
            if target_lang not in available_by_lang:
                similar_langs = [lang for lang in available_by_lang.keys() if 'C' in lang]
                if similar_langs:
                    target_lang = similar_langs[0]
                else:
                    target_lang = list(available_by_lang.keys())[0] if available_by_lang else 'Unknown'
            
            available_indices = available_by_lang.get(target_lang, [])
            
            if available_indices:
                random_idx = random.choice(available_indices)
                domain_match = domain_df.loc[random_idx]
                domain_idx = random_idx
            
                available_indices.remove(random_idx)
                available_by_lang[target_lang] = available_indices
                
                random_assignments.append({
                    'bivul_id': idx,
                    'bivul_cwe': cwe_id,
                    'domain_cwe': domain_match['CWE-ID'],
                    'language': target_lang
                })
            else:
                domain_match = pd.Series({
                    'CWE-ID': 'NO_DOMAIN',
                    'Language': target_lang,
                    'Sample_code': '// No domain sample available for this language'
                })
        
        # Get CVE description from cache or extract it
        cve_description = "CVE description not available"
        if cve_id and pd.notna(cve_id):
            if cve_id in cve_descriptions_cache:
                cve_description = cve_descriptions_cache[cve_id]
            else:
                # Try to extract it now
                cve_description = extract_cve_description(cve_id, cve_base_path)
                cve_descriptions_cache[cve_id] = cve_description
        
        # Create merged row
        merged_row = bigvul_row.to_dict()
        
        merged_row.update({
            'Domain_CWE-ID': domain_match['CWE-ID'],
            'Domain_decsriptions': cve_description,  # Actual CVE description
            'Description_Mitre': bigvul_row.get('Description_Mitre', 'CWE description not available'),
            'P Language': bigvul_row.get('P Language', 'C/C++'),
            'Sample Code': domain_match.get('Sample_code', '// Sample code not available'),
            'diff_lines': '',
            'Domain_Match_Type': 'Exact' if (cwe_id and 'CWE-ID' in domain_match and 
                                           domain_match['CWE-ID'] == cwe_id) else 'Random'
        })
        
        if domain_idx is not None and domain_match['CWE-ID'] != 'NO_DOMAIN':
            used_domain_samples.add(domain_idx)
        
        result_rows.append(merged_row)
    
    return pd.DataFrame(result_rows)

print("\nMerging datasets with fallback mechanism...")
bigvul_merged = merge_with_fallback(ddf, samplec)

required_columns = ['id', 'commit_ID', 'CVE-ID', 'CWE-ID', 'Domain_CWE-ID', 'project', 
                    'func_before', 'func_after', 'diff_lines', 'vul',
                    'Domain_decsriptions', 'Description_Mitre', 'P Language', 
                    'Sample Code', 'Domain_Match_Type']

missing_columns = []
for col in required_columns:
    if col not in bigvul_merged.columns:
        missing_columns.append(col)
        if col == 'project':
            bigvul_merged['project'] = 'unknown_project'
        elif col == 'func_before':
            func_cols = [c for c in bigvul_merged.columns if 'func' in c.lower() or 'code' in c.lower()]
            if func_cols:
                bigvul_merged['func_before'] = bigvul_merged[func_cols[0]]
            else:
                bigvul_merged['func_before'] = '// Function code not available'
        elif col == 'func_after':
            bigvul_merged['func_after'] = '// Patched function not available'
        elif col == 'vul':
            vul_cols = [c for c in bigvul_merged.columns if 'vul' in c.lower() or 'label' in c.lower()]
            if vul_cols:
                bigvul_merged['vul'] = bigvul_merged[vul_cols[0]]
            else:
                bigvul_merged['vul'] = 1  
        elif col == 'diff_lines':
            bigvul_merged['diff_lines'] = ''
        elif col == 'Domain_Match_Type':
            if 'Domain_Match_Type' not in bigvul_merged.columns:
                bigvul_merged['Domain_Match_Type'] = 'Unknown'

if missing_columns:
    print(f"Created missing columns: {missing_columns}")

available_cols = [col for col in required_columns if col in bigvul_merged.columns]
print(f"\nAvailable columns for final dataset: {available_cols}")

bigvul_merged = bigvul_merged[available_cols]


if 'func_before' in bigvul_merged.columns:
    before_len = len(bigvul_merged)
    bigvul_merged = bigvul_merged.dropna(subset=['func_before'])
    bigvul_merged = bigvul_merged[bigvul_merged['func_before'].astype(str).str.strip() != '']
    after_len = len(bigvul_merged)
    print(f"\nDropped {before_len - after_len} rows with missing or empty func_before")
    print(f"Final dataset size: {after_len}")

output_path1 = "/mnt/c/Users/cholo/Downloads"
output_path = f"{output_path1}/Prepocess_bigvul_domain.csv"
bigvul_merged.to_csv(output_path, index=False)
print(f"\nSaved processed data to: {output_path}")

# Print CVE description statistics
print("\n" + "="*80)
print("CVE DESCRIPTION STATISTICS")
print("="*80)

if cve_descriptions_cache:
    successful_extractions = sum(1 for desc in cve_descriptions_cache.values() 
                                 if not desc.startswith(('CVE JSON file not found', 'Error extracting', 'Invalid CVE')))
    failed_extractions = len(cve_descriptions_cache) - successful_extractions
    
    print(f"\nCVE Description Extraction Results:")
    print(f"  Total CVE-IDs processed: {len(cve_descriptions_cache)}")
    print(f"  Successful extractions: {successful_extractions} ({successful_extractions/len(cve_descriptions_cache)*100:.1f}%)")
    print(f"  Failed extractions: {failed_extractions} ({failed_extractions/len(cve_descriptions_cache)*100:.1f}%)")
    
    # Show some examples
    print("\nSample CVE descriptions extracted:")
    sample_cves = list(cve_descriptions_cache.keys())[:3]
    for cve in sample_cves:
        desc = cve_descriptions_cache[cve]
        print(f"\n  {cve}:")
        print(f"    {desc[:100]}..." if len(desc) > 100 else f"    {desc}")

# Print detailed report
print("\n" + "="*80)
print("DOMAIN MATCHING REPORT")
print("="*80)

# Calculate statistics
total_samples = len(bigvul_merged)
exact_matches_count = len(correct_matches)
random_assignments_count = len(random_assignments)

print(f"\nTotal BigVul samples processed: {total_samples}")
print(f"Exact CWE-ID matches: {exact_matches_count} ({exact_matches_count/total_samples*100:.1f}%)")
print(f"Random domain assignments: {random_assignments_count} ({random_assignments_count/total_samples*100:.1f}%)")

# Check for samples without domain info
if 'Sample Code' in bigvul_merged.columns:
    no_domain_count = bigvul_merged['Sample Code'].apply(lambda x: 'No domain sample' in str(x)).sum()
    print(f"Samples with no domain info (placeholders): {no_domain_count}")

# Domain sample utilization
print(f"\nDomain samples used: {len(used_domain_samples)} out of {len(samplec)} available")
print(f"Domain sample utilization: {len(used_domain_samples)/len(samplec)*100:.1f}%")

# Check that all required domain columns are present
print("\n" + "="*80)
print("DOMAIN COLUMNS CHECK")
print("="*80)

required_domain_cols = ['Description_Mitre', 'Domain_decsriptions', 'Sample Code']
for col in required_domain_cols:
    if col in bigvul_merged.columns:
        non_empty = bigvul_merged[col].apply(lambda x: str(x).strip() != '').sum()
        print(f">>> {col}: Present with {non_empty}/{total_samples} non-empty values ({non_empty/total_samples*100:.1f}%)")
    else:
        print(f"<<< {col}: MISSING from final dataset!")

# Verify a few sample rows
print(f"\nSample of final data (first 3 rows):")
for i in range(min(3, len(bigvul_merged))):
    row = bigvul_merged.iloc[i]
    print(f"\nRow {i+1}:")
    print(f"  CVE-ID: {row.get('CVE-ID', 'N/A')}")
    print(f"  CWE-ID: {row.get('CWE-ID', 'N/A')}")
    print(f"  Domain CWE-ID: {row.get('Domain_CWE-ID', 'N/A')}")
    print(f"  Match Type: {row.get('Domain_Match_Type', 'N/A')}")
    print(f"  Description_Mitre length: {len(str(row.get('Description_Mitre', '')))} chars")
    print(f"  Domain_decsriptions: {str(row.get('Domain_decsriptions', ''))[:100]}...")
    print(f"  Sample Code preview: {str(row.get('Sample Code', ''))[:50]}...")

print("\n" + "="*20)
print("PROCESSING COMPLETE")
print("="*20)