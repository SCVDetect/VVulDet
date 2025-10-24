
import pandas as pd
import boto3
import json
import time
import random
from botocore.exceptions import ClientError
from tqdm import tqdm


df = pd.read_csv("Testdata/projectkbtest.csv")
df = df[1658:]
df['Code'] = df['func_after'] 
# Bedrock client for Llama 3
client = boto3.client("bedrock-runtime", region_name="eu-west-1")
model_id = "arn:aws:bedrock:eu-west-1:742833394292:inference-profile/eu.meta.llama3-2-3b-instruct-v1:0"

def analyze_code_with_llama(code_snippet: str, unique_id: str):
    messages = [
        {
            "role": "user",
            "content": [{
                "text": (
                    "You are a software vulnerability detection assistant.\n"
                    "Analyze the following source code and respond strictly in JSON with keys:\n"
                    "  - vul_status: 'yes' or 'no'\n"
                    "  - vulnerable_lines: a list of vulnerable line numbers, or empty list if none.\n\n"
                    f"Code:\n{code_snippet}\n\n"
                    "Answer only in JSON format, no explanations."
                )
            }]
        }
    ]

    inference_config = {
        "maxTokens": 512,
        "temperature": 0.5,
        "topP": 0.9,
    }

    retries = 0
    while retries < 5:
        try:
            response = client.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig=inference_config,
                additionalModelRequestFields={}
            )

            output = response.get("output", {}).get("message", {}).get("content", [])
            if not output or "text" not in output[0]:
                raise ValueError("Unexpected response format from model.")
            output_text = output[0]["text"].strip()

            # Try parsing JSON
            try:
                parsed = json.loads(output_text)
                vul_status = parsed.get("vul_status", "").lower()
                vulnerable_lines = parsed.get("vulnerable_lines", [])
                
                
                if isinstance(vulnerable_lines, str):
                    vulnerable_lines = [int(x.strip()) for x in vulnerable_lines.replace('-',',').split(',') if x.strip().isdigit()]
                elif not isinstance(vulnerable_lines, list):
                    vulnerable_lines = []

            except json.JSONDecodeError:
                vul_status = "yes" if "yes" in output_text.lower() else "no"
                vulnerable_lines = []

            return vul_status, vulnerable_lines

        except ClientError as e:
            if "ThrottlingException" in str(e):
                wait = (2 ** retries) + random.uniform(1, 3)
                time.sleep(wait)
                retries += 1
            else:
                return "error", []
        except Exception as e:
            return "error", []

    return "error", []


df['vul_status'] = ""
df['vulnerable_lines'] = None 

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    vul_status, vulnerable_lines = analyze_code_with_llama(row["Code"], row["id"])
    df.at[idx, 'vul_status'] = vul_status
    df.at[idx, 'vulnerable_lines'] = vulnerable_lines
    time.sleep(random.uniform(10, 20)) 


df.to_csv("./llama/llama_vulnerability_results_full23.csv", index=False)

print("Analysis complete! Results saved to llama_vulnerability_results_full.csv")
print(df.head())



# analysis of the results


import pandas as pd
import numpy as np
import os 

df = pd.read_csv("llama/llama_vulnerability_results_full.csv")




df['vul_status'] = df['vul_status'].map({'yes':1, 'no':0})


print("stat:", df['vul_status'].value_counts())
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import numpy as np

y_true = df['target'].fillna(0).astype(int)
y_pred = df['vul_status'].astype(int)
y_pred_proba = (y_pred == 1).astype(int)  

precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

if len(np.unique(y_true)) > 1:
    PRAUC = average_precision_score(y_true, y_pred_proba)
else:
    PRAUC = np.nan
    print("Skipping PR-AUC — y_true contains only one class.")

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
print(f"PR-AUC: {PRAUC}")


# ------------------ statememt level-----------------
import difflib

def compare_functions_line_by_line(func_before, func_after):
   
    if pd.isna(func_before) or pd.isna(func_after):
        return [], []
    
    lines_before = func_before.split('\n')
    lines_after = func_after.split('\n')
    
    diff_lines = []
    code_lines = []
    

    diff = difflib.unified_diff(lines_before, lines_after, lineterm='', n=0)
    diff_list = list(diff)
    
    for line in diff_list:
        if line.startswith('@@'):
           
            parts = line.split(' ')
            before_part = parts[1]
            after_part = parts[2]
            
            start_before = int(before_part.split(',')[0].replace('-', ''))
            start_after = int(after_part.split(',')[0].replace('+', ''))
            
            current_line = start_after
            continue
        
        elif line.startswith('+') and not line.startswith('+++'):
            
            diff_lines.append(current_line)
            code_lines.append(line[1:])  
            current_line += 1
            
        elif line.startswith('-') and not line.startswith('---'):
        
            pass
        else:
            
            current_line += 1
    
    return diff_lines, code_lines


def simple_line_comparison(func_before, func_after):
    """
    Simple line-by-line comparison
    """
    if pd.isna(func_before) or pd.isna(func_after):
        return [], []
    
    lines_before = func_before.split('\n')
    lines_after = func_after.split('\n')
    
    diff_lines = []
    code_lines = []
    
  
    max_lines = max(len(lines_before), len(lines_after))
    
    for i in range(max_lines):
        line_before = lines_before[i] if i < len(lines_before) else ""
        line_after = lines_after[i] if i < len(lines_after) else ""
        
        if line_before != line_after:
            diff_lines.append(i + 1)  
            code_lines.append(line_after)
    
    return diff_lines, code_lines


df[['line_level', 'code_line']] = df.apply(
    lambda row: pd.Series(simple_line_comparison(row['func_before'], row['func_after'])),
    axis=1
)

df.to_csv("./llama/dfllama__final.csv", index=False)

dffinal = pd.read_csv("./llama/dfllama__final.csv", converters={
    'line_level': pd.eval,
    'vulnerable_lines': pd.eval
})


def expand_to_line_level(df, max_lines=None):

    records = []
    for _, row in df.iterrows():
        true_set = set(row['line_level']) if isinstance(row['line_level'], list) else set()
        pred_set = set(row['vulnerable_lines']) if isinstance(row['vulnerable_lines'], list) else set()
        if max_lines is None:
            max_ln = max(true_set.union(pred_set)) if (true_set or pred_set) else 0
        else:
            max_ln = max_lines
        for ln in range(1, max_ln+1):
            records.append({
                'id': row['id'],
                'line_no': ln,
                'true_vul': 1 if ln in true_set else 0,
                'pred_vul': 1 if ln in pred_set else 0
            })
    return pd.DataFrame.from_records(records)

# Expand
expanded = expand_to_line_level(dffinal)

y_true = expanded['true_vul']
y_pred = expanded['pred_vul']

precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
PRAUC     = average_precision_score(y_true, y_pred)

print(f"Line-Level Precision: {precision:.3f}")
print(f"Line-Level Recall:   {recall:.3f}")
print(f"Line-Level F1:       {f1:.3f}")
print(f"Line-Level PR-AUC:   {PRAUC:.3f}")
