import re
import requests as req
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from tqdm import tqdm
import json
import zipfile
import os

path_driver = "./chromedriver.exe"
driver = webdriver.Chrome()

##  create a list of link for all pages.
list_link = []
for i in range(1, 1430):
    listi = f"https://cwe.mitre.org/data/definitions/{i}.html"
    list_link.append(listi)
    
print(f"INFO[] --> List link contains {len(list_link)}.")
# example with a single page
driver.get('https://cwe.mitre.org/data/definitions/95.html')

language = driver.find_elements(By.CLASS_NAME, 'CodeHead')
code1 = driver.find_elements(By.CLASS_NAME, 'top')

codes = [c.text for c in code1]

language= [i.text for i in language]
print(language)
try:
    for i in range(len(language)):
        lang = language[i].split('\n')[1].split(':')[1]
        code = codes[i]
        # print(lang)
        # print("-------")
        # print(code)
except:
    pass


# All pages

List_CWE_ID = []
List_Language = []
List_Code = []

for link in tqdm(list_link):
    driver.get(link)
    try:
        CWE_ID = driver.find_element(By.CLASS_NAME, 'status')
        ID     = CWE_ID.text.split('\n')[0].split(' ')[2]
    except:
        CWE_ID = ""
        ID = ""
    try:
        Pro_Language = driver.find_elements(By.CLASS_NAME, 'CodeHead')
        language     = [i.text for i in Pro_Language]
    except:
        Pro_Language = ""
        language = ""
    try:
        Codes = driver.find_elements(By.CLASS_NAME, 'top')
        Codes = [c.text for c in Codes]
    except:
        Codes = ""
        Codes = ""
        
    try:
        for i in range(len(language)):
            lang = language[i].split('\n')[1].split(':')[1]
            code = codes[i]

            List_CWE_ID.append(ID)
            List_Language.append(lang)
            List_Code.append(code) 
    except:
        pass
    
knowledge_data = pd.DataFrame({"CWE-ID": ["CWE-"+str(i) for i in List_CWE_ID], 
                              "Language": List_Language,
                              "Sample_code": List_Code})

pathdata = "."
knowledge_data.to_csv(f"{pathdata}/sample_domain_data.csv", index= False)

