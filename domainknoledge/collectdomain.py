import re
import requests as req
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import pandas as pd
from tqdm import tqdm
import json
import zipfile
import os
import time

chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--headless')  
chrome_options.add_argument('--disable-gpu')

from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager


driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), 
                         options=chrome_options)


list_link = []
for i in range(1, 1430):
    listi = f"https://cwe.mitre.org/data/definitions/{i}.html"
    list_link.append(listi)
    
print(f"INFO[] --> List link contains {len(list_link)}.")


print("Testing with single page...")
driver.get('https://cwe.mitre.org/data/definitions/95.html')
time.sleep(2)  

try:
    language = driver.find_elements(By.CLASS_NAME, 'CodeHead')
    code1 = driver.find_elements(By.CLASS_NAME, 'top')
    codes = [c.text for c in code1]
    language = [i.text for i in language]
    print(f"Found {len(language)} code examples")
    
    for i in range(len(language)):
        try:
            lang = language[i].split('\n')[1].split(':')[1]
            code = codes[i]
            print(f"Language: {lang}")
            print("-------")
            print(code[:100] + "..." if len(code) > 100 else code)
            print("\n")
        except:
            print(f"Error parsing language at index {i}")
            pass
except Exception as e:
    print(f"Error in test: {e}")


print("\nStarting to scrape all pages...")
List_CWE_ID = []
List_Language = []
List_Code = []

for link in tqdm(list_link):
    try:
        driver.get(link)
        time.sleep(0.5)  
        
        try:
            CWE_ID = driver.find_element(By.CLASS_NAME, 'status')
            ID = CWE_ID.text.split('\n')[0].split(' ')[2]
        except:
            ID = ""
        
        try:
            Pro_Language = driver.find_elements(By.CLASS_NAME, 'CodeHead')
            language = [i.text for i in Pro_Language]
        except:
            language = []
        
        try:
            Codes = driver.find_elements(By.CLASS_NAME, 'top')
            Codes = [c.text for c in Codes]
        except:
            Codes = []
        
        try:
            for i in range(len(language)):
                lang = language[i].split('\n')[1].split(':')[1]
                code = Codes[i] if i < len(Codes) else ""

                List_CWE_ID.append(ID)
                List_Language.append(lang)
                List_Code.append(code) 
        except Exception as e:
            
            if ID:  # Only add if we have an ID
                List_CWE_ID.append(ID)
                List_Language.append("")
                List_Code.append("")
            
    except Exception as e:
        print(f"\nError processing {link}: {e}")
        continue


print(f"\nScraped {len(List_CWE_ID)} code examples")
knowledge_data = pd.DataFrame({
    "CWE-ID": ["CWE-" + str(i) for i in List_CWE_ID], 
    "Language": List_Language,
    "Sample_code": List_Code
})


pathdata = "."
knowledge_data.to_csv(f"{pathdata}/sample_domain_data.csv", index=False)
print(f"Data saved to {pathdata}/sample_domain_data.csv")

driver.quit()



# # Install Chrome browser
# sudo apt-get install -y google-chrome-stable

# sudo apt install -y wget
# wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
# sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
# sudo apt update
# sudo apt install -y google-chrome-stable

# google-chrome --version