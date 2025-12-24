Domain Knowledge Data Sources.

1- Collect the sample function and the CVE and CWE descriptions.
- Download a specific Chrome driver that matches your browser, then place it in a particular location and change the driver path accordingly in the code.

- You can consider searching here: [info chrome-driver](https://developer.chrome.com/docs/chromedriver/downloads)

2 - For the ProjectKB dataset, CVE descriptions are extracted from the [ProjectKB repository](https://github.com/SAP/project-kb/tree/vulnerability-data/statements).

3- [collectreffunction.py]() processes several Mitre web pages to search for reference functions, while [exampletoadd.py]() is used to add domain information to the dataset.


