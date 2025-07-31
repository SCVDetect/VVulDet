### Domain-Aware Graph Neural Networks for Source Code Vulnerability Detection
Paper submitted to XXX Journal.

### Section 1: Experiment Replication

The source code for training the model is located in `./sourcescripts`, and the dataset construction instructions can be found in `./domainknowledge`.

1. **Clone the project repository.**

```bash
   git clone https://github.com/SCVDetect/VVulDet.git
```

2. **Install the required Python packages.**

```bash
pip install -r requirements.txt
```
3. **Dataset and CPG Extraction.**

Dataset: We used publicly available datasets named BigVul-C/C++, Project_KB-Java, MegaVul-Java, and CVEFixes-Python.

CPG Extraction: We use Joern to parse the source code, extracting relevant nodes and edge data.

Running the following commands will install a specific Joern version for CPG extraction and download the Python version of the CVEFixes dataset from our drive.

```bash
chmod +x ./run.sh
./run.sh
./zrun/getjoern.sh
```

4. **Train/Test.**

```bash
./zrun/Process_train_test.sh
```
5. **Results.**
The results will be stored in ```storage/outputs/```. We provided a pre-trained model on Megavul data, a fine-tuned CodeBERT for feature embedding, and a set of functions constructed for testing. These can be downloaded while running ```.zrun/getJoern.sh```. Alternatively, a fully constructed function can be downloaded directly from Zenodo at [Link](https://zenodo.org/records/16629448?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM3YTE1NzY0LTViN2UtNGE0NS1hOTVkLTA3NzdiYTU4YzkzYiIsImRhdGEiOnt9LCJyYW5kb20iOiIwNTQ2MWVlNjAxOWQ3OGE1NWMwNWMyZWIyYWViNDU4NyJ9.xQtMKVIZkkUWTmHSOrjU85PB3S6VMTTe85v8TgAlVxEHD-CmWWv4iPrdG1jYAtGAvs_ZMfyD8QbQ1FulpIzriA).

An example of the returned results in the form of a CSV is available at ```./sourcescripts/storage/output```, containing some metrics.

Below is a preview of the contents of `cwe_level_metrics.csv`:

| CWE_ID   | num_functions | num_statements | func_accuracy | func_precision | func_recall | func_f1 | stmt_accuracy | stmt_precision | stmt_recall | stmt_f1 | func_pr_auc | stmt_pr_auc |
|----------|---------------|----------------|---------------|----------------|-------------|---------|---------------|----------------|-------------|---------|--------------|--------------|
| CWE-400  | 160           | 1041           | 0.95          | 0.6364         | 0.9745      | 0.7012  | 0.9837        | 0.7423         | 0.5289      | 0.5485  | 0.3010       | 0.0799       |
| CWE-284  | 114           | 1283           | 0.9912        | 0.75           | 0.9956      | 0.8311  | 0.9992        | 0.4996         | 0.5         | 0.4998  | 0.25         | 0.0116       |
| CWE-89   | 183           | 1875           | 0.9891        | 0.9167         | 0.9942      | 0.9516  | 0.9349        | 0.5173         | 0.5464      | 0.5210  | 0.9313       | 0.0801       |
| CWE-502  | 123           | 1062           | 1.0           | 1.0            | 1.0         | 1.0     | 0.9868        | 0.4943         | 0.4990      | 0.4967  | 1.0          | 0.0119       |
| CWE-79   | 283           | 2443           | 0.9929        | 0.9474         | 0.9962      | 0.9703  | 0.9824        | 0.6583         | 0.5115      | 0.5178  | 0.8009       | 0.0949       |
| CWE-863  | 207           | 1728           | 0.9952        | 0.75           | 0.9976      | 0.8321  | 0.9988        | 0.9994         | 0.6667      | 0.7497  | 0.25         | 0.5917       |
| CWE-22   | 185           | 1985           | 0.9784        | 0.875          | 0.9884      | 0.9227  | 0.9879        | 0.8275         | 0.5397      | 0.5684  | 0.8745       | 0.2195       |



### Section 2: **Gathering Domain Data.**

Navigate to ```./domainknowledge```.




### Citation

To be provided


##### Acknowledgment:
###### We thank [LineVD](https://github.com/davidhin/linevd) for providing the source code of their project, which has served as a foundation for the current research project.
