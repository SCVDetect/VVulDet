
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import MatthewsCorrCoef
import pandas as pd
import os

try:
    import utills as imp  
    import preprocessdata as prep      
except:
    import utils.utills as imp
    import utils.preprocessdata as prep

class DatasetDatasetNLP:
    """Override getitem for codebert."""

    def __init__(self, partition="train", random_labels=False):
        """Init."""
        self.df = prep.Dataset()
        self.df = self.df[self.df.label == partition]
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul), random_state=0)
            self.df = pd.concat([vul, nonvul])
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        text = [tokenizer.sep_token + " " + ct for ct in self.df.before.tolist()]
        tokenized = tokenizer(text, **tk_args)
        self.labels = self.df.vul.tolist()
        if random_labels:
            self.labels = torch.randint(0, 2, (len(self.df),)).tolist()
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Override getitem."""
        return self.ids[idx], self.att_mask[idx], self.labels[idx]


class DatasetDatasetNLPLine:
    """Override getitem for codebert."""

    def __init__(self, partition="train"):
        """Init."""
        linedict = prep.get_dep_add_lines_Dataset()
        df = prep.Dataset()
        df = df[df.label == partition]
        df = df[df.vul == 1].copy()
        df = df.sample(min(1000, len(df))) 

        texts = []
        self.labels = []

        for row in df.itertuples():
            line_info = linedict[row.id]
            vuln_lines = set(list(line_info["removed"]) + line_info["depadd"])
            for idx, line in enumerate(row.before.splitlines(), start=1):
                line = line.strip()
                if len(line) < 5:
                    continue
                if line[:2] == "//":
                    continue
                texts.append(line.strip())
                self.labels.append(1 if idx in vuln_lines else 0)

        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        tk_args = {"padding": True, "truncation": True, "return_tensors": "pt"}
        text = [tokenizer.sep_token + " " + ct for ct in texts]
        tokenized = tokenizer(text, **tk_args)
        self.ids = tokenized["input_ids"]
        self.att_mask = tokenized["attention_mask"]

    def __len__(self):
        """Get length of dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Override getitem."""
        return self.ids[idx], self.att_mask[idx], self.labels[idx]



class DatasetDatasetNLPDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Dataset."""

    def __init__(self, DataClass, batch_size: int = 32, sample: int = -1):
        """Init class from Dataset dataset."""
        super().__init__()
        self.train = DataClass(partition="train")
        self.val = DataClass(partition="val")
        self.test = DataClass(partition="test")
        self.batch_size = batch_size

    def train_dataloader(self):
        """Return train dataloader."""
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers = 15)

    def val_dataloader(self):
        """Return val dataloader."""
        return DataLoader(self.val, batch_size=self.batch_size, num_workers = 15)

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(self.test, batch_size=self.batch_size, num_workers = 15)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LitCodeBERT(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.fc1 = nn.Linear(768, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 2)
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)
        self.auroc = torchmetrics.AUROC(task="binary", num_classes=2)

        self.mcc = MatthewsCorrCoef(task="binary", num_classes=2)
    
    def forward(self, ids, mask):
        bert_out = self.bert(ids, attention_mask=mask).pooler_output
        x = self.fc1(bert_out)
        x = self.dropout1(F.relu(x))
        x = self.fc2(x)
        x = self.dropout2(F.relu(x))
        x = self.fc3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        ids, att_mask, labels = batch
        labels = labels.long().to(self.device)
        logits = self(ids, att_mask)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# Fine-tune the model and Save the fine-tuned model
save_path = f"{imp.cache_dir()}/codebert_finetuned"
if not os.path.exists(save_path):
    print(f"Fine-tuning CodeBert model")
    save_path = imp.get_dir(f"{imp.cache_dir()}/codebert_finetuned")
    datamodule = DatasetDatasetNLPDataModule(DatasetDatasetNLP, batch_size=16)
    model = LitCodeBERT(lr=2e-5)
    trainer = pl.Trainer(max_epochs=3, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    trainer.fit(model, datamodule)
    os.makedirs(save_path, exist_ok=True)
    model.bert.save_pretrained(save_path)
    model.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model.tokenizer.save_pretrained(save_path)
else:
    print(f"CodeBERT already exist")

# Load the fine-tuned model for embedding
class CodeBertEmbedder:
    def __init__(self, model_path):
        
        if os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        else:   
            cache_dir = imp.get_dir(f"{cache_dir()}/codebert_model")
            print("[Info] Loading Codebert...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                "microsoft/codebert-base", cache_dir=cache_dir
            )
            self._dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self._dev)
    
    def embed(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            return self.model(tokens["input_ids"], tokens["attention_mask"]).pooler_output


# # test functionallity
# embedder = CodeBertEmbedder(save_path)
# text = """def themain(): on vera bien ce tu prepare chez toi le weekend"""
# embedding = embedder.embed(text)
# print(embedding)
