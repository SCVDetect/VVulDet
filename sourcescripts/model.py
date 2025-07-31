import pytorch_lightning as pl
import torch.nn as nn
import torch as th
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torchmetrics
import torch     
import dgl
from dgl import load_graphs
import os
from dgl.nn import GATConv, GraphConv
from torch.optim import AdamW
from utils.preprocessdata import Dataset
import utils.allcvecwefeaturemanip as gpht
import utils.utills as imp 

torch.cuda.empty_cache()

class LitGNN(pl.LightningModule):
    """Main Trainer."""

    def __init__(
        self,
        hfeat: int = 512, 
        embtype: str = "codebert",
        embfeat: int = -1,  
        num_heads: int = 4,
        lr: float = 1e-4, #1e-3, # 1e-4
        hdropout: float = 0.2,
        mlpdropout: float = 0.2,
        gatdropout: float = 0.2,
        methodlevel: bool = False,
        nsampling: bool = False,
        model: str = "gat2layer",
        loss: str = "ce", # "sce", # 
        multitask: str = "linemethod",
        stmtweight: int = 1, # 5
        gnntype: str = "gat",
        random: bool = False,
        scea: float = 0.7,
    ):
        """Initialization."""
        super().__init__()
        self.lr = lr
        self.random = random
        self.save_hyperparameters()
        
        self.n2vec_projector = nn.Linear(1536, 768)
        self.despt_projector = nn.Linear(1536, 768)
        self.cve_projector = nn.Linear(768, 384)  
        self.cwe_projector = nn.Linear(768, 384)  
        self.embedding_combiner = nn.Linear(768 + 384 + 384, 768)  
        
        self.test_step_outputs = []

        if self.hparams.embtype == "codebert":
            self.hparams.embfeat = 768
            self.EMBED = "_CODEBERT"

        if self.hparams.loss == "sce":
            self.loss = gpht.SCELoss(self.hparams.scea, 1 - self.hparams.scea)
            self.loss_f = th.nn.CrossEntropyLoss()
        else:
            self.loss = th.nn.CrossEntropyLoss(
                weight=th.Tensor([1, self.hparams.stmtweight]) 
            )
            self.loss_f = th.nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2, average='macro')
        self.mcc = torchmetrics.MatthewsCorrCoef(task="binary", num_classes=2)

        hfeat = self.hparams.hfeat
        gatdrop = self.hparams.gatdropout
        numheads = self.hparams.num_heads
        embfeat = self.hparams.embfeat
        gnn_args = {"out_feats": hfeat}
        if self.hparams.gnntype == "gat":
            gnn = GATConv
            gat_args = {"num_heads": numheads, "feat_drop": gatdrop}
            gnn1_args = {**gnn_args, **gat_args, "in_feats": embfeat}
            gnn2_args = {**gnn_args, **gat_args, "in_feats": hfeat * numheads}
        elif self.hparams.gnntype == "gcn":
            gnn = GraphConv
            gnn1_args = {"in_feats": embfeat, **gnn_args}
            gnn2_args = {"in_feats": hfeat, **gnn_args}

        if "gat" in self.hparams.model:
            self.gat = gnn(**gnn1_args)
            self.gat2 = gnn(**gnn2_args)
            fcin = hfeat * numheads if self.hparams.gnntype == "gat" else hfeat
            self.fc = th.nn.Linear(fcin, self.hparams.hfeat)
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        if "mlponly" in self.hparams.model:
            self.fconly = th.nn.Linear(embfeat, self.hparams.hfeat)
            self.mlpdropout = th.nn.Dropout(self.hparams.mlpdropout)

        if "+femb" in self.hparams.model:
            self.fc_femb = th.nn.Linear(embfeat * 2, self.hparams.hfeat)

        self.codebertfc = th.nn.Linear(768, self.hparams.hfeat)

        self.fch = []
        for _ in range(8):
            self.fch.append(th.nn.Linear(self.hparams.hfeat, self.hparams.hfeat))
        self.hidden = th.nn.ModuleList(self.fch)
        self.hdropout = th.nn.Dropout(self.hparams.hdropout)
        self.fc2 = th.nn.Linear(self.hparams.hfeat, 2)

    def forward(self, g, test=False, e_weights=[], feat_override=""):
        """Forward pass."""
        if self.hparams.nsampling and not test:
            hdst = g[2][-1].dstdata[self.EMBED]
            h_func = g[2][-1].dstdata["_FUNC_EMB"]
            g2 = g[2][1]
            g = g[2][0]
            if "gat2layer" in self.hparams.model:
                n2vec = g.ndata['node_embedding']['_N']
                n2vec = torch.nan_to_num(n2vec, nan=0.0, posinf=0.0, neginf=0.0)
                cve_desc = g.ndata['_CVEVuldesc']['_N']
                cwe_desc = g.ndata['_CWEVuldesc']['_N']
                cve_proj = self.cve_projector(cve_desc)
                cwe_proj = self.cwe_projector(cwe_desc)
                h = g.srcdata[self.EMBED]
                h = th.cat([h, n2vec], dim=1)
                h = self.n2vec_projector(h)
                h = th.cat([h, cve_proj, cwe_proj], dim=1)
                h = self.embedding_combiner(h)
            elif "gat1layer" in self.hparams.model:
                h = g2.srcdata[self.EMBED]
                n2vec = g.ndata['node_embedding']['_N']
                n2vec = torch.nan_to_num(n2vec, nan=0.0, posinf=0.0, neginf=0.0)
                cve_desc = g.ndata['_CVEVuldesc']['_N']
                cwe_desc = g.ndata['_CWEVuldesc']['_N']
                cve_proj = self.cve_projector(cve_desc)
                cwe_proj = self.cwe_projector(cwe_desc)
                h = th.cat([h, n2vec, cve_proj, cwe_proj], dim=1)
                
        else:
            g2 = g
            n2vec = g.ndata.get('node_embedding')
            n2vec = torch.nan_to_num(n2vec, nan=0.0, posinf=0.0, neginf=0.0)
            cve_desc = g.ndata['_CVEVuldesc']
            cwe_desc = g.ndata['_CWEVuldesc']
            
            cve_proj = self.cve_projector(cve_desc)
            cwe_proj = self.cwe_projector(cwe_desc)
            
            h = g.srcdata[self.EMBED]
            h = th.cat([h, n2vec], dim=1)
            h = self.n2vec_projector(h)
            h = th.cat([h, cve_proj, cwe_proj], dim=1)
            h = self.embedding_combiner(h)
            
            if len(feat_override) > 0:
                h = g.ndata[feat_override]
            h_func = g.ndata["_FUNC_EMB"]
            hdst = h

        if self.random:
            return th.rand((h.shape[0], 2)).to(self.device), th.rand(
                h_func.shape[0], 2
            ).to(self.device)

        if "+femb" in self.hparams.model:
            h = th.cat([h, h_func], dim=1)
            h = F.elu(self.fc_femb(h))

        if self.hparams.embfeat != 768:
            h_func = self.codebertfc(h_func)

        if "gat" in self.hparams.model:
            if "gat2layer" in self.hparams.model:
                h = self.gat(g, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
                h = self.gat2(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            elif "gat1layer" in self.hparams.model:
                h = self.gat(g2, h)
                if self.hparams.gnntype == "gat":
                    h = h.view(-1, h.size(1) * h.size(2))
            h = self.mlpdropout(F.elu(self.fc(h)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        if test and len(e_weights) > 0:
            g.ndata["h"] = h
            g.edata["ew"] = e_weights
            g.update_all(
                dgl.function.u_mul_e("h", "ew", "m"), dgl.function.mean("m", "h")
            )
            h = g.ndata["h"]

        if "mlponly" in self.hparams.model:
            h = self.mlpdropout(F.elu(self.fconly(hdst)))
            h_func = self.mlpdropout(F.elu(self.fconly(h_func)))

        for idx, hlayer in enumerate(self.hidden):
            h = self.hdropout(F.elu(hlayer(h)))
            h_func = self.hdropout(F.elu(hlayer(h_func)))
        h = self.fc2(h)
        h_func = self.fc2(h_func)

        if self.hparams.methodlevel:
            g.ndata["h"] = h
            return dgl.mean_nodes(g, "h"), None
        else:
            return h, h_func

    def shared_step(self, batch, test=False):
        """Shared step."""
        logits = self(batch, test)
        if self.hparams.methodlevel:
            if self.hparams.nsampling:
                raise ValueError("Cannot train on method level with nsampling.")
            labels = dgl.max_nodes(batch, "_VULN").long()
            labels_func = None
        else:
            if self.hparams.nsampling and not test:
                labels = batch[2][-1].dstdata["_VULN"].long()
                labels_func = batch[2][-1].dstdata["_FVULN"].long()
            else:
                labels = batch.ndata["_VULN"].long()
                labels_func = batch.ndata["_FVULN"].long()
        return logits, labels, labels_func

    def training_step(self, batch, batch_idx):
        """Training step."""
        logits, labels, labels_func = self.shared_step(batch)
        loss1 = self.loss(logits[0], labels)
        logits1 = logits[0]
        
        if not self.hparams.methodlevel:
            loss2 = self.loss_f(logits[1], labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
            
        else:
            loss = loss1
            acc_func = self.accuracy(logits, labels_func)
            self.log("train_acc_func", acc_func, prog_bar=True, logger=True, batch_size=batch_idx)
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None
        
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_idx)
        self.log("train_loss_func", loss2, on_epoch=True, prog_bar=True, batch_size=batch_idx)
        self.log("train_acc", self.accuracy(preds, labels), prog_bar=True, batch_size=batch_idx)
        
        train_mcc = self.mcc(preds, labels).float()
        self.log("train_mcc", train_mcc, prog_bar=True, batch_size=batch_idx)
        
        if not self.hparams.methodlevel:
            self.log("train_acc_func", self.accuracy(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)
            
            train_mcc_func = self.mcc(preds_func, labels_func).float()
            self.log("train_mcc_func", train_mcc_func, prog_bar=True, batch_size=batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits, labels, labels_func = self.shared_step(batch)
        logits1 = logits[0]
        loss1 = self.loss(logits1, labels)
        if not self.hparams.methodlevel:
            logits2 = logits[1]
            loss2 = self.loss_f(logits2, labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None
        
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_idx)
        self.log("val_acc", self.accuracy(preds, labels), prog_bar=True, batch_size=batch_idx)
        
        val_mcc = self.mcc(preds, labels).float()
        self.log("val_mcc", val_mcc, prog_bar=True, batch_size=batch_idx)

        if not self.hparams.methodlevel:
            self.log("val_acc_func", self.accuracy(preds_func, labels_func), prog_bar=True, batch_size=batch_idx)

            val_mcc_func = self.mcc(preds_func, labels_func).float()
            self.log("val_mcc_func", val_mcc_func, prog_bar=True, batch_size=batch_idx)
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        logits, labels, labels_func = self.shared_step(batch, test=True)
        logits1 = logits[0]
        loss1 = self.loss(logits1, labels)
        if not self.hparams.methodlevel:
            logits2 = logits[1]
            loss2 = self.loss_f(logits2, labels_func)
            loss = (loss1 + self.hparams.stmtweight * loss2) / 2
        else:
            loss = loss1
        preds = th.argmax(logits1, dim=1)
        preds_func = th.argmax(logits[1], dim=1) if not self.hparams.methodlevel else None

        metrics = {
            "test_loss": loss,
            "test_acc": self.accuracy(preds, labels),
            "test_mcc": self.mcc(preds, labels).float(), 
        }
        
        if not self.hparams.methodlevel:
            metrics["t_acc_func"] = self.accuracy(preds_func, labels_func)
            metrics["t_mcc_func"] = self.mcc(preds_func, labels_func).float()  

        self.test_step_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):
        """Test epoch end."""
        avg_metrics = {
            key: th.mean(th.stack([x[key] for x in self.test_step_outputs]))
            for key in self.test_step_outputs[0].keys()
        }
        self.test_step_outputs.clear()
        self.log_dict(avg_metrics)
        return
        
    def configure_optimizers(self):
        """Configure optimizers."""
        return AdamW(self.parameters(), lr=self.lr)

