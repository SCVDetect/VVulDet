import numpy as np   
from dgl.dataloading import GraphDataLoader
from dgl import load_graphs, save_graphs
import torch.nn.functional as F
import pytorch_lightning as pl
from node2vec import Node2Vec
from pathlib import Path
import networkx as nx
from tqdm import tqdm
from glob import glob
import pickle as pkl
import pandas as pd
import torch as th
import torch
import json
import dgl
import os

try:
    import utills as imp
    import CodeBERT as cb  
    import preprocessdata as prep      
except:
    import utils.utills as imp
    import utils.CodeBERT as cb
    import utils.preprocessdata as prep

class DatasetDataset:
    """Represent Dataset as graph dataset."""

    def __init__(self, partition="train", vulonly=False, sample=-1, splits="default"):
        """Init class."""

        self.finished = [
            int(Path(i).name.split(".")[0])
            for i in glob(str(imp.processed_dir() / "Dataset/before/*nodes*"))
        ]
        self.df = prep.Dataset(splits=splits)
        self.partition = partition
        self.df = self.df[self.df.label == partition]
        self.df = self.df[self.df.id.isin(self.finished)]
        
        if partition == "train" or partition == "val":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0].sample(len(vul) + ((len(vul)*20)%100 ) , random_state=0)
            self.df = pd.concat([vul, nonvul])
        
        if partition == "test":
            vul = self.df[self.df.vul == 1]
            nonvul = self.df[self.df.vul == 0]
            nonvul = nonvul#.sample(min(len(nonvul), len(vul) * 20), random_state=0)
            self.df = pd.concat([vul, nonvul])

        if sample > 0:
            self.df = self.df.sample(sample, random_state=0)
        
        if vulonly:
            self.df = self.df[self.df.vul == 1]
        
        self.df["valid"] = imp.dfmp(
            self.df, DatasetDataset.check_validity, "id", desc="Validate Samples: "
        )
        self.df = self.df[self.df.valid]
        self.df = self.df.reset_index(drop=True).reset_index()
        self.df = self.df.rename(columns={"index": "idx"})
        self.idx2id = pd.Series(self.df.id.values, index=self.df.idx).to_dict()

    
    def itempath(_id):
        """Get itempath path from item id."""
        return imp.processed_dir() / f"Dataset/before/{_id}.c"

    def check_validity(_id):
        """Check whether sample with id=_id has node/edges.

        Example:
        _id = 1320
        with open(str(svd.processed_dir() / f"Dataset/before/{_id}.c") + ".nodes.json", "r") as f:
            nodes = json.load(f)
        """
        valid = 0
        try:
            with open(str(DatasetDataset.itempath(_id)) + ".nodes.json", "r") as f:
                nodes = json.load(f)
                lineNums = set()
                for n in nodes:
                    if "lineNumber" in n.keys():
                        lineNums.add(n["lineNumber"])
                        if len(lineNums) > 1:
                            valid = 1
                            break
                if valid == 0:
                    return False
            with open(str(DatasetDataset.itempath(_id)) + ".edges.json", "r") as f:
                edges = json.load(f)
                edge_set = set([i[2] for i in edges])
                if "REACHING_DEF" not in edge_set and "CDG" not in edge_set:
                    return False
                return True
        except Exception as E:
            print(E, str(DatasetDataset.itempath(_id)))
            return False

    def get_vuln_indices(self, _id):
        """Obtain vulnerable lines from sample ID."""
        df = self.df[self.df.id == _id]
        removed = df.removed.item()
        return dict([(i, 1) for i in removed])

    def stats(self):
        """Print dataset stats."""
        print(self.df.groupby(["label", "vul"]).count()[["id"]])

    def __getitem__(self, idx):
        """Must override."""
        return self.df.iloc[idx].to_dict()

    def __len__(self):
        """Get length of dataset."""
        return len(self.df)

    def __repr__(self):
        """Override representation."""
        vulnperc = round(len(self.df[self.df.vul == 1]) / len(self), 3)
        return f"DatasetDataset(partition={self.partition}, samples={len(self)}, vulnperc={vulnperc})"

def get_sast_lines(sast_pkl_path):
    """Get sast lines from path to sast dump."""
    ret = dict()
    ret["cppcheck"] = set()
    ret["rats"] = set()
    ret["flawfinder"] = set()

    try:
        with open(sast_pkl_path, "rb") as f:
            sast_data = pkl.load(f)
        for i in sast_data:
            if i["sast"] == "cppcheck":
                if i["severity"] == "error" and i["id"] != "syntaxError":
                    ret["cppcheck"].add(i["line"])
            elif i["sast"] == "flawfinder":
                if "CWE" in i["message"]:
                    ret["flawfinder"].add(i["line"])
            elif i["sast"] == "rats":
                ret["rats"].add(i["line"])
    except Exception as E:
        print(E)
        pass
    return ret

def read_CWEvuldescription(_idd):
    """Read CWE vulnerability description."""
    pathd = os.path.join(imp.processed_dir(), "Dataset", "CWEdescription")
    file_path = os.path.join(pathd, f"{_idd}.txt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CWE description file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return f.read()

def read_CVEvuldescription(_idd):
    """Read CVE vulnerability description."""
    pathd = os.path.join(imp.processed_dir(), "Dataset", "CVEdescription")
    file_path = os.path.join(pathd, f"{_idd}.txt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CVE description file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return f.read()

def read_CWESample(_idd):
    """Read CWE sample functions."""
    pathd = os.path.join(imp.processed_dir(), "Dataset", "CWE_Samples")
    file_path = os.path.join(pathd, f"{_idd}.txt")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CWE sample file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        return f.read()

def cosine_similarity(vara, varb):
    """Compute cosine similarity between two variables."""
    dot_product = np.dot(vara, varb.T)
    norm_vara = np.linalg.norm(vara, axis=1, keepdims=True)
    norm_varb = np.linalg.norm(varb, axis=1, keepdims=True)
    
    return dot_product / (norm_vara * norm_varb.T)

class DatasetDatasetVvuldet(DatasetDataset):
    """IVDetect version of Dataset."""

    def __init__(self, gtype="pdg", feat="all", **kwargs):
        """Init."""
        super(DatasetDatasetVvuldet, self).__init__(**kwargs)
        lines = prep.get_dep_add_lines_Dataset()
        lines = {k: set(list(v["removed"]) + v["depadd"]) for k, v in lines.items()}
        self.lines = lines
        self.graph_type = gtype 
        self.feat = feat
        self.delete_cat = []  
        self.processed_ids = set()  

    def item(self, _id, codebert=None):
        """Cache item."""
        savedir = imp.get_dir(
            imp.cache_dir() / f"Dataset_Vvuldet_codebert_{self.graph_type}"
        ) / str(_id)
        if os.path.exists(savedir):
            try:
                g = load_graphs(str(savedir))[0][0]
                
                if "_CODEBERT" in g.ndata:
                    if self.feat == "codebert":
                        for i in ["_GLOVE", "_DOC2VEC", "_RANDFEAT"]:
                            try:
                                g.ndata.pop(i, None)
                            except:
                                print(f"No {i} in nodes feature")
                return g
            except Exception as e:
                print(f"Error loading graph for ID {_id}: {str(e)}")
                self.delete_cat.append(_id)
                raise

        try:
            code, lineno, ei, eo, et = prep.feature_extraction(
                DatasetDataset.itempath(_id), self.graph_type
            )
        except Exception as e:
            print(f"Error in feature extraction for ID {_id}: {str(e)}")
            self.delete_cat.append(_id)
            raise

        if _id in self.lines:
            vuln = [1 if i in self.lines[_id] else 0 for i in lineno]
        else:
            vuln = [0 for _ in lineno]
        g = dgl.graph((eo, ei))
        
        if codebert:
            try:
                code = [c.replace("\\t", "").replace("\\n", "") for c in code]
                chunked_batches = imp.chunks(code, 128)
                features = [codebert.embed(c).detach().cpu() for c in chunked_batches]
                g.ndata["_CODEBERT"] = th.cat(features)
            except Exception as e:
                print(f"Error in CodeBERT embedding for ID {_id}: {str(e)}")
                self.delete_cat.append(_id)
                raise
        
        try:
            g.ndata["_RANDFEAT"] = th.rand(size=(g.number_of_nodes(), 100))
            g.ndata["_LINE"] = th.Tensor(lineno).int()
            g.ndata["_VULN"] = th.Tensor(vuln).float()
            g.ndata["_FVULN"] = g.ndata["_VULN"].max().repeat((g.number_of_nodes()))
            g.edata["_ETYPE"] = th.Tensor(et).long()
            
            emb_path = imp.cache_dir() / f"codebert_method_level/{_id}.pt"
            g.ndata["_FUNC_EMB"] = th.load(emb_path).repeat((g.number_of_nodes(), 1))
            
            nx_graph = g.to_networkx() 
            node2vec = Node2Vec(nx_graph, dimensions=768, walk_length= 5, num_walks= 10, workers=4)
            model = node2vec.fit(window = 5, min_count = 1, batch_words = 8)
            embeddings = model.wv
            node_embeddings = {int(node): embeddings[str(node)] for node in nx_graph.nodes}
            embedding_matrix = torch.tensor([node_embeddings[node.item()] for node in g.nodes()], dtype=torch.float)
            g.ndata['node_embedding'] = embedding_matrix
            src, dst = g.edges()
            src_embeddings = g.ndata['node_embedding'][src]
            dst_embeddings = g.ndata['node_embedding'][dst]
            edge_embeddings = (src_embeddings + dst_embeddings) / 2
            g.edata['edge_embedding'] = edge_embeddings
            g.ndata['node_embedding'] = (g.ndata['node_embedding'] - th.mean(g.ndata['node_embedding'], dim = 0))/th.std(g.ndata['node_embedding'], dim = 0)
            g.edata['edge_embedding'] = (g.edata['edge_embedding'] - th.mean(g.edata['edge_embedding'], dim = 0))/th.std(g.edata['edge_embedding'], dim = 0)
            
            embedder = cb.CodeBertEmbedder(model_path=save_path)
            
            
            desc_ = read_CVEvuldescription(_idd=_id)
            text_ = [desc_]
            embedd_text = embedder.embed(text=text_).detach().cpu()
            g.ndata['_CVEVuldesc'] = th.Tensor(embedd_text).repeat((g.number_of_nodes(), 1))
            
            desc_ = read_CWEvuldescription(_idd=_id)
            text_ = [desc_]
            embedd_text = embedder.embed(text=text_).detach().cpu()
            g.ndata['_CWEVuldesc'] = th.Tensor(embedd_text).repeat((g.number_of_nodes(), 1))
            
            desc_ = read_CWESample(_idd=_id)
            text_ = [desc_]
            embedd_text = embedder.embed(text=text_).detach().cpu()
            g.ndata['_CWESample'] = th.Tensor(embedd_text).repeat((g.number_of_nodes(), 1))
            
            cos_similarity = cosine_similarity(g.ndata["_FUNC_EMB"].numpy(), g.ndata['_CWESample'].numpy())
            max_cos = np.max(cos_similarity)
            
            g.ndata["_FUNC_EMB"] = max_cos * g.ndata["_FUNC_EMB"]
            g.ndata["_CODEBERT"] = max_cos * g.ndata["_CODEBERT"]
            g.ndata["_RANDFEAT"] = max_cos * g.ndata["_RANDFEAT"]
            
            g = dgl.add_self_loop(g)
            save_graphs(str(savedir), [g])
            return g
            
        except Exception as e:
            print(f"Error processing ID {_id}: {str(e)}")
            self.delete_cat.append(_id)
            raise

    def __getitem__(self, idx):
        """Override getitem with error tracking."""
        item_id = self.idx2id[idx]
        if item_id in self.processed_ids:
            return None  
            
        self.processed_ids.add(item_id)
        
        try:
            return self.item(item_id)
        except Exception as e:
            print(f"Error processing ID {item_id}: {str(e)}")
            self.delete_cat.append(item_id)
            return None

    def get_delete_cat(self):
        """Return list of IDs to be deleted."""
        return list(set(self.delete_cat)) 

    def clean_problematic_items(self):
        """Delete all problematic items from cache directories."""
        delete_ids = self.get_delete_cat()
        if not delete_ids:
            print("No problematic items to delete")
            return

        print(f"Cleaning up {len(delete_ids)} problematic items...")
        
        # Delete from codebert_method_level directory
        method_level_dir = imp.get_dir(imp.cache_dir() / "codebert_method_level")
        for item_id in delete_ids:
            pt_file = method_level_dir / f"{item_id}.pt"
            if pt_file.exists():
                try:
                    os.remove(str(pt_file))
                    print(f"Deleted: {pt_file}")
                except Exception as e:
                    print(f"Error deleting {pt_file}: {str(e)}")

        # Delete from graph cache directory
        graph_cache_dir = imp.get_dir(
            imp.cache_dir() / f"Dataset_Vvuldet_codebert_{self.graph_type}"
        )
        for item_id in delete_ids:
            graph_dir = graph_cache_dir / str(item_id)
            if graph_dir.exists():
                try:
                    for f in graph_dir.glob("*"):
                        os.remove(str(f))
                    os.rmdir(str(graph_dir))
                    print(f"Deleted graph cache for ID {item_id}")
                except Exception as e:
                    print(f"Error deleting graph cache for ID {item_id}: {str(e)}")

        print("Cleanup completed")

    def cache_items(self, codebert):
        """Cache all items with error tracking."""
        for i in tqdm(self.df.sample(len(self.df)).id.tolist()):
            if i in self.processed_ids:
                continue
            try:
                self.item(i, codebert)
            except Exception as E:
                print(f"Error caching item {i}: {str(E)}")
                self.delete_cat.append(i)

    def cache_codebert_method_level(self, codebert):
        """Cache method-level embeddings using Codebert."""
        savedir = imp.get_dir(imp.cache_dir() / "codebert_method_level")
        done = [int(i.split("/")[-1].split(".")[0]) for i in glob(str(savedir / "*"))]
        done = set(done)
        batches = imp.chunks((range(len(self.df))), 128)
        
        for idx_batch in tqdm(batches):
            batch_ids = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].id.tolist()
            batch_ids = [i for i in batch_ids if i not in self.delete_cat]  
            
            if not batch_ids:
                continue
                
            batch_texts = self.df.iloc[idx_batch[0] : idx_batch[-1] + 1].before.tolist()
            
            if set(batch_ids).issubset(done):
                continue
                
            texts = ["</s> " + ct for ct in batch_texts]
            try:
                embedded = codebert.embed(texts).detach().cpu()
                assert len(batch_texts) == len(batch_ids)
                for i in range(len(batch_texts)):
                    th.save(embedded[i], savedir / f"{batch_ids[i]}.pt")
            except Exception as e:
                print(f"Error caching batch: {str(e)}")
                for item_id in batch_ids:
                    self.delete_cat.append(item_id)


save_path = f"{imp.cache_dir()}/codebert_finetuned"

class DatasetDatasetVvuldetDataModule(pl.LightningDataModule):
    """Pytorch Lightning Datamodule for Dataset."""

    def __init__(
        self,
        batch_size: int = 32,
        sample: int = -1,
        methodlevel: bool = False,
        nsampling: bool = False,
        nsampling_hops: int = 1,
        gtype: str = "cfgcdg",
        splits: str = "default",
        feat: str = "all",
    ):
        """Init class from Dataset dataset."""
        super().__init__()
        dataargs = {"sample": sample, "gtype": gtype, "splits": splits, "feat": feat}
        self.train = DatasetDatasetVvuldet(partition="train", **dataargs)
        self.val = DatasetDatasetVvuldet(partition="val", **dataargs)
        self.test = DatasetDatasetVvuldet(partition="test", **dataargs)
        codebert = cb.CodeBertEmbedder(model_path = save_path) # save_path is from CodeBERT file
        self.train.cache_codebert_method_level(codebert)
        self.val.cache_codebert_method_level(codebert)
        self.test.cache_codebert_method_level(codebert)
        self.train.cache_items(codebert)
        self.val.cache_items(codebert)
        self.test.cache_items(codebert)
        self.batch_size = batch_size
        self.nsampling = nsampling
        self.nsampling_hops = nsampling_hops
                
     
    def node_dl(self, g, shuffle=False):
        """Return node dataloader."""
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.nsampling_hops)
        return dgl.dataloading.DataLoader(
            g,
            g.nodes(),
            sampler,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=10,
        ) 

    def train_dataloader(self):
        """Return train dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.train, batch_size=len(self.train))))
            return self.node_dl(g, shuffle=True)
        return GraphDataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=15)

    def val_dataloader(self):
        """Return val dataloader."""
        if self.nsampling:
            g = next(iter(GraphDataLoader(self.val, batch_size=len(self.val), num_workers=15)))
            return self.node_dl(g)
        return GraphDataLoader(self.val, shuffle = False, batch_size=self.batch_size, num_workers=15)

    def val_graph_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.val, shuffle = False, batch_size=32, num_workers=15)

    def test_dataloader(self):
        """Return test dataloader."""
        return GraphDataLoader(self.test, shuffle = False, batch_size=32, num_workers=15)


class SCELoss(torch.nn.Module):
    """Symmetric Cross Entropy Loss.

    https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
    """

    def __init__(self, alpha=1, beta=1, num_classes=2):
        """init."""
        super(SCELoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        """Forward."""
        # CCE
        ce = self.cross_entropy(pred, labels)

        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels, self.num_classes)
            .float()
            .to(self.device)
        )
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss
    
