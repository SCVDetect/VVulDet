import pytorch_lightning as pl
import torch.nn as nn
import torch as th
import pandas as pd
import numpy as np
import torch.nn.functional as F
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
from model import *
from tqdm import tqdm
from sklearn.metrics import (f1_score, precision_score, accuracy_score,
                           recall_score, roc_auc_score, matthews_corrcoef,
                           precision_recall_curve, auc)
from pytorch_lightning.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

class VulnerabilityAnalyzer:
    def __init__(self, output_dir, max_epochs=30):
        """
        Initialize the vulnerability analyzer.
        
        Args:
            output_dir: Directory to save output files
            max_epochs: Number of training epochs if training new model
        """
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.function_data = []
        self.statement_data = []
        self.cwe_data = {}
        
        self.checkpoint_path = os.path.join(output_dir, "checkpoints")
        self.model_path = os.path.join(self.checkpoint_path, "model-checkpoint.ckpt")
        
    def run(self):
        self.model = self._get_model()
        self._prepare_test_data()
        self.analyze()
    
    def _get_model(self):
        """Load trained model or train new one if needed."""
        if os.path.exists(self.model_path):
            print("Loading pretrained model...")
            return LitGNN.load_from_checkpoint(self.model_path)
        else:
            print("Training new model...")
            return self._train_model()
    
    def _train_model(self):
        """Train and save new model."""
        model = LitGNN(
            hfeat=512,
            embtype="codebert",
            methodlevel=False,
            nsampling=True,
            model="gat2layer",
            loss="ce",
            hdropout=0.5,
            gatdropout=0.3,
            num_heads=4,
            multitask="linemethod",
            stmtweight=1,
            gnntype="gat",
            scea=0.5,
            lr=2e-6 # 1e-5, 1e-3, 1e-4
        )
        
        data = gpht.DatasetDatasetVvuldetDataModule(
            batch_size=64,
            sample=-1,
            methodlevel=False,
            nsampling=True,
            nsampling_hops=2,
            gtype="pdg+raw",
            splits="default",
        )
        
        os.makedirs(self.checkpoint_path, exist_ok=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=self.checkpoint_path,
            filename="model-checkpoint",
            monitor="val_loss"
        )
        
        trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            default_root_dir=self.output_dir,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback],
            max_epochs=self.max_epochs,
            strategy=DDPStrategy(find_unused_parameters=True)
        )
        
        trainer.fit(model, data)
        trainer.save_checkpoint(self.model_path)
        return model
    
    def _prepare_test_data(self):
        """Prepare test data for analysis."""
        print("Preparing test data...")
        df = Dataset()
        dftest = df[df['label'] == 'test']
        gpath = f"{imp.cache_dir()}/Dataset_Vvuldet_codebert_pdg+raw"
        
        self.test_graphs = []
        self.dataset_df = dftest
        
        for graph_file in tqdm(os.listdir(gpath), desc="Loading test graphs"):
            if int(graph_file) in dftest['id'].tolist():
                try:
                    g = load_graphs(os.path.join(gpath, graph_file))[0][0]
                    g.path = os.path.join(gpath, graph_file) 
                    self.test_graphs.append(g)
                except Exception as e:
                    print(f"Error loading graph {graph_file}: {e}")
    
    def analyze(self):
        """Run the full analysis pipeline."""
        print("Starting vulnerability analysis...")
        
        for graph in tqdm(self.test_graphs, desc="Analyzing graphs"):
            self._process_graph(graph)
        
        self._generate_function_level_csv()
        self._generate_cwe_level_csv()
        self._calculate_metrics()
        
        print("Analysis completed successfully!")
    
    def _process_graph(self, graph):
        """Process a single graph and collect predictions."""
        graph_id = int(os.path.basename(graph.path))
        graph_data = self.dataset_df[self.dataset_df['id'] == graph_id].iloc[0]
        
        with torch.no_grad():
            # graph = graph.to(self.device)
            graph
            logits, _, _ = self.model.shared_step(graph, test=True)
    
            node_probs = torch.softmax(logits[0], dim=1).cpu().numpy() 
            node_probs_rounded = np.where(node_probs >= 0.5, 1, 0)
            # node_preds = np.argmax(node_probs, axis=1)
            node_preds = np.argmax(node_probs_rounded, axis=1)
            func_prob = torch.softmax(logits[1], dim=1).cpu().numpy()[0]
            func_pred = np.argmax(func_prob)
            node_labels = graph.ndata['_VULN'].cpu().numpy()
            func_label = graph.ndata['_FVULN'][0].item()
            
            line_numbers = graph.ndata['_LINE'].cpu().numpy()
            
            self.function_data.append({
                'CWE_ID': graph_data['CWE_ID'],
                'func_id': graph_id,
                'func_label': func_label,
                'func_pred': func_pred,
                'func_prob_0': func_prob[0],
                'func_prob_1': func_prob[1]
            })
            
            for i in range(len(node_labels)):
                self.statement_data.append({
                    'CWE_ID': graph_data['CWE_ID'],
                    'func_id': graph_id,
                    'line_number': line_numbers[i],
                    'node_label': node_labels[i],
                    'node_pred': node_preds[i],
                    'node_prob_0': node_probs[i][0],
                    'node_prob_1': node_probs[i][1]
                })
            
            if graph_data['CWE_ID'] not in self.cwe_data:
                self.cwe_data[graph_data['CWE_ID']] = {
                    'func_labels': [],
                    'func_preds': [],
                    'func_probs': [], 
                    'node_labels': [],
                    'node_preds': [],
                    'node_probs': []   
                }
            
            self.cwe_data[graph_data['CWE_ID']]['func_labels'].append(func_label)
            self.cwe_data[graph_data['CWE_ID']]['func_preds'].append(func_pred)
            self.cwe_data[graph_data['CWE_ID']]['func_probs'].append(func_prob[1])  
            self.cwe_data[graph_data['CWE_ID']]['node_labels'].extend(node_labels)
            self.cwe_data[graph_data['CWE_ID']]['node_preds'].extend(node_preds)
            self.cwe_data[graph_data['CWE_ID']]['node_probs'].extend(node_probs[:, 1])  
    
    def _generate_function_level_csv(self):
        """Generate CSV with function-level predictions and line numbers."""
        func_df = pd.DataFrame(self.function_data)
        stmt_df = pd.DataFrame(self.statement_data)
        
        stmt_grouped = stmt_df.groupby(['CWE_ID', 'func_id']).agg({
            'line_number': list,
            'node_label': list,
            'node_pred': list
        }).reset_index()
        
        merged_df = pd.merge(func_df, stmt_grouped, on=['CWE_ID', 'func_id'])
        output_path = os.path.join(self.output_dir, "function_level_predictions.csv")
        columns_to_drop = ['node_labels', 'node_probs', 'node_preds', 'func_probs',
                           'func_prob_0','func_prob_1','line_number','node_label','node_pred']
        existing_columns = [col for col in columns_to_drop if col in merged_df.columns]
        merged_df = merged_df.drop(columns=existing_columns)
        merged_df.reset_index(drop=True, inplace=True)
        merged_df.to_csv(output_path, index=False)
        print(f"Saved function-level predictions to {output_path}")
    
    def _generate_cwe_level_csv(self):
        """Generate CSV with CWE-level predictions and metrics."""
        cwe_results = []
        
        for cwe_id, data in self.cwe_data.items():
            func_metrics = self._calculate_metrics_array(
                data['func_labels'], data['func_preds'])
            stmt_metrics = self._calculate_metrics_array(
                data['node_labels'], data['node_preds'])
            
            if len(data['func_labels']) > 0:
                precision, recall, _ = precision_recall_curve(
                    data['func_labels'], data['func_probs'])
                func_pr_auc = auc(recall, precision)
            else:
                func_pr_auc = float('nan')
            
            if len(data['node_labels']) > 0:
                precision, recall, _ = precision_recall_curve(
                    data['node_labels'], data['node_probs'])
                stmt_pr_auc = auc(recall, precision)
            else:
                stmt_pr_auc = float('nan')
            
            cwe_results.append({
                'CWE_ID': cwe_id,
                'num_functions': len(data['func_labels']),
                'num_statements': len(data['node_labels']),
                **{f'func_{k}': v for k, v in func_metrics.items()},
                **{f'stmt_{k}': v for k, v in stmt_metrics.items()},
                'func_pr_auc': func_pr_auc,
                'stmt_pr_auc': stmt_pr_auc
            })
        
        output_path = os.path.join(self.output_dir, "cwe_level_metrics.csv")
        columns_to_drop = ['node_labels', 'node_probs', 'node_preds', 'func_probs']
        cwe_results = pd.DataFrame(cwe_results)
        existing_columns = [col for col in columns_to_drop if col in cwe_results.columns]
        cwe_results = cwe_results.drop(columns=existing_columns)
        cwe_results.reset_index(drop=True, inplace=True)
        cwe_results.to_csv(output_path, index=False)
        print(f"Saved CWE-level metrics to {output_path}")
    
    def _calculate_metrics(self):
        """Calculate and save overall metrics."""
        
        all_func_labels = [d['func_label'] for d in self.function_data]
        all_func_preds = [d['func_pred'] for d in self.function_data]
        all_func_probs = [d['func_prob_1'] for d in self.function_data]  
        
        all_node_labels = [d['node_label'] for d in self.statement_data]
        all_node_preds = [d['node_pred'] for d in self.statement_data]
        all_node_probs = [d['node_prob_1'] for d in self.statement_data] 
        
       
        func_metrics = self._calculate_metrics_array(all_func_labels, all_func_preds)
        stmt_metrics = self._calculate_metrics_array(all_node_labels, all_node_preds)
        
        
        if len(all_func_labels) > 0:
            precision, recall, _ = precision_recall_curve(all_func_labels, all_func_probs)
            func_pr_auc = auc(recall, precision)
        else:
            func_pr_auc = float('nan')
        
    
        if len(all_node_labels) > 0:
            precision, recall, _ = precision_recall_curve(all_node_labels, all_node_probs)
            stmt_pr_auc = auc(recall, precision)
        else:
            stmt_pr_auc = float('nan')
        
        func_metrics['pr_auc'] = func_pr_auc
        stmt_metrics['pr_auc'] = stmt_pr_auc
        
        pd.DataFrame([func_metrics]).to_csv(
            os.path.join(self.output_dir, "function_level_metrics.csv"), 
            index=False
        )
        pd.DataFrame([stmt_metrics]).to_csv(
            os.path.join(self.output_dir, "statement_level_metrics.csv"), 
            index=False
        )
        
        print("Saved overall metrics files")
    
    def _calculate_metrics_array(self, true_labels, pred_labels):
        """Calculate metrics for given true and predicted labels."""
        return {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='macro'), 
            'recall': recall_score(true_labels, pred_labels, average='macro'), 
            'f1': f1_score(true_labels, pred_labels, average='macro'), 
           # 'roc_auc': roc_auc_score(true_labels, pred_labels, average='macro'), 
            # 'mcc': matthews_corrcoef(true_labels, pred_labels)
        }


if __name__ == "__main__":
    analyzer = VulnerabilityAnalyzer(
        output_dir=imp.outputs_dir(),
        max_epochs=200 # 130 # 30, 150
    )
    analyzer.run()
    
torch.cuda.empty_cache()
