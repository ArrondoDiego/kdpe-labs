"""
======================================================================
EXAM CHEATSHEET - KNOWLEDGE DISCOVERY & PATTERN EXTRACTION
======================================================================
Indice:
1. Setup e Import comuni
2. NetworkX: Analisi di base e Centralità
3. PyTorch Geometric: Creazione Grafo e Data Object
4. Modelli GNN (GCN, GraphSAGE) per Node Classification
5. Training Loop Standard (PyTorch)
6. Node2Vec (Embeddings topologici)
7. Recommender Systems (Matrix Factorization / Embeddings)
8. Knowledge Graphs (R-GCN e TransE)
9. Similarità e LSH (Locality Sensitive Hashing)
10. PCA (PRINCIPAL COMPONENT ANALYSIS) E VISUALIZZAZIONE
======================================================================
"""

# ====================================================================
# 1. SETUP E IMPORT COMUNI
# ====================================================================
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric (PyG)
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, RGCNConv, TransE
from torch_geometric.nn import Node2Vec

# Sklearn per metriche e split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42) # Fissa il seed per riproducibilità

# ====================================================================
# 2. NETWORKX: ANALISI E CENTRALITA'
# ====================================================================
def networkx_basics(edge_list_path):
    # Creazione grafo da pandas
    # df = pd.read_csv(edge_list_path)
    # G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph())
    
    G = nx.karate_club_graph() # Esempio
    
    # Metriche di base
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Calcolo delle centralità
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # Estrarre il nodo più importante
    top_node = max(pagerank, key=pagerank.get)
    return G, top_node

# ====================================================================
# 3. PYTORCH GEOMETRIC: COSTRUIRE IL GRAFO
# ====================================================================
def build_pyg_data():
    # edge_index: shape [2, num_edges], tipo torch.long
    edge_index = torch.tensor([[0, 1, 1, 2], 
                               [1, 0, 2, 1]], dtype=torch.long)
    
    # Node features: shape [num_nodes, num_features]
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    
    # Labels (es. classificazione): shape [num_nodes]
    y = torch.tensor([0, 1, 0], dtype=torch.long)
    
    # Maschere per train/test
    train_mask = torch.tensor([True, True, False], dtype=torch.bool)
    test_mask = torch.tensor([False, False, True], dtype=torch.bool)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

# ====================================================================
# 4. MODELLI GNN (GCN, GraphSAGE)
# ====================================================================
class GNNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, model_type="GCN"):
        super(GNNClassifier, self).__init__()
        
        if model_type == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif model_type == "SAGE":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
            
    def forward(self, x, edge_index):
        # Primo layer + ReLU + Dropout
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        
        # Secondo layer
        out = self.conv2(h, edge_index)
        return out # Restituisce i logit (non applicare softmax se usi CrossEntropyLoss)

# ====================================================================
# 5. TRAINING LOOP STANDARD (PyTorch)
# ====================================================================
def train_node_classifier(model, data, epochs=100, lr=0.01):
    model = model.to(DEVICE)
    data = data.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            # Valutazione
            model.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

# ====================================================================
# 6. NODE2VEC
# ====================================================================
def run_node2vec(data):
    model = Node2Vec(data.edge_index, embedding_dim=64, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(DEVICE)
    
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    model.train()
    for epoch in range(1, 51): # Solitamente bastano 50-100 epoche
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Estrai gli embeddings finali
    embeddings = model()
    return embeddings

# ====================================================================
# 7. RECOMMENDER SYSTEMS (Embeddings Utente-Item)
# ====================================================================
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_nodes=num_users, embedding_dim=embedding_dim)
        self.item_emb = nn.Embedding(num_nodes=num_items, embedding_dim=embedding_dim)
        
    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        # Prodotto scalare tra utente e item (Dot Product)
        return (u * i).sum(dim=1) 
    
    def predict(self, user_idx, item_idx):
        return torch.sigmoid(self.forward(user_idx, item_idx))

# ====================================================================
# 8. KNOWLEDGE GRAPHS (R-GCN e TransE)
# ====================================================================
class RGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, num_classes, num_relations):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_dim) # Features iniziali
        self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.emb.weight
        h = F.relu(self.conv1(x, edge_index, edge_type))
        out = self.conv2(h, edge_index, edge_type)
        return out

def train_transe(head_index, rel_type, tail_index, num_nodes, num_relations):
    model = TransE(num_nodes=num_nodes, num_relations=num_relations, hidden_channels=32).to(DEVICE)
    loader = model.loader(head_index, rel_type, tail_index, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(100):
        for h, r, t in loader:
            optimizer.zero_grad()
            loss = model.loss(h, r, t)
            loss.backward()
            optimizer.step()
    return model

# ====================================================================
# 9. LSH E SIMILARITA' (Jaccard)
# ====================================================================
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# (Per LSH avanzato, in genere si usano librerie come datasketch, 
# ma all'esame spesso si chiede di implementare Jaccard o MinHash manuale)

# ====================================================================
# 10. PCA (PRINCIPAL COMPONENT ANALYSIS) E VISUALIZZAZIONE
# ====================================================================
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# --- Metodo 1: Scikit-Learn (Standard, ideale per array NumPy) ---
def apply_sklearn_pca(features, n_components=2):
    """
    Applica la PCA usando scikit-learn.
    features: array numpy o tensore convertito in numpy (es. x.cpu().numpy())
    """
    pca = PCA(n_components=n_components)
    embeddings_2d = pca.fit_transform(features)
    
    # Stampa la varianza spiegata (utile se richiesto all'esame)
    explained_var = pca.explained_variance_ratio_
    print(f"Varianza spiegata totale (primi {n_components} PC): {explained_var.sum():.2%}")
    
    return embeddings_2d

# --- Metodo 2: PyTorch SVD (Direttamente dai tuoi lab per R-GCN) ---
def pca_pytorch_svd(x):
    """
    Calcola la PCA proiettando le prime 2 componenti tramite SVD.
    x: tensore PyTorch di shape [num_nodes, hidden_dim]
    """
    # 1. Centra i dati sottraendo la media
    x = x - x.mean(0, keepdim=True) 
    # 2. Calcola la Singular Value Decomposition
    u, s, v = torch.linalg.svd(x, full_matrices=False) 
    # 3. Proietta la matrice sulle prime 2 componenti principali
    return (x @ v[:2].T).numpy() 

# --- Utility: Plot di base degli Embeddings ---
def plot_embeddings(embeddings_2d, labels, title="PCA Embeddings"):
    """
    embeddings_2d: array numpy di shape [num_nodes, 2]
    labels: array numpy o lista con le classi dei nodi
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], title="Classes")
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.show()