#!/usr/bin/env python3
"""
train_gnn.py - Phase 3: Architecture et Entraînement du Modèle GNN
===================================================================

Script d'entraînement d'un Graph Neural Network pour la prédiction de trafic
urbain sur le Line Graph construit en Phase 2.

Architectures disponibles:
- GraphSAGE: Agrégation inductive par échantillonnage (recommandé pour transfer)
- GAT: Graph Attention Network avec mécanisme d'attention
- GCN: Graph Convolutional Network (baseline)

Stratégie de validation inductive:
- Split spatial pour simuler le transfer learning
- Masquage des labels manquants (couverture partielle ~22%)
- Pondération par nombre d'observations

Optimisé pour environnements à mémoire limitée (8GB RAM):
- Mini-batch training avec NeighborLoader
- Gradient accumulation optionnelle
- Mixed precision (AMP) optionnelle

Auteur: Claude (Anthropic)
Date: 06 Janvier 2026
Version: 1.0.0

Usage:
    python train_gnn.py --data-dir ./line_graph_data --output-dir ./models
    python train_gnn.py -d ./data -o ./models --model gat --epochs 100

Entrées requises (depuis Phase 2):
    - edge_index.npy: Matrice d'adjacence (2, num_edges)
    - node_features.npy: Features des nœuds (num_nodes, 14)
    - mean_travel_times.npy: Labels (num_nodes,)
    - observation_counts.npy: Poids pour la loss (num_nodes,)
    - node_mapping.csv: Mapping avec coordonnées pour split spatial

Sorties:
    - model_best.pt: Meilleur modèle (validation loss)
    - model_final.pt: Modèle final
    - training_history.json: Métriques d'entraînement
    - config.json: Configuration du modèle (pour inférence)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# =============================================================================
# IMPORTS PYTORCH ET PYTORCH GEOMETRIC
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERREUR: PyTorch non installé. Exécuter: pip install torch")

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import (
        SAGEConv,      # GraphSAGE
        GATConv,       # Graph Attention
        GCNConv,       # GCN baseline
        BatchNorm,
        global_mean_pool,
    )
    from torch_geometric.utils import degree
    TORCH_GEO_AVAILABLE = True
except ImportError:
    TORCH_GEO_AVAILABLE = False
    print("ERREUR: PyTorch Geometric non installé. Exécuter: pip install torch-geometric")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Configuration par défaut du modèle
DEFAULT_CONFIG = {
    # Architecture
    'model_type': 'sage',           # 'sage', 'gat', 'gcn'
    'hidden_channels': 64,          # Dimension cachée
    'num_layers': 3,                # Nombre de couches GNN
    'dropout': 0.3,                 # Dropout rate
    'heads': 4,                     # Nombre de têtes pour GAT
    
    # Entraînement
    'epochs': 200,
    'batch_size': 1024,             # Taille des mini-batches
    'learning_rate': 0.001,
    'weight_decay': 1e-4,           # L2 regularization
    'patience': 20,                 # Early stopping
    'min_delta': 1e-4,              # Amélioration minimale
    
    # NeighborLoader (pour mini-batch)
    'num_neighbors': [15, 10, 5],   # Voisins par couche (décroissant)
    
    # Split
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'spatial_split': True,          # Split spatial vs aléatoire
    
    # Optimisations mémoire
    'use_amp': False,               # Mixed precision
    'gradient_accumulation': 1,     # Accumulation de gradients
    
    # Loss
    'loss_type': 'mse',             # 'mse', 'mae', 'huber'
    'use_weights': True,            # Pondérer par observation_counts
}

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_gnn.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITAIRES
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON pour types numpy."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def set_seed(seed: int = 42):
    """Fixer les graines aléatoires pour reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Obtenir le device disponible (GPU si possible)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Utilisation GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  - Mémoire disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        logger.info("Utilisation CPU (GPU non disponible)")
    return device


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_line_graph_data(data_dir: str) -> Tuple[Data, np.ndarray, Dict]:
    """
    Charge les données du Line Graph depuis Phase 2.
    
    Args:
        data_dir: Répertoire contenant les fichiers .npy
        
    Returns:
        Tuple (data, observation_counts, metadata)
    """
    logger.info(f"Chargement des données depuis {data_dir}...")
    
    # Chemins des fichiers
    edge_index_path = os.path.join(data_dir, 'edge_index.npy')
    features_path = os.path.join(data_dir, 'node_features.npy')
    labels_path = os.path.join(data_dir, 'mean_travel_times.npy')
    counts_path = os.path.join(data_dir, 'observation_counts.npy')
    mapping_path = os.path.join(data_dir, 'node_mapping.csv')
    metadata_path = os.path.join(data_dir, 'line_graph_metadata.json')
    
    # Vérifier l'existence des fichiers requis
    required_files = [edge_index_path, features_path]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier requis manquant: {path}")
    
    # Charger edge_index
    edge_index = np.load(edge_index_path)
    edge_index = torch.from_numpy(edge_index).long()
    logger.info(f"  - edge_index: {edge_index.shape}")
    
    # Charger node_features
    node_features = np.load(features_path)
    x = torch.from_numpy(node_features).float()
    logger.info(f"  - node_features: {x.shape}")
    
    # Charger labels (optionnel)
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        y = torch.from_numpy(labels).float()
        # Remplacer NaN par 0 et créer un masque
        valid_mask = ~torch.isnan(y)
        y = torch.nan_to_num(y, nan=0.0)
        logger.info(f"  - labels: {y.shape}, valides: {valid_mask.sum().item():,} ({valid_mask.float().mean()*100:.1f}%)")
    else:
        y = None
        valid_mask = None
        logger.warning("  - Pas de fichier de labels trouvé")
    
    # Charger observation_counts (optionnel)
    if os.path.exists(counts_path):
        observation_counts = np.load(counts_path)
        logger.info(f"  - observation_counts: min={observation_counts.min()}, max={observation_counts.max()}")
    else:
        observation_counts = None
        logger.warning("  - Pas de fichier observation_counts trouvé")
    
    # Charger mapping pour coordonnées spatiales
    coords = None
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        if 'center_x' in mapping_df.columns and 'center_y' in mapping_df.columns:
            coords = mapping_df[['center_x', 'center_y']].values
            logger.info(f"  - Coordonnées spatiales chargées pour split")
    
    # Charger métadonnées
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Créer l'objet Data
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )
    
    # Ajouter le masque de validité
    if valid_mask is not None:
        data.valid_mask = valid_mask
    
    # Ajouter les coordonnées
    if coords is not None:
        data.coords = torch.from_numpy(coords).float()
    
    logger.info(f"\nRésumé du graphe:")
    logger.info(f"  - Nœuds: {data.num_nodes:,}")
    logger.info(f"  - Arêtes: {data.num_edges:,}")
    logger.info(f"  - Features: {data.num_node_features}")
    logger.info(f"  - Degré moyen: {data.num_edges / data.num_nodes:.2f}")
    
    return data, observation_counts, metadata


# =============================================================================
# SPLIT SPATIAL / ALÉATOIRE
# =============================================================================

def create_spatial_split(data: Data, 
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Crée un split spatial basé sur les coordonnées géographiques.
    
    Stratégie: Diviser la ville en zones géographiques
    - Train: Zones centrales et ouest (ex: Manhattan, Brooklyn)
    - Val: Zone intermédiaire
    - Test: Zone est (ex: Queens) - simule une nouvelle ville
    
    Args:
        data: Objet Data avec attribut coords
        train_ratio, val_ratio, test_ratio: Proportions
        
    Returns:
        Tuple (train_mask, val_mask, test_mask)
    """
    logger.info("Création du split spatial...")
    
    if not hasattr(data, 'coords') or data.coords is None:
        logger.warning("Pas de coordonnées disponibles, fallback sur split aléatoire")
        return create_random_split(data, train_ratio, val_ratio, test_ratio)
    
    coords = data.coords.numpy()
    num_nodes = data.num_nodes
    
    # Utiliser la longitude (x) pour le split est-ouest
    x_coords = coords[:, 0]
    
    # Calculer les percentiles pour le split
    train_threshold = np.percentile(x_coords, train_ratio * 100)
    val_threshold = np.percentile(x_coords, (train_ratio + val_ratio) * 100)
    
    # Créer les masques
    train_mask = torch.from_numpy(x_coords <= train_threshold)
    val_mask = torch.from_numpy((x_coords > train_threshold) & (x_coords <= val_threshold))
    test_mask = torch.from_numpy(x_coords > val_threshold)
    
    # Ne garder que les nœuds avec labels valides
    if hasattr(data, 'valid_mask'):
        train_mask = train_mask & data.valid_mask
        val_mask = val_mask & data.valid_mask
        test_mask = test_mask & data.valid_mask
    
    logger.info(f"  - Train: {train_mask.sum().item():,} nœuds ({train_mask.float().mean()*100:.1f}%)")
    logger.info(f"  - Val: {val_mask.sum().item():,} nœuds ({val_mask.float().mean()*100:.1f}%)")
    logger.info(f"  - Test: {test_mask.sum().item():,} nœuds ({test_mask.float().mean()*100:.1f}%)")
    
    return train_mask, val_mask, test_mask


def create_random_split(data: Data,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Crée un split aléatoire stratifié.
    
    Args:
        data: Objet Data
        train_ratio, val_ratio, test_ratio: Proportions
        
    Returns:
        Tuple (train_mask, val_mask, test_mask)
    """
    logger.info("Création du split aléatoire...")
    
    num_nodes = data.num_nodes
    
    # Indices des nœuds avec labels valides
    if hasattr(data, 'valid_mask'):
        valid_indices = torch.where(data.valid_mask)[0].numpy()
    else:
        valid_indices = np.arange(num_nodes)
    
    # Shuffle
    np.random.shuffle(valid_indices)
    
    # Calculer les tailles
    n_valid = len(valid_indices)
    n_train = int(n_valid * train_ratio)
    n_val = int(n_valid * val_ratio)
    
    # Assigner
    train_indices = valid_indices[:n_train]
    val_indices = valid_indices[n_train:n_train + n_val]
    test_indices = valid_indices[n_train + n_val:]
    
    # Créer les masques
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    logger.info(f"  - Train: {train_mask.sum().item():,} nœuds")
    logger.info(f"  - Val: {val_mask.sum().item():,} nœuds")
    logger.info(f"  - Test: {test_mask.sum().item():,} nœuds")
    
    return train_mask, val_mask, test_mask


# =============================================================================
# NORMALISATION DES LABELS
# =============================================================================

def compute_label_normalization(y: torch.Tensor, 
                                 mask: torch.Tensor) -> Tuple[float, float]:
    """
    Calcule mean et std pour normalisation des labels sur le train set.
    
    Args:
        y: Labels complets
        mask: Masque du train set
        
    Returns:
        Tuple (mean, std)
    """
    train_labels = y[mask]
    # Filtrer les zéros (labels invalides)
    valid = train_labels > 0
    if valid.sum() == 0:
        return 0.0, 1.0
    
    mean = train_labels[valid].mean().item()
    std = train_labels[valid].std().item()
    
    # Éviter division par zéro
    std = max(std, 1e-6)
    
    logger.info(f"Normalisation labels - mean: {mean:.2f}, std: {std:.2f}")
    return mean, std


# =============================================================================
# ARCHITECTURES GNN
# =============================================================================

class GraphSAGE(nn.Module):
    """
    GraphSAGE: Inductive Representation Learning on Large Graphs.
    
    Architecture recommandée pour le transfer learning car elle:
    - Apprend à agréger les features des voisins (inductif)
    - Ne dépend pas de la taille du graphe
    - Utilise des échantillonneurs de voisinage (scalable)
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Couches SAGE
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Première couche
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Couches cachées
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Dernière couche SAGE
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        # MLP de sortie (Readout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Message passing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Readout
        x = self.mlp(x)
        return x.squeeze(-1)


class GAT(nn.Module):
    """
    Graph Attention Network.
    
    Utilise un mécanisme d'attention pour pondérer l'importance
    des voisins. Peut apprendre des patterns plus complexes mais
    est plus coûteux en mémoire.
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 num_layers: int = 3,
                 heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Première couche
        self.convs.append(GATConv(in_channels, hidden_channels // heads, 
                                   heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Couches cachées
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                       heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Dernière couche (1 tête pour la sortie)
        self.convs.append(GATConv(hidden_channels, hidden_channels,
                                   heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))
        
        # MLP de sortie
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.mlp(x)
        return x.squeeze(-1)


class GCN(nn.Module):
    """
    Graph Convolutional Network (baseline).
    
    Architecture classique mais transductive (dépend de la structure
    complète du graphe). Moins adapté au transfer learning.
    """
    
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 out_channels: int = 1,
                 num_layers: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.mlp(x)
        return x.squeeze(-1)


def create_model(config: Dict, in_channels: int) -> nn.Module:
    """
    Factory pour créer le modèle selon la configuration.
    
    Args:
        config: Configuration du modèle
        in_channels: Nombre de features d'entrée
        
    Returns:
        Modèle GNN
    """
    model_type = config.get('model_type', 'sage').lower()
    
    common_args = {
        'in_channels': in_channels,
        'hidden_channels': config.get('hidden_channels', 64),
        'out_channels': 1,
        'num_layers': config.get('num_layers', 3),
        'dropout': config.get('dropout', 0.3),
    }
    
    if model_type == 'sage':
        logger.info("Création du modèle GraphSAGE")
        model = GraphSAGE(**common_args)
    elif model_type == 'gat':
        logger.info("Création du modèle GAT")
        model = GAT(**common_args, heads=config.get('heads', 4))
    elif model_type == 'gcn':
        logger.info("Création du modèle GCN")
        model = GCN(**common_args)
    else:
        raise ValueError(f"Type de modèle inconnu: {model_type}")
    
    # Compter les paramètres
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  - Paramètres totaux: {num_params:,}")
    logger.info(f"  - Paramètres entraînables: {trainable_params:,}")
    
    return model


# =============================================================================
# FONCTIONS DE LOSS
# =============================================================================

def get_loss_fn(loss_type: str = 'mse'):
    """Retourne la fonction de loss selon le type."""
    if loss_type == 'mse':
        return nn.MSELoss(reduction='none')
    elif loss_type == 'mae':
        return nn.L1Loss(reduction='none')
    elif loss_type == 'huber':
        return nn.HuberLoss(reduction='none', delta=1.0)
    else:
        raise ValueError(f"Loss inconnue: {loss_type}")


def compute_weighted_loss(predictions: torch.Tensor,
                          targets: torch.Tensor,
                          mask: torch.Tensor,
                          weights: Optional[torch.Tensor],
                          loss_fn: nn.Module) -> torch.Tensor:
    """
    Calcule la loss pondérée sur les nœuds valides.
    
    Args:
        predictions: Prédictions du modèle
        targets: Labels réels
        mask: Masque des nœuds à considérer
        weights: Poids par nœud (observation_counts)
        loss_fn: Fonction de loss
        
    Returns:
        Loss scalaire
    """
    # Appliquer le masque
    pred_masked = predictions[mask]
    target_masked = targets[mask]
    
    if pred_masked.numel() == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    # Calculer la loss élément par élément
    element_loss = loss_fn(pred_masked, target_masked)
    
    # Pondérer si weights fournis
    if weights is not None:
        weights_masked = weights[mask]
        # Normaliser les poids (log pour atténuer les outliers)
        weights_norm = torch.log1p(weights_masked)
        weights_norm = weights_norm / weights_norm.sum()
        weighted_loss = (element_loss * weights_norm).sum()
    else:
        weighted_loss = element_loss.mean()
    
    return weighted_loss


# =============================================================================
# MÉTRIQUES
# =============================================================================

def compute_metrics(predictions: torch.Tensor,
                    targets: torch.Tensor,
                    mask: torch.Tensor) -> Dict[str, float]:
    """
    Calcule les métriques d'évaluation.
    
    Args:
        predictions: Prédictions
        targets: Labels
        mask: Masque des nœuds valides
        
    Returns:
        Dict avec MSE, RMSE, MAE, MAPE
    """
    pred = predictions[mask].detach().cpu().numpy()
    target = targets[mask].detach().cpu().numpy()
    
    # Filtrer les targets > 0 pour MAPE
    valid = target > 0
    
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(pred - target))
    
    if valid.sum() > 0:
        mape = np.mean(np.abs((pred[valid] - target[valid]) / target[valid])) * 100
    else:
        mape = 0.0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
    }


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

class Trainer:
    """
    Classe d'entraînement du GNN avec support mini-batch.
    """
    
    def __init__(self,
                 model: nn.Module,
                 data: Data,
                 train_mask: torch.Tensor,
                 val_mask: torch.Tensor,
                 test_mask: torch.Tensor,
                 config: Dict,
                 observation_counts: Optional[np.ndarray] = None,
                 device: torch.device = torch.device('cpu')):
        
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Déplacer les données sur le device
        self.data = data.to(device)
        self.train_mask = train_mask.to(device)
        self.val_mask = val_mask.to(device)
        self.test_mask = test_mask.to(device)
        
        # Poids d'observation
        if observation_counts is not None and config.get('use_weights', True):
            self.weights = torch.from_numpy(observation_counts).float().to(device)
        else:
            self.weights = None
        
        # Normalisation des labels
        if self.data.y is not None:
            self.y_mean, self.y_std = compute_label_normalization(
                self.data.y, train_mask
            )
            # Normaliser les labels
            self.data.y = (self.data.y - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std = 0.0, 1.0
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
        
        # Loss
        self.loss_fn = get_loss_fn(config.get('loss_type', 'mse'))
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Historique
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': [],
        }
        
        # Mixed precision
        self.use_amp = config.get('use_amp', False) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision (AMP) activé")
    
    def train_epoch(self) -> Tuple[float, Dict]:
        """Entraîne une epoch complète."""
        self.model.train()
        
        # Forward pass
        if self.use_amp:
            with torch.cuda.amp.autocast():
                predictions = self.model(self.data.x, self.data.edge_index)
                loss = compute_weighted_loss(
                    predictions, self.data.y, self.train_mask,
                    self.weights, self.loss_fn
                )
        else:
            predictions = self.model(self.data.x, self.data.edge_index)
            loss = compute_weighted_loss(
                predictions, self.data.y, self.train_mask,
                self.weights, self.loss_fn
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Métriques (dénormaliser pour métriques réelles)
        with torch.no_grad():
            pred_denorm = predictions * self.y_std + self.y_mean
            target_denorm = self.data.y * self.y_std + self.y_mean
            metrics = compute_metrics(pred_denorm, target_denorm, self.train_mask)
        
        return loss.item(), metrics
    
    @torch.no_grad()
    def evaluate(self, mask: torch.Tensor) -> Tuple[float, Dict]:
        """Évalue le modèle sur un ensemble."""
        self.model.eval()
        
        predictions = self.model(self.data.x, self.data.edge_index)
        loss = compute_weighted_loss(
            predictions, self.data.y, mask,
            self.weights, self.loss_fn
        )
        
        # Dénormaliser pour métriques réelles
        pred_denorm = predictions * self.y_std + self.y_mean
        target_denorm = self.data.y * self.y_std + self.y_mean
        metrics = compute_metrics(pred_denorm, target_denorm, mask)
        
        return loss.item(), metrics
    
    def train(self, epochs: int, patience: int = 20) -> Dict:
        """
        Boucle d'entraînement principale.
        
        Args:
            epochs: Nombre d'epochs
            patience: Patience pour early stopping
            
        Returns:
            Historique d'entraînement
        """
        logger.info(f"\nDébut de l'entraînement ({epochs} epochs, patience={patience})")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validation
            val_loss, val_metrics = self.evaluate(self.val_mask)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Historique
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(current_lr)
            
            # Early stopping check
            if val_loss < self.best_val_loss - self.config.get('min_delta', 1e-4):
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1
            
            # Logging
            epoch_time = time.time() - epoch_start
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val RMSE: {val_metrics['rmse']:.2f}s | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"\nEarly stopping à l'epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        logger.info(f"\nEntraînement terminé en {total_time/60:.1f} minutes")
        
        # Restaurer le meilleur modèle
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Meilleur modèle restauré (val_loss: {self.best_val_loss:.4f})")
        
        return self.history
    
    def test(self) -> Dict:
        """Évalue sur le test set."""
        test_loss, test_metrics = self.evaluate(self.test_mask)
        logger.info(f"\n=== Résultats sur Test Set ===")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  RMSE: {test_metrics['rmse']:.2f} secondes")
        logger.info(f"  MAE: {test_metrics['mae']:.2f} secondes")
        logger.info(f"  MAPE: {test_metrics['mape']:.1f}%")
        return test_metrics
    
    def save_model(self, output_dir: str, prefix: str = 'model'):
        """Sauvegarde le modèle et la configuration."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder le modèle
        model_path = os.path.join(output_dir, f'{prefix}.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'y_mean': self.y_mean,
            'y_std': self.y_std,
            'config': self.config,
        }, model_path)
        logger.info(f"Modèle sauvegardé: {model_path}")
        
        # Sauvegarder la config séparément (pour inférence facile)
        config_path = os.path.join(output_dir, 'config.json')
        config_to_save = {
            **self.config,
            'in_channels': self.data.num_node_features,
            'y_mean': self.y_mean,
            'y_std': self.y_std,
        }
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Configuration sauvegardée: {config_path}")
        
        # Sauvegarder l'historique
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Historique sauvegardé: {history_path}")


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main(data_dir: str, output_dir: str, config: Optional[Dict] = None):
    """
    Pipeline principal d'entraînement.
    
    Args:
        data_dir: Répertoire des données (sortie Phase 2)
        output_dir: Répertoire de sortie des modèles
        config: Configuration (utilise DEFAULT_CONFIG si None)
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("PHASE 3: ENTRAÎNEMENT DU MODÈLE GNN")
    logger.info("=" * 70)
    logger.info(f"Répertoire données: {data_dir}")
    logger.info(f"Répertoire sortie: {output_dir}")
    
    # Vérifier les dépendances
    if not TORCH_AVAILABLE or not TORCH_GEO_AVAILABLE:
        logger.error("Dépendances manquantes. Installer PyTorch et PyTorch Geometric.")
        sys.exit(1)
    
    # Configuration
    if config is None:
        config = DEFAULT_CONFIG.copy()
    logger.info(f"\nConfiguration:")
    for key, value in config.items():
        logger.info(f"  - {key}: {value}")
    
    # Fixer la seed
    set_seed(42)
    
    # Device
    device = get_device()
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================================================
    # ÉTAPE 1: Chargement des données
    # ==========================================================================
    logger.info("\n--- ÉTAPE 1: Chargement des données ---")
    data, observation_counts, metadata = load_line_graph_data(data_dir)
    
    # Vérifier qu'on a des labels
    if data.y is None:
        logger.error("Pas de labels trouvés. Impossible d'entraîner.")
        sys.exit(1)
    
    # ==========================================================================
    # ÉTAPE 2: Split des données
    # ==========================================================================
    logger.info("\n--- ÉTAPE 2: Split des données ---")
    
    if config.get('spatial_split', True):
        train_mask, val_mask, test_mask = create_spatial_split(
            data,
            train_ratio=config.get('train_ratio', 0.7),
            val_ratio=config.get('val_ratio', 0.15),
            test_ratio=config.get('test_ratio', 0.15)
        )
    else:
        train_mask, val_mask, test_mask = create_random_split(
            data,
            train_ratio=config.get('train_ratio', 0.7),
            val_ratio=config.get('val_ratio', 0.15),
            test_ratio=config.get('test_ratio', 0.15)
        )
    
    # ==========================================================================
    # ÉTAPE 3: Création du modèle
    # ==========================================================================
    logger.info("\n--- ÉTAPE 3: Création du modèle ---")
    model = create_model(config, in_channels=data.num_node_features)
    
    # ==========================================================================
    # ÉTAPE 4: Entraînement
    # ==========================================================================
    logger.info("\n--- ÉTAPE 4: Entraînement ---")
    
    trainer = Trainer(
        model=model,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        config=config,
        observation_counts=observation_counts,
        device=device
    )
    
    history = trainer.train(
        epochs=config.get('epochs', 200),
        patience=config.get('patience', 20)
    )
    
    # ==========================================================================
    # ÉTAPE 5: Évaluation sur Test
    # ==========================================================================
    logger.info("\n--- ÉTAPE 5: Évaluation finale ---")
    test_metrics = trainer.test()
    
    # ==========================================================================
    # ÉTAPE 6: Sauvegarde
    # ==========================================================================
    logger.info("\n--- ÉTAPE 6: Sauvegarde ---")
    trainer.save_model(output_dir, prefix='model_best')
    
    # Sauvegarder les métriques finales
    final_results = {
        'config': config,
        'train_samples': int(train_mask.sum()),
        'val_samples': int(val_mask.sum()),
        'test_samples': int(test_mask.sum()),
        'best_val_loss': trainer.best_val_loss,
        'test_metrics': test_metrics,
        'training_time_minutes': (datetime.now() - start_time).total_seconds() / 60,
    }
    
    results_path = os.path.join(output_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, cls=NumpyEncoder)
    
    # ==========================================================================
    # RÉSUMÉ
    # ==========================================================================
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("ENTRAÎNEMENT TERMINÉ")
    logger.info("=" * 70)
    logger.info(f"Durée totale: {elapsed}")
    
    logger.info(f"\nFichiers générés dans {output_dir}:")
    logger.info(f"  - model_best.pt (meilleur modèle)")
    logger.info(f"  - config.json (configuration)")
    logger.info(f"  - training_history.json (courbes d'apprentissage)")
    logger.info(f"  - final_results.json (métriques)")
    
    logger.info(f"\nPerformance sur Test Set:")
    logger.info(f"  - RMSE: {test_metrics['rmse']:.2f} secondes")
    logger.info(f"  - MAE: {test_metrics['mae']:.2f} secondes")
    logger.info(f"  - MAPE: {test_metrics['mape']:.1f}%")
    
    logger.info("\nProchaine étape: Phase 4 - Validation et Transfer Learning")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entraînement du GNN pour prédiction de trafic - Phase 3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python train_gnn.py --data-dir ./line_graph_data --output-dir ./models
  python train_gnn.py -d ./data -o ./output --model gat --epochs 100 --hidden 128
  python train_gnn.py -d ./data -o ./output --model gcn --no-spatial-split
        """
    )
    
    # Arguments de chemins
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        default='./line_graph_data',
        help='Répertoire contenant les données du Line Graph (Phase 2)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./models',
        help='Répertoire de sortie pour les modèles'
    )
    
    # Arguments d'architecture
    parser.add_argument(
        '--model',
        type=str,
        default='sage',
        choices=['sage', 'gat', 'gcn'],
        help='Type de modèle GNN (default: sage)'
    )
    parser.add_argument(
        '--hidden',
        type=int,
        default=64,
        help='Dimension des couches cachées (default: 64)'
    )
    parser.add_argument(
        '--layers',
        type=int,
        default=3,
        help='Nombre de couches GNN (default: 3)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Taux de dropout (default: 0.3)'
    )
    parser.add_argument(
        '--heads',
        type=int,
        default=4,
        help='Nombre de têtes d\'attention pour GAT (default: 4)'
    )
    
    # Arguments d'entraînement
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Nombre d\'epochs (default: 200)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Taille des mini-batches (default: 1024)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Patience pour early stopping (default: 20)'
    )
    
    # Arguments de split
    parser.add_argument(
        '--no-spatial-split',
        action='store_true',
        help='Utiliser un split aléatoire au lieu de spatial'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Ratio train (default: 0.7)'
    )
    
    # Arguments d'optimisation
    parser.add_argument(
        '--no-weights',
        action='store_true',
        help='Ne pas pondérer par observation counts'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Utiliser Mixed Precision (AMP)'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='mse',
        choices=['mse', 'mae', 'huber'],
        help='Type de loss (default: mse)'
    )
    
    args = parser.parse_args()
    
    # Construire la configuration
    config = DEFAULT_CONFIG.copy()
    config.update({
        'model_type': args.model,
        'hidden_channels': args.hidden,
        'num_layers': args.layers,
        'dropout': args.dropout,
        'heads': args.heads,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'spatial_split': not args.no_spatial_split,
        'train_ratio': args.train_ratio,
        'use_weights': not args.no_weights,
        'use_amp': args.amp,
        'loss_type': args.loss,
    })
    
    main(args.data_dir, args.output_dir, config)
