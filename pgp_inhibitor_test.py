
##### import libraries

import os
import numpy as np
np.bool = np.bool_
import pandas as pd
import logging
import matplotlib.pyplot as plt
from rdkit import Chem
import argparse
import torch
import mxnet
import pickle
import torchmetrics
from torch_geometric.loader import DataLoader

from torch.nn import Linear
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F 
from torch_geometric.nn import GraphConv, GCNConv
from torch_geometric.nn import global_add_pool as gap, ASAPooling as asp
from torch.nn import Dropout
from torch_geometric.data import Data, Batch

from deepchem_utils import encode_with_one_hot, atom_type_one_hot, get_hybridization_one_hot, construct_hydrogen_bonding_info, get_atom_formal_charge
from deepchem_utils import get_atom_is_in_aromatic_one_hot, get_atom_total_degree_one_hot, get_atom_total_num_Hs_one_hot, get_chirality_one_hot
from deepchem_utils import smiles_to_edge_indices


# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import pandas as pd
import torch
import argparse

# Argument parser
# Argument parser
parser = argparse.ArgumentParser(description="Run inhibitor test with configurable input files.")
parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
parser.add_argument("--model_weights", type=str, required=True, help="Path to the model weights file")
parser.add_argument("--results_csv", type=str, required=True, help="Path to save the results CSV file")
parser.add_argument("--output_dir", type=str, required=True, help="Parent directory to save all outputs")
args = parser.parse_args()


# Create directories dynamically
os.makedirs(args.output_dir, exist_ok=True)
barplots_dir = os.path.join(args.output_dir, "inhibitor_barplots")
os.makedirs(barplots_dir, exist_ok=True)

# prepare dataset

test_molecules = pd.read_csv(args.input_csv, names=['SMILES'])


class MolGraphConvFeaturizerWoLabels():

    def __init__(self):
        pass
    
    def check_valid_smiles(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError("Invalid SMILES string provided")
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 1:
            raise ValueError("The provided SMILES represents a single atom, not a compound.")

    def featurize_one(self, smile):
        
        self.check_valid_smiles(smile)
        
        encoder = MolGraphConvFeaturizerWoLabels()  
        atom_type = atom_type_one_hot(smile)
        formal_charge = get_atom_formal_charge(smile)
        hybridization = get_hybridization_one_hot(smile)
        hydrogen_bonding_info = construct_hydrogen_bonding_info(smile)
        aromatic = get_atom_is_in_aromatic_one_hot(smile)
        degree = get_atom_total_degree_one_hot(smile)
        num_Hs = get_atom_total_num_Hs_one_hot(smile)
        chirality = get_chirality_one_hot(smile)

        # Concatenate features
        node_features = torch.cat((atom_type, formal_charge, hybridization, hydrogen_bonding_info, 
                                           aromatic, degree, num_Hs, chirality), dim=1)
        
        return node_features 
    
    def get_edge_index(self, smile):

        encoder = MolGraphConvFeaturizerWoLabels()
        edge_index = smiles_to_edge_indices(smile)

        return edge_index
    
    def create_data_object(self, smile):
        node_features = self.featurize_one(smile)
        edge_index = self.get_edge_index(smile)
        data = Data(x=node_features, edge_index=edge_index)
        return data
    
    def get_graphs(self, smiles_series):
        
        smiles_list = smiles_series.tolist()
        
        graphs = []

        for smile in smiles_list:
            data_object = self.create_data_object(smile)
            graphs.append(data_object)
        
        graph_object = Batch.from_data_list(graphs)

        return graph_object
    
featurizerwolabels = MolGraphConvFeaturizerWoLabels()
mol_graphs = featurizerwolabels.get_graphs(test_molecules['SMILES'])

mol_graphs.x = mol_graphs.x.to(torch.float32).cuda()
mol_graphs.edge_index = mol_graphs.edge_index.cuda()

class GraphConvModel(torch.nn.Module):
    def __init__(self, embedding_size, num_classes):
        super(GraphConvModel, self).__init__()
        torch.manual_seed(185)

        self.initial_conv = GraphConv(mol_graphs.num_features, embedding_size)
        self.conv1 = GraphConv(embedding_size, embedding_size)
        self.conv2 = GraphConv(embedding_size, embedding_size)
        self.conv3 = GraphConv(embedding_size, embedding_size)

        #self.pool = asp(embedding_size, ratio=0.2)
        
        self.out = torch.nn.Linear(embedding_size, 1)  
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv1(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv2(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = self.conv3(hidden, edge_index)
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)

        hidden = gap(hidden, batch_index)
        
        out = self.out(hidden)
        return out

model = GraphConvModel(embedding_size=128, num_classes=2).to(device)
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

state_dict = torch.load(args.model_weights)

model.load_state_dict(state_dict, strict=True)

test_loader = DataLoader(mol_graphs, batch_size=1, shuffle=False)


y_prob = []
y_pred = []
threshold = 0.0

model.eval()
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch.x.float(), batch.edge_index.long(), batch.batch.long())
        predicted_labels = (outputs >= threshold).view(-1).int()
        predicted_probs = torch.sigmoid(outputs)
        y_pred += predicted_labels.tolist()
        y_prob += predicted_probs.view(-1).tolist()

# Create a DataFrame with the real and predicted labels
test_results = pd.DataFrame({'y_pred': y_pred, 'y_prob': y_prob})



def label_mols(x):
    if x == 1:
        return 'Inhibitor'
    else:
        return 'Non-Inhibitor'
    
test_results['status'] = test_results['y_pred'].apply(label_mols)

test_results_df = pd.concat([test_molecules['SMILES'], test_results], axis=1)

print(test_results_df)



""" save the results """

# Save the results
results_path = os.path.join(args.output_dir, "inhibitor_example_results.csv")
test_results_df.to_csv(results_path)


""" Explainer Model """

from torch_geometric.explain import Explainer, GNNExplainer, unfaithfulness
import random

# explainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(lr=0.001, epoch=700),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='binary_classification',
        task_level='graph',
        return_type='raw',
    ),
)

# Function to set the seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 121
all_explanations = []
unfaith_metrics = []

with torch.no_grad():
    with torch.set_grad_enabled(True):
        for data in test_loader:
            # Set the seed for reproducibility in each iteration
            set_seed(seed)
            
            # Extract the graph attributes from the DataLoader
            X = data.x.float()
            Edge_index = data.edge_index.long()
            Batch_index = data.batch.long()
        
            # Apply the explainer to get the explanations for the current graph
            explanation = explainer(x=X, edge_index=Edge_index, batch_index=Batch_index)
            metric = unfaithfulness(explainer, explanation)
        
            # Append explanations to the list
            all_explanations.append(explanation)
            unfaith_metrics.append(metric)

        test_explain_all = Batch.from_data_list(all_explanations)

def stack_score(graphs):
    scores = []

    for i in range(len(graphs)):
        feature_score = graphs[i].node_mask
        sums = feature_score.sum(dim=0)     # sum across the nodes for each feature
        scores.append(sums)
    
    stacked_tensors = torch.stack(scores)
    return stacked_tensors

test_explain_pm = stack_score(test_explain_all).cpu() # get the tensors for node-feature importance for each molecule. 

deepchem_features = ["Carbon", "Nitrogen", "Oxygen", "Fluorine", "Phosphorus", 
                     "Sulphur", "Chlorine", "Bromine", "Iodine", "Others",
                     "Formal Charge: -2", "Formal Charge: -1", "Formal Charge: 0", "Formal Charge: 1", "Formal Charge: 2", "Hyb. SP", "Hyb. SP2", "Hyb. SP3", "H-Bond Donor",
                     "H-Bond Acceptor", "Aromaticity", "Degree: 1", "Degree: 2", "Degree: 3", "Degree: 4", "Degree: 5", "Degree: 6", 
                     "No. of H: 0", "No. of H: 1", "No. of H: 2", "No. of H: 3", "Chirality: R", "Chirality: S"]
    
explain_df = pd.DataFrame(columns=deepchem_features)

explain_df = pd.DataFrame(test_explain_pm.numpy(), columns=deepchem_features)

# Create output directory
output_dir = os.path.join(args.output_dir, "inhibitor_barplots")
os.makedirs(output_dir, exist_ok=True)

for idx, row in explain_df.iterrows():
    plt.figure(figsize=(8, 6))
    
    # Sort features by their values in descending order
    row_sorted = row.sort_values(ascending=False)
    
    # Plot horizontal bar plot without solid black borders
    bars = plt.barh(row_sorted.index, row_sorted.values, color="red")
    
    # Extend the x-axis range dynamically
    x_max = row_sorted.max()  # Find the maximum value
    plt.xlim(0, x_max * 1.1)  # Extend the x-axis range by 10%
    
    # Dynamically adjust ylim to prevent bar clipping
    plt.ylim(-0.7, len(row_sorted) - 0.7)

    # Add values over each bar
    for bar in bars:
        plt.text(
            bar.get_width() + (x_max * 0.02),  # Position slightly to the right of the bar
            bar.get_y() + bar.get_height() / 2,  # Center vertically on the bar
            f"{bar.get_width():.2f}",  # Display the value with two decimals
            va="center",  # Align text vertically centered
            fontsize=10  # Font size for the text
        )
    
    # Add labels and title
    plt.title(f"Feature importance for Molecule Index:{idx}", fontsize=14, fontweight='bold')
    plt.xlabel("Value", fontsize=12, fontweight='bold')
    plt.ylabel("Feature", fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()  # Ensure the highest value is at the top
    plt.yticks(fontsize=10)  # Reduce font size of y-axis labels
    
    # Save the plot
    plt.tight_layout(pad=1.5)  # Adjust layout padding to avoid cutoff
    plot_path = os.path.join(output_dir, f"{idx}_barplot.png")
    plt.savefig(plot_path)
    plt.close()


