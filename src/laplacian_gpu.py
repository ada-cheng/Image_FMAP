import torch
from .dino_to_graph_gpu import Image_Graph, check_and_derive_sigma
import torch_geometric
import torch_geometric.utils.get_laplacian
import networkx as nx
import numpy as np
#import cupy as cp
import scipy
#from cupyx.scipy.sparse.linalg import eigsh
#from cupyx.scipy.sparse import csr_matrix
import torch.nn.functional as F

class Laplacian():
    def __init__(self, graph, laplacian_type='normalized_cut',device_id = 0):
        self.graph = graph # a data object
        self.num_nodes = graph.num_nodes
        self.laplacian_type = laplacian_type
        self.Laplacian = None 
        self.eigenvalues = None # not all eigen_values
        self.eigenvectors = None
        self.device_id = device_id
        self.device = torch.device(f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu')
        self.compute_laplacian()
        
    def compute_laplacian(self):
        """
        compute the laplacian matrix of the graph
        """
        if self.laplacian_type == 'normalized_cut':
            e_i = self.graph.edge_index
            e_w = self.graph.edge_attr

            e_i_undirected = torch.cat([e_i, e_i.flip(dims=[0])], dim=1)
       
            e_w_undirected = torch.cat([e_w, e_w], dim=0)
        
            self.Laplacian = torch_geometric.utils.get_laplacian(e_i_undirected, e_w_undirected, normalization='sym')
            
        elif self.laplacian_type == 'unnormalized':
            e_i = self.graph.edge_index
            e_w = self.graph.edge_attr
            e_i_undirected = torch.cat([e_i, e_i.flip(dims=[0])], dim=1)
            e_w_undirected = torch.cat([e_w, e_w], dim=0)
            self.Laplacian = torch_geometric.utils.get_laplacian(e_i, e_w)
        else:
            raise ValueError("Unimplemented.")
    
    def compute_spectra(self,eigen_num):
        """
        returns the smallest eigen_num eigenvalues and eigenvectors
        eigenvectors : (num_nodes, eigen_num)
        """
        if self.Laplacian is None:
            self.compute_laplacian()
    
        coo = torch.sparse_coo_tensor(self.Laplacian[0], self.Laplacian[1], size=(self.num_nodes, self.num_nodes))
        self.eigenvalues, self.eigenvectors = torch.lobpcg(coo, k=eigen_num, largest=False)
        self.eigenvectors = torch.linalg.qr(self.eigenvectors)[0]
        self.eigenvectors = F.normalize(self.eigenvectors, p=2, dim=0).to(self.device)
        print("eigenvectors:", torch.sum(self.eigenvectors))
        return self.eigenvalues, self.eigenvectors
        
        
   
    def project_functions(self, eigen_num, functions):
        """
        project functions onto the eigenbasis
        function : (num_nodes, dim)
        
        """
        if self.eigenvalues is None or self.eigenvectors is None:
            self.compute_spectra(eigen_num)
        return torch.matmul(self.eigenvectors[:, :eigen_num].T, functions) #shape (eigen_num, dim)
            


