import torch
from dino_to_graph import Image_Graph, check_and_derive_sigma
import networkx as nx
import numpy as np
import scipy

class Laplacian():
    def __init__(self, graph, laplacian_type='normalized_cut'):
        self.graph = graph
        self.laplacian_type = laplacian_type
        self.Laplacian = None # scipy.sparse array
        self.eigenvalues = None
        self.eigenvectors = None
        self.compute_laplacian()
    
    def compute_laplacian(self):
        """
        compute the laplacian matrix of the graph
        """
        if self.laplacian_type == 'normalized_cut':
            self.Laplacian = nx.normalized_laplacian_matrix(self.graph)
        elif self.laplacian_type == 'unnormalized':
            self.Laplacian = nx.laplacian_matrix(self.graph)
    
    def compute_spectra(self,eigen_num):
        """
        returns the smallest eigen_num eigenvalues and eigenvectors
        eigenvectors : (num_nodes, eigen_num)
        """
        if self.Laplacian is None:
            self.compute_laplacian()
        if self.eigenvalues is None or self.eigenvectors is None:
            self.eigenvalues, self.eigenvectors = scipy.sparse.linalg.eigsh(self.Laplacian, k=eigen_num, which='SM')
            
        return self.eigenvalues[:eigen_num], self.eigenvectors[:,:eigen_num]
    
    def project_functions(self, eigen_num, functions):
        """
        project functions onto the eigenbasis
        function : (num_nodes, dim)
        
        """
        if self.eigenvalues is None or self.eigenvectors is None:
            self.compute_spectra(eigen_num)
        return np.matmul(self.eigenvectors[:, :eigen_num].T, functions) #shape (eigen_num, dim)

    
    
if __name__ == "__main__":
    
    pass
   
    