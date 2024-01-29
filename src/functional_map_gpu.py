import numpy as np
import torch
from matplotlib import pyplot as plt
from src.pca_visualizer import pca_
from src.laplacian_gpu import Laplacian
from src.dino_to_graph_gpu import Image_Graph
#import cvxpy as cp
import torch.nn.functional as F


def l1_and_frobenius_norms_cvx(sourcefeat, targetfeat): # F.matmul(sourcefeat) = targetfeat
    pass # TODO: implement this

methods = {"l1":l1_and_frobenius_norms_cvx}


class Functional_Map:
    def __init__(self, src_data, tgt_data, eigen_num = "10", method = "l1",dim = 384,src_basis = None, transition_matrix = None, use_built_in = True,use_dino = True,device_id = 0):
        """
        method : "l1"
        """
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.eigen_num = eigen_num
        self.method = method
        self.dim = dim
        self.laplacian_matrix = None # laplacian of src
        self.laplacian_matrix_ = None # laplacian of tgt
        self.device_id = device_id
        self.device = torch.device(f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu')
        self.use_dino = use_dino
        self.src_eigenvalues = None
        self.tgt_eigenvalues = None
        self.src_basis = self.get_basis_functions(src_data,self.use_dino) if src_basis is None else torch.load(src_basis, map_location = self.device)
        self.tgt_basis = self.get_basis_functions(tgt_data,self.use_dino,src = False) 
        if use_built_in:
            self.transition_matrix = self.cal_functional_map() if transition_matrix is None else torch.load(transition_matrix, map_location = self.device)
        else:
            self.transition_matrix = transition_matrix if transition_matrix is not None else None
        
    def get_basis_functions(self,data,use_dino=True,src = True):
        if use_dino:
            feat_graph = Image_Graph(data, "2d_grid")
        else:
            feat_graph = Image_Graph(data, "2d_grid",False)
        if src == True:
            self.laplacian_matrix = Laplacian(feat_graph.graph,device_id=self.device_id)
            self.src_eigenvalues,basis_functions = self.laplacian_matrix.compute_spectra(self.eigen_num)
        else:
            self.laplacian_matrix_ = Laplacian(feat_graph.graph,device_id=self.device_id)
            self.tgt_eigenvalues,basis_functions = self.laplacian_matrix_.compute_spectra(self.eigen_num)
        return basis_functions

    def cal_functional_map(self):
        if self.laplacian_matrix is None:
            raise ValueError("laplace matrix is None")
        if self.dim != self.src_data.shape[-1]:
            raise ValueError("dim is not equal to src_data.shape[-1]")
        proj_data1 = self.laplacian_matrix.project_functions(self.eigen_num, self.src_data.reshape((-1,self.dim)))
        proj_data2 = self.laplacian_matrix_.project_functions(self.eigen_num, self.tgt_data.reshape((-1,self.dim)))
        transition_matrix,_ = methods[self.method](proj_data1, proj_data2)
        print("res",torch.norm(torch.matmul(transition_matrix, proj_data1) - proj_data2, p = "fro"))
        return transition_matrix
    
    def project(self, data):
        if self.transition_matrix is None:
            raise ValueError("transition matrix is None")
        if self.src_basis is None:
            raise ValueError("src basis is None")
        self.src_basis = self.src_basis.to(self.device)
        self.tgt_basis = self.tgt_basis.to(self.device)
        self.transition_matrix = self.transition_matrix.to(self.device)
        data = data.to(self.device)
        proj_data = torch.matmul(self.src_basis.T, data.reshape((-1,self.dim)))
        transfer_data = torch.matmul(self.transition_matrix, proj_data)
        inverse_transfer_data = torch.matmul(torch.linalg.pinv(self.tgt_basis).T, transfer_data)
        return transfer_data, inverse_transfer_data
        
    
    
if __name__ == "__main__":
    pass