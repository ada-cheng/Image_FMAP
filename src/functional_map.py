import numpy as np
import torch
from matplotlib import pyplot as plt
from laplacian import Laplacian
from dino_to_graph import Image_Graph
import cvxpy as cp
from pca_visualizer import pca_

def l1_and_frobenius_norms_cvx(sourcefeat, targetfeat): # F.matmul(sourcefeat) = targetfeat
    F = cp.Variable((sourcefeat.shape[0], targetfeat.shape[0]))
    residual = cp.norm(cp.matmul(F, sourcefeat) - targetfeat, 'fro')
    problem = cp.Problem(cp.Minimize(residual), [F >= 0])
    # Solve the problem
    problem.solve()

    return F.value, residual.value   

methods = {"l1":l1_and_frobenius_norms_cvx}


class Functional_Map:
    def __init__(self, src_data, tgt_data, eigen_num = "10", method = "l1",dim = 384):
        """
        method : "l1"
        """
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.eigen_num = eigen_num
        self.method = method
        self.dim = dim
        self.laplacian_matrix = None # laplacian of src
        self.src_basis = self.get_basis_functions(src_data)
        self.transition_matrix = self.cal_functional_map()
        
    
    def get_basis_functions(self,data):
        feat_graph = Image_Graph(data, "2d_grid")
        self.laplacian_matrix = Laplacian(feat_graph.graph)
        _,basis_functions = self.laplacian_matrix.compute_spectra(self.eigen_num)
        return basis_functions
    
    def cal_functional_map(self):
        if self.laplacian_matrix is None:
            raise ValueError("laplace matrix is None")
        proj_data1 = self.laplacian_matrix.project_functions(self.eigen_num, self.src_data.reshape((-1,self.dim)))
        proj_data2 = self.laplacian_matrix.project_functions(self.eigen_num, self.tgt_data.reshape((-1,self.dim)))
        transition_matrix,_ = methods[self.method](proj_data1, proj_data2)
        return transition_matrix
    
    
    def project(self, data):
        if self.transition_matrix is None:
            raise ValueError("transition matrix is None")
        if self.laplacian_matrix is None:
            raise ValueError("laplace matrix is None")
        if self.src_basis is None:
            raise ValueError("src basis is None")
        proj_data = self.laplacian_matrix.project_functions(self.eigen_num, data.reshape((-1,self.dim)))
        transfer_data = np.matmul(self.transition_matrix, proj_data)
        inverse_transfer_data = np.matmul(np.linalg.pinv(self.src_basis).T, transfer_data)
        return transfer_data, inverse_transfer_data


    
if __name__ == "__main__":
    pass
    