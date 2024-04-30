import torch
import torch.nn as nn

class FMDINOloss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_ij = None
        self.f_features_ij = None
        self.f_reg_ij = None
        self.f_cons = None
        self.device_id = 0
        
    def forward(self,T,feat_x,eval_x,evec_x,feat_y,eval_y,evec_y,Y_x,Y_y,sigma,lam,miu):

        batch_size = feat_x.size(0)  # 获取批量大小
        # cal wij
        self.w_ij = 1  # torch.exp(-torch.norm(feat_x - feat_y, p=2)**2/2*sigma**2)
    
        # align feat
        proj_x = torch.matmul(evec_x.transpose(1, 2), feat_x.view(batch_size, -1, feat_x.shape[-1]))  # shape (batch_size, eigen_num, num_nodes)
        proj_y = torch.matmul(evec_y.transpose(1, 2), feat_y.view(batch_size, -1, feat_y.shape[-1]))  # shape (batch_size, eigen_num, num_nodes)
        self.f_features_ij = torch.norm((torch.matmul(T, proj_x) - proj_y), p="fro").sum()
        self.f_features_ij = self.f_features_ij * 0.1 
    
        # regularizer
        diff_matrix = eval_x.unsqueeze(2) - eval_y.unsqueeze(1) 
        diff_matrix = torch.abs(diff_matrix)
        diff_matrix = diff_matrix * T
        self.f_reg_ij = torch.norm(diff_matrix, p="fro").pow(2).sum() 
      
        # consistency
        self.f_cons = torch.norm(torch.bmm(T, Y_x) - Y_y, p="fro").pow(2).sum() 
   
        return self.w_ij * self.f_features_ij + self.f_reg_ij * miu * self.w_ij + self.f_cons * lam * self.w_ij
