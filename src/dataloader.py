import os
from os import listdir
from os.path import isfile, join
from itertools import permutations

import torch
from torch.utils.data import Dataset
from src.functional_map_gpu import Functional_Map
from pathlib import Path


class FMVIDataset(Dataset):
    def __init__(self,root,feat_dim,eigen_num,device_id = 0,batch_size = 1,root_cons = None,tss = False):
        super().__init__()
        self.root = root
        self.feat_dim = feat_dim
        self.eigen_num = eigen_num
        self.batch_size = batch_size
        self.samples = [join(root, f) for f in listdir(root) if isfile(join(root, f)) and f.endswith(".pt") and not f.endswith("_mask.pt") and not f.startswith("Transition_Matrix") and not f.endswith("_img.pt") and not f.endswith("mask1.pt") and not f.endswith("mask2.pt")] 
        self.samples = sorted(self.samples,key = lambda x:int(str(x.split("/")[-1].split(".")[0].replace("_img","")))) if not tss else sorted(self.samples)

        self.combinations = list(permutations(range(len(self.samples)), 2))
        self.sample_num = len(self.samples)
        assert self.sample_num % 2 == 0
        self.combinations = [[i,i+1] for i in range(0,self.sample_num,2)]
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.fmap = None
        self.eval_cache = {}
        if root_cons:
            self.cons_samples = [join(root_cons, f) for f in listdir(root_cons) if isfile(join(root_cons, f)) and f.endswith(".pt") and not f.endswith("_mask.pt") and not f.startswith("Transition_Matrix")  and not f.endswith("_img.pt") and not f.endswith("mask1.pt") and not f.endswith("mask2.pt")]
            self.cons_samples = sorted(self.cons_samples,key = lambda x:int(str(x.split("/")[-1].split(".")[0].replace("_img","")))) if not tss else sorted(self.cons_samples)   
        else:
            self.cons_samples = self.samples
    
    
    def loader(self,path):
        """
        path : directory : Path object or string
        """
        feat = torch.load(path, map_location = self.device)
        return feat
    
    def cons_block(self,i,j,transiton_matrix,omega,eigen_num = None,device = None):
        
        transiton_matrix = transiton_matrix.squeeze(0)
        omega = omega.squeeze(0)
        image_num = 2
        block = torch.zeros((self.eigen_num, self.eigen_num),device = self.device)
        cur_transition = transiton_matrix.squeeze(0)
        if i == 1:
            cur_transition = cur_transition.t()
    
        if i==j:
            for k in range(image_num):
                if k != i:
                    
                    block += omega[k,i] * (torch.eye(eigen_num,device = device) + transiton_matrix.t() @ transiton_matrix)
                    
        else:
            block = - omega[i,j] *(cur_transition.t()  + transiton_matrix.t())
      
        
        if torch.isnan(block).any():
            print(omega[i,j])
            print(transiton_matrix)
            raise ValueError("block has nan")
        return block
    
    def compute_Y(self,W):
      
        batch_size = W.shape[0]
        eigen_num = self.eigen_num
        W_ = W.detach()
        Y = torch.randn((batch_size, 2 * eigen_num, 2 * eigen_num), requires_grad=True, device=self.device)
        def objective(Y):
            batch_traces = torch.einsum('bij, bji -> b', [Y.transpose(1, 2), torch.matmul(W_, Y)])
            return torch.sum(batch_traces)
        for _ in range(1000):
            optimizer_Y = torch.optim.Adam([Y], lr=0.01)
            optimizer_Y.zero_grad()
            loss = objective(Y)
            loss.backward()
            optimizer_Y.step()
        with torch.no_grad():
            # Apply Gram-Schmidt orthogonalization to each Y[batch_idx]
            for batch_idx in range(batch_size):
                Y[batch_idx] = torch.linalg.qr(Y[batch_idx])[0]
        return Y
    
    
    def __getitem__(self, index):

        batch_data = []

        for _ in range(self.batch_size):
            batch_index = index * self.batch_size + _
        
            if batch_index not in self.eval_cache:
                idx1, idx2 = self.combinations[batch_index]
                path1, path2 = self.samples[idx1], self.samples[idx2]
                mask_path1, mask_path2 = path1.replace(".pt", "_mask.pt"), path2.replace(".pt", "_mask.pt")
                feat_x = self.loader(path1).permute(1, 2, 0)
                feat_y = self.loader(path2).permute(1, 2, 0)
                fmap1 = Functional_Map(feat_x, feat_y, eigen_num=205, dim=self.feat_dim, use_built_in=False, use_dino=False,device_id = self.device_id)
                eval_x = fmap1.src_eigenvalues[:self.eigen_num]
                evec_x = fmap1.src_basis[:,:self.eigen_num]
                eval_y = fmap1.tgt_eigenvalues[:self.eigen_num]
                evec_y = fmap1.tgt_basis[:,:self.eigen_num]
            
                mask1 = self.loader(mask_path1) if os.path.exists(mask_path1) else torch.zeros_like(feat_x[:,:,0])
                mask2 = self.loader(mask_path2) if os.path.exists(mask_path2) else torch.zeros_like(feat_y[:,:,0])
 
                self.eval_cache[batch_index] = {'feat_x': feat_x, 'eval_x': eval_x, 'evec_x': evec_x, 'feat_y': feat_y, 'eval_y': eval_y, 'evec_y': evec_y, 'fmap1': fmap1, 'mask1': mask1, 'mask2': mask2}
            
            # Add the data to the batch
            batch_data.append(self.eval_cache[batch_index])

        # Stack the data in the batch
        batch = {
            'feat_x': torch.stack([item['feat_x'] for item in batch_data]),
            'eval_x': torch.stack([item['eval_x'] for item in batch_data]),
            'evec_x': torch.stack([item['evec_x'] for item in batch_data]),
            'feat_y': torch.stack([item['feat_y'] for item in batch_data]),
            'eval_y': torch.stack([item['eval_y'] for item in batch_data]),
            'evec_y': torch.stack([item['evec_y'] for item in batch_data]),
            'mask1': torch.stack([item['mask1'] for item in batch_data]),
            'mask2': torch.stack([item['mask2'] for item in batch_data]),
            'fmap1': [item['fmap1'] for item in batch_data]  # You can choose to keep this list of objects as-is
        }

        return batch    
    
    def __len__(self):
       
        return len(self.combinations)//self.batch_size
