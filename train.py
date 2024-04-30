import argparse
import os
import torch
from src.loss import FMDINOloss
from src.model import CrossAttentionRefinementNet
from src.dataloader import FMVIDataset
import torch.nn.functional as F
import logging
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=f"src/ifmap.log",  # Choose the log file name and path
                    filemode='w')

# Define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)  # Choose the log level for console output

# Set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger('').addHandler(console)

def train_epoch(args):
    device = torch.device(f'cuda:{args.device_id}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    device_ids = [args.device_id]
    
    use_dino = False
    if not 'sd' in args.root_cons.lower():
        use_dino = True 
        
    dataset = FMVIDataset(args.root,args.cons_feat_dim,args.eigen_num,device_ids[0],args.batch_size,args.root_cons, args.TSS,args.case,args.Tool, use_dino)
    model_1 = CrossAttentionRefinementNet(n_in=args.cons_feat_dim, num_head=4, gnn_dim=512,  n_layers=2, cross_sampling_ratio=1, attention_type="normal").to(device)

    Transition_Matrix = nn.Parameter(torch.stack([torch.eye(200,device=torch.device(f'cuda:{args.device_id}'))]*1))
    lr_model = args.lr_model
    lr_matrix = args.lr
    param_group = [{'params': model_1.parameters(), 'lr': lr_model}, {'params': Transition_Matrix, 'lr': lr_matrix}]
    optimizer = torch.optim.Adam(param_group)
    criterion = FMDINOloss().to(device)
    batch_size = args.batch_size
    
    print("Start training...")
    
    f_feat = []
    f_reg = []
    f_cons = []
 
    for epoch in tqdm(range(args.epochs)):
        # calculate omega
        omega = torch.zeros((batch_size,2,2), device = device)
        for batch_idx in range(batch_size):
            for i in range(2):
                for j in range(2):
                    sample_idx_i = batch_idx*2+i
                    sample_idx_j = batch_idx*2+j
                    feat_i = dataset.loader(dataset.samples[sample_idx_i]) #loss
                    feat_j = dataset.loader(dataset.samples[sample_idx_j])
                   
                    feat_i = feat_i.reshape((1,-1,args.cons_feat_dim))
                    feat_j = feat_j.reshape((1,-1,args.cons_feat_dim))
                    num_patch = feat_i.shape[1] if feat_i.shape[1]<feat_j.shape[1] else feat_j.shape[1]

                    norm = torch.norm(feat_i[:,:num_patch,:]-feat_j[:,:num_patch,:],p=2)
               
                    args.sigma = torch.median(norm)
                  
                    omega[batch_idx,i,j] = torch.exp(-norm**2/(2*args.sigma**2) ) if i!=j else 1

        W = torch.zeros((batch_size, args.eigen_num * 2, args.eigen_num * 2), device = device)
        for batch_idx in range(batch_size):
            for i in range(2):
                for j in range(2):
                    W[batch_idx, i*args.eigen_num:(i+1)*args.eigen_num, j*args.eigen_num:(j+1)*args.eigen_num] = dataset.cons_block(i,j,Transition_Matrix[batch_idx,:,:],omega[batch_idx],args.eigen_num,device)
        Y = dataset.compute_Y(W, epoch)

        model_1.train()
     
     
        for i, batch in enumerate(dataset):
           
            lam =  0 if epoch<500 else args.lam

            feat_x,eval_x,evec_x,feat_y,eval_y,evec_y = batch['feat_x'],batch['eval_x'],batch['evec_x'],batch['feat_y'],batch['eval_y'],batch['evec_y'] # basis
            feat_x,eval_x,evec_x,feat_y,eval_y,evec_y = feat_x.to(device),eval_x.to(device),evec_x.to(device),feat_y.to(device),eval_y.to(device),evec_y.to(device)
            mask_x,mask_y = batch['mask1'],batch['mask2']
            mask_x,mask_y = mask_x.to(device),mask_y.to(device)
            _,h1,w1,_ = feat_x.shape
            _,h2,w2,_ = feat_y.shape
            feat_x,feat_y = feat_x.reshape((batch_size,h1*w1,args.cons_feat_dim)),feat_y.reshape((batch_size,h2*w2,args.cons_feat_dim))
        
            feat_x_list = []
            feat_y_list = []
            feat_x_1,feat_y_1 = model_1(feat_x[0].unsqueeze(0),feat_y[0].unsqueeze(0),mask_x[0].unsqueeze(0),mask_y[0].unsqueeze(0))
            feat_x_list.append(feat_x_1)
            feat_y_list.append(feat_y_1)

            feat_x = torch.stack(feat_x_list)
            feat_y = torch.stack(feat_y_list)
      
            
            
            
            feat_x = feat_x.reshape((batch_size,h1,w1,args.cons_feat_dim))
            feat_y = feat_y.reshape((batch_size,h2,w2,args.cons_feat_dim))
            
            idx_x = 0
            idx_y = 1
            Y_x,Y_y = Y[torch.arange(batch_size), idx_x * args.eigen_num:(idx_x + 1) * args.eigen_num, :], Y[torch.arange(batch_size), idx_y * args.eigen_num:(idx_y + 1) * args.eigen_num, :]

            loss = criterion(Transition_Matrix,feat_x,eval_x,evec_x,feat_y,eval_y,evec_y,Y_x,Y_y,args.sigma,lam,args.miu)
            f_feat.append((criterion.f_features_ij).cpu().detach().numpy())
            f_reg.append((criterion.f_reg_ij*args.miu).cpu().detach().numpy())
            f_cons.append((criterion.f_cons*args.lam).cpu().detach().numpy())
            
            loss.backward()
            torch.cuda.empty_cache()
            # keep data  in [0,1]
            optimizer.step()
            Transition_Matrix.data.clamp_(0,1)
            optimizer.zero_grad()
            if i % 100 == 0:
                pass
                #plot the loss

    
    for idx in range(batch_size):
        h1,w1,_ = dataset.eval_cache[idx]['feat_x'].shape # basis feat
        h2,w2,_ = dataset.eval_cache[idx]['feat_y'].shape
        feat_from = dataset.eval_cache[idx]['feat_x']
        feat_from = feat_from.reshape((h1*w1,args.cons_feat_dim))
        evec_1 = dataset.eval_cache[idx]['evec_x']
        feat_proj = torch.matmul(evec_1.T, feat_from)
        transfer_feat = torch.matmul(Transition_Matrix[idx,:,:], feat_proj)
        evec_2 = dataset.eval_cache[idx]['evec_y']
        inverse_transfer_data = torch.matmul(torch.linalg.pinv(evec_2).T, transfer_feat)
        res_feat = inverse_transfer_data.reshape((h2, w2, -1))
     
        torch.save(res_feat, f"{args.save_dir}/{args.name}_1999_batch_{idx}_test.pt")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--feat_dim', type=int, default=768)
    parser.add_argument('--cons_feat_dim', type=int, default=768)
    parser.add_argument('--eigen_num', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sigma', type=float, default=1e-3)
    parser.add_argument('--lam', type=float, default=1e-3)
    parser.add_argument('--miu', type=float, default=5)
    parser.add_argument('--eta', type=float, default = 1)   
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--name', type=str, default="FMAP")
    parser.add_argument('--mode',type=str, default="correspondence")
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--root_cons', type = str)  
    parser.add_argument('--TSS', type = bool, default = False)
    parser.add_argument('--Tool', type = bool, default = False)
    parser.add_argument('--device_id', type = int)
    parser.add_argument('--lr_model', type=float, default=1e-9)
    parser.add_argument('--case', type = int)
    args = parser.parse_args()
    train_epoch(args)
    
    
    
