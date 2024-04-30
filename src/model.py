import torch
import torch.nn as nn
from copy import deepcopy
from flash_attn import flash_attn_func
import math

def positional_encoding(seq_length, feature_dim):
    position = torch.arange(0, seq_length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * -(math.log(10000.0) / feature_dim))
    pos_enc = torch.zeros(1, seq_length, feature_dim)
    pos_enc[0, :, 0::2] = torch.sin(position * div_term)
    pos_enc[0, :, 1::2] = torch.cos(position * div_term)
    return pos_enc

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
  
    scores = torch.matmul(query.permute(0, 2, 3, 1), key.permute(0, 2, 1, 3)) / dim ** 0.5
     
    torch.cuda.empty_cache()
    prob = torch.nn.functional.softmax(scores, dim=-1) # shape: (batch_size, num_heads, num_points, num_points)
    torch.cuda.empty_cache()
  
    result = torch.matmul(prob, value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    torch.cuda.empty_cache()
    return result, prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [ll(x).view(batch_dim, self.dim, self.num_heads, -1) for ll, x in zip(self.proj, (query, key, value))]
        # q : (batch_size, seqlen, nheads, headdim)
        
        q = query.permute(0, 3, 2, 1).reshape(batch_dim,-1, self.num_heads, self.dim).half()
        k = key.permute(0, 3, 2, 1).reshape(batch_dim, -1, self.num_heads, self.dim).half()
        v = value.permute(0, 3, 2, 1).reshape(batch_dim, -1, self.num_heads, self.dim).half()
        out = flash_attn_func(q, k, v)
        x = out.permute(0, 2, 3, 1).reshape(batch_dim, self.dim * self.num_heads, -1).float()
    #    x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))


    
class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)

        return  self.mlp(torch.cat([x, message], dim=1)) 

class CrossAttentionRefinementNet(nn.Module):
    def __init__(self, n_in=128, num_head=4, gnn_dim=512,  n_layers=2, cross_sampling_ratio=0.15, attention_type="normal"):
        super().__init__()

        self.attention_type = attention_type
        if attention_type == "normal":
            additional_dim = 0 # 这个目前没有什么用
        else:
            raise Exception("Attention type not recognized")

        self.n_in = n_in
        self.cross_sampling_ratio = cross_sampling_ratio
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(AttentionalPropagation(gnn_dim + additional_dim, num_head))
        self.layers = nn.ModuleList(self.layers)
        self.first_lin = nn.Linear(n_in, gnn_dim + additional_dim)
        self.last_lin = nn.Linear(gnn_dim + additional_dim, n_in + additional_dim)


    def forward(self, features_x, features_y, mask_x, mask_y):
        """
        coords0: shape: (batch_size, num_points, 2)
        features_x: shape: (batch_size, num_points, n_in)
        """
        
        
        seq_length = features_x.size(1)
        pos_enc = positional_encoding(seq_length, self.n_in)
        pos_enc = pos_enc.to(features_x.device)
        features_x = features_x + pos_enc
        seq_length = features_y.size(1)
        pos_enc = positional_encoding(seq_length, self.n_in)
        pos_enc = pos_enc.to(features_y.device)
        features_y = features_y + pos_enc
        
        
        desc0, desc1 = self.first_lin(features_x).transpose(1, 2), self.first_lin(features_y).transpose(1, 2) # shape: (batch_size, n_in, num_points)
        

        for layer in self.layers:
            if self.cross_sampling_ratio == 1:
                desc0 = desc0 + layer(desc0, desc1) # xi = xi + MLP([xi||mE→i])
                torch.cuda.empty_cache()
                desc1 = desc1 + layer(desc1, desc0)
                torch.cuda.empty_cache()
            else:
                sampled_desc0 = desc0[:, :, mask_x.view(-1) != 0]
                sampled_desc1 = desc1[:, :, mask_y.view(-1) != 0]
                torch.cuda.empty_cache()
                sampled_desc0 = sampled_desc0 + layer(sampled_desc0, sampled_desc1)
                torch.cuda.empty_cache()
                sampled_desc1 = sampled_desc1 + layer(sampled_desc1, sampled_desc0)
                torch.cuda.empty_cache()
                desc0[:, :, mask_x.view(-1) != 0] = sampled_desc0
                desc1[:, :, mask_y.view(-1) != 0] = sampled_desc1

        augmented_features_x = self.last_lin(desc0.transpose(1, 2))
        augmented_features_y = self.last_lin(desc1.transpose(1, 2))

        ref_feat_x, ref_feat_y = augmented_features_x[:, :, :self.n_in], augmented_features_y[:, :, :self.n_in]
        
        return ref_feat_x, ref_feat_y


