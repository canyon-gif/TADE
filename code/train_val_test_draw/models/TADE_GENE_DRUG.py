import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import einops
import numpy as np
from scipy.stats import pearsonr

class EfficientAdditiveAttention(nn.Module):
    def __init__(self, in_dims, token_dim, num_heads=1):
        super().__init__()
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = torch.nn.functional.normalize(self.to_query(x), dim=-1)
        key = torch.nn.functional.normalize(self.to_key(x), dim=-1)
        query_weight = query @ self.w_g
        A = torch.nn.functional.normalize(query_weight * self.scale_factor, dim=1)
        G = torch.sum(A * query, dim=1)
        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])
        out = self.final(self.Proj(G * key) + query)
        return out

class FunctionalGroupPrompt(nn.Module):
    def __init__(self, hidden_feats, fg2emb, fg_emb_dim=133):
        super(FunctionalGroupPrompt, self).__init__()
        self.fg_embedding = nn.Embedding(num_embeddings=len(fg2emb), embedding_dim=fg_emb_dim)
        self.linear = nn.Linear(fg_emb_dim, hidden_feats)
    
    def forward(self, fg_indices):
        fg_embs = self.fg_embedding(fg_indices)
        return fg_embs.mean(dim=1)

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, activation):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(in_feats, out_feats * num_heads)
        self.a = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.activation = activation

    def forward(self, graph, h):
        h = self.linear(h).view(-1, self.num_heads, self.linear.out_features // self.num_heads)
        graph.ndata['h'] = h
        graph.apply_edges(fn.u_add_v('h', 'h', 'e'))
        e = self.activation(torch.matmul(torch.tanh(graph.edata['e'][:, None, :]), self.a.transpose(1, 2)).squeeze(1))
        attention = F.softmax(e, dim=1)
        graph.edata['a'] = attention.unsqueeze(-1)
        graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h'))
        return graph.ndata['h'].mean(dim=1)

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, activation=F.elu):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(in_feats, hidden_feats, num_heads, activation)
        self.layer2 = GATLayer(hidden_feats, hidden_feats, num_heads, activation)
        self.linear = nn.Linear(hidden_feats, hidden_feats)

    def forward(self, g, features):
        h = self.layer1(g, features)
        h = self.linear(h.mean(1))
        h = self.layer2(g, h)
        return g.edata['a'], h

class Classifier(nn.Module):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(dim, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.out(x))

class DrugGenePredictor(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, genomic_feats, fg2emb):
        super(DrugGenePredictor, self).__init__()
        self.drug_encoder = GAT(in_feats, hidden_feats, num_heads)
        self.fg_prompt_module = FunctionalGroupPrompt(hidden_feats, fg2emb)
        self.genomic_attention = EfficientAdditiveAttention(in_dims=genomic_feats, token_dim=hidden_feats)
        self.classifier = Classifier(133 + 133)
    
    def forward(self, g, genomic_feats, fg_indices):
        device = genomic_feats.device
        _, drug_feats = self.drug_encoder(g, g.ndata['h']) 
        drug_feats = drug_feats.mean(dim=[0, 1]).unsqueeze(0).expand(genomic_feats.size(0), -1)
        
        genomic_feats = self.genomic_attention(genomic_feats.unsqueeze(1)).squeeze(1)
        fg_prompt = self.fg_prompt_module(fg_indices.to(device))
        drug_feats = drug_feats + fg_prompt
        
        drug_np = drug_feats.cpu().detach().numpy()
        geno_np = genomic_feats.cpu().detach().numpy()
        correlations = [pearsonr(drug_np[i], geno_np[i])[0] for i in range(drug_np.shape[0])]
        
        correlations = torch.tensor(correlations, device=device, dtype=torch.float32).unsqueeze(1)
        boosting_factors = 1 + torch.sigmoid(correlations) * 1.0 
        
        combined_feats = torch.cat([drug_feats * boosting_factors, genomic_feats * boosting_factors], dim=1)
        return self.classifier(combined_feats)